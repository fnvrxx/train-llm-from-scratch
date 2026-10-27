"""Pemeriksaan perilaku yang dapat dibuktikan tanpa korpus besar."""

from __future__ import annotations

import tempfile
import unittest
from math import isfinite
from pathlib import Path

import torch

from small_llm.data import load_stories, sample_batch
from small_llm.config import ExperimentConfig, SparseConfig, TrainConfig
from small_llm.generate import load_model
from small_llm.model import ModelConfig, StoryTransformer
from small_llm.sparse import MoEMLP, TopKMLP, active_connections, convert_model, rewire_set
from small_llm.tokenizer import END_OF_STORY, VOCAB_SIZE, decode, encode
from small_llm.train import train_variant


def tiny_model() -> StoryTransformer:
    return StoryTransformer(ModelConfig(context_length=16, n_embed=16, n_head=4, n_blocks=2))


class TokenizerAndDataTests(unittest.TestCase):
    def test_utf8_round_trip_and_story_end(self):
        text = "Hujan turun di desa."
        ids = encode(text, end_of_story=True)
        self.assertEqual(ids[-1], END_OF_STORY)
        self.assertEqual(decode(ids), text)

    def test_story_split_and_shift(self):
        with tempfile.TemporaryDirectory() as folder:
            for i in range(5):
                Path(folder, f"{i}.txt").write_text(f"Cerita {i}. " * 20, encoding="utf-8")
            corpus = load_stories(folder, seed=3)
            self.assertFalse(set(corpus.train_files) & set(corpus.valid_files))
            x, y = sample_batch(corpus.train, batch_size=2, context_length=8,
                                generator=torch.Generator().manual_seed(1), device="cpu")
            self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))


class ModelTests(unittest.TestCase):
    def test_generation_keeps_utf8_valid(self):
        class BiasedModel(StoryTransformer):
            def forward(self, tokens, targets=None):
                logits = torch.full((*tokens.shape, VOCAB_SIZE), -100.0,
                                    device=tokens.device)
                logits[..., 0x80] = 100.0  # Invalid as the first UTF-8 byte.
                logits[..., 0xE0] = 90.0
                logits[..., 0xA0] = 80.0
                logits[..., 0x20] = 70.0
                return logits, None

        model = BiasedModel(ModelConfig(context_length=8, n_embed=16, n_head=4, n_blocks=1))
        prefix = torch.tensor([[ord("A")]])
        three_bytes = model.generate(prefix, 3).squeeze(0).tolist()
        self.assertEqual(three_bytes[1:], [0xE0, 0xA0, 0x80])
        bytes(three_bytes).decode("utf-8", errors="strict")
        one_byte = model.generate(prefix, 1).squeeze(0).tolist()
        self.assertEqual(one_byte[1:], [0x20])

    def test_attention_cannot_read_future_token(self):
        torch.manual_seed(1)
        model = tiny_model().eval()
        first = torch.tensor([[1, 2, 3, 4]])
        changed = torch.tensor([[1, 2, 3, 200]])
        with torch.no_grad():
            a, _ = model(first)
            b, _ = model(changed)
        self.assertTrue(torch.allclose(a[:, :3], b[:, :3], atol=1e-6))

    def test_topk_all_neurons_matches_dense(self):
        torch.manual_seed(2)
        source = tiny_model()
        selected = TopKMLP(source.blocks[0].mlp, active_fraction=1.0)
        x = torch.randn(2, 3, 16)
        self.assertTrue(torch.allclose(source.blocks[0].mlp(x), selected(x)))

    def test_moe_all_experts_can_reconstruct_dense_mlp(self):
        torch.manual_seed(3)
        source = tiny_model().blocks[0].mlp
        moe = MoEMLP(source, n_experts=4, top_k=4)
        with torch.no_grad():
            moe.gate.weight.zero_()
        x = torch.randn(2, 3, 16)
        self.assertTrue(torch.allclose(source(x), moe(x), atol=1e-6))

    def test_set_rewire_keeps_density_and_clears_optimizer_state(self):
        torch.manual_seed(4)
        model = convert_model(tiny_model(), "set", {"density": 0.5})
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        tokens = torch.randint(0, 257, (2, 8))
        _, loss = model(tokens, tokens)
        loss.backward()
        optimizer.step()
        for block in model.blocks:
            block.mlp.enforce_masks()
        before = [block.mlp.hidden_mask.clone() for block in model.blocks]
        active_before = active_connections(model)
        changed = rewire_set(model, optimizer, fraction=0.1)
        self.assertGreater(changed, 0)
        self.assertEqual(active_before, active_connections(model))
        for block, old in zip(model.blocks, before):
            mask = block.mlp.hidden_mask
            positions = old ^ mask
            self.assertTrue(positions.any())
            for state in optimizer.state[block.mlp.hidden.weight].values():
                if isinstance(state, torch.Tensor) and state.shape == mask.shape:
                    self.assertTrue(torch.all(state[positions] == 0))

    def test_checkpoint_rebuilds_sparse_architecture(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = ModelConfig(context_length=16, n_embed=16, n_head=4, n_blocks=2)
            torch.manual_seed(5)
            model = convert_model(StoryTransformer(cfg), "pruned", {"sparsity": 0.5}).eval()
            path = Path(folder, "model.pt")
            torch.save({"model": model.state_dict(), "model_config": vars(cfg),
                        "sparse_config": {"sparsity": 0.5}, "variant": "pruned"}, path)
            restored = load_model(path, "cpu")
            tokens = torch.randint(0, 257, (2, 8))
            with torch.no_grad():
                expected, _ = model(tokens)
                actual, _ = restored(tokens)
            self.assertTrue(torch.allclose(expected, actual))


class TrainingTests(unittest.TestCase):
    def test_each_variant_trains_and_saves_a_loadable_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            stories = Path(folder, "stories")
            stories.mkdir()
            for i in range(5):
                Path(stories, f"{i}.txt").write_text(f"Cerita pendek nomor {i}. " * 15,
                                                      encoding="utf-8")
            cfg = ExperimentConfig(
                model=ModelConfig(context_length=16, n_embed=16, n_head=4, n_blocks=1),
                train=TrainConfig(data_dir=str(stories), out_dir=str(Path(folder, "output")),
                                  device="cpu", batch_size=2, grad_accum=2, steps=1,
                                  eval_every=1, eval_batches=1, log_every=1),
                sparse=SparseConfig(rewire_every=1),
            )
            for variant in ("dense", "pruned", "topk", "moe", "set"):
                with self.subTest(variant=variant):
                    summary = train_variant(cfg, variant)
                    self.assertTrue(isfinite(summary["valid_loss"]))
                    checkpoint = Path(cfg.train.out_dir, variant, "model.pt")
                    model = load_model(checkpoint, "cpu")
                    self.assertEqual(model.cfg, cfg.model)


if __name__ == "__main__":
    unittest.main()
