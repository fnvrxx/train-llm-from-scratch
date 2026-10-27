"""Focused invariants for all three experimental feed-forward variants."""

import copy
import unittest

import torch

from experiments.moe.model import convert_model as to_moe
from experiments.pruned.model import convert_model as to_pruned
from experiments.topk.model import convert_model as to_topk
from src.models.transformer import Transformer


class SparseExperimentsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.dense = Transformer(2, 8, 8, 32, 2)
        self.tokens = torch.randint(32, (2, 5))

    def test_pruned_masks_survive_checkpoint_and_mask_gradients(self):
        model = to_pruned(copy.deepcopy(self.dense), 0.5)
        mlp = model.attn_blocks[0].mlp
        self.assertEqual(int((mlp.hidden_mask == 0).sum()), mlp.hidden_mask.numel() // 2)
        logits, loss = model(self.tokens, self.tokens)
        self.assertEqual(tuple(logits.shape), (2, 5, 32))
        loss.backward()
        self.assertTrue(torch.all(mlp.hidden.weight.grad[mlp.hidden_mask == 0] == 0))
        restored = to_pruned(copy.deepcopy(self.dense), 0.25)
        restored.load_state_dict(model.state_dict(), strict=True)
        self.assertTrue(torch.equal(restored.attn_blocks[0].mlp.hidden_mask, mlp.hidden_mask))

    def test_topk_full_is_dense_and_partial_has_gradients(self):
        full = to_topk(copy.deepcopy(self.dense), 1.0)
        self.assertTrue(torch.allclose(full(self.tokens)[0], self.dense(self.tokens)[0]))
        partial = to_topk(copy.deepcopy(self.dense), 0.25)
        logits, loss = partial(self.tokens, self.tokens)
        self.assertEqual(tuple(logits.shape), (2, 5, 32))
        loss.backward()
        self.assertIsNotNone(partial.attn_blocks[0].mlp.proj.weight.grad)
        self.assertGreater(partial.attn_blocks[0].mlp.proj.weight.grad.abs().sum().item(), 0)

    def test_moe_all_experts_reproduce_dense_and_router_trains(self):
        all_experts = to_moe(copy.deepcopy(self.dense), 4, 4)
        for block in all_experts.attn_blocks:
            torch.nn.init.zeros_(block.mlp.gate.weight)
        self.assertTrue(torch.allclose(all_experts(self.tokens)[0], self.dense(self.tokens)[0], atol=1e-5))
        partial = to_moe(copy.deepcopy(self.dense), 4, 2)
        logits, loss = partial(self.tokens, self.tokens)
        self.assertEqual(tuple(logits.shape), (2, 5, 32))
        loss = loss + 0.01 * sum(block.mlp.aux_loss for block in partial.attn_blocks)
        loss.backward()
        self.assertGreater(partial.attn_blocks[0].mlp.gate.weight.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
