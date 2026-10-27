"""Small correctness checks intended for the machine that runs the experiment."""

import copy
import io
import unittest

import torch

from experiments.evolutionary.model import convert_model
from experiments.evolutionary.rewire import active_connections, enforce_model_masks, rewire_model
from src.models.transformer import Transformer


class EvolutionaryMLPTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.base = Transformer(2, 8, 8, 32, 2)
        self.model = convert_model(copy.deepcopy(self.base), density=0.25)

    def test_initial_density_and_backward(self):
        active, total = active_connections(self.model)
        self.assertEqual(active, total // 4)
        ids = torch.randint(32, (2, 5))
        logits, loss = self.model(ids, ids)
        self.assertEqual(tuple(logits.shape), (2, 5, 32))
        loss.backward()
        mlp = self.model.attn_blocks[0].mlp
        self.assertTrue(torch.all(mlp.hidden.weight.grad[~mlp.hidden_mask] == 0))

    def test_rewire_preserves_count_and_resets_moments(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        ids = torch.randint(32, (2, 5))
        self.model(ids, ids)[1].backward()
        optimizer.step()
        enforce_model_masks(self.model)
        mlp = self.model.attn_blocks[0].mlp
        before_mask = mlp.hidden_mask.clone()
        before_count = active_connections(self.model)
        changed = rewire_model(self.model, optimizer, fraction=0.2)
        self.assertGreater(changed, 0)
        self.assertEqual(active_connections(self.model), before_count)
        positions = before_mask ^ mlp.hidden_mask
        self.assertTrue(positions.any())
        self.assertTrue(torch.all(mlp.hidden.weight[positions] == 0))
        state = optimizer.state[mlp.hidden.weight]
        self.assertTrue(torch.all(state["exp_avg"][positions] == 0))
        self.assertTrue(torch.all(state["exp_avg_sq"][positions] == 0))

    def test_masks_survive_checkpoint_round_trip(self):
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        buffer.seek(0)
        restored = convert_model(copy.deepcopy(self.base), density=0.5)
        restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        for original, recovered in zip(self.model.attn_blocks, restored.attn_blocks):
            self.assertTrue(torch.equal(original.mlp.hidden_mask, recovered.mlp.hidden_mask))
            self.assertTrue(torch.equal(original.mlp.proj_mask, recovered.mlp.proj_mask))


if __name__ == "__main__":
    unittest.main()
