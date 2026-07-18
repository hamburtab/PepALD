"""Regression checks for PepALD w/o ALD and the unchanged default ALD path."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pepar_diff.config import ALDConfig
from pepar_diff.diffusion.engine import DiffusionEngine
from pepar_diff.models.ald_model import AutoregressiveLatentDiffusion
from pepar_diff.models.token_constraints import TokenConstraintSampler
from pepar_diff.models.token_mapper import TokenMapper


class LMOnlyAblationTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.embeddings_dir = self.data_dir / "unimol_embeddings"
        self.embeddings_dir.mkdir()

        full = np.arange(8 * 4 * 8, dtype=np.float32).reshape(8, 4, 8)
        np.save(self.embeddings_dir / "full_embeddings.npy", full)
        np.save(self.embeddings_dir / "embeddings_matrix.npy", full[:, 0, :])
        (self.embeddings_dir / "metadata.json").write_text(
            json.dumps({"num_monomers": 8}), encoding="utf-8"
        )

        self.vocab = {
            "M0": 0,
            "<BOS>": 1,
            "M1": 2,
            "<EOS>": 3,
            "M2": 4,
            "M3": 5,
            "M4": 6,
            "M5": 7,
            "<PAD>": 8,
        }
        rows = [
            {"Symbol": "M0", "R1": "-", "R2": "[*:2]", "R3": "-"},
            {"Symbol": "M1", "R1": "[*:1]", "R2": "[*:2]", "R3": "-"},
            {"Symbol": "M2", "R1": "[*:1]", "R2": "-", "R3": "-"},
            {"Symbol": "M3", "R1": "[*:1]", "R2": "[*:2]", "R3": "-"},
            {"Symbol": "M4", "R1": "[*:1]", "R2": "[*:2]", "R3": "-"},
            {"Symbol": "M5", "R1": "[*:1]", "R2": "[*:2]", "R3": "-"},
        ]
        pd.DataFrame(rows).to_csv(self.data_dir / "monomer_library.csv", index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def config(self, variant="lm_only"):
        config = ALDConfig()
        config.model.model_variant = variant
        config.model.embedding_dim = 8
        config.model.d_model = 8
        config.model.n_heads = 2
        config.model.context_layers = 1
        config.model.denoiser_layers = 1
        config.model.d_ff = 16
        config.model.max_seq_len = 8
        config.model.dropout = 0.0
        config.model.num_diffusion_steps = 4
        config.model.r_group_dim = 8
        config.training.embeddings_dir = str(self.embeddings_dir)
        config.training.data_dir = str(self.data_dir)
        config.training.ring_loss_weight = 0.5
        config.training.ce_loss_weight = 0.5
        config.generation.history_embedding_mode = "token"
        config.generation.mapping_sample = False
        return config

    def make_model(self, variant="lm_only"):
        return AutoregressiveLatentDiffusion(
            self.vocab, self.config(variant), verbose=False
        )

    def test_lm_only_structure_has_no_denoiser_or_embedding_mapper(self):
        lm_only = self.make_model()
        ald = self.make_model("ald")

        self.assertIsNone(lm_only.diffusion_engine)
        self.assertIsInstance(lm_only.token_mapper, TokenConstraintSampler)
        self.assertFalse(hasattr(lm_only.token_mapper, "reference_embeddings"))
        self.assertFalse(hasattr(lm_only.token_mapper, "_compute_distances"))
        self.assertFalse(
            any(key.startswith("diffusion_engine.") for key in lm_only.state_dict())
        )
        self.assertIsInstance(ald.diffusion_engine, DiffusionEngine)
        self.assertIsInstance(ald.token_mapper, TokenMapper)
        self.assertTrue(
            any(key.startswith("diffusion_engine.") for key in ald.state_dict())
        )
        self.assertLess(
            sum(p.numel() for p in lm_only.parameters()),
            sum(p.numel() for p in ald.parameters()),
        )

        # The constraint sets are exactly the same as the main TokenMapper path.
        self.assertEqual(lm_only.token_mapper.class1_tokens, ald.token_mapper.class1_tokens)
        self.assertEqual(lm_only.token_mapper.class2_tokens, ald.token_mapper.class2_tokens)
        self.assertEqual(lm_only.token_mapper.class3_tokens, ald.token_mapper.class3_tokens)

    def test_lm_only_training_is_exact_ce_and_keeps_ring_loss(self):
        model = self.make_model()
        token_ids = torch.tensor([[0, 2, 5, 4]])
        mask = torch.ones_like(token_ids, dtype=torch.bool)

        _, contexts = model._prepare_contexts(token_ids, mask)
        expected_ce = F.cross_entropy(
            model.lm_head(contexts)[mask], token_ids[mask]
        )
        pretrain_result = model(token_ids, mask)
        self.assertTrue(torch.allclose(pretrain_result["loss"], expected_ce))
        self.assertTrue(torch.allclose(pretrain_result["ce_loss"], expected_ce))
        self.assertEqual(pretrain_result["diffusion_loss"].item(), 0.0)

        ring_bonds = [[{"i": 0, "j": 3, "type": 1}]]
        finetune_result = model(
            token_ids,
            mask,
            ring_bonds=ring_bonds,
            compute_ring_loss=True,
        )
        expected_total = (
            finetune_result["ce_loss"]
            + 0.5 * finetune_result["ring_bond_loss"]
        )
        self.assertTrue(torch.allclose(finetune_result["loss"], expected_total))
        self.assertGreater(finetune_result["ring_bond_loss"].item(), 0.0)

        finetune_result["loss"].backward()
        self.assertIsNotNone(model.lm_head.weight.grad)
        self.assertFalse(model.embedding_loader._embeddings.requires_grad)

    def test_lm_only_generation_uses_logits_constraints_and_token_history(self):
        model = self.make_model()
        with torch.no_grad():
            model.lm_head.weight.zero_()
            model.lm_head.bias.zero_()
            model.lm_head.bias[0] = 20.0  # M0: allowed only in the first position.
            model.lm_head.bias[5] = 10.0  # M3: allowed at every position.

        results = model.sample(
            num_samples=2,
            min_seq_len=4,
            max_seq_len=4,
            predict_ring_bonds=False,
        )
        for result in results:
            self.assertEqual(result["tokens"].tolist(), [0, 5, 5, 5])
            expected_history = model.context_encoder.get_token_embedding(result["tokens"])
            self.assertTrue(torch.equal(result["embeddings"], expected_history))
            self.assertEqual(result["history_embedding_mode"], "token")

        # The unchanged autoregressive ring-prediction path also runs at inference.
        ring_result = model.sample(
            num_samples=1,
            min_seq_len=4,
            max_seq_len=4,
            predict_ring_bonds=True,
            ring_threshold=1.0,
        )[0]
        self.assertIn("ring_bonds", ring_result)
        self.assertIn("ring_connections", ring_result)

        with self.assertRaisesRegex(ValueError, "requires history_embedding_mode='token'"):
            model.sample(
                num_samples=1,
                min_seq_len=2,
                max_seq_len=2,
                history_embedding_mode="latent",
                predict_ring_bonds=False,
            )

    def test_generation_can_disable_r1r2_constraints_without_selecting_specials(self):
        model = self.make_model()
        with torch.no_grad():
            model.lm_head.weight.zero_()
            model.lm_head.bias.zero_()
            model.lm_head.bias[8] = 100.0  # <PAD> must never become a monomer.
            model.lm_head.bias[4] = 20.0   # M2 has R1 but no R2.

        model.config.generation.enforce_r1r2_constraints = False
        unconstrained = model.sample(
            num_samples=1,
            min_seq_len=4,
            max_seq_len=4,
            predict_ring_bonds=False,
        )[0]["tokens"].tolist()
        self.assertEqual(unconstrained, [4, 4, 4, 4])

        constrained = model.sample(
            num_samples=1,
            min_seq_len=4,
            max_seq_len=4,
            enforce_r1r2_constraints=True,
            predict_ring_bonds=False,
        )[0]["tokens"].tolist()
        self.assertNotEqual(constrained[0], 4)
        self.assertNotEqual(constrained[1], 4)
        self.assertEqual(constrained[-1], 4)

    def test_default_ald_still_calls_diffusion_and_mapper(self):
        self.assertEqual(ALDConfig().model.model_variant, "ald")
        model = self.make_model("ald")
        token_ids = torch.tensor([[0, 2, 5, 4]])
        mask = torch.ones_like(token_ids, dtype=torch.bool)

        with patch.object(
            model.diffusion_engine,
            "training_step",
            wraps=model.diffusion_engine.training_step,
        ) as training_step:
            model(token_ids, mask)
            training_step.assert_called_once()

        fake_latent = torch.zeros(1, 1, 8)
        with patch.object(
            model.diffusion_engine,
            "sample_ddim",
            return_value=fake_latent,
        ) as sample_ddim, patch.object(
            model.token_mapper,
            "batch_map",
            return_value=torch.tensor([5]),
        ) as batch_map:
            model.sample(
                num_samples=1,
                min_seq_len=2,
                max_seq_len=2,
                use_ddim=True,
                lambda_gpt=0.0,
                predict_ring_bonds=False,
            )
            self.assertEqual(sample_ddim.call_count, 2)
            self.assertEqual(batch_map.call_count, 2)

    def test_checkpoint_variants_cannot_be_mixed(self):
        lm_only = self.make_model()
        lm_state = copy.deepcopy(lm_only.state_dict())
        restored = self.make_model()
        restored.load_state_dict(lm_state)

        ald = self.make_model("ald")
        with self.assertRaisesRegex(RuntimeError, "Cannot load an ALD checkpoint"):
            restored.load_state_dict(ald.state_dict(), strict=False)


if __name__ == "__main__":
    unittest.main()
