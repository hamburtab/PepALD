"""Regression checks for the PepALD w/o R-site ablation."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pepar_diff.config import ALDConfig
from pepar_diff.models.ald_model import AutoregressiveLatentDiffusion
from pepar_diff.models.ring_predictor import (
    AutoregressiveRingPredictor,
    ContextOnlyAutoregressiveRingPredictor,
)


class RSiteAblationTest(unittest.TestCase):
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
            {
                "Symbol": name,
                "R1": "[*:1]",
                "R2": "[*:2]",
                "R3": "[*:3]" if name in {"M0", "M5"} else "-",
            }
            for name in ("M0", "M1", "M2", "M3", "M4", "M5")
        ]
        pd.DataFrame(rows).to_csv(self.data_dir / "monomer_library.csv", index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def config(self, ring_feature_mode="context_only"):
        config = ALDConfig()
        config.model.model_variant = "ald"
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
        config.model.ring_feature_mode = ring_feature_mode
        config.training.embeddings_dir = str(self.embeddings_dir)
        config.training.data_dir = str(self.data_dir)
        config.training.ring_loss_weight = 0.5
        config.training.ce_loss_weight = 0.5
        config.generation.history_embedding_mode = "token"
        config.generation.mapping_sample = False
        return config

    def make_model(self, ring_feature_mode="context_only"):
        return AutoregressiveLatentDiffusion(
            self.vocab,
            self.config(ring_feature_mode),
            verbose=False,
        )

    def test_context_only_predictor_has_no_mlp_r(self):
        model = self.make_model()
        predictor = model.ar_ring_predictor

        self.assertIsInstance(
            predictor, ContextOnlyAutoregressiveRingPredictor
        )
        self.assertFalse(model.use_r_site_embeddings)
        self.assertFalse(hasattr(predictor, "r_encoder"))
        self.assertFalse(hasattr(predictor, "position_head"))
        self.assertFalse(hasattr(predictor, "type_head"))
        self.assertFalse(
            any("r_encoder" in key for key in predictor.state_dict())
        )

        current_context = torch.randn(2, 8)
        history_context = torch.randn(2, 3, 8)
        position_scores, type_logits = predictor(
            current_context, history_context
        )
        self.assertEqual(tuple(position_scores.shape), (2, 3))
        self.assertEqual(tuple(type_logits.shape), (2, 3, 4))

    def test_training_never_requests_rsite_embeddings(self):
        model = self.make_model()
        token_ids = torch.tensor([[0, 2, 5, 4]])
        mask = torch.ones_like(token_ids, dtype=torch.bool)
        ring_bonds = [[{"i": 0, "j": 3, "type": 1}]]

        with patch.object(
            model.embedding_loader,
            "forward",
            wraps=model.embedding_loader.forward,
        ) as embedding_forward:
            result = model(
                token_ids,
                mask,
                ring_bonds=ring_bonds,
                compute_ring_loss=True,
            )

        self.assertGreater(result["ring_bond_loss"].item(), 0.0)
        self.assertFalse(
            any(
                call.kwargs.get("return_r_groups", False)
                or (len(call.args) > 1 and call.args[1] is True)
                for call in embedding_forward.call_args_list
            )
        )
        result["loss"].backward()
        self.assertIsNotNone(
            model.ar_ring_predictor.context_position_head[0].weight.grad
        )
        self.assertIsNotNone(
            model.ar_ring_predictor.context_type_head[0].weight.grad
        )

    def test_generation_never_requests_rsite_embeddings(self):
        model = self.make_model()
        fake_latent = torch.zeros(1, 1, 8)

        with patch.object(
            model.diffusion_engine,
            "sample_ddim",
            return_value=fake_latent,
        ), patch.object(
            model.token_mapper,
            "batch_map",
            return_value=torch.tensor([5]),
        ), patch.object(
            model.embedding_loader,
            "forward",
            wraps=model.embedding_loader.forward,
        ) as embedding_forward:
            result = model.sample(
                num_samples=1,
                min_seq_len=4,
                max_seq_len=4,
                use_ddim=True,
                lambda_gpt=0.0,
                predict_ring_bonds=True,
                ring_threshold=1.0,
            )[0]

        self.assertIn("ring_bonds", result)
        self.assertFalse(
            any(
                call.kwargs.get("return_r_groups", False)
                or (len(call.args) > 1 and call.args[1] is True)
                for call in embedding_forward.call_args_list
            )
        )

    def test_default_predictor_still_uses_rsite_embeddings(self):
        self.assertEqual(
            ALDConfig().model.ring_feature_mode, "context_rsite"
        )
        model = self.make_model("context_rsite")
        self.assertTrue(model.use_r_site_embeddings)
        self.assertIsInstance(
            model.ar_ring_predictor, AutoregressiveRingPredictor
        )
        self.assertTrue(hasattr(model.ar_ring_predictor, "r_encoder"))

        token_ids = torch.tensor([[0, 2, 5, 4]])
        mask = torch.ones_like(token_ids, dtype=torch.bool)
        ring_bonds = [[{"i": 0, "j": 3, "type": 1}]]
        with patch.object(
            model.embedding_loader,
            "forward",
            wraps=model.embedding_loader.forward,
        ) as embedding_forward:
            model(
                token_ids,
                mask,
                ring_bonds=ring_bonds,
                compute_ring_loss=True,
            )

        self.assertTrue(
            any(
                call.kwargs.get("return_r_groups", False)
                for call in embedding_forward.call_args_list
            )
        )

    def test_main_checkpoint_loads_except_new_ring_predictor(self):
        main_model = self.make_model("context_rsite")
        context_only = self.make_model()
        main_state = copy.deepcopy(main_model.state_dict())
        filtered_state = {
            key: value for key, value in main_state.items()
            if not key.startswith("ar_ring_predictor.")
        }

        incompatible = context_only.load_state_dict(
            filtered_state, strict=False
        )
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(
            all(
                key.startswith("ar_ring_predictor.")
                for key in incompatible.missing_keys
            )
        )
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertTrue(
            torch.equal(
                context_only.lm_head.weight,
                main_model.lm_head.weight,
            )
        )

        # A context-only fine-tuning checkpoint restores strictly.
        restored = self.make_model()
        restored.load_state_dict(copy.deepcopy(context_only.state_dict()))


if __name__ == "__main__":
    unittest.main()
