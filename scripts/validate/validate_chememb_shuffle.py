"""Regression tests for the fixed shuffled-ChemEmb ablation."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pepar_diff.config import ALDConfig
from pepar_diff.core.embeddings import UniMolEmbeddingLoader
from pepar_diff.models.ald_model import AutoregressiveLatentDiffusion


class ChemEmbShuffleTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.embeddings_dir = self.data_dir / "unimol_embeddings"
        self.embeddings_dir.mkdir()

        # Eight non-PAD rows: six ordinary monomers plus BOS/EOS control tokens.
        full = np.arange(8 * 4 * 8, dtype=np.float32).reshape(8, 4, 8)
        np.save(self.embeddings_dir / "full_embeddings.npy", full)
        np.save(self.embeddings_dir / "embeddings_matrix.npy", full[:, 0, :])
        (self.embeddings_dir / "metadata.json").write_text(
            json.dumps({"num_monomers": 8}), encoding="utf-8"
        )
        self.original_full = torch.from_numpy(full)

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
        monomer_rows = [
            {"Symbol": name, "R1": "[*:1]", "R2": "[*:2]", "R3": "-"}
            for name in ("M0", "M1", "M2", "M3", "M4", "M5")
        ]
        pd.DataFrame(monomer_rows).to_csv(
            self.data_dir / "monomer_library.csv", index=False
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def model_config(self, mode="shuffled", seed=42):
        config = ALDConfig()
        config.model.embedding_dim = 8
        config.model.d_model = 8
        config.model.n_heads = 2
        config.model.context_layers = 1
        config.model.denoiser_layers = 1
        config.model.d_ff = 16
        config.model.max_seq_len = 8
        config.model.dropout = 0.0
        config.model.num_diffusion_steps = 4
        config.model.chememb_mode = mode
        config.model.chememb_shuffle_seed = seed
        config.training.embeddings_dir = str(self.embeddings_dir)
        config.training.data_dir = str(self.data_dir)
        return config

    def test_deterministic_cls_only_shuffle_and_frozen_special_rows(self):
        ordinary_ids = [0, 2, 4, 5, 6, 7]
        first = UniMolEmbeddingLoader(
            str(self.embeddings_dir),
            chememb_mode="shuffled",
            chememb_shuffle_seed=42,
            shuffle_token_ids=ordinary_ids,
        )
        second = UniMolEmbeddingLoader(
            str(self.embeddings_dir),
            chememb_mode="shuffled",
            chememb_shuffle_seed=42,
            shuffle_token_ids=ordinary_ids,
        )

        self.assertTrue(torch.equal(first.chememb_permutation, second.chememb_permutation))
        self.assertEqual(
            set(first.chememb_permutation[ordinary_ids].tolist()), set(ordinary_ids)
        )
        for target_id in ordinary_ids:
            source_id = first.chememb_permutation[target_id]
            self.assertTrue(
                torch.equal(first._embeddings[target_id, 0], self.original_full[source_id, 0])
            )

        # BOS, EOS, PAD and all attachment-site embeddings remain untouched.
        self.assertEqual(first.chememb_permutation[[1, 3, 8]].tolist(), [1, 3, 8])
        self.assertTrue(torch.equal(first._embeddings[1, 0], self.original_full[1, 0]))
        self.assertTrue(torch.equal(first._embeddings[3, 0], self.original_full[3, 0]))
        self.assertTrue(torch.equal(first._embeddings[:8, 1:], self.original_full[:, 1:]))
        self.assertEqual(torch.count_nonzero(first._embeddings[8]).item(), 0)
        self.assertFalse(first._embeddings.requires_grad)
        self.assertFalse(first.chememb_permutation.requires_grad)
        self.assertEqual(list(first.parameters()), [])
        self.assertEqual(self.vocab["M0"], 0)
        self.assertEqual(self.vocab["<PAD>"], 8)

    def test_original_mode_is_identity_and_matches_previous_codebook(self):
        loader = UniMolEmbeddingLoader(
            str(self.embeddings_dir), chememb_mode="original"
        )
        expected = torch.cat([self.original_full, torch.zeros(1, 4, 8)], dim=0)

        self.assertTrue(torch.equal(loader.chememb_permutation, torch.arange(9)))
        self.assertTrue(torch.equal(loader._embeddings, expected))
        self.assertTrue(torch.equal(loader.get_codebook(), expected[:, 0, :]))
        self.assertEqual(ALDConfig().model.chememb_mode, "original")

    def test_context_z0_mapper_share_codebook_without_touching_metadata(self):
        shuffled = AutoregressiveLatentDiffusion(
            self.vocab, self.model_config(), verbose=False
        )
        original = AutoregressiveLatentDiffusion(
            self.vocab, self.model_config(mode="original"), verbose=False
        )
        token_ids = torch.tensor([[0, 2, 4, 8]])
        mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)

        context_inputs = shuffled.context_encoder.get_token_embedding(token_ids)
        z0, _ = shuffled._prepare_contexts(token_ids, mask)
        mapper_refs = shuffled.token_mapper.reference_embeddings[token_ids]
        self.assertTrue(torch.equal(context_inputs, z0))
        self.assertTrue(torch.equal(z0, mapper_refs))
        self.assertTrue(
            torch.equal(
                shuffled.embedding_loader.get_r_embeddings(token_ids),
                original.embedding_loader.get_r_embeddings(token_ids),
            )
        )
        self.assertEqual(shuffled.vocab, original.vocab)
        self.assertEqual(original.vocab, self.vocab)
        self.assertEqual(shuffled.token_rgroups, original.token_rgroups)
        self.assertEqual(
            shuffled.token_mapper.class1_tokens, original.token_mapper.class1_tokens
        )
        self.assertEqual(
            shuffled.token_mapper.class2_tokens, original.token_mapper.class2_tokens
        )
        self.assertEqual(
            shuffled.token_mapper.class3_tokens, original.token_mapper.class3_tokens
        )
        self.assertFalse(shuffled.token_mapper.reference_embeddings.requires_grad)

    def test_checkpoint_restores_and_validates_permutation(self):
        model = AutoregressiveLatentDiffusion(
            self.vocab, self.model_config(seed=42), verbose=False
        )
        state = copy.deepcopy(model.state_dict())
        key = "context_encoder.embedding.chememb_permutation"
        self.assertIn(key, state)

        same_seed = AutoregressiveLatentDiffusion(
            self.vocab, self.model_config(seed=42), verbose=False
        )
        same_seed.load_state_dict(state)
        self.assertTrue(
            torch.equal(
                same_seed.embedding_loader.chememb_permutation,
                model.embedding_loader.chememb_permutation,
            )
        )

        different_seed = AutoregressiveLatentDiffusion(
            self.vocab, self.model_config(seed=7), verbose=False
        )
        with self.assertRaisesRegex(RuntimeError, "permutation does not match"):
            different_seed.load_state_dict(state)

        # Legacy checkpoints are compatible in original mode: missing means identity.
        original = AutoregressiveLatentDiffusion(
            self.vocab, self.model_config(mode="original"), verbose=False
        )
        legacy_state = copy.deepcopy(original.state_dict())
        legacy_state.pop(key)
        restored_original = AutoregressiveLatentDiffusion(
            self.vocab, self.model_config(mode="original"), verbose=False
        )
        restored_original.load_state_dict(legacy_state)
        self.assertTrue(
            torch.equal(
                restored_original.embedding_loader.chememb_permutation,
                torch.arange(len(self.vocab)),
            )
        )

        with self.assertRaisesRegex(
            RuntimeError, "without a saved ChemEmb permutation"
        ):
            same_seed.load_state_dict(legacy_state, strict=False)


if __name__ == "__main__":
    unittest.main()
