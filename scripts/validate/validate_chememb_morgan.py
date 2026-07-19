"""Regression tests for the Morgan-fingerprint ChemEmb ablation."""

import copy
import csv
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
from pepar_diff.embeddings.morgan_fingerprint import get_morgan_fingerprints
from pepar_diff.models.ald_model import AutoregressiveLatentDiffusion


class ChemEmbMorganTest(unittest.TestCase):
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
        self.smiles = {
            "M0": "C",
            "M1": "CC",
            "M2": "CCC",
            "M3": "CCO",
            "M4": "CCN",
            "M5": "c1ccccc1",
        }
        with (self.embeddings_dir / "monomer_mapping.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=["symbol", "smiles"])
            writer.writeheader()
            for symbol, smiles in self.smiles.items():
                writer.writerow({"symbol": symbol, "smiles": smiles})

        monomer_rows = [
            {"Symbol": name, "R1": "[*:1]", "R2": "[*:2]", "R3": "-"}
            for name in self.smiles
        ]
        pd.DataFrame(monomer_rows).to_csv(
            self.data_dir / "monomer_library.csv", index=False
        )
        self.ordinary_ids = [0, 2, 4, 5, 6, 7]

    def tearDown(self):
        self.temp_dir.cleanup()

    def model_config(self, mode="morgan", radius=2, chirality=False):
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
        config.model.r_group_dim = 8
        config.model.chememb_mode = mode
        config.model.morgan_radius = radius
        config.model.morgan_n_bits = 8
        config.model.morgan_include_chirality = chirality
        config.training.embeddings_dir = str(self.embeddings_dir)
        config.training.data_dir = str(self.data_dir)
        return config

    def make_loader(self, mode="morgan", radius=2):
        return UniMolEmbeddingLoader(
            str(self.embeddings_dir),
            chememb_mode=mode,
            vocab=self.vocab,
            fingerprint_token_ids=self.ordinary_ids,
            morgan_radius=radius,
            morgan_n_bits=8,
            morgan_include_chirality=False,
        )

    def test_morgan_replaces_only_ground_truth_cls(self):
        loader = self.make_loader()

        for symbol, token_id in self.vocab.items():
            if token_id not in self.ordinary_ids:
                continue
            expected, atom_fps = get_morgan_fingerprints(
                self.smiles[symbol], (), radius=2, n_bits=8,
                include_chirality=False,
            )
            self.assertEqual(atom_fps.shape, (0, 8))
            self.assertTrue(
                torch.equal(
                    loader._embeddings[token_id, 0],
                    torch.from_numpy(expected).float(),
                )
            )

        # R1/R2/R3, BOS/EOS, and PAD are unchanged from the main representation.
        self.assertTrue(torch.equal(loader._embeddings[:8, 1:], self.original_full[:, 1:]))
        self.assertTrue(torch.equal(loader._embeddings[1, 0], self.original_full[1, 0]))
        self.assertTrue(torch.equal(loader._embeddings[3, 0], self.original_full[3, 0]))
        self.assertEqual(torch.count_nonzero(loader._embeddings[8]).item(), 0)
        self.assertFalse(loader._embeddings.requires_grad)
        self.assertEqual(list(loader.parameters()), [])

    def test_original_main_codebook_is_bitwise_unchanged(self):
        loader = self.make_loader(mode="original")
        expected = torch.cat([self.original_full, torch.zeros(1, 4, 8)], dim=0)

        self.assertTrue(torch.equal(loader._embeddings, expected))
        self.assertTrue(torch.equal(loader.get_codebook(), expected[:, 0, :]))
        self.assertTrue(torch.equal(loader.chememb_permutation, torch.arange(9)))
        self.assertFalse(hasattr(loader, "morgan_signature"))
        self.assertEqual(ALDConfig().model.chememb_mode, "original")

    def test_context_target_history_and_mapper_share_morgan_codebook(self):
        morgan = AutoregressiveLatentDiffusion(
            self.vocab, self.model_config(), verbose=False
        )
        original = AutoregressiveLatentDiffusion(
            self.vocab, self.model_config(mode="original"), verbose=False
        )
        token_ids = torch.tensor([[0, 2, 4, 8]])
        mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)

        context_inputs = morgan.context_encoder.get_token_embedding(token_ids)
        z0, _ = morgan._prepare_contexts(token_ids, mask)
        mapper_refs = morgan.token_mapper.reference_embeddings[token_ids]
        self.assertTrue(torch.equal(context_inputs, z0))
        self.assertTrue(torch.equal(z0, mapper_refs))
        self.assertTrue(
            torch.equal(
                morgan.embedding_loader.get_r_embeddings(token_ids),
                original.embedding_loader.get_r_embeddings(token_ids),
            )
        )
        self.assertEqual(
            sum(p.numel() for p in morgan.parameters()),
            sum(p.numel() for p in original.parameters()),
        )
        self.assertEqual(morgan.token_rgroups, original.token_rgroups)
        self.assertFalse(morgan.token_mapper.reference_embeddings.requires_grad)

    def test_checkpoint_modes_and_morgan_signature_cannot_be_mixed(self):
        morgan = AutoregressiveLatentDiffusion(
            self.vocab, self.model_config(), verbose=False
        )
        morgan_state = copy.deepcopy(morgan.state_dict())
        signature_key = "context_encoder.embedding.morgan_signature"
        permutation_key = "context_encoder.embedding.chememb_permutation"
        self.assertIn(signature_key, morgan_state)

        restored = AutoregressiveLatentDiffusion(
            self.vocab, self.model_config(), verbose=False
        )
        restored.load_state_dict(morgan_state)
        self.assertTrue(
            torch.equal(
                restored.embedding_loader.morgan_signature,
                morgan.embedding_loader.morgan_signature,
            )
        )

        different_radius = AutoregressiveLatentDiffusion(
            self.vocab, self.model_config(radius=3), verbose=False
        )
        with self.assertRaisesRegex(RuntimeError, "signature does not match"):
            different_radius.load_state_dict(morgan_state)

        original = AutoregressiveLatentDiffusion(
            self.vocab, self.model_config(mode="original"), verbose=False
        )
        original_state = copy.deepcopy(original.state_dict())
        with self.assertRaisesRegex(RuntimeError, "into chememb_mode='morgan'"):
            morgan.load_state_dict(original_state, strict=False)
        with self.assertRaisesRegex(RuntimeError, "into the main"):
            original.load_state_dict(morgan_state, strict=False)

        # Main checkpoints predating the identity marker still load unchanged.
        legacy_original = copy.deepcopy(original_state)
        legacy_original.pop(permutation_key)
        original.load_state_dict(legacy_original)

        # Old shuffled checkpoints are explicitly rejected instead of being reused.
        legacy_shuffled = copy.deepcopy(original_state)
        legacy_shuffled[permutation_key][0] = 2
        legacy_shuffled[permutation_key][2] = 0
        with self.assertRaisesRegex(RuntimeError, "legacy ChemEmb permutation"):
            original.load_state_dict(legacy_shuffled, strict=False)

    def test_removed_shuffled_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "original.*morgan"):
            self.make_loader(mode="shuffled")


if __name__ == "__main__":
    unittest.main()
