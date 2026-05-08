"""PepTune solubility scoring for HELM or SMILES peptide candidates."""

from __future__ import annotations

import importlib.util
import inspect
import os
import shutil
import sys
from pathlib import Path
from typing import Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np


PEPTUNE_REPO_ID = "ChatterjeeLab/PepTune"
PEPTIDECLM_CHECKPOINT = "aaronfeller/PeptideCLM-23M-all"

MODEL_CANDIDATES = (
    "src/scoring/functions/classifiers/solubility-xgboost.json",
    "classifiers/solubility-xgboost.json",
    "src/scoring/classifiers/solubility-xgboost.json",
    "src/scoring/functions/solubility/new_train/best_model.json",
    "scoring/functions/solubility/new_train/best_model.json",
)
VOCAB_CANDIDATES = (
    "src/scoring/tokenizer/new_vocab.txt",
    "src/tokenizer/new_vocab.txt",
    "src/scoring/functions/tokenizer/new_vocab.txt",
    "tokenizer/new_vocab.txt",
)
SPLITS_CANDIDATES = (
    "src/scoring/tokenizer/new_splits.txt",
    "src/tokenizer/new_splits.txt",
    "src/scoring/functions/tokenizer/new_splits.txt",
    "tokenizer/new_splits.txt",
)
TOKENIZER_MODULE_CANDIDATES = (
    "src/scoring/tokenizer/my_tokenizers.py",
    "src/tokenizer/my_tokenizers.py",
    "tokenizer/my_tokenizers.py",
)

_HELM_TO_SMILES = None
_RDKIT_CHEM = None


def _expand_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _require_file(path: str | Path, label: str) -> Path:
    resolved = _expand_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{label} file not found: {resolved}")
    return resolved


def _missing_dependency(package: str, install_hint: str) -> ImportError:
    return ImportError(
        "PepTune solubility scoring requires optional dependencies that are not "
        f"installed. Missing: {package}. Install with `{install_hint}` or update "
        "the `pepardiff` conda environment from envs/pepardiff.yml."
    )


def _get_rdkit_chem():
    global _RDKIT_CHEM
    if _RDKIT_CHEM is not None:
        return _RDKIT_CHEM

    try:
        from rdkit import Chem, rdBase
    except ImportError as exc:
        raise _missing_dependency("rdkit", "conda install -c conda-forge rdkit") from exc

    rdBase.DisableLog("rdApp.error")
    _RDKIT_CHEM = Chem
    return _RDKIT_CHEM


def _get_helm_to_smiles():
    global _HELM_TO_SMILES
    if _HELM_TO_SMILES is not None:
        return _HELM_TO_SMILES

    try:
        from pepar_diff.utils.helm import get_cycpep_smi_from_helm
    except ImportError as exc:
        if exc.name in {"rdkit", "pandas", "loguru"}:
            raise _missing_dependency(exc.name, "conda env update -f envs/pepardiff.yml") from exc
        raise

    _HELM_TO_SMILES = get_cycpep_smi_from_helm
    return _HELM_TO_SMILES


def _direct_download_hf_file(repo_id: str, filename: str) -> Path:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    quoted_filename = quote(filename, safe="/")
    url = f"{endpoint}/{repo_id}/resolve/main/{quoted_filename}"
    cache_file = (
        Path.home()
        / ".cache"
        / "pepar_diff"
        / "peptune"
        / repo_id.replace("/", "--")
        / "main"
        / filename
    )
    if cache_file.exists() and cache_file.stat().st_size > 0:
        return cache_file

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = cache_file.with_suffix(cache_file.suffix + ".tmp")
    request = Request(url, headers={"User-Agent": "pepar-diff"})
    try:
        with urlopen(request, timeout=120) as response, open(tmp_file, "wb") as f:
            status = getattr(response, "status", 200)
            if status >= 400:
                raise HTTPError(url, status, "download failed", response.headers, None)
            shutil.copyfileobj(response, f)
    except (HTTPError, URLError, OSError):
        if tmp_file.exists():
            tmp_file.unlink()
        raise

    tmp_file.replace(cache_file)
    return cache_file


def _download_hf_file(repo_id: str, candidates: Sequence[str], label: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise _missing_dependency("huggingface_hub", "pip install huggingface-hub") from exc

    errors = []
    for filename in candidates:
        try:
            return Path(hf_hub_download(repo_id=repo_id, filename=filename))
        except Exception as hf_exc:  # noqa: BLE001 - direct URL fallback below.
            try:
                return _direct_download_hf_file(repo_id, filename)
            except Exception as direct_exc:  # noqa: BLE001 - report all failed candidates.
                errors.append(
                    f"{filename}: hf_hub_download={hf_exc}; direct_download={direct_exc}"
                )

    checked = "\n  ".join(errors)
    raise FileNotFoundError(
        f"Could not download PepTune {label} from Hugging Face repo {repo_id}. "
        f"Checked candidates:\n  {checked}"
    )


def _resolve_asset(
    explicit_path: str | Path | None,
    peptune_dir: Path | None,
    candidates: Sequence[str],
    label: str,
    repo_id: str,
) -> Path:
    if explicit_path is not None:
        return _require_file(explicit_path, label)

    if peptune_dir is not None:
        for rel_path in candidates:
            candidate = peptune_dir / rel_path
            if candidate.exists():
                return candidate.resolve()
        checked = "\n  ".join(str(peptune_dir / rel_path) for rel_path in candidates)
        raise FileNotFoundError(
            f"Could not find PepTune {label} under {peptune_dir}. "
            f"Checked:\n  {checked}"
        )

    return _download_hf_file(repo_id, candidates, label)


def _is_valid_smiles(smiles: str | None) -> bool:
    if not smiles:
        return False
    Chem = _get_rdkit_chem()
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False


class PepTuneSolubility:
    """Predict PepTune solubility positive-class probabilities.

    This mirrors PepTune's published scoring path:
    SMILES -> PepTune SPE tokenizer -> PeptideCLM hidden-state mean embedding
    -> XGBoost solubility classifier positive-class probability.
    """

    def __init__(
        self,
        peptune_dir: str | Path | None = None,
        model_path: str | Path | None = None,
        vocab_path: str | Path | None = None,
        splits_path: str | Path | None = None,
        tokenizer_module_path: str | Path | None = None,
        batch_size: int = 1,
        input_type: str = "helm",
        device: str | None = None,
        invalid_score: float = np.nan,
        hf_repo_id: str = PEPTUNE_REPO_ID,
        hf_endpoint: str | None = None,
        peptideclm_checkpoint: str = PEPTIDECLM_CHECKPOINT,
        load_on_init: bool = True,
    ) -> None:
        if input_type not in {"helm", "smiles"}:
            raise ValueError("input_type must be either 'helm' or 'smiles'")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        self.peptune_dir = _expand_path(peptune_dir) if peptune_dir is not None else None
        self.model_path_arg = model_path
        self.vocab_path_arg = vocab_path
        self.splits_path_arg = splits_path
        self.tokenizer_module_path_arg = tokenizer_module_path
        self.batch_size = batch_size
        self.input_type = input_type
        self.device_arg = device
        self.invalid_score = invalid_score
        self.hf_repo_id = hf_repo_id
        self.hf_endpoint = hf_endpoint
        self.peptideclm_checkpoint = peptideclm_checkpoint

        self.model_path: Path | None = None
        self.vocab_path: Path | None = None
        self.splits_path: Path | None = None
        self.tokenizer_module_path: Path | None = None
        self.tokenizer = None
        self.embedding_model = None
        self.device = None
        self.xgb_model = None
        self._xgb_mode: str | None = None
        self._xgb_module = None
        self._loaded = False

        if load_on_init:
            self.load()

    def load(self) -> None:
        """Load PepTune tokenizer, PeptideCLM model, and XGBoost classifier."""
        if self._loaded:
            return

        if self.hf_endpoint:
            os.environ["HF_ENDPOINT"] = self.hf_endpoint

        try:
            import torch
            from transformers import AutoModelForMaskedLM
            from transformers.utils import logging as hf_logging
        except ImportError as exc:
            raise _missing_dependency(
                "torch/transformers",
                "pip install torch transformers",
            ) from exc
        hf_logging.set_verbosity_error()

        try:
            import xgboost as xgb
        except ImportError as exc:
            raise _missing_dependency("xgboost", "pip install xgboost") from exc

        self.model_path = _resolve_asset(
            self.model_path_arg,
            self.peptune_dir,
            MODEL_CANDIDATES,
            "solubility model",
            self.hf_repo_id,
        )
        self.vocab_path = _resolve_asset(
            self.vocab_path_arg,
            self.peptune_dir,
            VOCAB_CANDIDATES,
            "tokenizer vocabulary",
            self.hf_repo_id,
        )
        self.splits_path = _resolve_asset(
            self.splits_path_arg,
            self.peptune_dir,
            SPLITS_CANDIDATES,
            "tokenizer SPE splits",
            self.hf_repo_id,
        )
        self.tokenizer_module_path = _resolve_asset(
            self.tokenizer_module_path_arg,
            self.peptune_dir,
            TOKENIZER_MODULE_CANDIDATES,
            "tokenizer module",
            self.hf_repo_id,
        )

        tokenizer_cls = self._load_peptune_tokenizer_class(self.tokenizer_module_path)
        self.tokenizer = self._make_tokenizer(tokenizer_cls)

        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_arg if self.device_arg else default_device)
        masked_lm = AutoModelForMaskedLM.from_pretrained(self.peptideclm_checkpoint)
        self.embedding_model = getattr(masked_lm, "roformer", masked_lm).to(self.device)
        self.embedding_model.eval()

        self.xgb_model, self._xgb_mode = self._load_xgb_model(xgb, self.model_path)
        self._xgb_module = xgb
        self._loaded = True

    def _make_tokenizer(self, tokenizer_cls):
        assert self.vocab_path is not None
        assert self.splits_path is not None

        try:
            signature = inspect.signature(tokenizer_cls)
            first_param = next(iter(signature.parameters.values())).name
        except Exception:
            first_param = "vocab_file"

        if "vocab" in first_param:
            return tokenizer_cls(str(self.vocab_path), str(self.splits_path))
        return tokenizer_cls(str(self.splits_path), str(self.vocab_path))

    @staticmethod
    def _load_peptune_tokenizer_class(module_path: Path):
        module_name = f"_peptune_tokenizer_{abs(hash(str(module_path)))}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load PepTune tokenizer module: {module_path}")

        module = importlib.util.module_from_spec(spec)
        added_paths = [
            str(module_path.parent),
            str(module_path.parent.parent),
            str(module_path.parent.parent.parent),
        ]
        old_sys_path = list(sys.path)
        try:
            for path in reversed(added_paths):
                if path not in sys.path:
                    sys.path.insert(0, path)
            spec.loader.exec_module(module)
        except ModuleNotFoundError as exc:
            if exc.name == "SmilesPE":
                raise _missing_dependency("SmilesPE", "pip install SmilesPE==0.0.3") from exc
            raise
        finally:
            sys.path = old_sys_path

        tokenizer_cls = getattr(module, "SMILES_SPE_Tokenizer", None)
        if tokenizer_cls is None:
            raise ImportError(
                f"PepTune tokenizer module does not define SMILES_SPE_Tokenizer: {module_path}"
            )
        return tokenizer_cls

    @staticmethod
    def _load_xgb_model(xgb, model_path: Path):
        booster_error: Exception | None = None
        try:
            booster = xgb.Booster(model_file=str(model_path))
            return booster, "booster"
        except Exception as exc:  # noqa: BLE001 - fall back below.
            booster_error = exc

        try:
            model = xgb.XGBClassifier()
            model.load_model(str(model_path))
            return model, "classifier"
        except Exception as classifier_error:  # noqa: BLE001
            raise RuntimeError(
                f"Could not load PepTune solubility XGBoost model from {model_path}. "
                f"Booster error: {booster_error}. XGBClassifier error: {classifier_error}"
            ) from classifier_error

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def prepare_inputs(self, input_seqs: Sequence[str]) -> tuple[list[str], list[str], list[int]]:
        """Convert inputs to valid SMILES and return aligned statuses."""
        smiles_list: list[str] = []
        statuses: list[str] = []
        valid_idxes: list[int] = []

        for idx, raw_seq in enumerate(input_seqs):
            seq = str(raw_seq).strip()
            if not seq:
                smiles_list.append("")
                statuses.append("empty_input")
                continue

            if self.input_type == "helm":
                try:
                    helm_to_smiles = _get_helm_to_smiles()
                    smiles = helm_to_smiles(seq)
                except ImportError:
                    raise
                except Exception:
                    smiles = None
                if not smiles:
                    smiles_list.append("")
                    statuses.append("helm_to_smiles_failed")
                    continue
            else:
                smiles = seq

            smiles_list.append(smiles)
            if _is_valid_smiles(smiles):
                valid_idxes.append(idx)
                statuses.append("ok")
            else:
                statuses.append("invalid_smiles")

        return smiles_list, statuses, valid_idxes

    def _encode_smiles_fallback(self, smiles: str) -> list[int]:
        assert self.tokenizer is not None
        encoded = self.tokenizer.encode(smiles)
        if isinstance(encoded, dict):
            ids = encoded.get("input_ids")
            if hasattr(ids, "squeeze"):
                ids = ids.squeeze(0)
        else:
            ids = getattr(encoded, "ids", encoded)
        ids = list(ids)
        if not ids:
            raise ValueError(f"PepTune tokenizer produced an empty token sequence for SMILES: {smiles}")
        return ids

    def _pad_token_id(self) -> int:
        assert self.tokenizer is not None
        tokenizer_obj = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        for token in ("<pad>", "[PAD]", "<PAD>", "PAD"):
            token_to_id = getattr(tokenizer_obj, "token_to_id", None)
            if token_to_id is not None:
                token_id = token_to_id(token)
                if token_id is not None:
                    return int(token_id)
        return 0

    def _tokenize_batch(self, smiles: Sequence[str]):
        assert self.tokenizer is not None
        import torch

        if callable(self.tokenizer):
            tokenizer_input = list(smiles) if len(smiles) > 1 else smiles[0]
            tokenized = self.tokenizer(
                tokenizer_input,
                return_tensors="pt",
                padding=len(smiles) > 1,
            )
            return {key: value.to(self.device) for key, value in tokenized.items()}

        encoded_batch = [self._encode_smiles_fallback(smi) for smi in smiles]
        max_len = max(len(ids) for ids in encoded_batch)
        pad_id = self._pad_token_id()
        input_ids = torch.full(
            (len(encoded_batch), max_len),
            pad_id,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros(
            (len(encoded_batch), max_len),
            dtype=torch.long,
            device=self.device,
        )
        for row_idx, ids in enumerate(encoded_batch):
            row = torch.tensor(ids, dtype=torch.long, device=self.device)
            input_ids[row_idx, : len(ids)] = row
            attention_mask[row_idx, : len(ids)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def _embed_smiles(self, smiles: Sequence[str]) -> np.ndarray:
        self._ensure_loaded()
        assert self.embedding_model is not None
        assert self.device is not None

        import torch

        embeddings: list[np.ndarray] = []

        for start in range(0, len(smiles), self.batch_size):
            batch = list(smiles[start:start + self.batch_size])
            tokenized = self._tokenize_batch(batch)

            with torch.no_grad():
                outputs = self.embedding_model(**tokenized)

                if hasattr(outputs, "last_hidden_state"):
                    hidden = outputs.last_hidden_state
                elif getattr(outputs, "hidden_states", None) is not None:
                    hidden = outputs.hidden_states[-1]
                else:
                    outputs = self.embedding_model(**tokenized, output_hidden_states=True)
                    hidden = outputs.hidden_states[-1]

                attention_mask = tokenized.get("attention_mask")
                if attention_mask is not None and len(batch) > 1:
                    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                else:
                    pooled = hidden.mean(dim=1)
                embeddings.append(pooled.cpu().numpy())

        if not embeddings:
            return np.zeros((0, 0), dtype=np.float32)
        return np.concatenate(embeddings, axis=0)

    def _predict_positive_probability(self, embeddings: np.ndarray) -> np.ndarray:
        assert self.xgb_model is not None
        if self._xgb_mode == "classifier":
            probs = self.xgb_model.predict_proba(embeddings)
            probs = np.asarray(probs)
            if probs.ndim == 2 and probs.shape[1] > 1:
                return probs[:, 1].astype(float)
            return probs.reshape(-1).astype(float)

        assert self._xgb_module is not None
        probs = self.xgb_model.predict(self._xgb_module.DMatrix(embeddings))
        return np.asarray(probs).reshape(-1).astype(float)

    def predict_with_details(self, input_seqs: Sequence[str]) -> tuple[np.ndarray, list[str], list[str]]:
        """Return scores plus aligned SMILES and status strings."""
        input_list = list(input_seqs)
        scores = np.full(len(input_list), self.invalid_score, dtype=float)
        smiles_list, statuses, valid_idxes = self.prepare_inputs(input_list)

        if not valid_idxes:
            return scores, smiles_list, statuses

        valid_smiles = [smiles_list[idx] for idx in valid_idxes]
        embeddings = self._embed_smiles(valid_smiles)
        predictions = self._predict_positive_probability(embeddings)
        scores[np.asarray(valid_idxes, dtype=int)] = predictions
        return scores, smiles_list, statuses

    def get_scores(self, input_seqs: Sequence[str]) -> np.ndarray:
        scores, _, _ = self.predict_with_details(input_seqs)
        return scores

    def __call__(self, input_seqs: Sequence[str]) -> np.ndarray:
        return self.get_scores(input_seqs)


Solubility = PepTuneSolubility
