#!/usr/bin/env python3
"""Embedding engine for the Kuku promo discovery pipeline.

Wraps Google's EmbeddingGemma 2 behind a swappable `EmbeddingProvider`
interface and adds: task-instruction handling (query/document prompts),
batching, cosine similarity utilities and a content-addressed on-disk cache.

Model contract implemented here (EmbeddingGemma 2 model card):
* sentence-transformers `SentenceTransformer("google/embeddinggemma-2")`.
* Asymmetric retrieval: queries use the documented ``SearchQuery`` instruction
  (``task: search result | query: {query}``); corpus items use the document
  instruction ``title: {title} | text: {content}`` (``title: none`` when no
  real title exists). Instructions are applied as text, byte-identical to the
  documented prompt strings, so every provider shares one formatting path.
* Symmetric tasks share one instruction on both sides
  (``task: sentence similarity | query: {content}``).
* float32 on CPU (never float16); bfloat16 only where CUDA supports it.
* Text-only deployment loads no vision/audio encoders (270M effective).
* Matryoshka truncation (128/256/512) happens after encoding, with L2
  renormalisation, as the model card requires.

No LLM calls are made here; this module is pure encoder + math.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import numpy as np

PROMPT_ASYMMETRIC_SEARCH = "ASYMMETRIC_SEARCH"
PROMPT_SYMMETRIC_SIMILARITY = "SYMMETRIC_SIMILARITY"
VALID_PROMPT_STRATEGIES = (PROMPT_ASYMMETRIC_SEARCH, PROMPT_SYMMETRIC_SIMILARITY)

# Documented task instruction strings (EmbeddingGemma 2 model card).
SEARCH_QUERY_INSTRUCTION = "task: search result | query: {text}"
SENTENCE_SIMILARITY_INSTRUCTION = "task: sentence similarity | query: {text}"
DOCUMENT_INSTRUCTION = "title: {title} | text: {text}"


@dataclass(frozen=True)
class EmbeddingConfig:
    """All engine knobs. Changing any of them re-embeds affected texts."""

    model_id: str = "google/embeddinggemma-2"
    local_model_dir: Optional[str] = None
    device: str = "cpu"
    batch_size: int = 8
    dtype: str = "float32"  # float32 | bfloat16 (never float16)
    truncate_dim: Optional[int] = None  # 128/256/512; None -> native 768
    text_only: bool = True
    max_chars_per_text: int = 4000

    def validated(self) -> "EmbeddingConfig":
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError("dtype must be float32 or bfloat16 (float16 is unsafe for this model).")
        if self.truncate_dim not in {None, 128, 256, 512}:
            raise ValueError("truncate_dim must be one of None/128/256/512 (native 768 when None).")
        if self.max_chars_per_text < 200:
            raise ValueError("max_chars_per_text is too small to describe a candidate.")
        return self


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (vectors / norms).astype(np.float32)


def truncate_and_normalize(vectors: np.ndarray, target_dim: int) -> np.ndarray:
    """Matryoshka truncation with renormalisation, as the model card requires."""
    if target_dim > vectors.shape[1]:
        raise ValueError(f"Cannot truncate {vectors.shape[1]} dims down to {target_dim}.")
    return _normalize_rows(vectors[:, :target_dim])


def cosine_similarity_matrix(rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """Cosine similarity between every row and every column (unit vectors assumed)."""
    rows = np.asarray(rows, dtype=np.float32)
    columns = np.asarray(columns, dtype=np.float32)
    return np.clip(rows @ columns.T, -1.0, 1.0)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Swappable embedding backend.

    Implementations MUST:
    * return a float32 numpy array of shape (len(texts), native_dimension);
    * L2-normalise every row, so cosine similarity == dot product;
    * keep one fixed output dimension (native resolution, e.g. 768);
    * be deterministic for identical inputs.
    """

    model_id: str
    dimension: int
    metadata: dict

    def embed_texts(self, texts: list[str], prompt_name: Optional[str] = None) -> np.ndarray: ...


class SentenceTransformerProvider:
    """Official-implementation provider: sentence-transformers + EmbeddingGemma 2.

    Encodes at the model's native dimension and float32 (bfloat16 on CUDA that
    supports it). Truncation/truncation-width is applied by the engine so that
    the cache always stores full-resolution, strategy-tagged vectors.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config: EmbeddingConfig = (config or EmbeddingConfig()).validated()
        self.model = None
        self.model_id = self.config.local_model_dir or self.config.model_id
        self.dimension = 768  # native output dimension of EmbeddingGemma 2
        self.prompts: dict = {}
        self.metadata: dict = {}
        self.load_calls: int = 0

    def load(self) -> None:
        if self.model is not None:
            return
        self.load_calls += 1
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - model stack missing
            raise RuntimeError(
                "The sentence-transformers package is required for the real provider; "
                "it is only available for the official EmbeddingGemma 2 implementation."
            ) from exc

        dtype = import_bf16_supported() if self.config.dtype == "bfloat16" else self.config.dtype
        model_kwargs: dict = {"torch_dtype": dtype}
        config_kwargs: dict = {"vision_config": None, "audio_config": None} if self.config.text_only else {}
        try:
            model = SentenceTransformer(
                self.model_id,
                device=self.config.device,
                model_kwargs=model_kwargs,
                config_kwargs=config_kwargs,
            )
            self.metadata = {"text_only": self.config.text_only, "torch_dtype": str(dtype)}
        except (ValueError, KeyError, TypeError) as exc:
            # Older sentence-transformers versions may reject config_kwargs;
            # retry once without selective encoder loading and flag it.
            model = SentenceTransformer(self.model_id, device=self.config.device, model_kwargs=model_kwargs)
            self.metadata = {"text_only": False, "torch_dtype": str(dtype),
                             "config_kwargs_fallback": True, "fallback_error": str(exc)}
        self.model = model
        self.prompts = dict(getattr(model, "prompts", {}) or {})
        self.metadata["available_prompts"] = sorted(self.prompts.keys())
        self.metadata["device"] = str(self.config.device)

    def embed_texts(self, texts: list[str], prompt_name: Optional[str] = None) -> np.ndarray:
        """Encode pre-formatted texts, or texts with a documented prompt name.

        The retrieval engine formats instructions itself (single deterministic
        path), so `prompt_name` is only used by callers who want the model's
        built-in prompt application instead.
        """
        self.load()
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        encode_kwargs: dict = {
            "batch_size": self.config.batch_size,
            "convert_to_numpy": True,
            "normalize_embeddings": True,
            "show_progress_bar": False,
        }
        if prompt_name is not None:
            encode_kwargs["prompt_name"] = prompt_name
        vectors = self.model.encode(texts, **encode_kwargs)
        return np.asarray(vectors, dtype=np.float32)


def import_bf16_supported() -> str:  # pragma: no cover - CPU-only environment
    """dtype for model_kwargs; bfloat16 only on CUDA supporting it."""
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return "bfloat16"
    except ImportError:
        pass
    return "float32"


class OfflineTestProvider:
    """Deterministic provider for unit tests and offline runs (no model download).

    Similarity semantics: token overlap. Distinct tokens map to fixed unit
    dimensions via sha1, so two texts have cosine similarity proportional to
    their weighted shared vocabulary. Useful for testing the retrieval math;
    it is never used by the production CLI.
    """

    def __init__(self, dimension: int = 64):
        self.dimension = dimension
        self.model_id = "offline-test-provider-v1"
        self.prompts: dict[str, str] = {}
        self.embed_calls = 0
        self.metadata: dict = {"deterministic": True, "similarity": "token-overlap"}

    @staticmethod
    def _tokens(text: str) -> dict[str, float]:
        tokens: dict[str, float] = {}
        for token in text.split():
            cleaned = "".join(ch for ch in token if ch.isalnum() or ch in "|:").strip("|:").lower()
            if cleaned:
                tokens[cleaned] = tokens.get(cleaned, 0.0) + 1.0
        return tokens

    def _dimension_index(self, token: str) -> int:
        return int(hashlib.sha1(token.encode("utf-8")).hexdigest(), 16) % self.dimension

    def embed_texts(self, texts: list[str], prompt_name: Optional[str] = None) -> np.ndarray:
        self.embed_calls += 1
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token, weight in self._tokens(text).items():
                vectors[row, self._dimension_index(token)] += weight
        return _normalize_rows(vectors)


class TaskInstructionFormatter:
    """Applies EmbeddingGemma 2 task instructions exactly as documented."""

    def __init__(self, strategy: str):
        if strategy not in VALID_PROMPT_STRATEGIES:
            raise ValueError(f"Unknown prompt strategy {strategy!r}; valid: {VALID_PROMPT_STRATEGIES}")
        self.strategy = strategy

    # -- query side (category descriptions) -----------------------------------
    def format_category_query(self, description: str) -> str:
        if self.strategy == PROMPT_ASYMMETRIC_SEARCH:
            return SEARCH_QUERY_INSTRUCTION.format(text=description)
        return SENTENCE_SIMILARITY_INSTRUCTION.format(text=description)

    # -- document side (candidate content) ------------------------------------
    def format_candidate_document(self, body: str, title: Optional[str] = None) -> str:
        if self.strategy == PROMPT_SYMMETRIC_SIMILARITY:
            return SENTENCE_SIMILARITY_INSTRUCTION.format(text=body)
        return DOCUMENT_INSTRUCTION.format(title=title or "none", text=body)


class EmbeddingCache:
    """Content-addressed on-disk cache (npz), atomic writes.

    Keys are sha1 digests over everything that could change a vector: engine
    token (provider id, strategy, dtype, dimension) plus the *exact formatted
    source text*. Two runs with the same config reuse vectors automatically;
    changed input text or settings re-embed only what changed.
    """

    _META_NAME = "cache_meta.json"

    def __init__(self, cache_dir: Path, engine_token: str, native_dim: int):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.engine_token = engine_token
        self._path = self.cache_dir / "embedding_cache.npz"
        self.native_dim = native_dim
        self._rows: dict[str, np.ndarray] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        with np.load(self._path) as archive:
            for key in archive.files:
                self._rows[key] = np.array(archive[key], dtype=np.float32)

    @staticmethod
    def cache_key(engine_token: str, role: str, text: str) -> str:
        digest = hashlib.sha1()
        for part in (engine_token, role, text):
            digest.update(part.encode("utf-8"))
            digest.update(b"\x00")
        return digest.hexdigest()

    def key_for(self, role: str, text: str) -> str:
        return self.cache_key(self.engine_token, role, text)

    def get(self, keys: list[str]) -> dict[str, np.ndarray]:
        return {key: self._rows[key].copy() for key in keys if key in self._rows}

    def put(self, keyed: dict[str, np.ndarray]) -> int:
        added = 0
        for key, vector in keyed.items():
            if key in self._rows:
                continue
            self._rows[key] = np.asarray(vector, dtype=np.float32)
            added += 1
        if added:
            self._flush()
        return added

    def _flush(self) -> None:
        temporary = self._path.with_name(self._path.name + ".tmp")
        keys = sorted(self._rows)
        arrays = {}
        for key in keys:
            vector = self._rows[key]
            if vector.shape[0] != self.native_dim:
                raise ValueError(f"Cached vector {key!r} has width {vector.shape[0]}, expected {self.native_dim}.")
            arrays[key] = vector
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False, dir=str(self.cache_dir)) as handle:
            np.savez_compressed(handle, **arrays)
            temporary = Path(handle.name)
        temporary.replace(self._path)
        (self.cache_dir / self._META_NAME).write_text(
            json.dumps({"engine_token": self.engine_token, "rows": len(self._rows)}, indent=2),
            encoding="utf-8",
        )

    def stats(self) -> dict:
        return {"entries": len(self._rows), "path": str(self._path), "engine_token": self.engine_token}


class EmbeddingEngine:
    """High-level embedding facade used by the retrieval stage.

    Embeds candidate documents and category queries with the documented task
    instructions for the configured strategy, consults/refreshes the cache,
    and returns vectors in a fixed, input-ordered row layout.
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
        config: Optional[EmbeddingConfig] = None,
        strategy: str = PROMPT_ASYMMETRIC_SEARCH,
        cache_dir: Optional[Path] = None,
    ):
        self.config = (config or EmbeddingConfig()).validated()
        self.provider = provider
        self.strategy = strategy
        self.formatter = TaskInstructionFormatter(strategy)
        if isinstance(provider, SentenceTransformerProvider):
            provider.config = self.config
        self.native_dimension = provider.dimension
        self.dimension = self.config.truncate_dim or self.native_dimension
        self.cache_dir = Path(cache_dir or Path("retrieval_output/embedding_cache"))
        engine_token = self._engine_token()
        self.cache = EmbeddingCache(self.cache_dir, engine_token, native_dim=self.native_dimension)

    def _engine_token(self) -> str:
        """Stable per-provider/per-strategy token used in cache keys."""
        return "|".join([
            getattr(self.provider, "model_id", "unknown-provider"),
            self.strategy,
            self.config.dtype,
            str(self.provider.dimension),
        ])

    def signature(self) -> str:
        """Stable signature attached to outputs for cross-run provenance checks."""
        return "|".join([
            getattr(self.provider, "model_id", "unknown-provider"),
            self.strategy,
            str(self.native_dimension),
            str(self.config.truncate_dim or self.native_dimension),
        ])

    # -- documents -------------------------------------------------------------
    def embed_candidate_documents(
        self, bodies: list[str], titles: Optional[list[Optional[str]]] = None
    ) -> tuple[np.ndarray, dict]:
        titles = titles or [None] * len(bodies)
        if len(titles) != len(bodies):
            raise ValueError("titles length must match bodies length.")
        bodies = [_clip_text(body, self.config.max_chars_per_text) for body in bodies]
        formatted = [
            self.formatter.format_candidate_document(body, title)
            for body, title in zip(bodies, titles)
        ]
        keys = [self.cache.key_for("doc", text) for text in formatted]
        return self._embed_with_cache(formatted, keys)

    # -- queries ---------------------------------------------------------------
    def embed_category_queries(self, descriptions: list[str]) -> tuple[np.ndarray, dict]:
        formatted = [self.formatter.format_category_query(description) for description in descriptions]
        keys = [self.cache.key_for("category", text) for text in formatted]
        return self._embed_with_cache(formatted, keys)

    # -- shared embedding path ---------------------------------------------------
    def _embed_with_cache(self, texts: list[str], keys: list[str]) -> tuple[np.ndarray, dict]:
        key_to_row = {key: index for index, key in enumerate(keys)}
        vectors = np.zeros((len(texts), self.native_dimension), dtype=np.float32)
        cached = self.cache.get(keys)
        missing_rows = [index for index, key in enumerate(keys) if key not in cached]
        for key, vector in cached.items():
            vectors[key_to_row[key]] = vector
        if missing_rows:
            encoded = self.provider.embed_texts([texts[index] for index in missing_rows], prompt_name=None)
            encoded = np.asarray(encoded, dtype=np.float32)
            if encoded.shape[0] != len(missing_rows):
                raise RuntimeError(f"Provider returned {encoded.shape[0]} vectors for {len(missing_rows)} texts.")
            if encoded.shape[1] != self.native_dimension:
                raise RuntimeError(
                    f"Provider dimension mismatch: expected {self.native_dimension}, got {encoded.shape[1]}."
                )
            encoded = _normalize_rows(encoded)
            for slot, index in enumerate(missing_rows):
                vectors[index] = encoded[slot]
            self.cache.put({keys[index]: vectors[index] for index in missing_rows})
        out_vectors = vectors
        if self.config.truncate_dim is not None and self.native_dimension > self.config.truncate_dim:
            out_vectors = truncate_and_normalize(vectors, self.config.truncate_dim)
        stats = {
            "total_texts": len(texts),
            "cache_hits": len(keys) - len(missing_rows),
            "encoded_now": len(missing_rows),
            "cache_entries": self.cache.stats()["entries"],
        }
        return out_vectors, stats


def _clip_text(text: str, limit: int) -> str:
    """Deterministic head+tail clipping of over-long texts (output <= limit)."""
    text = (text or "").replace("\x00", " ").strip()
    if len(text) <= limit:
        return text
    separator = " … "
    body_limit = limit - len(separator)
    head = body_limit // 2
    tail = body_limit - head
    return text[:head].rstrip() + separator + text[-tail:].lstrip()


def memory_usage_bytes() -> dict:
    """Current/peak RSS of this process (Linux ru_maxrss is KiB)."""
    usage = {"rss_peak_bytes": 0, "source": "unknown"}
    try:
        import resource
        usage["rss_peak_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        usage["source"] = "resource.ru_maxrss"
    except (ImportError, AttributeError):  # pragma: no cover
        pass
    return usage
