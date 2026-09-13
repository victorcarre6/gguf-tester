"""llama-cpp-python wrapper for dynamic scanning.

Loads a GGUF model, exposes chat completion + embedding helpers, manages cleanup.
"""

from __future__ import annotations

import gc
import logging
from typing import Any

import numpy as np

from ggufscan.parser import ParsedGGUF

log = logging.getLogger(__name__)


ARCH_TO_CHAT_FORMAT = {
    "qwen": "chatml",
    "qwen2": "chatml",
    "qwen3": "chatml",
    "qwen35": "chatml",
    "llama": "llama-3",
    "mistral": "mistral-instruct",
    "mixtral": "mistral-instruct",
    "gemma": "gemma",
    "gemma2": "gemma",
    "phi3": "phi3",
    "granite": "chatml",
}

# Architectures that emit <think>...</think> reasoning blocks by default.
# Used by the CLI to auto-bump max_tokens so the actual answer has room.
THINKING_ARCHS = {"qwen3", "qwen35", "glm4", "deepseek_r1", "deepseek_v2"}


def is_thinking_model(parsed: ParsedGGUF) -> bool:
    arch = parsed.architecture.lower()
    return any(t in arch for t in THINKING_ARCHS)


def detect_chat_format(parsed: ParsedGGUF, override: str | None = None) -> str | None:
    """Return the chat_format string to pass to llama-cpp-python, or None.

    None tells llama-cpp-python to use the embedded `tokenizer.chat_template`
    directly (its minja engine executes it). load_guard must have cleared the
    template (CLEAN or WARN verdict) before this path is taken.

    Priority: explicit override → embedded template (None) → arch map → chatml.
    """
    if override:
        return override
    # If the GGUF carries an embedded chat_template, let llama-cpp-python run it
    # natively instead of maintaining a parallel hand-coded arch map.
    if parsed.metadata.get("tokenizer.chat_template"):
        return None
    arch = parsed.architecture.lower()
    for key, fmt in ARCH_TO_CHAT_FORMAT.items():
        if key in arch:
            return fmt
    log.warning("Unknown architecture %r, defaulting to chatml", arch)
    return "chatml"


class ModelHandle:
    """Context-manager wrapping llama_cpp.Llama.

    Use:
        with ModelHandle(parsed, n_ctx=4096, tensor_split=[0.5, 0.5]) as h:
            text = h.complete([{"role": "user", "content": "hi"}])
            vec  = h.embed("some text")  # requires separate handle (see below)

    `embed` and chat completion are mutually exclusive in llama.cpp:
    set `embedding=True` to use embeddings, otherwise chat only.
    """

    def __init__(
        self,
        parsed: ParsedGGUF,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        tensor_split: list[float] | None = None,
        chat_format: str | None = None,
        embedding: bool = False,
        seed: int = -1,
        verbose: bool = False,
        n_batch: int = 512,
    ):
        self.parsed = parsed
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.tensor_split = tensor_split
        self.chat_format = detect_chat_format(parsed, chat_format)
        self.embedding = embedding
        self.seed = seed
        self.verbose = verbose
        self.n_batch = n_batch
        self.llm = None
        self._embed_seq_slot: int | None = None

    _CHARS_PER_TOKEN = 3
    # llama-cpp-python forces n_seq_max=256 in embedding mode (regardless of
    # what we pass), so n_ctx_seq = n_ctx/256 stays small and bumping n_ctx
    # only wastes VRAM. We just cap batch + truncate input to the slot size.
    _EMBED_SLOT_TOKENS = 256

    def __enter__(self) -> "ModelHandle":
        from llama_cpp import Llama

        if self.embedding:
            self._embed_seq_slot = self._EMBED_SLOT_TOKENS
            n_batch = min(self.n_batch, self._EMBED_SLOT_TOKENS)
        else:
            self._embed_seq_slot = None
            n_batch = self.n_batch

        kwargs: dict[str, Any] = dict(
            model_path=self.parsed.file_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            n_batch=n_batch,
            embedding=self.embedding,
            verbose=self.verbose,
            seed=self.seed,
        )
        if self.embedding:
            kwargs["n_ubatch"] = n_batch
            kwargs["pooling_type"] = 1  # LLAMA_POOLING_TYPE_MEAN
        else:
            kwargs["chat_format"] = self.chat_format
        if self.tensor_split:
            kwargs["tensor_split"] = self.tensor_split

        log.info("Loading Llama (chat_format=%s, embedding=%s, n_ctx=%d, n_batch=%d, tensor_split=%s)",
                 self.chat_format, self.embedding, self.n_ctx, n_batch, self.tensor_split)
        self.llm = Llama(**kwargs)
        return self

    def __exit__(self, exc_type, exc, tb):
        # finally guarantees gc.collect() runs even if del raises, so VRAM
        # is released before the next ModelHandle enters (embedding pass).
        try:
            if self.llm is not None:
                del self.llm
                self.llm = None
        finally:
            gc.collect()
        return False

    def complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.3,
        stop: list[str] | None = None,
        **kwargs,
    ) -> str:
        if self.embedding:
            raise RuntimeError("Handle initialised with embedding=True; cannot run chat completion.")
        resp = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or [],
            **kwargs,
        )
        return resp["choices"][0]["message"].get("content", "") or ""

    def embed(self, text: str, max_chars: int | None = None) -> np.ndarray:
        if not self.embedding:
            raise RuntimeError("Handle initialised with embedding=False; cannot extract embeddings.")
        # Truncate to fit one per-sequence slot (n_ctx_seq * chars_per_token,
        # with a margin to stay clear of the boundary).
        if max_chars is None and self._embed_seq_slot:
            max_chars = int(self._embed_seq_slot * self._CHARS_PER_TOKEN * 0.8)
        if max_chars and len(text) > max_chars:
            text = text[:max_chars]
        vec = np.asarray(self.llm.embed(text), dtype=np.float32)
        # If pooling_type=MEAN is honored, vec is 1D. Otherwise mean-pool
        # over the token axis ourselves.
        if vec.ndim == 2:
            vec = vec.mean(axis=0)
        return vec

    def n_layers(self) -> int:
        return int(self.llm.n_layers()) if self.llm else 0

    def n_vocab(self) -> int:
        return int(self.llm.n_vocab()) if self.llm else 0
