"""Lazy ONNX Runtime session management.

Sessions are created on first use and cached for the lifetime of the process.
A missing artefact never crashes the API: the registry reports the file as
unavailable, the health endpoint degrades gracefully, and only the affected
workspace returns a 503 with an actionable message. That behaviour matters
because the evaluator clones the repository before downloading the ONNX weights.
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import onnxruntime as ort

from .config import MODEL_SPECS, MODEL_SPECS_BY_KEY, ONNX_THREADS, ModelSpec, relative_model_path

logger = logging.getLogger("onnx.registry")


class ModelUnavailableError(RuntimeError):
    """Raised when a required ONNX artefact is missing or cannot be loaded."""

    def __init__(self, key: str, path: str, reason: str) -> None:
        super().__init__(f"{MODEL_SPECS_BY_KEY[key].description} is unavailable: {reason}")
        self.key = key
        self.path = path
        self.reason = reason


@dataclass
class _Entry:
    session: ort.InferenceSession | None = None
    error: str | None = None
    load_ms: float | None = None
    input_names: tuple[str, ...] = ()
    output_names: tuple[str, ...] = ()
    providers: tuple[str, ...] = ()
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


@functools.lru_cache(maxsize=None)
def _external_data_present(path_string: str, size_bytes: int) -> bool:
    """Report whether an ONNX graph file is self-contained.

    PyTorch splits large exports into a small graph file plus a ``.onnx.data``
    weight sidecar. If the sidecar is absent the graph cannot be executed, so
    this has to be verified before handing the file to onnxruntime.
    """
    from pathlib import Path

    path = Path(path_string)
    if path.with_name(path.name + ".data").is_file():
        return True
    if not path.is_file():
        return False
    if size_bytes < 2_000_000_000:
        # Small graphs are always written inline; only scan when that is unclear.
        with path.open("rb") as handle:
            return b".onnx.data" not in handle.read()
    return False


class ModelRegistry:
    """Thread-safe cache of ONNX Runtime sessions keyed by :attr:`ModelSpec.key`."""

    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {spec.key: _Entry() for spec in MODEL_SPECS}

    # -- introspection -----------------------------------------------------

    def exists(self, key: str) -> bool:
        return self._entries[key].session is not None or MODEL_SPECS_BY_KEY[key].path.is_file()

    def is_loaded(self, key: str) -> bool:
        return self._entries[key].session is not None

    def info(self, key: str) -> dict[str, Any]:
        """Public metadata for a single model, without exposing the cache."""
        spec = MODEL_SPECS_BY_KEY[key]
        entry = self._entries[key]
        available, reason = self._availability(spec)
        return {
            "key": spec.key,
            "task": spec.task,
            "role": spec.role,
            "description": spec.description,
            "path": relative_model_path(spec.path),
            "available": available,
            "loaded": entry.session is not None,
            "size_bytes": spec.path.stat().st_size if spec.path.is_file() else 0,
            "providers": list(entry.providers),
            "inputs": list(entry.input_names),
            "outputs": list(entry.output_names),
            "load_ms": round(entry.load_ms, 1) if entry.load_ms else None,
            "reason": reason,
        }

    def _availability(self, spec: ModelSpec) -> tuple[bool, str | None]:
        entry = self._entries[spec.key]
        if entry.session is not None:
            return True, None
        if entry.error:
            return False, entry.error
        if not spec.path.is_file():
            return False, f"{relative_model_path(spec.path)} does not exist"
        size = spec.path.stat().st_size
        if not _external_data_present(str(spec.path), size):
            return False, f"{spec.path.name}.data sidecar file is missing"
        return True, None

    def status(self) -> list[dict[str, Any]]:
        """Report every configured model; used by ``/api/health``."""
        report: list[dict[str, Any]] = []
        for spec in MODEL_SPECS:
            entry = self._entries[spec.key]
            available, reason = self._availability(spec)
            present = spec.path.is_file()
            report.append(
                {
                    "key": spec.key,
                    "task": spec.task,
                    "role": spec.role,
                    "description": spec.description,
                    "path": relative_model_path(spec.path),
                    "available": available,
                    "loaded": entry.session is not None,
                    "size_bytes": spec.path.stat().st_size if present else 0,
                    "providers": list(entry.providers),
                    "inputs": list(entry.input_names),
                    "outputs": list(entry.output_names),
                    "load_ms": round(entry.load_ms, 1) if entry.load_ms else None,
                    "reason": reason,
                }
            )
        return report

    # -- loading -----------------------------------------------------------

    def load(self, key: str) -> ort.InferenceSession:
        spec = MODEL_SPECS_BY_KEY[key]
        entry = self._entries[key]
        if entry.session is not None:
            return entry.session

        with entry.lock:
            if entry.session is not None:
                return entry.session

            available, reason = self._availability(spec)
            if not available:
                entry.error = reason
                raise ModelUnavailableError(key, relative_model_path(spec.path), reason or "unknown")

            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            if ONNX_THREADS:
                options.intra_op_num_threads = ONNX_THREADS

            started = time.perf_counter()
            try:
                session = ort.InferenceSession(
                    str(spec.path), sess_options=options, providers=["CPUExecutionProvider"]
                )
            except Exception as exc:  # noqa: BLE001 - surfaced to the caller as 503
                entry.error = f"{type(exc).__name__}: {exc}"
                logger.exception("failed to load %s", spec.path)
                raise ModelUnavailableError(
                    key, relative_model_path(spec.path), entry.error
                ) from exc

            entry.load_ms = (time.perf_counter() - started) * 1000.0
            entry.session = session
            entry.error = None
            entry.input_names = tuple(item.name for item in session.get_inputs())
            entry.output_names = tuple(item.name for item in session.get_outputs())
            entry.providers = tuple(session.get_providers())
            logger.info(
                "loaded %s in %.0f ms (providers=%s)",
                spec.path.name,
                entry.load_ms,
                entry.providers,
            )
            return session

    def preload(self, keys: tuple[str, ...] | None = None) -> dict[str, str]:
        """Warm the cache; returns ``{key: "ok" | reason}`` for every attempt."""
        outcomes: dict[str, str] = {}
        for spec in MODEL_SPECS:
            if keys is not None and spec.key not in keys:
                continue
            try:
                self.load(spec.key)
            except ModelUnavailableError as exc:
                outcomes[spec.key] = exc.reason
            else:
                outcomes[spec.key] = "ok"
        return outcomes

    # -- inference ---------------------------------------------------------

    def run(self, key: str, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        """Run one model, mapping logical input names onto the graph's names.

        ``feed`` is ordered: each entry is matched positionally against the
        graph's declared inputs, so callers can use readable names such as
        ``photo``/``style_id`` without hard-coding the exporter's naming.
        """
        session = self.load(key)
        entry = self._entries[key]
        graph_inputs = list(entry.input_names)
        if not graph_inputs:
            graph_inputs = [item.name for item in session.get_inputs()]

        resolved: dict[str, np.ndarray] = {}
        for index, (logical_name, array) in enumerate(feed.items()):
            graph_name = graph_inputs[index] if index < len(graph_inputs) else logical_name
            if array.dtype.kind == "f":
                value = np.ascontiguousarray(array, dtype=np.float32)
            elif array.dtype.kind in "iu":
                value = np.ascontiguousarray(array, dtype=np.int64)
            else:
                value = np.ascontiguousarray(array)
            resolved[graph_name] = value

        outputs = list(entry.output_names) or None
        return session.run(outputs, resolved)


registry = ModelRegistry()
