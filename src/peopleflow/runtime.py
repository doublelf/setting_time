"""CPU runtime tuning.

Running a YOLO detector and a ReID encoder on the same machine easily oversubscribes
the CPU: ONNX Runtime creates one intra-op thread per core for *each* session, and
PyTorch adds its own pool on top.  On a 16-thread machine that is ~48 runnable threads
fighting over 16 cores, which turns a 40 ms frame into a 400 ms frame.

This module caps the thread pools so the three engines share the cores instead of
fighting for them.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_INTRA = 4
DEFAULT_INTER = 1


def configure_process_threads(intra: int = DEFAULT_INTRA) -> None:
    """Limit PyTorch's own thread pools."""
    import torch

    torch.set_num_threads(max(1, intra))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Interop pool can only be set once per process; ignore if already initialised.
        pass


def set_env_defaults(intra: int = DEFAULT_INTRA) -> None:
    """Export the usual OpenMP environment variables before the runtimes start."""
    os.environ.setdefault("OMP_NUM_THREADS", str(intra))
    os.environ.setdefault("MKL_NUM_THREADS", str(intra))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(intra))


def retune_onnx_session(backend, model_path: str | Path, intra: int = DEFAULT_INTRA, inter: int = DEFAULT_INTER) -> bool:
    """Rebuild an ONNX Runtime session with a bounded thread pool.

    The thread pool is fixed at session creation time, so the only way to change it is
    to recreate the session.  Returns True when the session was replaced.
    """
    session = getattr(backend, "session", None)
    if session is None:
        return False
    try:
        import onnxruntime as ort
    except ImportError:
        return False

    path = Path(model_path)
    if not path.is_file():
        return False

    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, intra)
    options.inter_op_num_threads = max(1, inter)
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Let idle worker threads sleep instead of spinning, which keeps them out of the
    # way of the other inference engines.
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")

    providers = session.get_providers()
    try:
        backend.session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
    except Exception:  # noqa: BLE001 - keep the original session if the rebuild fails
        return False
    return True


def tune_yolo_model(model, model_path: str | Path, intra: int = DEFAULT_INTRA, inter: int = DEFAULT_INTER) -> bool:
    """Apply the thread cap to a YOLO model's ONNX backend, if it has one.

    For exported models the AutoBackend is not reachable through ``model.model`` (that
    attribute keeps the checkpoint path); it only exists once a predictor has been
    created, at ``model.predictor.model``.
    """
    candidates = []
    predictor = getattr(model, "predictor", None)
    if predictor is not None:
        candidates.append(getattr(predictor, "model", None))
    candidates.append(getattr(model, "model", None))

    for backend in candidates:
        if backend is None or isinstance(backend, str):
            continue
        if retune_onnx_session(backend, model_path, intra, inter):
            return True
    return False


def tune_reid_encoder(encoder, model_path: str | Path, intra: int = DEFAULT_INTRA, inter: int = DEFAULT_INTER) -> bool:
    """Apply the thread cap to a ReID encoder's ONNX backend, if it has one."""
    backend = getattr(encoder, "model", None)
    if backend is None:
        return False
    return retune_onnx_session(backend, model_path, intra, inter)
