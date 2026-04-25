"""Tests for the embeddings service."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_embeddings_module():
    module_path = Path(__file__).resolve().parents[1] / "src/services/embeddings.py"
    spec = importlib.util.spec_from_file_location("test_embeddings_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_embedding_service_loads_models_from_local_cache_only(monkeypatch):
    """Runtime model loads should stay on the local cache warmed during deploy."""
    embeddings = _load_embeddings_module()

    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    service = embeddings.EmbeddingsService(embeddings.MODEL_NAME)
    service._load_model()

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (embeddings.MODEL_NAME,)
    assert kwargs["device"] == "cpu"
    assert kwargs["local_files_only"] is True
