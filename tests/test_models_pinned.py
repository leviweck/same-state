import inspect
import re

import pytest

from plm_kit.models import EMBEDDERS, RERANKERS

SHA = re.compile(r"^[0-9a-f]{40}$")


def test_all_models_pinned_and_permissive():
    for spec in {**RERANKERS, **EMBEDDERS}.values():
        assert SHA.match(spec["revision"]), spec["hf_id"]
        assert spec["license"] in ("apache-2.0", "mit")
        assert "marco" not in spec["hf_id"].lower()


def test_fastembed_signature_matches_pinned_version():
    fe = pytest.importorskip("fastembed")
    assert fe.__version__ == "0.9.0"
    from fastembed.common.model_description import ModelSource  # noqa: F401
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    params = inspect.signature(TextCrossEncoder.add_custom_model).parameters
    for p in ("model", "sources", "model_file", "license", "size_in_gb"):
        assert p in params
