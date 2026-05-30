import importlib

import pytest

REQUIRED = {
    "GOOGLE_API_KEY": "g",
    "PINECONE_API_KEY": "p",
    "HUGGINGFACE_TOKEN": "h",
    "AWS_REGION": "us-east-1",
    "S3_BUCKET_NAME": "b",
    "AWS_ACCESS_KEY_ID": "ak",
    "AWS_SECRET_ACCESS_KEY": "sk",
}


def _fresh(monkeypatch, env):
    for k in list(REQUIRED) + [
        "UPLOADTHING_API_KEY",
        "PINECONE_INDEX_NAME",
        "LOG_JSON",
        "ENVIRONMENT",
        "S3_ENDPOINT_URL",
    ]:
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import config

    importlib.reload(config)
    return config


def test_loads_required(monkeypatch):
    c = _fresh(monkeypatch, REQUIRED)
    assert c.settings.GOOGLE_API_KEY == "g"


def test_index_name_default(monkeypatch):
    c = _fresh(monkeypatch, REQUIRED)
    assert c.settings.PINECONE_INDEX_NAME == "rag-knowledge-base"


def test_optionals_default(monkeypatch):
    c = _fresh(monkeypatch, REQUIRED)
    assert c.settings.UPLOADTHING_API_KEY is None and c.settings.LOG_JSON is False


def test_missing_required_raises(monkeypatch):
    bad = dict(REQUIRED)
    del bad["GOOGLE_API_KEY"]
    with pytest.raises(Exception):
        _fresh(monkeypatch, bad)


def test_environment_default(monkeypatch):
    c = _fresh(monkeypatch, REQUIRED)
    assert c.settings.ENVIRONMENT == "development"


def test_s3_endpoint_optional(monkeypatch):
    c = _fresh(monkeypatch, REQUIRED)
    assert c.settings.S3_ENDPOINT_URL is None
