"""Inject dummy secrets BEFORE any app import.

config.Settings() and several integration modules construct external clients at import time and read
required env vars. Tests are fully mocked/offline, so we fill harmless dummies. No constructor here
performs network I/O (pydantic validates only; genai.configure stores the key; Pinecone v8 is lazy;
boto3.client builds an object without contacting AWS).
"""

import os

_DUMMY = {
    "GOOGLE_API_KEY": "test-google-key",
    "PINECONE_API_KEY": "test-pinecone-key",
    "HUGGINGFACE_TOKEN": "test-hf-token",
    "AWS_REGION": "us-east-1",
    "S3_BUCKET_NAME": "test-bucket",
    "AWS_ACCESS_KEY_ID": "test-akid",
    "AWS_SECRET_ACCESS_KEY": "test-secret",
    "PINECONE_INDEX_NAME": "rag-knowledge-base",
    "LOG_JSON": "false",
}
for _k, _v in _DUMMY.items():
    os.environ.setdefault(_k, _v)  # a real shell/.env value still wins
