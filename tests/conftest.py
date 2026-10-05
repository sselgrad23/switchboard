"""Shared fixtures. Everything runs on BM25 + scripted models: no GPU, no network."""

from __future__ import annotations

import os

os.environ.setdefault("SWB_RETRIEVER", "bm25")

import pytest  # noqa: E402

from switchboard.retrieval import BM25Retriever  # noqa: E402
from switchboard.tools import Session  # noqa: E402
from switchboard.world.db import session_db  # noqa: E402


@pytest.fixture()
def retriever() -> BM25Retriever:
    return BM25Retriever()


@pytest.fixture()
def session(retriever: BM25Retriever) -> Session:
    return Session("LK-1003", session_db(), retriever, "2026-09-14")
