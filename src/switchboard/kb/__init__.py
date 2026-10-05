"""The Larkspur help centre: the corpus behind the ``search_help`` tool."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel

KB_PATH = Path(__file__).with_name("articles.json")


class Article(BaseModel):
    """One help-centre article. ``source`` is "official" or "community"."""

    id: str
    title: str
    text: str
    source: str = "official"

    @property
    def indexed_text(self) -> str:
        return f"{self.title}. {self.text}"


@lru_cache(maxsize=1)
def load_articles() -> tuple[Article, ...]:
    """All articles, in file order."""
    raw = json.loads(KB_PATH.read_text(encoding="utf-8"))
    return tuple(Article(**a) for a in raw)
