"""
Source registry.

"Train on online data that's allowed to be used by the public" is
implemented literally: this file is the ONLY place that decides what
counts as fair game, and both the bulk ingestion pipeline (ingest.py)
and the router's opportunistic self-training queue (router.py) check
against it before anything gets queued for training. Nothing is
scraped or trained on just because it was fetched to answer a
question -- fetching to *read* and fetching to *train on* are
deliberately different gates.

Add a source by adding an entry here with an honest license note.
Do not add anything paywalled, ToS-restricted, or copyrighted without
a license that permits reuse -- that defeats the point of the gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List
from urllib.parse import urlparse


@dataclass
class Source:
    key: str
    domains: List[str]              # hostnames this source covers
    license: str                    # human-readable license summary
    notes: str


REGISTRY: List[Source] = [
    Source(
        key="gutenberg",
        domains=["gutenberg.org", "www.gutenberg.org"],
        license="Public domain (US) for texts whose copyright has expired",
        notes="Project Gutenberg. Always check the per-book license header; "
              "a small fraction of texts carry additional restrictions.",
    ),
    Source(
        key="wikipedia",
        domains=["wikipedia.org", "en.wikipedia.org", "simple.wikipedia.org"],
        license="CC BY-SA 4.0",
        notes="Attribution + share-alike required if redistributed verbatim; "
              "fine for training on with attribution retained in the manifest.",
    ),
    Source(
        key="wikisource",
        domains=["wikisource.org"],
        license="Public domain / CC BY-SA depending on the work",
        notes="Primary-source texts; check per-page license same as Wikipedia.",
    ),
    Source(
        key="arxiv",
        domains=["arxiv.org"],
        license="Per-paper (often CC BY or CC0); abstracts generally reusable",
        notes="Use the abstracts/metadata API, not full PDFs, unless a "
              "paper's specific license permits full-text reuse.",
    ),
    Source(
        key="usgov",
        domains=["nih.gov", "nasa.gov", "usa.gov", "loc.gov"],
        license="US federal government works are public domain (17 U.S.C. 105)",
        notes="Restricted to *.gov domains listed here, not every .gov site.",
    ),
]

_DOMAIN_INDEX = {d: s for s in REGISTRY for d in s.domains}


def _host(url: str) -> str:
    # .hostname, not .netloc: netloc keeps "user@" and ":port", so
    # "https://www.gutenberg.org:443/x" was refused while the lookalike
    # "https://gutenberg.org@evil.example/" only passed by luck of the suffix rule.
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def is_allow_listed(url: str) -> bool:
    host = _host(url)
    return any(host == d or host.endswith("." + d) for d in _DOMAIN_INDEX)


def source_for(url: str) -> Source | None:
    host = _host(url)
    for d, s in _DOMAIN_INDEX.items():
        if host == d or host.endswith("." + d):
            return s
    return None
