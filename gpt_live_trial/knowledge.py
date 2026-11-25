"""Search only the public Chert reference documents bundled with the app."""

from functools import lru_cache
import re
import sqlite3

from realtime.config import ROOT
from realtime.web_search import SearchResult


SOURCES = (
    ("README.md", "Demo", "Waiting queue and claiming your turn"),
    (
        "README.md",
        "Known limitations",
        "Session duration, simultaneous callers and frame rate",
    ),
    (
        "realtime/demo-avatars/README.md",
        "Demo reference portraits",
        "Gallery, portraits and stock voices",
    ),
)
STOP = set(
    "what which who how when me my your the a an of to for by do is are in on from does can you tell according chert demo documentation docs public about please".split()
)
PUBLIC = "https://github.com/cherthq/realtime-research/blob/main/"


@lru_cache(maxsize=1)
def index():
    db = sqlite3.connect(":memory:")
    db.execute(
        "CREATE VIRTUAL TABLE passages USING fts5(title, body, path UNINDEXED, tokenize='porter unicode61')"
    )
    for path, heading, label in SOURCES:
        markdown = (ROOT / path).read_text(encoding="utf-8")
        section = re.search(
            r"^#{1,2} " + re.escape(heading) + r"[ \t]*$\n(.*?)(?=^#{1,2} |\Z)",
            markdown,
            re.M | re.S,
        )
        if not section:
            continue
        for paragraph in re.split(r"\n\s*\n", section.group(1)):
            paragraph = (
                re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", paragraph)
                .replace("**", "")
                .strip()
            )
            if not paragraph or paragraph.startswith(
                ("https:", "[", "Try the live demo")
            ):
                continue
            db.execute("INSERT INTO passages VALUES(?,?,?)", (label, paragraph, path))
    return db


def retrieve(query):
    terms = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if w not in STOP][:32]
    if not terms:
        return SearchResult("no_matching_documents")
    match = " OR ".join(f'"{word}"' for word in terms)
    rows = (
        index()
        .execute(
            "SELECT title,body,path FROM passages WHERE passages MATCH ? ORDER BY bm25(passages,0.45,2.0,0) LIMIT 3",
            (match,),
        )
        .fetchall()
    )
    if not rows:
        return SearchResult("no_matching_documents")
    links = {path: {"url": PUBLIC + path, "title": label} for label, _, path in rows}
    return SearchResult(
        "ok", "\n".join(body for _, body, _ in rows)[:2600], tuple(links.values())
    )
