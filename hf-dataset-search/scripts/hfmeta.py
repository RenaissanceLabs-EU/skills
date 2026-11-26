#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["model2vec>=0.6", "numpy"]
# ///
"""hfmeta - a local, searchable index of Hugging Face dataset *metadata*.

It never downloads dataset contents. It reads the public Hub API
(1000 dataset records per request) and, on demand, the dataset viewer API
for exact row counts and column schemas, and stores it all in one SQLite file.

Standard library only, Python 3.9+. The one exception is semantic search
(`embed`, `search --semantic`), which needs numpy and model2vec: run those
with `uv run hfmeta.py ...` and uv installs them from the header above.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

HUB = "https://huggingface.co/api/datasets"
VIEWER = "https://datasets-server.huggingface.co"
USER_AGENT = "hfmeta/0.1"
EMBED_MODEL = "minishlab/potion-multilingual-128M"


def default_db():
    """Where the index lives: $HFMETA_DB; else an existing hfmeta.db beside this script
    or in a folder above it (also inside an `hfmeta/` folder there); else ~/.cache/hfmeta."""
    if os.environ.get("HFMETA_DB"):
        return os.environ["HFMETA_DB"]
    cache = os.path.join(os.path.expanduser("~"), ".cache", "hfmeta", "hfmeta.db")
    here = os.path.dirname(os.path.realpath(__file__))
    for _ in range(4):
        for cand in (os.path.join(here, "hfmeta.db"), os.path.join(here, "hfmeta", "hfmeta.db")):
            if os.path.exists(cand):
                return cand
        here = os.path.dirname(here)
    return cache


DEFAULT_DB = default_db()

# ---------------------------------------------------------------- schema

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
  rid INTEGER PRIMARY KEY,
  id TEXT UNIQUE NOT NULL,
  author TEXT,
  name TEXT,
  pretty_name TEXT,
  description TEXT,
  created_at TEXT,
  last_modified TEXT,
  downloads INTEGER,
  likes INTEGER,
  trending REAL,
  gated TEXT,
  disabled INTEGER,
  license TEXT,
  license_class TEXT,
  size_category TEXT,
  size_rank INTEGER,
  card_rows INTEGER,
  card_bytes INTEGER,
  n_configs INTEGER,
  has_card INTEGER,
  sha TEXT,
  extra TEXT,
  crawled_at TEXT,
  viewer_status TEXT,
  viewer_rows INTEGER,
  viewer_bytes INTEGER,
  enriched_at TEXT
);
CREATE INDEX IF NOT EXISTS d_downloads ON datasets(downloads);
CREATE INDEX IF NOT EXISTS d_modified ON datasets(last_modified);
CREATE INDEX IF NOT EXISTS d_author ON datasets(author);

CREATE TABLE IF NOT EXISTS tags (
  rid INTEGER NOT NULL, kind TEXT NOT NULL, value TEXT NOT NULL,
  PRIMARY KEY (rid, kind, value)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS t_kv ON tags(kind, value);

CREATE TABLE IF NOT EXISTS columns (
  rid INTEGER NOT NULL, source TEXT NOT NULL, name TEXT NOT NULL, dtype TEXT,
  PRIMARY KEY (rid, source, name)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS c_name ON columns(name);

CREATE TABLE IF NOT EXISTS splits (
  rid INTEGER NOT NULL, source TEXT NOT NULL, config TEXT NOT NULL,
  split TEXT NOT NULL, num_rows INTEGER, num_bytes INTEGER,
  PRIMARY KEY (rid, source, config, split)
) WITHOUT ROWID;

CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(
  id, pretty_name, description, tags, tokenize='unicode61 remove_diacritics 0'
);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""

# Hub tag prefixes that carry structure ("language:sv" -> kind=language, value=sv).
TAG_KINDS = {
    "language": "language",
    "task_categories": "task",
    "task_ids": "task_id",
    "modality": "modality",
    "format": "format",
    "library": "library",
    "license": "license",
    "size_categories": "size",
    "source_datasets": "source",
    "annotations_creators": "annotations",
    "language_creators": "language_creators",
    "multilinguality": "multilinguality",
    "arxiv": "arxiv",
    "doi": "doi",
    "region": "region",
    "benchmark": "benchmark",
}

SIZE_ORDER = [
    "n<1K", "1K<n<10K", "10K<n<100K", "100K<n<1M", "1M<n<10M", "10M<n<100M",
    "100M<n<1B", "1B<n<10B", "10B<n<100B", "100B<n<1T", "n>1T",
]

# A dataset with thousands of configs should not bloat the index.
MAX_COLUMNS, MAX_SPLITS = 300, 200

# Card keys that are bulky or already stored elsewhere.
EXTRA_DROP = {
    "configs", "dataset_info", "license", "language", "task_categories", "task_ids",
    "size_categories", "tags", "pretty_name", "source_datasets", "annotations_creators",
    "language_creators", "multilinguality", "train-eval-index",
}


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def connect(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.executescript(SCHEMA)
    return db


# ---------------------------------------------------------------- classification

def license_class(lic):
    """Rough bucket for filtering. A heuristic on the declared tag, not legal advice."""
    if not lic:
        return "none"
    l = lic.lower()
    if "-nc" in l or l.endswith("nc") or "noncommercial" in l or "non-commercial" in l:
        return "noncommercial"
    if l.startswith(("mit", "apache", "bsd", "cc0", "isc", "unlicense", "pddl", "wtfpl",
                     "odc-by", "cdla-permissive", "zlib", "postgresql", "ecl", "bsl")):
        return "permissive"
    if l.startswith("cc-by-nd"):
        return "noderivs"
    if l.startswith(("cc-by-sa", "gpl", "agpl", "lgpl", "odbl", "cdla-sharing", "gfdl",
                     "mpl", "eupl", "osl", "cecill")):
        return "sharealike"
    if re.match(r"cc-by-\d", l) or l == "cc-by" or l == "cc":
        return "permissive"
    return "other"


def shapes(cols):
    """Guess what kind of training examples a schema holds, from (name, dtype) pairs."""
    names = {n.lower() for n, _ in cols}
    dtypes = {(t or "").lower() for _, t in cols}
    has = lambda *xs: any(x in names for x in xs)
    out = set()
    if has("messages", "conversations", "conversation", "dialogue", "dialog", "turns", "chat"):
        out.add("chat")
    if has("chosen") and has("rejected"):
        out.add("preference")
    if has("instruction") and has("output", "response", "answer", "completion"):
        out.add("instruction")
    if has("prompt") and has("completion", "response", "output", "answer"):
        out.add("prompt_completion")
    if has("question") and has("choices", "options"):
        out.add("multiple_choice")
    elif has("question", "query") and has("answer", "answers"):
        out.add("qa")
    if has("translation") or (has("source", "src") and has("target", "tgt")):
        out.add("translation")
    if has("summary") and has("document", "article", "text", "dialogue"):
        out.add("summarization")
    if has("label", "labels") and has("text", "sentence", "sentence1", "premise"):
        out.add("classification")
    if "audio" in dtypes or has("audio"):
        out.add("speech" if has("text", "transcription", "transcript", "sentence") else "audio")
    if "image" in dtypes or has("image"):
        out.add("image_text" if has("caption", "text", "prompt") else "image")
    if has("reasoning", "thinking", "thought", "cot", "rationale", "reasoning_content"):
        out.add("reasoning")
    if has("tools", "tool_calls", "functions"):
        out.add("tool_use")
    if has("reward", "rewards", "trajectory", "observations", "actions"):
        out.add("trajectory")
    if not out and has("text", "content"):
        out.add("text")
    return out


def card_dtype(f):
    """dtype of one feature entry in a dataset card's YAML."""
    if not isinstance(f, dict):
        return None
    if isinstance(f.get("dtype"), str):
        return f["dtype"]
    for k in ("image", "audio", "video", "class_label", "struct", "list", "sequence"):
        if k in f:
            return k
    if isinstance(f.get("dtype"), dict):
        return next(iter(f["dtype"]), "struct")
    return None


def viewer_dtype(f):
    """dtype of one feature entry from the dataset viewer /info endpoint."""
    if isinstance(f, list):
        return "list"
    if not isinstance(f, dict):
        return None
    t = f.get("_type")
    if t == "Value":
        return f.get("dtype")
    if t in ("Sequence", "List", "LargeList"):
        return "list"
    if t == "ClassLabel":
        return "class_label"
    if t:
        return t.lower()
    return "struct"


# ---------------------------------------------------------------- http

class RateLimited(Exception):
    pass


def http_get(url, token=None, retries=5):
    """GET returning (status, parsed json or None, headers). Waits out rate limits."""
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        if token:
            req.add_header("Authorization", "Bearer " + token)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                body, status, headers = r.read(), r.status, r.headers
        except urllib.error.HTTPError as e:
            body, status, headers = e.read(), e.code, e.headers
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            wait = 2 ** attempt * 2
            print("  network error (%s), retrying in %ds" % (e, wait), file=sys.stderr)
            time.sleep(wait)
            continue
        if status in (429, 500, 502, 503, 504):
            if attempt == retries - 1:
                return status, None, headers
            wait = _reset_seconds(headers) or int(headers.get("Retry-After") or 0) or 2 ** attempt * 5
            print("  HTTP %d, waiting %ds" % (status, wait), file=sys.stderr)
            time.sleep(min(wait, 330) + 1)
            continue
        # Hub sends: ratelimit: "api";r=<remaining>;t=<seconds to reset>
        m = re.search(r"r=(\d+);t=(\d+)", headers.get("ratelimit") or "")
        if m and int(m.group(1)) <= 2:
            print("  rate limit window used up, waiting %ss" % m.group(2), file=sys.stderr)
            time.sleep(int(m.group(2)) + 1)
        try:
            data = json.loads(body) if body else None
        except ValueError:
            data = None
        return status, data, headers
    return 0, None, {}


def _reset_seconds(headers):
    m = re.search(r"t=(\d+)", headers.get("ratelimit") or "")
    return int(m.group(1)) if m else 0


def next_link(headers):
    m = re.search(r'<([^>]+)>;\s*rel="next"', headers.get("Link") or "")
    return m.group(1) if m else None


# ---------------------------------------------------------------- ingest

def as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def sane_int(x):
    """A non-negative integer SQLite can hold, else None. Cards contain junk values."""
    return x if isinstance(x, int) and not isinstance(x, bool) and 0 <= x < 2 ** 62 else None


def parse_card_info(card):
    """Columns, splits and totals declared in the card's dataset_info block."""
    cols, splits, rows, nbytes = [], [], None, None
    infos = [i for i in as_list(card.get("dataset_info")) if isinstance(i, dict)]
    for info in infos:
        cfg = str(info.get("config_name") or "default")
        for f in as_list(info.get("features")):
            if isinstance(f, dict) and f.get("name") is not None:
                cols.append((cfg, str(f["name"]), card_dtype(f)))
        for s in as_list(info.get("splits")):
            if isinstance(s, dict) and s.get("name") is not None:
                n = sane_int(s.get("num_examples"))
                b = sane_int(s.get("num_bytes"))
                splits.append((cfg, str(s["name"]), n, b))
                if n is not None:
                    rows = (rows or 0) + n
        b = sane_int(info.get("dataset_size"))
        if b is not None:
            nbytes = (nbytes or 0) + b
    n_configs = max(len(infos), len(as_list(card.get("configs"))))
    return cols, splits, sane_int(rows), sane_int(nbytes), n_configs


def refresh_shapes(db, rid):
    cols = db.execute("SELECT DISTINCT name, dtype FROM columns WHERE rid=?", (rid,)).fetchall()
    db.execute("DELETE FROM tags WHERE rid=? AND kind='shape'", (rid,))
    db.executemany("INSERT OR IGNORE INTO tags VALUES (?,?,?)",
                   [(rid, "shape", s) for s in shapes([(c[0], c[1]) for c in cols])])


def upsert(db, r, stamp):
    card = r.get("cardData") if isinstance(r.get("cardData"), dict) else {}
    did = r["id"]
    author, _, name = did.rpartition("/")
    lic = as_list(card.get("license"))
    lic = str(lic[0]) if lic else None

    tagrows, free, size = set(), [], None
    for t in r.get("tags") or []:
        if not isinstance(t, str):
            continue
        prefix, sep, value = t.partition(":")
        if sep and prefix in TAG_KINDS and value:
            tagrows.add((TAG_KINDS[prefix], value))
            if prefix == "license" and not lic:
                lic = value
            if prefix == "size_categories" and (
                size is None or _size_rank(value) > _size_rank(size)
            ):
                size = value
        else:
            tagrows.add(("tag", t))
            free.append(t)

    cols, splits, rows, nbytes, n_configs = parse_card_info(card)
    extra = {k: v for k, v in card.items() if k not in EXTRA_DROP and not k.startswith("extra_gated")}
    for k in ("paperswithcode_id", "citation"):
        if r.get(k):
            extra[k] = r[k]
    extra = json.dumps(extra, ensure_ascii=False)
    if len(extra) > 6000:
        extra = None
    desc = re.sub(r"\s+", " ", r.get("description") or "").strip()[:2000] or None
    pretty = card.get("pretty_name")
    pretty = str(pretty) if pretty else None
    gated = r.get("gated")

    db.execute(
        """INSERT INTO datasets (id, author, name, pretty_name, description, created_at,
             last_modified, downloads, likes, trending, gated, disabled, license,
             license_class, size_category, size_rank, card_rows, card_bytes, n_configs,
             has_card, sha, extra, crawled_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(id) DO UPDATE SET
             author=excluded.author, name=excluded.name, pretty_name=excluded.pretty_name,
             description=excluded.description, created_at=excluded.created_at,
             last_modified=excluded.last_modified, downloads=excluded.downloads,
             likes=excluded.likes, trending=excluded.trending, gated=excluded.gated,
             disabled=excluded.disabled, license=excluded.license,
             license_class=excluded.license_class, size_category=excluded.size_category,
             size_rank=excluded.size_rank, card_rows=excluded.card_rows,
             card_bytes=excluded.card_bytes, n_configs=excluded.n_configs,
             has_card=excluded.has_card, sha=excluded.sha, extra=excluded.extra,
             crawled_at=excluded.crawled_at""",
        (did, author or None, name, pretty, desc, r.get("createdAt"), r.get("lastModified"),
         r.get("downloads") or 0, r.get("likes") or 0, r.get("trendingScore"),
         "no" if not gated else str(gated), int(bool(r.get("disabled"))), lic,
         license_class(lic), size, _size_rank(size), rows, nbytes, n_configs,
         int(bool(card)), r.get("sha"), extra, stamp),
    )
    rid = db.execute("SELECT rid FROM datasets WHERE id=?", (did,)).fetchone()[0]

    db.execute("DELETE FROM tags WHERE rid=? AND kind!='shape'", (rid,))
    db.executemany("INSERT OR IGNORE INTO tags VALUES (?,?,?)", [(rid, k, v) for k, v in tagrows])
    db.execute("DELETE FROM columns WHERE rid=? AND source='card'", (rid,))
    db.executemany("INSERT OR IGNORE INTO columns VALUES (?,?,?,?)",
                   [(rid, "card", n, t) for _, n, t in cols[:MAX_COLUMNS]])
    db.execute("DELETE FROM splits WHERE rid=? AND source='card'", (rid,))
    db.executemany("INSERT OR IGNORE INTO splits VALUES (?,?,?,?,?,?)",
                   [(rid, "card", c, s, n, b) for c, s, n, b in splits[:MAX_SPLITS]])
    refresh_shapes(db, rid)
    db.execute("DELETE FROM fts WHERE rowid=?", (rid,))
    db.execute("INSERT INTO fts(rowid, id, pretty_name, description, tags) VALUES (?,?,?,?,?)",
               (rid, did, pretty, desc, " ".join(free)))
    return rid


def _size_rank(s):
    return SIZE_ORDER.index(s) if s in SIZE_ORDER else None if s is None else -1


# ---------------------------------------------------------------- commands

def cmd_crawl(args):
    db = connect(args.db)
    token = os.environ.get("HF_TOKEN")
    params = {"limit": 1000, "full": "true"}
    scoped = bool(args.search or args.author or args.filter)
    if args.search:
        params["search"] = args.search
    if args.author:
        params["author"] = args.author
    if args.filter:
        params["filter"] = ",".join(args.filter)
    if args.update:
        # Newest changes first; stop once a whole page is already up to date.
        params.update(sort="lastModified", direction="-1")
    else:
        # Oldest first: a stable order, so a crawl can stop and resume.
        params.update(sort="createdAt", direction="1")
    url = HUB + "?" + urllib.parse.urlencode(params)
    state_key = None if (scoped or args.update) else "crawl_next"
    if state_key and not args.restart:
        saved = db.execute("SELECT value FROM meta WHERE key=?", (state_key,)).fetchone()
        if saved and saved[0]:
            url = saved[0]
            print("resuming earlier crawl (use --restart to start over)")

    stamp, total, pages, t0 = now(), 0, 0, time.time()
    while url:
        status, data, headers = http_get(url, token)
        if status != 200 or not isinstance(data, list):
            sys.exit("Hub API returned HTTP %s for %s" % (status, url))
        unchanged = 0
        if args.update:
            for r in data:
                row = db.execute("SELECT last_modified FROM datasets WHERE id=?", (r["id"],)).fetchone()
                if row and row[0] == r.get("lastModified"):
                    unchanged += 1
        with db:
            for r in data:
                upsert(db, r, stamp)
            url = next_link(headers) if data else None
            if state_key:
                db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (state_key, url or ""))
        total += len(data)
        pages += 1
        last = data[-1].get("lastModified" if args.update else "createdAt", "")[:10] if data else ""
        print("  %7d datasets  page %d  at %s  %.0fs" % (total, pages, last, time.time() - t0),
              flush=True)
        if args.update and data and unchanged == len(data):
            print("reached datasets that are already up to date")
            break
        if args.max_pages and pages >= args.max_pages:
            print("stopped at --max-pages; run crawl again to continue" if state_key else
                  "stopped at --max-pages")
            break
    with db:
        if not url and state_key:
            db.execute("INSERT OR REPLACE INTO meta VALUES ('full_crawl_done', ?)", (stamp,))
        db.execute("INSERT OR REPLACE INTO meta VALUES ('last_crawl', ?)", (stamp,))
    n = db.execute("SELECT count(*) FROM datasets").fetchone()[0]
    print("done: %d records this run, %d datasets in %s" % (total, n, args.db))


TRANSIENT = ("network_error", "http_429", "http_500", "http_502", "http_503", "http_504")


def fetch_viewer(did, token):
    """One /info request per dataset; /size only when the viewer holds a partial copy."""
    q = urllib.parse.urlencode({"dataset": did})
    st, info, _ = http_get("%s/info?%s" % (VIEWER, q), token, retries=3)
    out = {"id": did, "status": "ok", "rows": None, "bytes": None, "cols": [], "splits": []}
    if st != 200 or not isinstance(info, dict):
        out["status"] = {401: "no_access", 404: "not_found", 501: "no_viewer",
                         0: "network_error"}.get(st, "http_%s" % st)
        return out
    rows = nbytes = None
    for cfg, ci in (info.get("dataset_info") or {}).items():
        if not isinstance(ci, dict):
            continue
        for n, f in (ci.get("features") or {}).items():
            out["cols"].append((str(n), viewer_dtype(f)))
        for sn, sp in (ci.get("splits") or {}).items():
            n = sane_int(sp.get("num_examples")) if isinstance(sp, dict) else None
            b = sane_int(sp.get("num_bytes")) if isinstance(sp, dict) else None
            out["splits"].append((str(cfg), str(sn), n, b))
            if n is not None:
                rows = (rows or 0) + n
        b = sane_int(ci.get("download_size")) or sane_int(ci.get("dataset_size"))
        if b:
            nbytes = (nbytes or 0) + b
    if not out["cols"] and not out["splits"]:
        out["status"] = "failed" if info.get("failed") else "empty"
    if info.get("partial"):
        # Only the first part was converted; /size carries an estimate of the real total.
        out["status"] = "partial"
        st2, size, _ = http_get("%s/size?%s" % (VIEWER, q), token, retries=2)
        if st2 == 200 and isinstance(size, dict):
            ds = (size.get("size") or {}).get("dataset") or {}
            rows = sane_int(ds.get("estimated_num_rows")) or sane_int(ds.get("num_rows")) or rows
            nbytes = sane_int(ds.get("num_bytes_parquet_files")) or nbytes
    out["rows"], out["bytes"] = sane_int(rows) or None, sane_int(nbytes) or None
    return out


def cmd_enrich(args):
    db = connect(args.db)
    token = os.environ.get("HF_TOKEN")
    where, params = build_filters(args)
    if not args.refresh:
        where.append("d.enriched_at IS NULL")
    if args.missing_schema:
        where.append("NOT EXISTS (SELECT 1 FROM columns c WHERE c.rid=d.rid)")
    if args.all:
        # The Hub adds a format: tag once the viewer has processed a dataset, so the
        # rest have nothing to fetch. Gated datasets need a token with access.
        where.append("EXISTS (SELECT 1 FROM tags t WHERE t.rid=d.rid AND t.kind='format')")
        where.append("d.disabled=0")
        if not token:
            where.append("d.gated='no'")
    w = " AND ".join(where) or "1"
    left = db.execute("SELECT count(*) FROM datasets d WHERE " + w, params).fetchone()[0]
    goal = left if args.all else min(left, args.max)
    if not goal:
        print("nothing to enrich for these filters")
        return
    print("asking the dataset viewer about %d datasets, most downloaded first" % goal, flush=True)
    done = ok = retry = 0
    t0 = time.time()
    # Keyset paging keeps memory flat and never revisits a dataset within one run.
    page_sql = ("SELECT d.rid, d.id, d.downloads FROM datasets d WHERE %s AND "
                "(d.downloads < ? OR (d.downloads = ? AND d.rid > ?)) "
                "ORDER BY d.downloads DESC, d.rid LIMIT ?" % w)
    key = (1 << 62, -1)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while done < goal:
            batch = db.execute(page_sql, params + [key[0], key[0], key[1], min(500, goal - done)]).fetchall()
            if not batch:
                break
            key = (batch[-1]["downloads"], batch[-1]["rid"])
            rids = {r["id"]: r["rid"] for r in batch}
            results = list(pool.map(lambda d: fetch_viewer(d, token), list(rids)))
            stamp = now()
            with db:
                for r in results:
                    rid = rids[r["id"]]
                    transient = r["status"] in TRANSIENT  # left un-enriched, so a rerun retries it
                    db.execute("UPDATE datasets SET viewer_status=?, viewer_rows=?, viewer_bytes=?, "
                               "enriched_at=? WHERE rid=?",
                               (r["status"], r["rows"], r["bytes"], None if transient else stamp, rid))
                    db.execute("DELETE FROM columns WHERE rid=? AND source='viewer'", (rid,))
                    db.executemany("INSERT OR IGNORE INTO columns VALUES (?,?,?,?)",
                                   [(rid, "viewer", n, t) for n, t in r["cols"][:MAX_COLUMNS]])
                    db.execute("DELETE FROM splits WHERE rid=? AND source='viewer'", (rid,))
                    db.executemany("INSERT OR IGNORE INTO splits VALUES (?,?,?,?,?,?)",
                                   [(rid, "viewer") + sp for sp in r["splits"][:MAX_SPLITS]])
                    refresh_shapes(db, rid)
                    ok += r["status"] in ("ok", "partial")
                    retry += transient
            done += len(batch)
            rate = done / max(time.time() - t0, 0.001)
            print("  %d/%d  %d with schema  %d to retry  %.0f/s  about %.0f min left" % (
                done, goal, ok, retry, rate, (goal - done) / rate / 60), flush=True)
    print("done: %d asked, %d with viewer data, %d temporary failures (rerun to retry)" % (done, ok, retry))


ROWS = "COALESCE(d.viewer_rows, d.card_rows)"
SORTS = {
    "downloads": "d.downloads DESC", "likes": "d.likes DESC",
    "rows": ROWS + " DESC", "modified": "d.last_modified DESC",
    "created": "d.created_at DESC", "trending": "d.trending DESC", "id": "d.id",
}


def add_filters(p):
    g = p.add_argument_group("filters")
    g.add_argument("text", nargs="*", help="full-text query over id, name, description and tags")
    g.add_argument("--lang", action="append", default=[], help="language code, e.g. sv (repeat = all must match)")
    g.add_argument("--task", action="append", default=[], help="task category, e.g. text-generation")
    g.add_argument("--modality", action="append", default=[], help="text, audio, image, tabular, ...")
    g.add_argument("--format", dest="fmt", action="append", default=[], help="parquet, json, csv, ...")
    g.add_argument("--shape", action="append", default=[],
                   help="guessed example shape: chat, instruction, qa, preference, speech, ...")
    g.add_argument("--tag", action="append", default=[], help="any kind:value, e.g. arxiv:2301.00001 or tag:medical")
    g.add_argument("--has-column", action="append", default=[], help="a column with this exact name")
    g.add_argument("--license", action="append", default=[], help="exact license id (repeat = any of)")
    g.add_argument("--license-class", action="append", default=[],
                   choices=["permissive", "sharealike", "noncommercial", "noderivs", "other", "none"],
                   help="rough license bucket (repeat = any of)")
    g.add_argument("--author", help="user or organisation")
    g.add_argument("--min-downloads", type=int)
    g.add_argument("--min-likes", type=int)
    g.add_argument("--min-rows", type=int, help="needs a known row count (card or viewer)")
    g.add_argument("--max-rows", type=int)
    g.add_argument("--min-size", choices=SIZE_ORDER, help="declared size category at least this")
    g.add_argument("--modified-since", metavar="YYYY-MM-DD")
    g.add_argument("--created-since", metavar="YYYY-MM-DD")
    g.add_argument("--open", action="store_true", help="exclude gated and disabled datasets")
    g.add_argument("--has-schema", action="store_true", help="only datasets with known columns")


def build_filters(a):
    where, params = [], []

    def tag(kind, value):
        where.append("EXISTS (SELECT 1 FROM tags t WHERE t.rid=d.rid AND t.kind=? AND t.value=?)")
        params.extend([kind, value])

    for kind, values in (("language", a.lang), ("task", a.task), ("modality", a.modality),
                         ("format", a.fmt), ("shape", a.shape)):
        for v in values:
            tag(kind, v)
    for kv in a.tag:
        k, sep, v = kv.partition(":")
        tag(k, v) if sep else tag("tag", kv)
    for c in a.has_column:
        where.append("EXISTS (SELECT 1 FROM columns c WHERE c.rid=d.rid AND c.name=?)")
        params.append(c)
    for col, values in (("d.license", a.license), ("d.license_class", a.license_class)):
        if values:
            where.append("%s IN (%s)" % (col, ",".join("?" * len(values))))
            params.extend(values)
    simple = (("d.author = ?", a.author), ("d.downloads >= ?", a.min_downloads),
              ("d.likes >= ?", a.min_likes), (ROWS + " >= ?", a.min_rows),
              (ROWS + " <= ?", a.max_rows), ("d.last_modified >= ?", a.modified_since),
              ("d.created_at >= ?", a.created_since),
              ("d.size_rank >= ?", SIZE_ORDER.index(a.min_size) if a.min_size else None))
    for clause, v in simple:
        if v is not None:
            where.append(clause)
            params.append(v)
    if a.open:
        where.append("d.gated='no' AND d.disabled=0")
    if a.has_schema:
        where.append("EXISTS (SELECT 1 FROM columns c WHERE c.rid=d.rid)")
    if a.text:
        q = " ".join(a.text)
        if not re.search(r'["*():]|\b(AND|OR|NOT|NEAR)\b', q):
            q = " ".join('"%s"' % w for w in q.split())  # plain words: all must appear
        where.append("d.rid IN (SELECT rowid FROM fts WHERE fts MATCH ?)")
        params.append(q)
    return where, params


def human(n):
    if n is None:
        return ""
    for unit in ("", "K", "M", "B", "T"):
        if abs(n) < 1000:
            return ("%d%s" % (n, unit)) if unit == "" or n >= 100 else ("%.1f%s" % (n, unit))
        n /= 1000.0
    return "%.0fQ" % n


def short_size(cat):
    """'10K<n<100K' -> '<100K' so a declared size fits the ROWS column."""
    if not cat:
        return ""
    return cat.replace("n", "") if cat.startswith("n") else "<" + cat.split("<")[-1]


def tag_values(db, rid, kind, limit=None):
    vals = [r[0] for r in db.execute(
        "SELECT value FROM tags WHERE rid=? AND kind=? ORDER BY value", (rid, kind))]
    return vals[:limit] + ["+%d" % (len(vals) - limit)] if limit and len(vals) > limit else vals


def record(db, d, full=False):
    rid = d["rid"]
    out = {
        "id": d["id"], "url": "https://huggingface.co/datasets/" + d["id"],
        "pretty_name": d["pretty_name"], "downloads": d["downloads"], "likes": d["likes"],
        "rows": d["viewer_rows"] if d["viewer_rows"] is not None else d["card_rows"],
        "rows_source": "viewer" if d["viewer_rows"] is not None else
                       "card" if d["card_rows"] is not None else None,
        "size_category": d["size_category"], "license": d["license"],
        "license_class": d["license_class"], "gated": d["gated"],
        "created_at": d["created_at"], "last_modified": d["last_modified"],
        "languages": tag_values(db, rid, "language"), "tasks": tag_values(db, rid, "task"),
        "modalities": tag_values(db, rid, "modality"), "formats": tag_values(db, rid, "format"),
        "shapes": tag_values(db, rid, "shape"),
    }
    if full:
        out["description"] = d["description"]
        out["tags"] = tag_values(db, rid, "tag")
        out["sources"] = tag_values(db, rid, "source")
        out["arxiv"] = tag_values(db, rid, "arxiv")
        out["bytes"] = d["viewer_bytes"] if d["viewer_bytes"] is not None else d["card_bytes"]
        out["viewer_status"] = d["viewer_status"]
        out["crawled_at"], out["enriched_at"] = d["crawled_at"], d["enriched_at"]
        # Prefer the viewer's schema (measured) over the card's (declared).
        src = "viewer" if db.execute("SELECT 1 FROM columns WHERE rid=? AND source='viewer'",
                                     (rid,)).fetchone() else "card"
        out["columns"] = {c[0]: c[1] for c in db.execute(
            "SELECT name, dtype FROM columns WHERE rid=? AND source=?", (rid, src))}
        out["schema_source"] = src if out["columns"] else None
        configs = {}
        ssrc = "viewer" if db.execute("SELECT 1 FROM splits WHERE rid=? AND source='viewer'",
                                      (rid,)).fetchone() else "card"
        for s in db.execute("SELECT config, split, num_rows FROM splits WHERE rid=? AND source=?",
                            (rid, ssrc)):
            configs.setdefault(s[0], {})[s[1]] = s[2]
        out["splits"] = configs
        out["extra"] = json.loads(d["extra"]) if d["extra"] else {}
    return out


# ---------------------------------------------------------------- semantic search

def vec_paths(db_path):
    return db_path + ".vec.npy", db_path + ".vec.rids.npy", db_path + ".vec.json"


def need_vector_libs():
    try:
        import numpy
        from model2vec import StaticModel
    except ImportError:
        sys.exit("semantic search needs numpy and model2vec. Run this command as\n"
                 "  uv run %s ...\nor: pip install numpy model2vec" % os.path.basename(__file__))
    return numpy, StaticModel


def embed_texts(db, lo, hi):
    """One short text per dataset with rid in (lo, hi]: what a person would say it is."""
    extra = {}
    for rid, kind, value in db.execute(
            "SELECT rid, kind, value FROM tags WHERE rid>? AND rid<=? AND kind IN "
            "('task','tag','shape','modality','language')", (lo, hi)):
        extra.setdefault(rid, {}).setdefault(kind, []).append(value)
    cols = {}
    for rid, name in db.execute("SELECT DISTINCT rid, name FROM columns WHERE rid>? AND rid<=?", (lo, hi)):
        cols.setdefault(rid, []).append(name)
    rids, texts = [], []
    for rid, did, pretty, desc in db.execute(
            "SELECT rid, id, pretty_name, description FROM datasets WHERE rid>? AND rid<=? ORDER BY rid",
            (lo, hi)):
        e = extra.get(rid, {})
        parts = [re.sub(r"[/_\-.]+", " ", did)]
        if pretty:
            parts.append(pretty)
        if desc:
            parts.append(desc[:700])
        for label, kind, cap in (("tasks", "task", 6), ("shape", "shape", 6), ("modality", "modality", 4),
                                 ("tags", "tag", 15)):
            if e.get(kind):
                parts.append("%s: %s" % (label, ", ".join(e[kind][:cap])))
        if 0 < len(e.get("language", [])) <= 6:  # a 100-language list says little
            parts.append("languages: " + ", ".join(e["language"]))
        if cols.get(rid):
            parts.append("columns: " + ", ".join(cols[rid][:25]))
        rids.append(rid)
        texts.append(". ".join(parts))
    return rids, texts


def cmd_embed(args):
    np, StaticModel = need_vector_libs()
    db = connect(args.db)
    vec_file, rid_file, meta_file = vec_paths(args.db)
    n = db.execute("SELECT count(*) FROM datasets").fetchone()[0]
    top = db.execute("SELECT max(rid) FROM datasets").fetchone()[0] or 0
    print("loading %s" % args.model, flush=True)
    model = StaticModel.from_pretrained(args.model)
    dim = int(model.encode(["probe"]).shape[1])
    # Written to .tmp files first, so an interrupted run never leaves a half-built index.
    vecs = np.lib.format.open_memmap(vec_file + ".tmp", mode="w+", dtype=np.int8, shape=(n, dim))
    rid_out = np.zeros(n, dtype=np.int64)
    done, lo, t0, step = 0, 0, time.time(), 20000
    while lo < top and done < n:
        rids, texts = embed_texts(db, lo, lo + step)
        lo += step
        if not rids:
            continue
        rids, texts = rids[:n - done], texts[:n - done]
        v = model.encode(texts).astype(np.float32)
        v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
        vecs[done:done + len(rids)] = np.round(v * 127).astype(np.int8)
        rid_out[done:done + len(rids)] = rids
        done += len(rids)
        print("  %d/%d  %.0fs" % (done, n, time.time() - t0), flush=True)
    vecs.flush()
    del vecs
    if done < n:  # rows vanished mid-run; keep only what was written
        full = np.load(vec_file + ".tmp", mmap_mode="r")
        np.save(vec_file + ".tmp2.npy", np.asarray(full[:done]))
        os.replace(vec_file + ".tmp2.npy", vec_file + ".tmp")
    np.save(rid_file + ".tmp.npy", rid_out[:done])
    os.replace(vec_file + ".tmp", vec_file)
    os.replace(rid_file + ".tmp.npy", rid_file)
    with open(meta_file, "w") as f:
        json.dump({"model": args.model, "dim": dim, "count": done, "built_at": now(),
                   "encoding": "unit vectors * 127, int8"}, f, indent=1)
    print("done: %d vectors of %d dims in %s (%.0f MB)" % (
        done, dim, vec_file, os.path.getsize(vec_file) / 1e6))


def semantic_rank(args, db, query, where, params):
    """Exact cosine ranking of every dataset against the query, within the SQL filters."""
    np, StaticModel = need_vector_libs()
    vec_file, rid_file, meta_file = vec_paths(args.db)
    if not os.path.exists(vec_file):
        sys.exit("no vector index yet: run `uv run %s embed` first" % os.path.basename(__file__))
    with open(meta_file) as f:
        meta = json.load(f)
    model = StaticModel.from_pretrained(meta["model"])
    q = model.encode([query]).astype(np.float32)[0]
    q /= max(float(np.linalg.norm(q)), 1e-9)
    vecs = np.load(vec_file, mmap_mode="r")
    rids = np.load(rid_file)
    scores = np.empty(len(rids), dtype=np.float32)
    for i in range(0, len(rids), 50000):  # chunked: about 100 MB of working memory at most
        scores[i:i + 50000] = vecs[i:i + 50000].astype(np.float32) @ q
    scores /= 127.0
    total = len(rids)
    if where:
        allowed = np.fromiter((r[0] for r in db.execute(
            "SELECT d.rid FROM datasets d WHERE " + " AND ".join(where), params)), dtype=np.int64)
        mask = np.isin(rids, allowed)
        scores[~mask] = -2.0
        total = int(mask.sum())
    k = min(args.limit, total)
    if k <= 0:
        return [], [], total
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]
    return [int(r) for r in rids[top]], [float(x) for x in scores[top]], total


def cmd_search(args):
    db = connect(args.db)
    scores = None
    if args.semantic:
        if not args.text:
            sys.exit("--semantic needs a query, e.g. search --semantic swedish court rulings")
        query, args.text = " ".join(args.text), []
        where, params = build_filters(args)
        top, scores, total = semantic_rank(args, db, query, where, params)
        rows = [db.execute("SELECT * FROM datasets WHERE rid=?", (r,)).fetchone() for r in top]
        scores = [s for s, r in zip(scores, rows) if r]
        rows = [r for r in rows if r]
    else:
        where, params = build_filters(args)
        w = " AND ".join(where) or "1"
        total = db.execute("SELECT count(*) FROM datasets d WHERE " + w, params).fetchone()[0]
        rows = db.execute("SELECT d.* FROM datasets d WHERE %s ORDER BY %s LIMIT ?" % (w, SORTS[args.sort]),
                          params + [args.limit]).fetchall()
    if args.out == "ids":
        print("\n".join(r["id"] for r in rows))
        return
    recs = [record(db, r, full=args.out in ("json", "jsonl")) for r in rows]
    if scores is not None:
        recs = [dict({"score": round(s, 3)}, **r) for s, r in zip(scores, recs)]
    if args.out == "json":
        print(json.dumps(recs, indent=1, ensure_ascii=False))
    elif args.out == "jsonl":
        for r in recs:
            print(json.dumps(r, ensure_ascii=False))
    elif args.out == "csv":
        wr = csv.writer(sys.stdout)
        keys = list(recs[0]) if recs else []
        wr.writerow(keys)
        for r in recs:
            wr.writerow([";".join(v) if isinstance(v, list) else v for v in (r[k] for k in keys)])
    else:
        # Ids are never cut: a truncated id cannot be passed to `show`.
        fmt = "%%-%ds %%7s %%5s %%7s %%-14s %%-12s %%s" % max([30] + [len(r["id"]) for r in recs])
        print(fmt % ("DATASET", "DOWNL", "SCORE" if scores is not None else "LIKES", "ROWS",
                     "LICENSE", "LANG", "SHAPE / TASK"))
        for d, r in zip(rows, recs):
            langs = tag_values(db, d["rid"], "language", 3)
            what = r["shapes"] or r["tasks"][:2]
            third = "%.2f" % r["score"] if scores is not None else human(r["likes"])
            print(fmt % (r["id"], human(r["downloads"]), third,
                         human(r["rows"]) or short_size(r["size_category"]),
                         (r["license"] or "-")[:14], ",".join(langs)[:12], ",".join(what)))
        if scores is not None:
            print("\ntop %d by similarity, out of %d datasets that pass the filters" % (len(rows), total))
        else:
            print("\n%d of %d matching datasets (sorted by %s)" % (len(rows), total, args.sort))


def cmd_show(args):
    db = connect(args.db)
    d = db.execute("SELECT * FROM datasets WHERE id=? COLLATE NOCASE", (args.id,)).fetchone()
    if not d:
        sys.exit("not in the index: %s (run crawl, or check the spelling)" % args.id)
    print(json.dumps(record(db, d, full=True), indent=1, ensure_ascii=False))


def cmd_stats(args):
    db = connect(args.db)
    where, params = build_filters(args)
    w = " AND ".join(where) or "1"
    if args.by:
        if args.by in ("license", "license_class", "author", "size_category", "gated", "viewer_status"):
            sql = ("SELECT d.%s AS v, count(*) n, sum(d.downloads) dl FROM datasets d WHERE %s "
                   "GROUP BY v ORDER BY n DESC LIMIT ?" % (args.by, w))
            rows = db.execute(sql, params + [args.limit]).fetchall()
        elif args.by == "column":
            sql = ("SELECT c.name AS v, count(DISTINCT d.rid) n, sum(d.downloads) dl FROM datasets d "
                   "JOIN (SELECT DISTINCT rid, name FROM columns) c ON c.rid=d.rid WHERE %s "
                   "GROUP BY v ORDER BY n DESC LIMIT ?" % w)
            rows = db.execute(sql, params + [args.limit]).fetchall()
        else:
            sql = ("SELECT t.value AS v, count(*) n, sum(d.downloads) dl FROM datasets d "
                   "JOIN tags t ON t.rid=d.rid AND t.kind=? WHERE %s "
                   "GROUP BY v ORDER BY n DESC LIMIT ?" % w)
            rows = db.execute(sql, [args.by] + params + [args.limit]).fetchall()
        print("%-40s %9s %10s" % (args.by.upper(), "DATASETS", "DOWNLOADS"))
        for r in rows:
            print("%-40s %9d %10s" % (str(r["v"])[:40], r["n"], human(r["dl"])))
        return
    one = lambda sql: db.execute(sql % w, params).fetchone()[0]
    print("index:            %s" % args.db)
    for k in ("last_crawl", "full_crawl_done"):
        v = db.execute("SELECT value FROM meta WHERE key=?", (k,)).fetchone()
        print("%-17s %s" % (k + ":", v[0] if v else "never"))
    print("datasets:         %d" % one("SELECT count(*) FROM datasets d WHERE %s"))
    print("  with a card:    %d" % one("SELECT count(*) FROM datasets d WHERE %s AND d.has_card=1"))
    print("  with a license: %d" % one("SELECT count(*) FROM datasets d WHERE %s AND d.license IS NOT NULL"))
    print("  with a schema:  %d" % one(
        "SELECT count(*) FROM datasets d WHERE %s AND EXISTS (SELECT 1 FROM columns c WHERE c.rid=d.rid)"))
    print("  with row count: %d" % one("SELECT count(*) FROM datasets d WHERE %s AND " + ROWS + " IS NOT NULL"))
    print("  asked viewer:   %d" % one("SELECT count(*) FROM datasets d WHERE %s AND d.enriched_at IS NOT NULL"))
    print("rows, where known: %s (all languages and configs of each dataset)" % human(
        one("SELECT sum(" + ROWS + ") FROM datasets d WHERE %s")))


def main():
    p = argparse.ArgumentParser(prog="hfmeta", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=DEFAULT_DB, help="SQLite file (default: %(default)s, or $HFMETA_DB)")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("crawl", help="fetch metadata for every dataset on the Hub (or a slice)")
    c.add_argument("--update", action="store_true", help="only fetch what changed since the last crawl")
    c.add_argument("--restart", action="store_true", help="ignore the saved position of a full crawl")
    c.add_argument("--search", help="only datasets whose id matches this text")
    c.add_argument("--author", help="only this user or organisation")
    c.add_argument("--filter", action="append", help="only this Hub tag, e.g. language:sv (repeatable)")
    c.add_argument("--max-pages", type=int, help="stop after N pages of 1000")
    c.set_defaults(fn=cmd_crawl)

    e = sub.add_parser("enrich", help="get exact rows and columns from the dataset viewer for matching datasets")
    add_filters(e)
    e.add_argument("--max", type=int, default=200, help="most-downloaded N matches (default 200)")
    e.add_argument("--all", action="store_true",
                   help="no cap: every matching dataset the viewer has processed (resumable)")
    e.add_argument("--missing-schema", action="store_true", help="skip datasets whose columns are already known")
    e.add_argument("--refresh", action="store_true", help="also redo datasets already enriched")
    e.add_argument("--workers", type=int, default=8)
    e.set_defaults(fn=cmd_enrich)

    s = sub.add_parser("search", help="query the local index")
    add_filters(s)
    s.add_argument("--semantic", action="store_true",
                   help="rank by meaning instead of matching words (needs `embed` to have run)")
    s.add_argument("--sort", choices=list(SORTS), default="downloads")
    s.add_argument("--limit", type=int, default=25)
    s.add_argument("--out", choices=["table", "json", "jsonl", "csv", "ids"], default="table")
    s.set_defaults(fn=cmd_search)

    em = sub.add_parser("embed", help="build the vector index for search --semantic")
    em.add_argument("--model", default=EMBED_MODEL, help="a model2vec static model (default: %(default)s)")
    em.set_defaults(fn=cmd_embed)

    sh = sub.add_parser("show", help="everything known about one dataset, as JSON")
    sh.add_argument("id")
    sh.set_defaults(fn=cmd_show)

    st = sub.add_parser("stats", help="index overview, or counts grouped --by a field")
    add_filters(st)
    st.add_argument("--by", help="language, task, modality, format, shape, tag, source, column, "
                                 "license, license_class, author, size_category, gated, viewer_status")
    st.add_argument("--limit", type=int, default=30)
    st.set_defaults(fn=cmd_stats)

    args = p.parse_args()
    try:
        args.fn(args)
    except KeyboardInterrupt:
        sys.exit("\ninterrupted (progress so far is saved)")
    except BrokenPipeError:
        pass


if __name__ == "__main__":
    main()
