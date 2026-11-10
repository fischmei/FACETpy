"""Transactional evaluation and reusable calculation storage."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_VERSION = 1


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def file_identity(path: Path) -> dict:
    """Hash recording payloads, including multi-file format companions.

    Streaming hashes cost one sequential read per session, and catch edits
    even when file size and modification time have been preserved.
    """
    path = Path(path)
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    if path.suffix.lower() in {".vhdr", ".set"}:
        files = sorted(set(files + [p for p in path.parent.glob(path.stem + ".*") if p.is_file()]))
    if path.suffix.lower() == ".vhdr":
        import re

        header = path.read_text(encoding="latin-1")
        for match in re.finditer(r"^(?:DataFile|MarkerFile)=(.+)$", header, flags=re.MULTILINE):
            companion = path.parent / match.group(1).strip().replace("\\", "/")
            if not companion.is_file():
                raise ValueError(f"Missing BrainVision companion {companion}.")
            files.append(companion)
    if path.suffix.lower() == ".set":
        # MNE supports conventional .set + .fdt pairs. Include all nearby
        # FDT payloads conservatively: external EEGLAB data need not share
        # the .set stem. This can over-invalidate, never silently reuse edits.
        files.extend(path.parent.glob("*.fdt"))
    files = sorted(set(files))
    result = []
    for item in files:
        checksum = hashlib.sha256()
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(block)
        result.append((str(item.resolve()), item.stat().st_size, checksum.hexdigest()))
    return {"files": result}


class EvaluationCache:
    """One committed row per signature, fidelity, recording and recipe.

    Independent worker processes open independent WAL connections. Only
    finished evaluations are committed; an interruption leaves no partial row.
    Failed rows remain available as diagnostics and may be explicitly rebuilt.
    """

    def __init__(self, path: Path, *, signature: str, rebuild: bool = False, compatible_signatures=()):
        self.path = Path(path)
        self.signature = signature
        database = self.path.with_suffix(".sqlite3")
        database.parent.mkdir(parents=True, exist_ok=True)
        self.database = database
        self.connection = sqlite3.connect(database, timeout=120)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS evaluations (
            signature TEXT, fidelity TEXT, configuration TEXT, dataset TEXT, payload TEXT,
            PRIMARY KEY(signature, fidelity, configuration, dataset))""")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS calculations (
            key TEXT PRIMARY KEY, payload BLOB)""")
        self.connection.execute("CREATE TABLE IF NOT EXISTS units (key TEXT PRIMARY KEY, payload TEXT)")
        self.unit_hits = 0
        self.unit_misses = 0
        if rebuild:
            with self.connection:
                self.connection.execute("DELETE FROM evaluations WHERE signature=?", (signature,))
        # Explicit migration only. Ordinary optimization never rewrites a CSV.
        if self.path.suffix == ".csv" and self.path.exists() and not rebuild:
            frame = pd.read_csv(self.path)
            observed = set(frame.get("evaluation_signature", pd.Series(dtype=str)).astype(str))
            if not observed or not observed.issubset({signature, *compatible_signatures}):
                raise RuntimeError("Legacy cache signature differs; use --rebuild-cache.")
            for row in frame.to_dict("records"):
                row["evaluation_signature"] = signature
                self.add(row)
            frame["evaluation_signature"] = signature
            frame.to_csv(self.path, index=False)
        self.hits = 0
        self.misses = 0

    @property
    def rows(self):
        return [
            json.loads(row[0])
            for row in self.connection.execute(
                "SELECT payload FROM evaluations WHERE signature=? ORDER BY fidelity,dataset,configuration",
                (self.signature,),
            )
        ]

    def get(self, configuration: str, dataset_id: str, fidelity: str = "full"):
        row = self.connection.execute(
            "SELECT payload FROM evaluations WHERE signature=? AND fidelity=? AND configuration=? AND dataset=?",
            (self.signature, fidelity, configuration, dataset_id),
        ).fetchone()
        if row:
            self.hits += 1
            return json.loads(row[0])
        self.misses += 1
        return None

    def add(self, row: dict):
        row = {**row, "evaluation_signature": self.signature}
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO evaluations VALUES (?,?,?,?,?)",
                (
                    self.signature,
                    row.get("fidelity", "full"),
                    row["configuration_id"],
                    row["dataset_id"],
                    json.dumps(row, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)),
                ),
            )

    def calculation(self, key: str, compute):
        row = self.connection.execute("SELECT payload FROM calculations WHERE key=?", (key,)).fetchone()
        if row is not None:
            return np.frombuffer(row[0], dtype=np.float64).copy()
        value = np.asarray(compute(), dtype=np.float64)
        with self.connection:
            self.connection.execute("INSERT OR IGNORE INTO calculations VALUES (?,?)", (key, value.tobytes()))
        return value

    def get_unit(self, key):
        row = self.connection.execute("SELECT payload FROM units WHERE key=?", (key,)).fetchone()
        if row is not None:
            self.unit_hits += 1
            return json.loads(row[0])
        self.unit_misses += 1
        return None

    def add_unit(self, key, metrics, aligned):
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO units VALUES (?,?)",
                (
                    key,
                    json.dumps(
                        {"metrics": metrics, "aligned": aligned.tolist()},
                        default=lambda x: x.item() if isinstance(x, np.generic) else str(x),
                    ),
                ),
            )

    def export(self, path: Path):
        pd.DataFrame(self.rows).to_csv(path, index=False)

    def close(self):
        self.connection.close()
