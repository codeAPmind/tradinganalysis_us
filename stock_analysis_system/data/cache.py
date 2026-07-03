import sqlite3
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path


class SQLiteCache:
    """基于 SQLite 的简单缓存层,同一 ticker 同一天的数据只拉一次。"""

    def __init__(self, db_path: str = "./cache/analysis_cache.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT,
                expires_at TEXT
            )
        """)
        self.conn.commit()

    def _make_key(self, namespace: str, *args) -> str:
        raw = f"{namespace}:{':'.join(str(a) for a in args)}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, namespace: str, *args):
        key = self._make_key(namespace, *args)
        row = self.conn.execute(
            "SELECT value, expires_at FROM cache WHERE key = ?", (key,)
        ).fetchone()
        if not row:
            return None
        value, expires_at = row
        if datetime.fromisoformat(expires_at) < datetime.now():
            self.conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            self.conn.commit()
            return None
        return json.loads(value)

    def set(self, namespace: str, *args, value, ttl_hours: int = 24):
        key = self._make_key(namespace, *args)
        expires_at = (datetime.now() + timedelta(hours=ttl_hours)).isoformat()
        self.conn.execute(
            "INSERT OR REPLACE INTO cache (key, value, expires_at) VALUES (?, ?, ?)",
            (key, json.dumps(value, default=str), expires_at),
        )
        self.conn.commit()
