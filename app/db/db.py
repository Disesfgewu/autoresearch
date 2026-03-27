import os
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta


class DB:
    def __init__(self):
        self.db_path = "./db/stock_agent.db"
        os.makedirs("./db", exist_ok=True)
        self.createTables()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def createTables(self):
        schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
        with open(schema_path, "r") as f:
            schema = f.read()
        with self._conn() as conn:
            conn.executescript(schema)

    # =====================================================
    # Conversations
    # =====================================================

    def add_message(self, channel_id: str, user_id: str, role: str, content: str):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO conversations (channel_id, user_id, role, content) VALUES (?, ?, ?, ?)",
                (channel_id, user_id, role, content),
            )

    def get_history(self, channel_id: str, limit: int = 10) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT role, content FROM conversations
                   WHERE channel_id = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (channel_id, limit),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def clear_history(self, channel_id: str):
        with self._conn() as conn:
            conn.execute("DELETE FROM conversations WHERE channel_id = ?", (channel_id,))

    # =====================================================
    # Stock Real-time Cache
    # =====================================================

    def cache_stock(self, symbol: str, price: float, data: dict):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO stock_cache (symbol, price, data) VALUES (?, ?, ?)",
                (symbol.upper(), price, json.dumps(data)),
            )

    def get_stock_cache(self, symbol: str, max_age_minutes: int = 5) -> dict | None:
        cutoff = datetime.utcnow() - timedelta(minutes=max_age_minutes)
        with self._conn() as conn:
            row = conn.execute(
                """SELECT price, data FROM stock_cache
                   WHERE symbol = ? AND fetched_at >= ?
                   ORDER BY fetched_at DESC LIMIT 1""",
                (symbol.upper(), cutoff.strftime("%Y-%m-%d %H:%M:%S")),
            ).fetchone()
        if row is None:
            return None
        return {"price": row["price"], **json.loads(row["data"])}

    # =====================================================
    # Search Cache
    # =====================================================

    def cache_search(self, query: str, results: list):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO search_cache (query, results) VALUES (?, ?)",
                (query, json.dumps(results)),
            )

    def get_search_cache(self, query: str, max_age_minutes: int = 30) -> list | None:
        cutoff = datetime.utcnow() - timedelta(minutes=max_age_minutes)
        with self._conn() as conn:
            row = conn.execute(
                """SELECT results FROM search_cache
                   WHERE query = ? AND fetched_at >= ?
                   ORDER BY fetched_at DESC LIMIT 1""",
                (query, cutoff.strftime("%Y-%m-%d %H:%M:%S")),
            ).fetchone()
        return json.loads(row["results"]) if row else None

    # =====================================================
    # Scheduled Events
    # =====================================================

    def add_event(self, channel_id: str, user_id: str, description: str, scheduled_at: datetime):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO scheduled_events (channel_id, user_id, description, scheduled_at) VALUES (?, ?, ?, ?)",
                (channel_id, user_id, description, scheduled_at.strftime("%Y-%m-%d %H:%M:%S")),
            )

    def get_events(self, channel_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT id, user_id, description, scheduled_at FROM scheduled_events
                   WHERE channel_id = ? ORDER BY scheduled_at""",
                (channel_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_event(self, event_id: int):
        with self._conn() as conn:
            conn.execute("DELETE FROM scheduled_events WHERE id = ?", (event_id,))

    # =====================================================
    # Library Catalog (RAG index)
    # =====================================================

    def library_add(
        self,
        category: str,
        description: str,
        file_path: str,
        file_types: list[str],
        embedding: list[float] | None = None,
        symbol: str | None = None,
        tags: list[str] | None = None,
    ):
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO library
                   (category, symbol, description, embedding, file_path, file_types, tags)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    category,
                    symbol,
                    description,
                    json.dumps(embedding) if embedding else None,
                    file_path,
                    json.dumps(file_types),
                    json.dumps(tags) if tags else None,
                ),
            )

    def library_get_all(
        self,
        category: str | None = None,
        symbol: str | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        conditions, params = [], []
        if category:
            conditions.append("category = ?")
            params.append(category)
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol.upper())

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        limit_clause = f"LIMIT {limit}" if limit else ""

        query = f"""SELECT id, category, symbol, description, embedding, file_path, file_types, tags, created_at
                    FROM library {where} ORDER BY created_at DESC {limit_clause}"""

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()

        return [
            {
                **dict(r),
                "embedding": json.loads(r["embedding"]) if r["embedding"] else None,
                "file_types": json.loads(r["file_types"]),
                "tags": json.loads(r["tags"]) if r["tags"] else [],
            }
            for r in rows
        ]

    def library_delete(self, file_path: str):
        with self._conn() as conn:
            conn.execute("DELETE FROM library WHERE file_path = ?", (file_path,))
