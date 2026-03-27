"""
Library — file-based knowledge store with DB catalog index.

Write flow:
    library.save_*(raw_data)
        → LLM generates detailed markdown content
        → LLM generates short description (for RAG index)
        → LLM generates embedding (for similarity search)
        → writes file(s) to /mnt/my_library/dest_db/{category}/
        → inserts lightweight record into DB (description + embedding + file_path)

Search flow (RAG):
    results = library.search(query_embedding, top_k=5)
    content = library.read_file(results[0]["file_path"], file_type="md")
"""

import csv
import json
import math
import os
from datetime import datetime

from db.db import DB
from router.LLMClient import LLMClient

BASE_DIR = "/mnt/my_library/dest_db"
CATEGORIES = ["market_data", "news", "indicators", "signals", "reports", "knowledge"]


class Library:
    def __init__(self, db: DB, llm: LLMClient):
        self.db = db
        self.llm = llm
        for cat in CATEGORIES:
            os.makedirs(os.path.join(BASE_DIR, cat), exist_ok=True)

    # =====================================================
    # LLM helpers
    # =====================================================

    def _reshape(self, raw: str, hint: str = "") -> str:
        """
        Reshape raw input (scraped text, mixed content, etc.) into clean markdown.
        LLM acts as a formatter, not a content generator.
        """
        system = (
            "You are a document formatter. "
            "Reformat the following raw input into clean, well-structured markdown. "
            "Preserve all factual content — do not add or invent information. "
            + hint
        )
        response = self.llm.discuss([
            {"role": "system", "content": system},
            {"role": "user",   "content": raw},
        ])
        return response["choices"][0]["message"]["content"].strip()

    def _auto_description(self, text: str) -> str:
        """Generate a short one-sentence description for RAG indexing."""
        response = self.llm.discuss([
            {"role": "system", "content": "Summarize the following in one concise sentence for search indexing. Output only the sentence."},
            {"role": "user",   "content": text[:600]},
        ])
        return response["choices"][0]["message"]["content"].strip()

    def _auto_embedding(self, text: str) -> list[float]:
        return self.llm.embedding(text)

    # =====================================================
    # Save: Market Data  →  CSV  (structured, no LLM content)
    # =====================================================

    def save_market_data(
        self,
        symbol: str,
        rows: list[dict],   # [{date, open, high, low, close, volume}, ...]
    ) -> str:
        """Save OHLCV rows as CSV. Returns base path."""
        if not rows:
            raise ValueError("rows must not be empty")

        dates = sorted(r["date"] for r in rows)
        filename = f"{symbol.upper()}_{dates[0]}_{dates[-1]}"
        base = os.path.join(BASE_DIR, "market_data", filename)

        fieldnames = ["date", "open", "high", "low", "close", "volume"]
        with open(base + ".csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda r: r["date"]))

        description = f"{symbol.upper()} OHLCV market data from {dates[0]} to {dates[-1]}"
        embedding = self._auto_embedding(description)

        self.db.library_add(
            category="market_data",
            description=description,
            file_path=base,
            file_types=["csv"],
            embedding=embedding,
            symbol=symbol.upper(),
            tags=["ohlcv", symbol.upper()],
        )
        return base

    # =====================================================
    # Save: News  →  LLM-generated Markdown
    # =====================================================

    def save_news(
        self,
        title: str,
        raw_content: str,
        symbol: str | None = None,
        url: str | None = None,
        sentiment_score: float | None = None,
        sentiment_label: str | None = None,
        published_at: datetime | None = None,
    ) -> str:
        sym = symbol.upper() if symbol else "MARKET"
        pub = published_at.strftime("%Y-%m-%d %H:%M:%S") if published_at else "N/A"
        sentiment_line = ""
        if sentiment_label and sentiment_score is not None:
            sentiment_line = f"**Sentiment:** {sentiment_label} ({sentiment_score:+.2f})  \n"

        # LLM generates detailed markdown analysis
        md_body = self._reshape(
            "You are a financial news analyst. "
            "Given the raw news article below, write a detailed markdown analysis. "
            "Include: key facts, market implications, potential impact on the stock, and a brief sentiment summary. "
            "Use markdown headers and bullet points.",
            f"Title: {title}\nSymbol: {sym}\nPublished: {pub}\n\n{raw_content}",
        )

        md = f"# {title}\n"
        md += f"**Symbol:** {sym}  \n"
        md += f"**Published:** {pub}  \n"
        md += sentiment_line
        if url:
            md += f"**Source:** {url}  \n"
        md += f"\n{md_body}\n"

        base = os.path.join(BASE_DIR, "news", f"{sym}_{_ts()}")
        with open(base + ".md", "w") as f:
            f.write(md)

        description = self._auto_description(f"{title}. {raw_content[:300]}")
        embedding = self._auto_embedding(description)

        self.db.library_add(
            category="news",
            description=description,
            file_path=base,
            file_types=["md"],
            embedding=embedding,
            symbol=symbol.upper() if symbol else None,
            tags=["news", sym],
        )
        return base

    # =====================================================
    # Save: Technical Indicators  →  JSON + CSV + LLM Markdown
    # =====================================================

    def save_indicators(
        self,
        symbol: str,
        date: str,          # "YYYY-MM-DD"
        indicators: dict,   # {"RSI": 65.3, "MACD": {"value": 0.42, "signal": 0.38, "hist": 0.04}}
    ) -> str:
        base = os.path.join(BASE_DIR, "indicators", f"{symbol.upper()}_{date}")

        # JSON
        payload = {
            "symbol": symbol.upper(),
            "date": date,
            "indicators": indicators,
            "generated_at": datetime.utcnow().isoformat(),
        }
        with open(base + ".json", "w") as f:
            json.dump(payload, f, indent=2)

        # CSV (flatten multi-value indicators)
        flat_rows = _flatten_indicators(indicators)
        with open(base + ".csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["indicator", "value"])
            writer.writeheader()
            writer.writerows(flat_rows)

        # LLM generates markdown interpretation
        md_body = self._reshape(
            "You are a quantitative analyst. "
            "Interpret the following technical indicators for the given stock and date. "
            "Explain what each indicator signals, highlight key observations, and give an overall technical outlook. "
            "Use markdown.",
            f"Symbol: {symbol.upper()}\nDate: {date}\n\nIndicators:\n{json.dumps(indicators, indent=2)}",
        )

        md = f"# {symbol.upper()} Technical Indicators — {date}\n\n"
        md += _indicators_to_md_table(indicators) + "\n"
        md += md_body + "\n"
        with open(base + ".md", "w") as f:
            f.write(md)

        description = self._auto_description(
            f"{symbol.upper()} technical indicators on {date}: {json.dumps(_flatten_indicators(indicators))}"
        )
        embedding = self._auto_embedding(description)

        self.db.library_add(
            category="indicators",
            description=description,
            file_path=base,
            file_types=["json", "csv", "md"],
            embedding=embedding,
            symbol=symbol.upper(),
            tags=["indicators", symbol.upper(), date],
        )
        return base

    # =====================================================
    # Save: Quant Signal  →  LLM Markdown + JSON
    # =====================================================

    def save_signal(
        self,
        symbol: str,
        signal: str,        # BUY | SELL | HOLD
        confidence: float,
        indicators: dict,
        timeframe: str | None = None,   # SHORT | MID | LONG
    ) -> str:
        now = datetime.utcnow()
        base = os.path.join(BASE_DIR, "signals", f"{symbol.upper()}_{_ts()}")
        tf = timeframe.upper() if timeframe else "N/A"

        # LLM generates detailed reasoning
        reasoning = self._reshape(
            "You are a quantitative trading analyst. "
            "Given the signal, confidence, and indicator data below, write a detailed markdown explanation of the reasoning behind this signal. "
            "Include: which indicators support the signal, risk factors, and suggested action. "
            "Use markdown.",
            f"Symbol: {symbol.upper()}\nSignal: {signal.upper()}\nConfidence: {confidence:.2f}\n"
            f"Timeframe: {tf}\n\nIndicators:\n{json.dumps(indicators, indent=2)}",
        )

        tf_line = f"**Timeframe:** {tf}  \n" if timeframe else ""
        md = f"# {symbol.upper()} Signal: {signal.upper()}\n"
        md += f"**Confidence:** {confidence:.2f}  \n"
        md += tf_line
        md += f"**Generated:** {now.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
        md += f"\n## Supporting Indicators\n\n{_indicators_to_md_table(indicators)}\n"
        md += f"\n## Reasoning\n\n{reasoning}\n"

        with open(base + ".md", "w") as f:
            f.write(md)

        payload = {
            "symbol": symbol.upper(),
            "signal": signal.upper(),
            "confidence": confidence,
            "timeframe": timeframe,
            "reasoning": reasoning,
            "indicators": indicators,
            "generated_at": now.isoformat(),
        }
        with open(base + ".json", "w") as f:
            json.dump(payload, f, indent=2)

        description = self._auto_description(
            f"{symbol.upper()} {signal.upper()} signal, confidence {confidence:.2f}, timeframe {tf}. "
            f"Indicators: {json.dumps(_flatten_indicators(indicators))}"
        )
        embedding = self._auto_embedding(description)

        self.db.library_add(
            category="signals",
            description=description,
            file_path=base,
            file_types=["md", "json"],
            embedding=embedding,
            symbol=symbol.upper(),
            tags=["signal", signal.upper(), symbol.upper()],
        )
        return base

    # =====================================================
    # Save: Analysis Report  →  LLM Markdown (+ JSON metadata)
    # =====================================================

    def save_report(
        self,
        symbol: str,
        report_type: str,   # TECHNICAL | FUNDAMENTAL | SENTIMENT | COMPREHENSIVE
        raw_data: str,      # raw input for LLM to write the report from
        confidence_score: float | None = None,
        metadata: dict | None = None,
    ) -> str:
        now = datetime.utcnow()
        base = os.path.join(BASE_DIR, "reports", f"{symbol.upper()}_{report_type.upper()}_{_ts()}")
        conf_line = f"**Confidence:** {confidence_score:.2f}  \n" if confidence_score is not None else ""

        # LLM writes the full report
        content = self._reshape(
            f"You are a professional financial analyst specializing in {report_type.lower()} analysis. "
            f"Write a comprehensive {report_type.upper()} analysis report for {symbol.upper()} in markdown format. "
            "Use headers, bullet points, tables where appropriate. Be detailed and data-driven.",
            raw_data,
        )

        md = f"# {symbol.upper()} — {report_type.upper()} Report\n"
        md += f"**Generated:** {now.strftime('%Y-%m-%d %H:%M:%S')} UTC  \n"
        md += conf_line
        md += f"\n{content}\n"
        if metadata:
            md += f"\n## Metadata\n\n```json\n{json.dumps(metadata, indent=2)}\n```\n"

        with open(base + ".md", "w") as f:
            f.write(md)

        file_types = ["md"]
        if metadata:
            meta_payload = {
                "symbol": symbol.upper(),
                "report_type": report_type.upper(),
                "confidence_score": confidence_score,
                "metadata": metadata,
                "generated_at": now.isoformat(),
            }
            with open(base + ".json", "w") as f:
                json.dump(meta_payload, f, indent=2)
            file_types.append("json")

        description = self._auto_description(f"{symbol.upper()} {report_type.upper()} report. {raw_data[:300]}")
        embedding = self._auto_embedding(description)

        self.db.library_add(
            category="reports",
            description=description,
            file_path=base,
            file_types=file_types,
            embedding=embedding,
            symbol=symbol.upper(),
            tags=["report", report_type.upper(), symbol.upper()],
        )
        return base

    # =====================================================
    # Save: Knowledge Chunk  →  Markdown
    # =====================================================

    def save_knowledge(
        self,
        content: str,
        source: str | None = None,
    ) -> str:
        now = datetime.utcnow()
        description = self._auto_description(content)
        embedding = self._auto_embedding(description)

        slug = description[:40].lower().replace(" ", "_").replace("/", "_")
        base = os.path.join(BASE_DIR, "knowledge", f"{slug}_{_ts()}")

        md = f"# Knowledge Entry\n"
        md += f"**Source:** {source or 'N/A'}  \n"
        md += f"**Created:** {now.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
        md += f"\n## Content\n\n{content}\n"

        with open(base + ".md", "w") as f:
            f.write(md)

        self.db.library_add(
            category="knowledge",
            description=description,
            file_path=base,
            file_types=["md"],
            embedding=embedding,
            symbol=None,
            tags=["knowledge"],
        )
        return base

    # =====================================================
    # Search (RAG) & Read
    # =====================================================

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        category: str | None = None,
        symbol: str | None = None,
    ) -> list[dict]:
        """
        Cosine similarity search over description embeddings.
        Returns top_k entries sorted by score descending.
        """
        entries = self.db.library_get_all(category=category, symbol=symbol)
        scored = []
        for entry in entries:
            if not entry["embedding"]:
                continue
            sim = _cosine_similarity(query_embedding, entry["embedding"])
            scored.append({**entry, "score": round(sim, 4)})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def read_file(self, file_path: str, file_type: str = "md") -> str:
        full = file_path if file_path.endswith(f".{file_type}") else f"{file_path}.{file_type}"
        with open(full, "r") as f:
            return f.read()

    def read_json(self, file_path: str) -> dict:
        full = file_path if file_path.endswith(".json") else f"{file_path}.json"
        with open(full, "r") as f:
            return json.load(f)

    def get_catalog(
        self,
        category: str | None = None,
        symbol: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        return self.db.library_get_all(category=category, symbol=symbol, limit=limit)

    def delete(self, file_path: str, delete_files: bool = True):
        if delete_files:
            entries = self.db.library_get_all()
            for e in entries:
                if e["file_path"] == file_path:
                    for ft in e["file_types"]:
                        fp = f"{file_path}.{ft}"
                        if os.path.exists(fp):
                            os.remove(fp)
                    break
        self.db.library_delete(file_path)


# =====================================================
# Helpers
# =====================================================

def _ts() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _flatten_indicators(indicators: dict) -> list[dict]:
    rows = []
    for name, val in indicators.items():
        if isinstance(val, dict):
            for sub_key, sub_val in val.items():
                rows.append({"indicator": f"{name}_{sub_key}", "value": sub_val})
        else:
            rows.append({"indicator": name, "value": val})
    return rows


def _indicators_to_md_table(indicators: dict) -> str:
    rows = _flatten_indicators(indicators)
    table = "| Indicator | Value |\n|-----------|-------|\n"
    for r in rows:
        table += f"| {r['indicator']} | {r['value']} |\n"
    return table


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
