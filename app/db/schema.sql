-- =====================================================
-- Operational Tables (lightweight, ephemeral)
-- =====================================================

-- Conversation history per Discord channel
CREATE TABLE IF NOT EXISTS conversations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id TEXT    NOT NULL,
    user_id    TEXT    NOT NULL,
    role       TEXT    NOT NULL CHECK(role IN ('user', 'assistant')),
    content    TEXT    NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Real-time stock price cache (short-lived)
CREATE TABLE IF NOT EXISTS stock_cache (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol     TEXT      NOT NULL,
    price      REAL,
    data       TEXT,               -- JSON blob
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Web search result cache
CREATE TABLE IF NOT EXISTS search_cache (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    query      TEXT      NOT NULL,
    results    TEXT      NOT NULL, -- JSON list
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Scheduled events (Event Scheduling intent)
CREATE TABLE IF NOT EXISTS scheduled_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id   TEXT      NOT NULL,
    user_id      TEXT      NOT NULL,
    description  TEXT      NOT NULL,
    scheduled_at TIMESTAMP NOT NULL,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- Library Catalog (RAG index → points to files)
-- =====================================================
--
-- All heavy content (reports, indicators, news, signals,
-- market data, knowledge) lives as files under
-- /mnt/my_library/dest_db/{category}/
--
-- This table is the lightweight search index:
--   1. embed the short `description`
--   2. similarity search → get file_path
--   3. read actual file for full detail
-- =====================================================

CREATE TABLE IF NOT EXISTS library (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    category    TEXT    NOT NULL,   -- market_data | news | indicators | signals | reports | knowledge
    symbol      TEXT,               -- NULL = not symbol-specific
    description TEXT    NOT NULL,   -- short summary, used for RAG search
    embedding   TEXT,               -- JSON list[float] of description embedding
    file_path   TEXT    NOT NULL UNIQUE, -- base path without extension
    file_types  TEXT    NOT NULL,   -- JSON list: ["md"] | ["json","csv"] | ["md","json","csv"]
    tags        TEXT,               -- JSON list of searchable tags
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_library_category ON library(category);
CREATE INDEX IF NOT EXISTS idx_library_symbol    ON library(symbol);
