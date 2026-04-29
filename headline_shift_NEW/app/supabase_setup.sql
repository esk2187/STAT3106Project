-- ============================================================
-- Headline Shift: Supabase database setup
-- Run this entire file in the Supabase SQL Editor once.
-- ============================================================

-- Table 1: Raw pairwise comparisons
-- Every time a user clicks a button in the app, one row is added here.
CREATE TABLE IF NOT EXISTS comparisons (
    id          BIGSERIAL PRIMARY KEY,
    headline_a  TEXT NOT NULL,
    headline_b  TEXT NOT NULL,
    choice      TEXT NOT NULL CHECK (choice IN ('A', 'B', 'equal')),
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Table 2: Aggregated win/loss counts per headline
-- Updated automatically by the app after each comparison.
CREATE TABLE IF NOT EXISTS headline_scores (
    headline    TEXT PRIMARY KEY,
    wins        INTEGER NOT NULL DEFAULT 0,
    losses      INTEGER NOT NULL DEFAULT 0,
    ties        INTEGER NOT NULL DEFAULT 0,
    comparisons INTEGER NOT NULL DEFAULT 0,
    uncertainty REAL NOT NULL DEFAULT 1.0,
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- ── Row Level Security ────────────────────────────────────────────────────────
-- Allow anyone (anon key) to read and write — this is a class annotation project.

ALTER TABLE comparisons     ENABLE ROW LEVEL SECURITY;
ALTER TABLE headline_scores ENABLE ROW LEVEL SECURITY;

-- comparisons: anyone can insert and read
CREATE POLICY "anon_insert_comparisons"
    ON comparisons FOR INSERT TO anon WITH CHECK (true);

CREATE POLICY "anon_select_comparisons"
    ON comparisons FOR SELECT TO anon USING (true);

-- headline_scores: anyone can read, insert, and update
CREATE POLICY "anon_select_scores"
    ON headline_scores FOR SELECT TO anon USING (true);

CREATE POLICY "anon_insert_scores"
    ON headline_scores FOR INSERT TO anon WITH CHECK (true);

CREATE POLICY "anon_update_scores"
    ON headline_scores FOR UPDATE TO anon USING (true);
