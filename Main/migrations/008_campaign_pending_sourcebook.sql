-- Migration 008: the sourcebook a campaign INTENDS to play
--
-- The wizard has to remember the DM's choice across a lobby that can sit for
-- days and survive a restart. The obvious place was `campaign_sourcebook`
-- (migration 007), and that was wrong: that table is the ACTIVE-CANON
-- boundary. Every claim query joins it, so binding at creation made the
-- book's claims queryable before a word of it was installed -- eight
-- effective claims and three discoverable ones visible for a campaign whose
-- world did not exist yet. Worse, an install later REFUSED left the binding
-- in place, so a campaign that never got its book still read as playing it.
--
-- Intent and installation are different facts and need different homes.
-- This column is intent: written when the DM picks, read at session start,
-- and cleared once `install_sourcebook` has done the real binding.

ALTER TABLE campaign ADD COLUMN pending_sourcebook_key TEXT;

INSERT INTO schema_migrations (version) VALUES (8);
