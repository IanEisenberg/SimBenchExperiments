---
# Scrye_Project-5oez
title: Publish SimBenchExperiments + single-commit Scrye submission repo
status: todo
type: task
priority: normal
created_at: 2026-06-23T05:15:57Z
updated_at: 2026-06-23T05:15:57Z
---

Two private GitHub repos under IanEisenberg. WAIT for user 'go' before executing — user is still finalizing the repo.

## Repo 1 — SimBenchExperiments (private)
- Full history (all 84 commits), pushed as-is.
- gh repo create IanEisenberg/SimBenchExperiments --private --source=. --remote=origin --push

## Repo 2 — submission (private, single commit)
- Built in a FRESH dir (copy tracked files -> git init -> ONE commit -> push) so it shares zero objects/history with repo 1.
- Content: everything EXCEPT .beans/ (confirm: also drop .beans.yml? it's beans config).
- Keep docs/experiments/, .superpowers/, src, notebooks, tests, README, final docs.
- Sharing: leave under IanEisenberg; user shares manually.
- Name: TBD (default e.g. SimBench-Scrye-Submission).

## Pre-push safety
- Verify .env, scrye-ian-openrouter.rtf (contains live OpenRouter key), .DS_Store stay untracked/excluded.
- Hard hook blocks pushes to main; final push may need user to run via ! prefix.

## Open at execution time
- Snapshot state (current working tree vs HEAD) — user deferred; will say which 'final commit' to capture.
- Submission repo name.

## Todo
- [ ] User gives go + final state ready
- [ ] Re-verify no secrets tracked
- [ ] Create + push SimBenchExperiments (full history)
- [ ] Build submission dir, exclude .beans, single commit
- [ ] Create + push submission repo
- [ ] Confirm both repos private, hand user the links
