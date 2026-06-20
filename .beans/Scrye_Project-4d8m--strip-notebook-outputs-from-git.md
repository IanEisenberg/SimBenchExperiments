---
# Scrye_Project-4d8m
title: Strip notebook outputs from git
status: completed
type: task
priority: normal
created_at: 2026-06-20T02:43:43Z
updated_at: 2026-06-20T02:45:25Z
---

Configure nbstripout so executed notebook outputs are not tracked; git only sees source-cell changes in notebooks/.

## Summary of Changes
- Added nbstripout (0.9.1) as a dev dependency via `uv add --optional dev`.
- Ran `nbstripout --install --attributes .gitattributes`: registered a local git clean-filter and wrote `.gitattributes` (`*.ipynb filter=nbstripout diff=ipynb`).
- git now tracks only notebook source cells; outputs/execution counts are stripped on commit while the working copy keeps rendered outputs.
- Documented the one-time `uv run nbstripout --install` step in README Setup.
