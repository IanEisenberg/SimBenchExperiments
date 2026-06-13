---
# Scrye_Project-rza4
title: 'Scrye mini-project: brainstorm gameplan and direction'
status: in-progress
type: task
priority: normal
created_at: 2026-06-10T21:21:21Z
updated_at: 2026-06-11T02:40:31Z
---

First brainstorming session for the Scrye AI Research Mini Project (SimBench survey distribution prediction + feedback architecture). Goal: map the direction space, converge on an approach, produce a design doc.

- [x] Explore project context (PDF brief, prep docs, SimBench paper)
- [x] Ask clarifying questions (constraints, time budget, compute)
- [x] Propose candidate approaches with trade-offs
- [x] Present design and get approval (A chosen as spine; options doc requested and delivered)
- [x] Write design doc and commit (docs/superpowers/specs/2026-06-10-scrye-mini-project-gameplan-design.md, commit ba1ff38)
- [x] Spec self-review
- [ ] User reviews spec
- [ ] Transition to writing-plans

## Constraints established
- Clock running; 5-10 hours total (part-time alongside full job)
- ~$100 API budget; cheap models fine (Gemini Flash, Together+Qwen) — model-as-commodity thesis
- Posture: rigor as the product, but must meaningfully improve the score; defensible good practices; limitations framed as clear areas for improvement

## SimBench facts (from research agent)
- arXiv 2510.17516, HF pitehu/SimBench; S = 100*(1 - TVD/TVD_uniform), uniform baseline = 0; best zero-shot Claude-3.7-Sonnet 40.8
- Verbalized beats logprobs; CoT does not help; mode-seeking on high-entropy questions is the diagnosed failure; grouped conditioning degrades scores
- Pop split 7,167 cases / Grouped 6,343; no train/dev split — must construct own holdout; ground truth public (leakage discussion ours to own)
- All 3 required questions present (LatinoBarometro 2023, ESS 2016) with 67-91 grouped variants each
