# Literature

The source PDFs live in [`papers/`](./papers/).
This list is the index: what each paper is and why it matters for simulating
human survey responses and behavior. Grouped by theme; filenames match `papers/`.

## Benchmark & core framing

- **SimBench** — Hu et al., *SimBench: A Large-Scale Benchmark for Group-Level
  Human Behavior Simulation* (ICLR 2026; arXiv 2510.17516).
  `SimBench_paper.pdf`. The benchmark this project targets: 20 harmonized
  surveys, >130 countries, distributional TVD score `S = 100·(1 − TVD(P,Q)/TVD(P,U))`,
  and the alignment–simulation tradeoff (instruction tuning helps consensus
  questions, hurts high-entropy/diverse ones; r ≈ −0.94).

- **Alignment / Calibration / Model Merging** — Hu et al.
  `Alignment_Calibration_ModelMerging_Hu.pdf`. Mechanism behind the tradeoff
  (mode-seeking vs. mass-covering KL) and merging as a calibration lever.

## Demographic conditioning & its pitfalls

- **The Persona Effect** — Hu & Collier, *Quantifying the Persona Effect in LLM
  Simulations* (ACL 2024). `Persona_Effect_Hu_Collier.pdf`. Explicit persona
  conditioning yields limited/negative accuracy gains — better as a validity
  probe than a prediction engine. Direct motivation for the strategy registry.

- **iNews** — Hu & Collier. `iNews_Hu_Collier.pdf`. Persona/individual-annotation
  effects on a news-opinion dataset.

- **Flattening identity groups** — Wang, Morgenstern & Dickerson, *LLMs that
  replace human participants can harmfully misportray and flatten identity
  groups* (Nature Machine Intelligence 2025). `Flatten_Identity_Groups_Wang.pdf`.
  Documents within-group homogenization / stereotype collapse — the failure the
  `anti_flattening` strategy targets.

- **Perils of LLMs for survey data** — Bisbee et al.
  `Perils_of_LLMs_Bisbee.pdf`. Instability and bias when substituting LLM
  output for human survey responses.

## Representation steering (validity probes)

- **Persona Vectors** — Chen et al. (Anthropic), *Persona Vectors: Monitoring
  and Controlling Character Traits in Language Models*.
  `Persona_Vectors_Anthropic.pdf`. Linear activation directions for traits;
  extraction, steering, and prompt-shift monitoring — the "validity probe" idea.

- **Representation Engineering** — Zou et al., *Representation Engineering: A
  Top-Down Approach to AI Transparency* (arXiv 2023 / ICLR 2025).
  `Representation_Engineering_Zou.pdf`. LAT pipeline; reading/control vectors.

## Individual-level simulation

- **Synthetic Personalities** — Kinzinger & Hartmann, *How Well Can LLMs Mimic
  Individual Respondents Using Socio-Economic Microdata?* (2026).
  `Synthetic_Personalities_Kinzinger_Hartmann.pdf`. Entropy-ranked information
  depth beats demographic labels; Pareto point ~75% entropy quartile. Motivates
  feeding behavioral history over "be this demographic" prompts.

- **Generative Agent Simulations of 1,000 People** — Park et al. (2024).
  `Generative_Agent_Simulations_1000_People.pdf`. Person-specific agents;
  ~0.83 normalized accuracy on GSS attitudes, weaker on staked behavior.

- **Generative Agents** — Park et al. (2023), *Interactive Simulacra of Human
  Behavior*. `Generative_Agents.pdf`. The memory-architecture foundation (not a
  predictive-accuracy result).

- **Centaur** — Binz et al., *A Foundation Model of Human Cognition*.
  `Centaur_Binz_FoundationModelHumanCognition.pdf`. Fine-tuning an LLM on large
  behavioral datasets to predict human choices.

## Behavioral validity & the value–action gap

- **Mind the Value–Action Gap** — Shen et al. `Value_Action_Gap_Shen.pdf`.
  Attitudes are easier to simulate than staked behavior — the central open gap
  in this literature.

- **Beyond Believability** — fine-tuning on real behavioral traces.
  `Beyond_Believability_FineTuned.pdf`. Fine-tuning on behavioral traces closes
  part of the behavior gap.

- **LLMs Assume People Are More Rational** — Liu et al.
  `LLMs_Assume_Rational_Liu.pdf`. Systematic rationality bias in simulated
  humans.

- **A Turing Test of Behavioral Similarity** — Mei et al. (PNAS 2024).
  `Turing_Test_Behavioral_Mei.pdf`. Whether chatbots behave like humans in
  economic games.

- **LLMs Replace Human Subjects? (156 replications)** —
  `LLMs_Replace_Human_Subjects_156Replication.pdf`. Large replication audit of
  LLM-as-participant across social-science studies.

## Distributional alignment & scale

- **Distributional Alignment** — Meister et al.
  `Distributional_Alignment_Meister.pdf`. Benchmarking and steering LLMs to
  match human response *distributions* (not just modes).

- **AgentSociety** — `AgentSociety.pdf`. Large-scale agent-based social
  simulation; a scale demonstration with face-validity-level evaluation.

---

See [`research_overview.md`](./research_overview.md) for the synthesized
literature map and open questions, and [`docs/experiments/README.md`](../experiments/README.md)
for the experiments run against SimBench.
