# Refusal judges across benchmarks: what is official, what is still usable

Parent: [Experiment index](experiments/README.md).

Survey made on 2026-10-03 for the judge-agreement study
([x102](experiments/x102-orbench-judge-agreement.md)). Every availability
statement was checked by live fetch on that day unless marked
**unverified**. "REFUSAL" means the judge decides declined-vs-complied;
"HARM" means it decides whether the response is harmful or completes the
behaviour, under which a refusal and a harmless compliance are both
"safe" — useless for measuring refusal.

## Two availability facts that frame everything

- **OpenAI retires `gpt-4-turbo-2024-04-09`, `gpt-4-0613`,
  `gpt-4-1106-preview`, `gpt-3.5-turbo(-0125)` and `gpt-4o-2024-05-13` on
  2026-10-23** (deprecation notice of 2026-04-22). OR-Bench's, XSTest's,
  OKTest's, PHTest's and CoCoNot's official API judges become
  unreproducible at the vendor on that date. The oldest OpenAI snapshots
  with no announced shutdown: `gpt-4o-2024-08-06` (USD 2.50 / 10 per M),
  `gpt-4o-mini` (0.15 / 0.60), `gpt-4.1` (2 / 8), `gpt-4.1-mini`
  (0.40 / 1.60).
- **Anthropic retired Claude 3.5 Sonnet on 2025-10-28**; FalseReject's
  official judge is gone.

## Over-refusal benchmarks

Table: [Over-refusal benchmarks](experiments/json/refusal-judges__over-refusal-benchmarks.json) — each over-refusal benchmark's official judge, label space, construct, 2026-10 availability and judge–human agreement; OR-Bench's GPT-4-turbo judge retires 2026-10-23 and has no human validation, while XSTest ships 2,250 human-labelled completions.

Also checked, no reusable judge: XSB/MS-XSB (manual), Health-ORSC-Bench
(keywords + Grok-4), RefusalBench bio (3-LLM council) and grounded
(Claude-4-Sonnet), AOR-Bench (audio). No benchmark named OverRefusalBench,
XSTest-v2, SG-Bench, SafeMT or OverThink exists.

## Jailbreak / harmfulness benchmarks

Table: [Jailbreak / harmfulness benchmarks](experiments/json/refusal-judges__jailbreak-harmfulness-benchmarks.json) — the same for jailbreak and harmfulness benchmarks; most official judges decide HARM, and SORRY-Bench's fine-tuned judge is the one REFUSAL judge with a published human κ (0.81).

## Guard models with an explicit refusal field

Table: [Guard models with an explicit refusal field](experiments/json/refusal-judges__guard-models.json) — open guard models that emit a refusal label, with size, gating and reported refusal quality; WildGuard has third-party human validation (96.3% / κ 0.93 on XSTest), Qwen3Guard-Gen-4B the best reported refusal F1 (94.6 XSTest) but no human number.

**Harm-only guards (no refusal label; excluded for this purpose):**
Llama Guard 1–4 (`meta-llama/Llama-Guard-3-8B`, `Llama-Guard-4-12B`;
forced into refusal detection F1 ≈ 54), ShieldGemma 1/2, NVIDIA
Aegis / NemoGuard / Nemotron Content Safety (through 3.5, 2026-06),
`openai/gpt-oss-safeguard-20b/120b` (policy-as-prompt, no fixed labels),
IBM Granite Guardian 3.x–4.1, ThinkGuard, RSafe, DuoGuard, X-Guard.

## Dedicated refusal classifiers (small encoders)

Table: [Dedicated refusal classifiers (small encoders)](experiments/json/refusal-judges__refusal-classifiers.json) — six small encoder refusal classifiers with base model, input and validation status.

None of these has an independent human-agreement number.

## Recommendation for re-scoring OR-Bench three-way labels on a binary axis

Mapping: `refusal := direct_refusal ∪ indirect_refusal`, exactly OR-Bench's
own metric. The hard stratum is `indirect_refusal` (declines-then-redirects
or redirects silently): FalseReject counts the analogue as compliance,
XSTest keeps it separate, binary judges split on it. Always report
agreement on that stratum separately.

1. **WildGuard** — the only open refusal judge with multiple independent
   human validations and the de-facto refusal judge of the 2025–26
   literature. Needs 8-bit or FP8 on a 16 GB laptop GPU.
2. **Qwen3Guard-Gen-4B** — ungated, best reported refusal F1, fits any
   GPU. Same vendor lineage as the incumbent judge, different training
   objective.
3. **SORRY-Bench judge** — the one benchmark-official judge whose construct
   is literally refusal-vs-fulfilment with a published κ. Trained only on
   unsafe instructions: in-distribution on OR-Bench Toxic, out of
   distribution on OR-Bench Hard (benign prompts). Secondary judge; report
   Toxic and Hard separately.
4. **API anchor: `gpt-4o-2024-08-06` with the verbatim OR-Bench prompt** —
   the closest surviving analogue to the official judge, no announced
   shutdown, USD 2.50 / 10 per M; zero-shot GPT-4o scored 94.0% / κ 0.88
   on refusal (Quan et al. 2026) and κ 0.789 on SORRY-Bench's human set.

**XSTest's 2,250 human-labelled completions** are the only public
human-labelled over-refusal responses: running the incumbent judge (and
any candidate) on them yields a direct human-agreement number that
OR-Bench's own materials cannot provide.

**Excluded as refusal judges:** every HARM-only model above; keyword
lists (κ 0.37–0.54) except as a reported baseline; CoCoNot (broader
construct); FalseReject (opposite polarity on the middle class).

## How OR-Bench responses are actually judged (benchmark and later papers)

Second survey, 2026-10-03, primary sources only.

**The benchmark's own protocol.**
- The three-way label lives only in the judge prompt and the judge's CSV.
  The reported metric is a **binary rejection rate**: Appendix S of the
  paper says "we count both direct refusal and indirect refusal as
  rejections"; no table reports the two refusal classes separately. The
  leaderboard (`bench-llms/or-bench-leaderboard`) is a static page of
  author-supplied binary rates and names no judge.
- The prompt is fixed in `response_checker/check_response.py`: one user
  message, no system prompt, `temperature 0`, model `gpt-4-turbo-2024-04-09`
  (or Llama-3-70b on Together, never used for a reported number). Parser:
  last `[[...]]`; anything that is not `direct_answer` or
  `indirect_refusal` counts as `direct_refusal` (fail-open); no bracket
  raises.
- The paper's only judge comparison for responses is keyword vs GPT-4
  (Table 8): gaps of 2.4 points on three models and 5.5 on
  gpt-3.5-turbo-0613. **No human validation and no inter-judge agreement
  exists for response judging**; all human numbers concern prompt
  moderation.
- The authors released no judged responses. The demo Space holds ~7,500
  unlabeled responses (150 per model × 25 models × safe/toxic).

**Later papers (16 found that report OR-Bench numbers, 2024-12 → 2026-09).**
Six use keyword matching; nine use an LLM or guard judge with at least
eight distinct judge models and five distinct prompts. Only two reuse
the verbatim official prompt with another model (DeepSeek-V4-Flash in
arXiv:2608.13482; an unstated model in arXiv:2602.12418), one reuses the
label scheme (MORBench, arXiv:2505.18325: Llama-3.1-70B and GPT-4o). The
only OR-Bench-specific agreement evidence: Hasan & Biswas
(arXiv:2605.05427) Llama-3.3-70B vs Qwen2.5-32B inter-judge κ 0.785 on
OR-Bench (own three-label prompt). ORFuzz's OR-Judge (Qwen2.5-14B LoRA)
is the only judge trained on human labels that include OR-Bench
responses (F1 0.967). DDOR (arXiv:2606.03601) used Qwen3Guard-Gen-0.6B
as the refusal detector, κ 0.97 vs three experts on 400 non-OR-Bench
responses. **No paper reports human agreement for any judge on OR-Bench
responses.** Numbers across papers are not comparable.

**Reading for this project.** "Closest to official" means the
verbatim prompt with a living model, stated as not the official judge.
"Best validated" points to WildGuard and Qwen3Guard-Gen for the binary
refusal axis, and to Llama-3.3-70B-Instruct as the open LLM judge with
OR-Bench inter-judge evidence. A hand-labelled stratified sample of
OR-Bench responses would be the first human validation of any judge on
this benchmark.

## Primary sources

- OR-Bench: <https://github.com/justincui03/or-bench>, <https://arxiv.org/abs/2405.20947>, <https://huggingface.co/datasets/bench-llm/or-bench>
- XSTest: <https://github.com/paul-rottger/xstest>, <https://arxiv.org/abs/2308.01263>, <https://huggingface.co/datasets/allenai/xstest-response>
- StrongREJECT: <https://github.com/dsbowen/strong_reject>, <https://huggingface.co/qylu4156/strongreject-15k-v1>
- SORRY-Bench: <https://github.com/sorry-bench/sorry-bench>, <https://huggingface.co/sorry-bench/ft-mistral-7b-instruct-v0.2-sorry-bench-202406>, <https://arxiv.org/abs/2406.14598>
- WildGuard: <https://huggingface.co/allenai/wildguard>, <https://github.com/allenai/wildguard>, <https://arxiv.org/abs/2406.18495>; third-party validations <https://arxiv.org/abs/2602.02132>, <https://arxiv.org/abs/2606.09864>, <https://arxiv.org/abs/2608.02684>
- Qwen3Guard: <https://huggingface.co/Qwen/Qwen3Guard-Gen-4B>, <https://github.com/QwenLM/Qwen3Guard>, <https://arxiv.org/abs/2510.14276>
- HarmBench: <https://github.com/centerforaisafety/HarmBench>; JailbreakBench: <https://github.com/JailbreakBench/jailbreakbench>; SALAD-Bench: <https://github.com/OpenSafetyLab/SALAD-BENCH>; CoCoNot: <https://github.com/allenai/noncompliance>; FalseReject: <https://huggingface.co/datasets/AmazonScience/FalseReject>; PHTest: <https://github.com/umd-huang-lab/FalseRefusal>; OKTest: <https://github.com/InvokerStark/OverKill>; ORFuzz: <https://github.com/HotBento/ORFuzz>
- GuardReasoner: <https://github.com/yueliu1999/GuardReasoner>; PolyGuard: <https://arxiv.org/abs/2504.04377>; Minos: <https://huggingface.co/NousResearch/Minos-v1>; garak detector: <https://huggingface.co/garak-llm/garak-refusal-detector>
- Deprecations and prices: <https://developers.openai.com/api/docs/deprecations>, <https://platform.claude.com/docs/en/docs/about-claude/model-deprecations>, <https://openrouter.ai/api/v1/models>
