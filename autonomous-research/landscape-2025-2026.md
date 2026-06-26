# Autonomous Research / "AI Scientist" Landscape (2025 - mid-2026)

A curated, ranked map of domain-general autonomous-research work: end-to-end discovery systems, hypothesis generation and validation, self-driving labs, agent frameworks and scientific foundation/reasoning models, benchmarks, and the peer-review milestones and critiques that frame the field. Scope: primarily Jan 2025 - Jun 2026 (today is 2026-06-25), with seminal 2023-2024 anchors marked as foundational. Domain-general: ML, biology, chemistry/materials, physics, math, and social science - not just biomedicine.

Method and trust: every arxiv ID, DOI, and date below was either fetched from the source abstract/page or corroborated across multiple independent outlets during six parallel web-research passes. Vendor "first ever" claims (Sakana, Autoscience, Intology) conflict with one another and are flagged claimed-vs-verified. Items that could not be fetched directly are collected in the final Verification Flags section. Maturity tags: demo (works, narrow) / benchmarked (measured on a benchmark or open-sourced) / peer-reviewed (passed real review) / deployed (a product or production use).

A central caution runs through everything: the most rigorous results in this space are negative or deflationary (the Ideation-Execution Gap, the A-Lab reanalysis, the AI Scientist failure-rate audits, Robin's 7.5x-vs-1.75x effect-size gap). Read "Open Problems & Critiques" alongside the headline systems.

---

## Must-Read Shortlist (ranked across all threads)

| #  | Work                          | Org                          | Date                | Link / ID                                                       | Maturity                  | Why it tops the list                                                            |
|----|-------------------------------|------------------------------|---------------------|----------------------------------------------------------------|---------------------------|--------------------------------------------------------------------------------|
| 1  | Towards an AI co-scientist    | Google (DeepMind/Research)   | Feb 2025            | [arxiv 2502.18864](https://arxiv.org/abs/2502.18864)           | deployed + peer-reviewed  | Largest effort; generate-debate-evolve + Elo tournament; wet-lab-validated AML/fibrosis/AMR leads; Nature 2026 + Gemini for Science product |
| 2  | The Virtual Lab               | Stanford / CZ Biohub         | Jul 2025            | [Nature s41586-025-09442-9](https://www.nature.com/articles/s41586-025-09442-9) | peer-reviewed | Only agent-discovery system in Nature with wet-lab-validated output (92 SARS-CoV-2 nanobodies; 2 validated) |
| 3  | Robin                         | FutureHouse + Oxford         | May 2025            | [arxiv 2505.13400](https://arxiv.org/abs/2505.13400)           | peer-reviewed             | First to automate every intellectual step of a wet-bio loop; surfaced ripasudil for dry AMD (Nature) |
| 4  | The AI Scientist v1 / v2      | Sakana AI                    | Aug 2024 / Apr 2025 | [2408.06292](https://arxiv.org/abs/2408.06292) / [2504.08066](https://arxiv.org/abs/2504.08066) | demo / peer-reviewed (workshop) | Field-defining progenitor; v2 = first fully-AI paper to clear (workshop) peer review |
| 5  | POPPER                        | Stanford (SNAP)              | Feb 2025            | [arxiv 2502.09858](https://arxiv.org/abs/2502.09858)           | peer-reviewed             | The hypothesis-validation engine: agentic sequential falsification with strict Type-I error control |
| 6  | AlphaEvolve                   | Google DeepMind              | May 2025            | [DeepMind blog](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) | deployed | Evolutionary coding agent; first improvement on 4x4 matrix-mult in 56 years; real Google datacenter/chip impact |
| 7  | MLE-bench                     | OpenAI                       | Oct 2024            | [arxiv 2410.07095](https://arxiv.org/abs/2410.07095)           | benchmarked               | De-facto industry ML-engineering agent benchmark; reported in model cards |
| 8  | RE-Bench                      | METR                         | Nov 2024            | [arxiv 2411.15114](https://arxiv.org/abs/2411.15114)           | benchmarked               | Frontier AI-R&D-automation safety eval; head-to-head vs 61 human experts; cited in lab safety frameworks |
| 9  | The Ideation-Execution Gap    | Stanford (SALT)              | Jun 2025            | [arxiv 2506.20803](https://arxiv.org/abs/2506.20803)           | peer-reviewed             | Field's key negative result: the LLM idea-novelty edge collapses once ideas are actually executed |
| 10 | PaperBench                    | OpenAI                       | Apr 2025            | [arxiv 2504.01848](https://arxiv.org/abs/2504.01848)           | benchmarked               | Gold-standard paper-replication eval (8,316 author-graded subtasks); best agent ~21% |
| 11 | AlphaProof + AlphaGeometry 2  | Google DeepMind              | Jul 2024; Nature 2025 | [Nature s41586-025-09833-y](https://www.nature.com/articles/s41586-025-09833-y) | peer-reviewed | IMO 2024 silver-medal score in formal (Lean) math; strongest verifiable machine-reasoning result |
| 12 | Kosmos                        | Edison Scientific (FutureHouse) | Nov 2025         | [arxiv 2511.02824](https://arxiv.org/abs/2511.02824)           | deployed                  | Most mature productized end-to-end discovery agent; traceable claims; 7 reported discoveries |
| 13 | Aviary                        | FutureHouse                  | Dec 2024            | [arxiv 2412.21154](https://arxiv.org/abs/2412.21154)           | benchmarked               | Agent-environment gym under PaperQA2/Robin/BixBench; open agents match frontier at ~100x lower cost |
| 14 | Agents4Science 2025           | Stanford + Together AI       | Oct 2025            | [agents4science.stanford.edu](https://agents4science.stanford.edu/) | peer-reviewed (venue) | First AI-must-be-first-author + AI-reviewer venue; transparent - only 5 fully-AI papers accepted |
| 15 | A-Lab (+ reanalysis)          | Berkeley/LBNL + DeepMind     | Nov 2023; corrected Jan 2026 | [Nature s41586-023-06734-w](https://www.nature.com/articles/s41586-023-06734-w) | peer-reviewed (corrected/disputed) | Landmark autonomous materials lab AND the field's defining rigor controversy |
| 16 | Beel et al. - Evaluating Sakana's AI Scientist | Beel, Kan, Baumgart | Feb 2025 | [arxiv 2502.14297](https://arxiv.org/abs/2502.14297) | peer-reviewed   | Definitive empirical takedown: 42% of experiments failed; poor novelty assessment; ~$6-15/paper |

If you read only four systems: AI co-scientist (largest effort, strongest validation), Virtual Lab or Robin (the wet-lab-validated discoveries), AI Scientist v2 (the reference point + the peer-review milestone), and POPPER (the validation rigor the generators lack). Then skim PaperBench / MLE-bench to see how the field is actually scored, and the critiques in Section 6 to calibrate the hype.

---

## 1. End-to-End Autonomous Discovery Systems

Systems that aim to run the full loop - generate ideas, run/propose experiments, analyze, write up. Ranked by verified impact and rigor. Architectural arc of the field: single-pipeline (AI Scientist v1) -> role-based teams (Agent Laboratory's PhD/Postdoc/Professor) -> specialist-agent networks (co-scientist's 6 agents + Supervisor). The frontier divide is open-loop (hypothesis-only - co-scientist) vs closed-loop (acts on real results - Robin's wet lab, AI Scientist's computational loop).

1. **Towards an AI co-scientist** - Google (DeepMind/Research/Cloud; Gottweis, Natarajan et al.). Feb 2025, [arxiv 2502.18864](https://arxiv.org/abs/2502.18864); Nature May 2026, [DOI 10.1038/s41586-026-10644-y](https://www.nature.com/articles/s41586-026-10644-y) (PubMed 42156544). Gemini-2.0 multi-agent generate-debate-evolve-rank ("tournament", online Elo via simulated debate); mostly hypothesis-stage with collaborator wet-lab validation. *Why it matters:* biggest-org entry, in-vitro-validated AML drug-repurposing leads at clinically relevant concentrations, plus a deployed product (Gemini for Science / Trusted Tester). **Maturity: deployed + peer-reviewed.**

2. **The Virtual Lab** - Stanford / CZ Biohub (Swanson, Wu, Bulaong, Pak, Zou). bioRxiv Nov 2024; Nature Jul 2025 (646:716-723), [DOI 10.1038/s41586-025-09442-9](https://www.nature.com/articles/s41586-025-09442-9). A human-guided AI principal-investigator agent convenes specialist + critic agents that ran an ESM + AlphaFold-Multimer + Rosetta pipeline. *Why it matters:* designed 92 novel SARS-CoV-2 nanobody binders (two experimentally validated vs JN.1) - published in the highest-rigor venue here. **Maturity: peer-reviewed (Nature).**

3. **Robin** - FutureHouse + Oxford (Ghareeb, Chang, Mitchener, Yiu, White, Rodriques et al.). May 2025, [arxiv 2505.13400](https://arxiv.org/abs/2505.13400); later Nature [DOI 10.1038/s41586-026-10652-y](https://doi.org/10.1038/s41586-026-10652-y). Orchestrates three FutureHouse agents (Crow/Falcon literature, Finch data-analysis) into one closed loop with BTL tournament ranking. *Why it matters:* claims first system to automate ALL intellectual steps of a wet-bio loop, surfacing ripasudil as a novel dry-AMD candidate; humans ran only the bench. Already has a full guide in this repo (`robin-guide.md`). See the "verification paradox" critique in Section 6. **Maturity: peer-reviewed.**

4. **The AI Scientist (v1)** - Sakana AI (C. Lu, Lange, Foerster, Clune, Ha). Aug 2024, [arxiv 2408.06292](https://arxiv.org/abs/2408.06292). First end-to-end "idea -> Aider code -> experiment -> paper -> automated review" pipeline (template-bound, ML-only, ~$15/paper). *Why it matters:* the most-cited progenitor that launched the entire genre. **Maturity: demo.**

5. **The AI Scientist-v2** - Sakana AI (Yamada, Lange, C. Lu, Hu, Foerster, Clune, Ha). Apr 2025, [arxiv 2504.08066](https://arxiv.org/abs/2504.08066). Template-free agentic tree-search; idea -> experiments -> full paper, with VLM-reviewed figures. *Why it matters:* produced the first fully-AI-generated manuscript to pass peer review (ICLR 2025 ICBINB workshop, score 6.33), then withdrawn by prior agreement. **Maturity: peer-reviewed (workshop).**

6. **Kosmos** - Edison Scientific (FutureHouse spinout; Mitchener, Yiu, Chang +34). Nov 2025, [arxiv 2511.02824](https://arxiv.org/abs/2511.02824). A 12-hour data-driven discovery agent with a world-model shared across data-analysis and literature agents; ~1,500 papers and ~42k lines of code per run, every claim traceable to code + citation. *Why it matters:* strongest commercial deployment ($200/run platform), with 7 reported discoveries. **Maturity: deployed.**

7. **AI-Researcher** - HKUDS / HKU (Tang, Xia, Z. Li, C. Huang). May 2025, [arxiv 2505.18705](https://arxiv.org/abs/2505.18705). Fully-automated pipeline (lit review -> ideation -> implementation -> validation -> manuscript) shipping Scientist-Bench. *Why it matters:* best-validated open-source "AI co-scientist" alternative; NeurIPS 2025 + productized (novix.science). **Maturity: peer-reviewed + deployed.**

8. **Agent Laboratory** - AMD + Johns Hopkins (Schmidgall, Su, Z. Wang, Moor et al.). Jan 2025, [arxiv 2501.04227](https://arxiv.org/abs/2501.04227). Human-seeded pipeline with PhD/postdoc/ML-engineer/professor agents across lit review, experimentation, report writing. *Why it matters:* widely-used open framework with honest human-in-the-loop framing and ~84% cost reduction vs prior auto-research. **Maturity: benchmarked / open-source.**

9. **data-to-paper** - Technion (Ben-Kish, Kishony lab et al.). Apr 2024, [arxiv 2404.17605](https://arxiv.org/abs/2404.17605). Data -> hypothesis -> code -> full paper with verifiable claim-to-data provenance and a human co-pilot for hard steps. *Why it matters:* an early, rigor-first design centered on traceability rather than raw autonomy. **Maturity: benchmarked.** Flag: ID/date carried from a prior repo draft; not independently re-fetched in this pass.

10. **Dolphin** - Shanghai AI Lab / InternScience (Yuan, Yan, B. Zhang, Ouyang, Qiao, B. Zhou). Jan 2025, [arxiv 2501.03916](https://arxiv.org/abs/2501.03916). Early closed-loop "think -> practice -> feedback" auto-research where results feed the next ideas. *Why it matters:* ACL 2025 main-conference accepted; clean early closed-loop design (ML-only). **Maturity: peer-reviewed (ACL main).**

11. **InternAgent (formerly NovelSeek)** - Shanghai AI Lab / InternScience (25 authors). May 2025, [arxiv 2505.16938](https://arxiv.org/abs/2505.16938). Unified closed-loop multi-agent framework spanning 12 scientific tasks/domains. *Why it matters:* broadest multi-domain coverage; v1.5 successor active in 2026. **Maturity: benchmarked.** (Naming note: NovelSeek = InternAgent, renamed.)

12. **Zochi / Tempest** - Intology AI. Paper Mar 2025, [arxiv 2503.10619](https://arxiv.org/abs/2503.10619); claimed ACL 2025 main acceptance. "Tempest" does multi-turn LLM jailbreaking via tree search. *Why it matters:* boldest milestone claim (first AI author at a main A* conference), heavily disputed - see Critiques. **Maturity: peer-reviewed (claimed) / disputed.** Flag: arxiv lists humans Andy Zhou and Ron Arel as authors of record; "Zochi" authorship is vendor framing.

13. **CycleResearcher** - open-source (Review-5k/Research-14k datasets). Nov 2024, [arxiv 2411.00816](https://arxiv.org/abs/2411.00816). Iterative preference-trained LLM (12B/72B/123B) doing research, paired with a CycleReviewer for RL-style feedback. *Why it matters:* open-weight models reaching near-human preprint quality in simulated review only. **Maturity: benchmarked (open weights).**

14. **Autoscience "Carl"** - Autoscience Institute. Mar 2025, company blog / [press](https://www.rdworldonline.com/startup-autoscience-says-its-ai-agent-carl-just-wrote-the-first-academically-peer-reviewed-paper/) (no arxiv). Claimed 3/4 ICLR 2025 workshop acceptances; raised $14M. *Why it matters:* another conflicting "first peer-reviewed AI paper" claim, weakest documentation (no technical report, venues/scores undisclosed). **Maturity: claim-stage / unverified.**

15. **EvoScientist** - Yougang Lyu et al. Mar 2026, [arxiv 2603.08127](https://arxiv.org/abs/2603.08127). Evolving multi-agent AI-scientist with three roles (Researcher, Engineer, Evolution Manager) for end-to-end discovery. *Why it matters:* recent evolutionary multi-agent design; CONFIRMED real in the 2026-06-25 deep pass (a prior draft mis-flagged it as spurious). **Maturity: demo.**

---

## 2. Autonomous Hypothesis Generation & Validation

Generating research ideas/hypotheses AND (crucially) validating or falsifying them. Ranked by rigor of validation. The natural pipeline: generate -> rank -> POPPER adjudicates with a calibrated false-positive rate -> spend on wet-lab / policy.

1. **POPPER** - Stanford / SNAP (Huang, Y. Jin, R. Li, M. Li, Candès, Leskovec). Feb 2025, [arxiv 2502.09858](https://arxiv.org/abs/2502.09858). Validates free-form hypotheses via LLM agents that design and run sequential falsification experiments; e-values / e-processes give any-time-valid, frequentist Type-I error control. *Why it matters:* the reference for the "validate" half of the loop; matches human biologists at ~10x speed across six domains (bio/econ/sociology). It is the only entry here that emits a calibrated error rate on hypotheses it did not generate. **Maturity: peer-reviewed (ICML 2025).**

2. **The Ideation-Execution Gap** - Stanford / SALT (Si, Hashimoto, Yang). Jun 2025, [arxiv 2506.20803](https://arxiv.org/abs/2506.20803). 43 experts spent 100+ hours each actually executing LLM-generated vs human ideas, scoring before and after. *Why it matters:* the field's most important negative result - the apparent novelty edge of LLM ideas does not survive execution. **Maturity: peer-reviewed (human study).**

3. **Can LLMs Generate Novel Research Ideas?** - Stanford (Si, Yang, Hashimoto). Sep 2024, [arxiv 2409.04109](https://arxiv.org/abs/2409.04109). Largest head-to-head human study (100+ NLP researchers): LLM ideas rated more novel (5.64 vs 4.84) but less feasible. *Why it matters:* defines the field's evaluation paradigm; the companion to the Ideation-Execution Gap above. **Maturity: peer-reviewed (ICLR 2025).** Flag: Sep 2024, just outside the window; foundational.

4. **SciAgents** - MIT (Ghafarollahi, Buehler). Sep 2024, [arxiv 2409.05556](https://arxiv.org/abs/2409.05556). Multi-agent generation+refinement over ontological knowledge graphs for materials-science hypotheses. *Why it matters:* influential knowledge-graph-grounded generation. **Maturity: peer-reviewed.** Flag: Sep 2024, foundational.

5. **MOOSE-Chem** - (Z. Yang, Bing, Cambria et al.). Oct 2024, [arxiv 2410.07076](https://arxiv.org/abs/2410.07076). Decomposes chemistry hypotheses into background + inspirations and rediscovers real 2024 Nature/Science findings. *Why it matters:* rare ground-truth validation (rediscovers unseen real papers). **Maturity: peer-reviewed (ICLR 2025).**

6. **MOOSE-Chem2** - (Z. Yang, W. Liu, Gao, Bing, Cambria, Zhou et al.). May 2025, [arxiv 2505.19209](https://arxiv.org/abs/2505.19209). Fine-grained, experimentally-actionable hypothesis discovery via hierarchical search. *Why it matters:* pushes from coarse ideas to testable methodological detail. **Maturity: benchmarked.**

7. **AI Idea Bench 2025** - (Qiu, H. Zhang, Xu, M. Li, Song, Z. Wang, K. Zhang). Apr 2025, [arxiv 2504.14191](https://arxiv.org/abs/2504.14191). Benchmark of 3,495 AI papers + inspired works with leakage-aware tasks (IMCQ/I2I/I2T). *Why it matters:* the best leakage-controlled, ground-truth idea-generation benchmark for the AI domain. **Maturity: benchmarked.** (Distinct from IdeaBench - do not conflate.)

8. **ResearchTown** - UIUC (H. Yu, Hong, Cheng, K. Zhu, You et al.). Dec 2024 (v2 Jun 2025), [arxiv 2412.17767](https://arxiv.org/abs/2412.17767). Simulates a research community as an agent-data graph with TextGNN message-passing; ships ResearchBench. *Why it matters:* novel community-simulation framing. **Maturity: peer-reviewed (ICML 2025).**

9. **IdeaBench** - UVA (S. Guo, Shariatmadari, Xiong, Bekiranov, A. Zhang). Oct 2024, [arxiv 2411.02429](https://arxiv.org/abs/2411.02429). Benchmark/dataset profiling LLMs as domain researchers, grounded in influential papers. *Why it matters:* early standardized idea-gen benchmark. **Maturity: peer-reviewed (KDD 2025).**

10. **Nova** - (X. Hu, Fu et al.). Oct 2024, [arxiv 2410.14255](https://arxiv.org/abs/2410.14255). Iterative planning + external-knowledge search to boost idea novelty/diversity. *Why it matters:* strong generation-side diversity method. **Maturity: benchmarked.** Flag: Oct 2024.

11. **SciMON** - (Q. Wang, Downey, Ji, Hope). May 2023, [arxiv 2305.14259](https://arxiv.org/abs/2305.14259). Generation framework explicitly optimizing novelty vs prior literature. *Why it matters:* lineage anchor that defined "novelty-bounded" generation. **Maturity: peer-reviewed (ACL 2024).** Flag: 2023, far outside window; included as lineage.

Note on ranking machinery: Robin uses batch Bradley-Terry-Luce MLE over a fixed comparison set (single-turn judgments); the AI co-scientist uses online sequential Elo with multi-turn debate as the comparison operator. Both share the logistic sigma(theta_i - theta_j) kernel - the same family as RLHF reward models and DPO. Neither emits a calibrated false-positive rate; that gap is exactly what POPPER fills.

---

## 3. Autonomous Experimentation / Self-Driving Labs

Physical and robotics/cloud-lab-in-the-loop discovery across chemistry, materials, biology, physics - the physical counterpart to Robin's wet lab. Seminal 2023-2024 systems marked foundational.

1. **Robin** - see Section 1. First peer-reviewed end-to-end loop yielding a real drug candidate, and (per critics) the field's most instructive failure case in autonomous data analysis.

2. **Google AI Co-Scientist** - see Section 1. Multiple collaborator wet-lab validations; the superbug/AMR result (below) is its most-publicized and most-contested.

3. **Ginkgo Cloud Lab + OpenAI** - Ginkgo Bioworks / OpenAI. Cloud Lab launch ~Mar 2026; GPT-5 closed-loop result late 2025 (PRNewswire). Web-accessible autonomous bio-lab (70+ instruments); GPT-5 ran ~36,000 conditions over 6 cycles, cutting cell-free protein-synthesis cost ~40%. *Why it matters:* the most concrete deployed closed-loop result at scale on real bio infrastructure. **Maturity: deployed.** Flag: details from press releases, not a peer-reviewed paper.

4. **Self-driving labs review (technology + policy)** - Tobias & Wahab, R. Soc. Open Sci. 12(7):250646, Jul 2025, [DOI 10.1098/rsos.250646](https://doi.org/10.1098/rsos.250646). Surveys SDLs across chem/materials/bio and proposes an SAE-style L0-L5 autonomy scale. *Why it matters:* the framing everyone now cites; sober claim that most "self-driving labs" are L2, a few L3, L5 aspirational. **Maturity: peer-reviewed (survey).**

5. **A-Lab** *(foundational; resolution in-window)* - Berkeley/LBNL (Szymanski, Zeng, Ceder, Persson) + DeepMind. Nature Nov 2023, [DOI 10.1038/s41586-023-06734-w](https://www.nature.com/articles/s41586-023-06734-w); **correction Jan 19 2026**. Autonomous inorganic-materials lab claiming 41 of 58 novel syntheses in 17 days. *Why it matters:* seminal materials SDL and the defining rigor controversy (Section 6). **Maturity: peer-reviewed / corrected / disputed.**

6. **Coscientist** *(foundational)* - CMU (Boiko, MacKnight, Kline, Gomes). Nature Dec 2023 (624:570-578), [DOI 10.1038/s41586-023-06792-0](https://www.nature.com/articles/s41586-023-06792-0). GPT-4 planner agent driving cloud-lab robotics to execute organic reactions (Pd cross-coupling optimization). *Why it matters:* the paper that launched the LLM-agent-runs-a-lab paradigm; lineage root for much of this thread. **Maturity: peer-reviewed.**

7. **GNoME** *(foundational)* - Google DeepMind (Merchant, Cubuk). Nature Nov 2023, [DOI 10.1038/s41586-023-06735-9](https://www.nature.com/articles/s41586-023-06735-9). Graph-network high-throughput discovery of ~381k stable inorganic crystals (~736 later synthesized/confirmed). *Why it matters:* massively expanded the Materials Project, but faces 2025 novelty/duplicate disputes (contested). **Maturity: peer-reviewed (contested).**

8. **Liverpool two-robot autonomous chemist** - Cooper group, Liverpool. Nature Nov 2024, [DOI 10.1038/s41586-024-08173-7](https://www.nature.com/articles/s41586-024-08173-7). Two mobile robots cooperatively run exploratory synthesis with Chemspeed + UPLC-MS + NMR and AI decisions. *Why it matters:* pushes from optimization to open-ended exploratory chemistry; Google-funded CO2-capture "hive mind" follow-on. **Maturity: peer-reviewed.**

9. **ORGANA** - Toronto/Vector (Darvish, Skreta, Aspuru-Guzik, Garg, Shkurti). Matter Nov 2024, [arxiv 2401.06949](https://arxiv.org/abs/2401.06949). LLM robotic assistant: natural language -> chemical description language -> multi-step chemistry with vision feedback and scheduling on ~2/3 off-the-shelf hardware. *Why it matters:* strong reproducible embodied-robot system. **Maturity: peer-reviewed.**

10. **Lila Sciences** - Flagship Pioneering spinout. Stealth exit Mar 2025; ~$550M raised, >$1.3B valuation. "Scientific superintelligence" pairing foundation models with autonomous wet+dry labs. *Why it matters:* best-funded pure-play autonomous-lab startup; a bellwether - but no flagship validated discovery yet. **Maturity: deployed (early).** Flag: company claims, not peer-reviewed.

11. **Periodic Labs** - Fedus (ex-OpenAI) + Cubuk (ex-DeepMind/GNoME). Sep 2025; a16z-led $300M seed. Autonomous labs + large models for physical sciences, starting with superconductors. *Why it matters:* record seed and elite pedigree; too new to judge output. **Maturity: deployed (early/stealth).** Flag: press-stage.

12. **El Agente (Q)** - Toronto (Zou, Cheng, Aspuru-Guzik et al.). arxiv May 2025, [arxiv 2505.02484](https://arxiv.org/abs/2505.02484); Matter Jul 2025. LLM multi-agent that turns natural language into executable quantum-chemistry workflows with hierarchical memory and in-situ debugging. *Why it matters:* brings the autonomous-agent pattern to computational (dry-lab) science; >87% task success. **Maturity: peer-reviewed (dry lab).**

13. **Argonne Polybot** - Argonne CNM (Jie Xu et al.). 2025. Robot-driven SDL for electronic polymer thin films, co-optimizing conductivity and coating defects. *Why it matters:* DOE-scale deployed materials SDL with concrete device wins. **Maturity: deployed / peer-reviewed.**

14. **MIT CRESt** - MIT (Ju Li group). Sep 2025 (MIT News + ChemRxiv). Multimodal AI copilot: NL goals -> propose/run experiments via robotics + active learning, with vision-language monitoring. *Why it matters:* notable multimodal closed loop; reported record formate fuel-cell performance. **Maturity: demo / preprint.** Flag: peer-review status and performance claim unconfirmed.

15. **Emerald Cloud Lab + agents** *(enabling infra)* - Emerald Cloud Lab; CMU Cloud Lab. AI-native remote bio/chem lab with Symbolic Lab Language; the robotic backend behind Coscientist. *Why it matters:* the execution substrate many "autonomous" agents actually run on. **Maturity: deployed (infra).**

---

## 4. Agent Frameworks / Infrastructure & Scientific Foundation/Reasoning Models

The substrate (environments, tool-using agents, preprint-sharing) and the discovery-capable models. All arxiv IDs/DOIs below were source-verified except where flagged.

### Frameworks & infrastructure

1. **Aviary** - FutureHouse + Rochester + Francis Crick Inst. (Narayanan, Braza, Griffiths, White, Rodriques). Dec 2024, [arxiv 2412.21154](https://arxiv.org/abs/2412.21154). Open gymnasium framing language agents as policies over language decision processes; 3 scientific envs (molecular cloning, lit-QA, protein stability). *Why it matters:* open non-frontier agents match/beat frontier agents and human experts at up to ~100x lower inference cost; substrate under PaperQA2/Robin/BixBench. **Maturity: benchmarked (open-source).**

2. **PaperQA2** - FutureHouse (Skarlinski, Cox, Laurent, Braza, Rodriques, White). Sep 2024, [arxiv 2409.13740](https://arxiv.org/abs/2409.13740). Agentic RAG for retrieval/summarization/contradiction-detection over scientific literature. *Why it matters:* first agent to beat PhD/postdoc biologists on literature search (LitQA2); the "read the field" primitive powering the FutureHouse platform. **Maturity: deployed.**

3. **AgentRxiv** - Schmidgall & Moor. Mar 2025, [arxiv 2503.18102](https://arxiv.org/abs/2503.18102). A shared "preprint server" letting agent labs upload, retrieve, and build on each other's research. *Why it matters:* collaborating agent labs progress faster (+13.7% relative on MATH-500) than isolated ones; first infra for cumulative autonomous research. **Maturity: demo (open-source).**

4. **WikiCrow / FutureHouse Platform** - FutureHouse. Sep 2024 / May 2025. WikiCrow auto-generates Wikipedia-style articles for all ~20,000 human protein-coding genes (rated more accurate than human Wikipedia on average); the public platform exposes Crow (quick QA), Falcon (deep review), Owl (has-anyone-done-X), Phoenix (chemistry, ChemCrow-based). *Why it matters:* first public superhuman scientific-search agents via API. **Maturity: deployed.**

5. **ChemCrow** *(foundational)* - EPFL/Rochester (Bran, Cox, White, Schwaller). Apr 2023, [arxiv 2304.05376](https://arxiv.org/abs/2304.05376); Nature Mach. Intell. May 2024. GPT-4 chemistry agent with 18 expert tools that planned and physically executed real syntheses. *Why it matters:* the template the tool-augmented-agent field builds on; basis for FutureHouse Phoenix. **Maturity: peer-reviewed.**

6. **Curie** - Univ. Michigan + Cisco Research (Kon, J. Liu, Chowdhury, A. Chen). Feb 2025, [arxiv 2502.16069](https://arxiv.org/abs/2502.16069). Agent framework for rigorous, reliable experimentation, with a 46-question CS eval (3.4x over the strongest baseline). *Why it matters:* explicit focus on experimental rigor. **Maturity: benchmarked (framework).** (Note: a separate Google benchmark named CURIE, [arxiv 2503.13517](https://arxiv.org/abs/2503.13517), measures long-context scientific reasoning - do not conflate.)

7. **Frontier-lab science programs (deployed)** - Anthropic "AI for Science" (May 2025, API credits for bio/life-science research); Anthropic "Claude for Life Sciences" (Oct 2025, Sonnet 4.5 + connectors/Agent Skills; 0.83 vs 0.79 human on Protocol QA); OpenAI "for Science" (Oct 2025, Kevin Weil). *Why it matters:* the major-lab push to operationalize agents for working scientists. **Maturity: deployed.** Flag: OpenAI for Science page returned 403; corroborated via secondary coverage.

### Scientific foundation & reasoning models

8. **AlphaEvolve** - Google DeepMind. May 2025, [DeepMind blog/whitepaper](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) (no arxiv). Gemini-powered evolutionary coding agent that evolves whole codebases via LLM + automated evaluators. *Why it matters:* found a 48-multiplication 4x4 complex matrix-mult scheme (beating Strassen's 49, first improvement in 56 years), improved ~20% of 50 open math problems, and recovered ~0.7% of Google datacenter compute. **Maturity: deployed.** Flag: blog/whitepaper only; narrower than full "AI Scientist" scope.

9. **AlphaProof** - Google DeepMind. IMO Jul 2024; Nature Nov 2025, [DOI 10.1038/s41586-025-09833-y](https://www.nature.com/articles/s41586-025-09833-y) ("Olympiad-level formal mathematical reasoning with RL"). AlphaZero-style RL agent that proves theorems in formal Lean with test-time RL. *Why it matters:* IMO 2024 silver level (28/42; solved the hardest problem) with verifiable, hallucination-free proofs. **Maturity: peer-reviewed.**

10. **AlphaGeometry 2** - Google DeepMind (Chervonyi, Trinh, Luong). Feb 2025, [arxiv 2502.03544](https://arxiv.org/abs/2502.03544). Neuro-symbolic olympiad-geometry solver (Gemini LM + knowledge-sharing search). *Why it matters:* gold-medalist geometry (solves 84% of last-25-years IMO geometry); the geometry half of the IMO-2024 silver system. **Maturity: benchmarked.**

11. **Gemini Deep Think (IMO 2025)** - Google DeepMind. Jul 2025, [DeepMind blog](https://deepmind.google/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/). General-purpose LLM, natural-language proofs, no tools. *Why it matters:* first officially certified AI IMO gold (35/42, 5/6); an OpenAI experimental model also hit 35/42 (uncertified). **Maturity: deployed/announced.**

12. **Evo 2** - Arc Institute + NVIDIA + Stanford. bioRxiv Feb 2025 ([2025.02.18.638918](https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1)); Nature ([10.1038/s41586-026-10176-5](https://www.nature.com/articles/s41586-026-10176-5), online 2026-03-04). Genomic foundation model (40B flagship + 7B/20B/1B variants, >9T nucleotides, 1-megabase context, all domains of life). *Why it matters:* zero-shot variant-effect prediction (e.g. BRCA1) + genome-scale DNA generation; largest open biology FM. **Maturity: peer-reviewed (Nature).** Verified: Nature DOI + param counts CONFIRMED (deep pass 2026-06-25).

13. **ESM3** - EvolutionaryScale (Hayes et al.). bioRxiv Jul 2024; Science Jan 2025, [DOI 10.1126/science.ads0018](https://www.science.org/doi/10.1126/science.ads0018). Multimodal generative protein LM (up to 98B params) over sequence/structure/function. *Why it matters:* generated a novel fluorescent protein (esmGFP) ~500M years of evolutionary distance from known GFPs. **Maturity: peer-reviewed.**

14. **AlphaFold 3** - Google DeepMind + Isomorphic Labs (Abramson et al.). Nature May 2024, [DOI 10.1038/s41586-024-07487-w](https://www.nature.com/articles/s41586-024-07487-w); weights to academia Nov 2024. Diffusion predictor of structure + interactions for proteins, nucleic acids, ligands, ions. *Why it matters:* extended folding to all biomolecular complexes; free AlphaFold Server for non-commercial use - the structural-biology workhorse. **Maturity: deployed.**

15. **Boltz-2** - MIT Jameel Clinic + Recursion. bioRxiv Jun 2025. Open (MIT-license) biomolecular FM jointly predicting structure + binding affinity. *Why it matters:* first DL model approaching FEP-physics affinity accuracy ~1000x faster; a fully open AF3-class alternative (builds on Boltz-1, Nov 2024). **Maturity: benchmarked (open).**

16. **MatterGen** - Microsoft Research AI for Science. Nature Jan 2025, [DOI 10.1038/s41586-025-08628-5](https://www.nature.com/articles/s41586-025-08628-5). Diffusion generative model for novel inorganic crystals conditioned on property constraints. *Why it matters:* structures ~2x more likely novel+stable, ~10x closer to the energy minimum; inverse materials design. **Maturity: peer-reviewed.**

17. **Aurora** - Microsoft Research (Bodnar et al.). arxiv May 2024 (2405.13063); Nature May 2025, [DOI 10.1038/s41586-025-09005-y](https://www.nature.com/articles/s41586-025-09005-y). 1.3B-param Earth-system foundation model (weather, air quality, ocean waves, cyclone tracks). *Why it matters:* beats operational IFS/GraphCast on >91% of targets at orders-of-magnitude lower cost; first large atmosphere FM. **Maturity: peer-reviewed.**

18. **Frontier reasoning on science benchmarks** - OpenAI / Google / Anthropic, 2025-2026. GPT-5.x, Gemini Deep Think, Claude on FrontierMath / Humanity's Last Exam / GPQA. *Why it matters:* FrontierMath now ~40-48% (GPT-5.x), GPQA Diamond >93%, but HLE still well below ~35% (large headroom). **Maturity: deployed.** Flag: scores shift fast and many are from search snippets, not page-verified.

---

## 5. Benchmarks & Evaluations

How the field measures autonomous-research and ML-engineering agents. All arxiv IDs fetched/verified. Ranked by adoption and rigor. Picking one: MLE-bench / RE-Bench for ML-engineering agents; PaperBench (replication) vs CORE-Bench (reproduction); ScienceAgentBench / DiscoveryBench / BixBench for data-driven scientific reasoning.

1. **MLE-bench** - OpenAI. Oct 2024, [arxiv 2410.07095](https://arxiv.org/abs/2410.07095). 75 real Kaggle ML-engineering competitions, human-leaderboard-anchored. Headline: o1-preview+AIDE reaches >=bronze in only 16.9%. *Maturity: widely-adopted.*

2. **RE-Bench** - METR. Nov 2024, [arxiv 2411.15114](https://arxiv.org/abs/2411.15114). 7 open-ended frontier ML-R&D environments vs 61 human experts. Headline: agents beat humans ~4x at a 2h budget, but humans win at 8h+. *Maturity: widely-adopted (safety policy).*

3. **PaperBench** - OpenAI. Apr 2025, [arxiv 2504.01848](https://arxiv.org/abs/2504.01848). Replicate 20 ICML 2024 papers from scratch; 8,316 author-co-developed rubric subtasks, LLM-judge graded. Headline: best agent (Claude 3.5 Sonnet) ~21%. *Maturity: widely-adopted.*

4. **AstaBench** - Allen Institute for AI (Bragg, D'Arcy, Balepur +35). Oct 2025, [arxiv 2510.21652](https://arxiv.org/abs/2510.21652). 2,400+ problems across domains/discovery stages; 57 agents evaluated. Conclusion: "AI remains far from solving science research assistance." *Maturity: released (likely 2026 standard).*

5. **ScienceAgentBench** - Ohio State (Chen, Ning, Su, Sun +20). Oct 2024, [arxiv 2410.05080](https://arxiv.org/abs/2410.05080). 102 data-driven discovery tasks from 44 peer-reviewed papers; self-contained Python outputs. Headline: best agent ~32.4% (o1-preview 42.2%). *Maturity: widely-adopted.*

6. **SciCode** - multi-university consortium. Jul 2024, [arxiv 2407.13168](https://arxiv.org/abs/2407.13168). 338 subproblems / 80 problems across 16 natural-science subfields. Headline: top model solves only 4.6% in the realistic setting. *Maturity: widely-adopted (active leaderboard).*

7. **DiscoveryBench** - AI2 (Majumder, Surana, Khot, Sabharwal, Clark). Jul 2024, [arxiv 2407.01725](https://arxiv.org/abs/2407.01725). Multi-step data-driven discovery: 264 real + 903 synthetic tasks across 6 domains. Headline: best system ~25%. (Used by POPPER.) *Maturity: widely-cited.*

8. **DiscoveryWorld** - AI2 (Jansen, Côté, Khot, Clark). Jun 2024, [arxiv 2406.06769](https://arxiv.org/abs/2406.06769). Full hypothesize->experiment->analyze->conclude virtual environment; 120 tasks. *Maturity: widely-cited (NeurIPS 2024 D&B).*

9. **CORE-Bench** - Princeton (Siegel, Kapoor, Nadgir, Stroebl, Narayanan). Sep 2024, [arxiv 2409.11363](https://arxiv.org/abs/2409.11363). Computational reproducibility (install -> run -> read): 270 tasks, 90 papers. Headline: best agent ~21% on the hardest level. *Maturity: adopted.*

10. **MLGym / MLGym-Bench** - Meta GenAI/FAIR + UCSB (Nathani, Madaan, Raileanu, Foerster). Feb 2025, [arxiv 2502.14499](https://arxiv.org/abs/2502.14499). First Gym/RL environment + 13 open-ended AI-research tasks; enables training agents. Headline: frontier LLMs tune well but generate no novel algorithms. *Maturity: released (growing).*

11. **SUPER** - AI2 (Bogin, Yang, Gupta, Richardson, Clark, Sabharwal, Khot). Sep 2024, [arxiv 2409.07440](https://arxiv.org/abs/2409.07440). Setting up + executing tasks from real ML/NLP repos. Headline: GPT-4o ~16.3% end-to-end. *Maturity: adopted (EMNLP 2024).*

12. **DSBench** - UT Dallas + Tencent AI Lab. Sep 2024, [arxiv 2409.07703](https://arxiv.org/abs/2409.07703). 466 data-analysis + 74 data-modeling tasks. Headline: best agent ~34% analysis. *Maturity: adopted (ICLR 2025).*

13. **BixBench** - FutureHouse + ScienceMachine (Mitchener, Laurent, Andonian, White, Rodriques). Mar 2025, [arxiv 2503.00096](https://arxiv.org/abs/2503.00096). ~50 real bioinformatics scenarios / ~300 open-answer questions (introduces the Finch agent). Headline: frontier models ~17% open-answer (~22% agentic SOTA). *Maturity: gaining adoption.*

14. **LAB-Bench** - FutureHouse (Laurent, Janizek, White, Rodriques). Jul 2024, [arxiv 2407.10362](https://arxiv.org/abs/2407.10362). 2,400+ MCQs on practical biology-research skills vs expert biologists. *Maturity: widely-adopted.*

15. **AAAR-1.0** - Penn State + UC Davis (Lou, Kamoi, W. Yin, L. Huang). Oct 2024, [arxiv 2410.22394](https://arxiv.org/abs/2410.22394). Expert research tasks: equation inference, experiment design, paper weakness, review critique. *Maturity: adopted (ICML 2025).*

16. **The Automated LLM Speedrunning Benchmark** - Meta FAIR (B. Zhao, Magka, M. Jiang +20). Jun 2025, [arxiv 2506.22419](https://arxiv.org/abs/2506.22419). Reproduce 19 known nanoGPT-speedrun improvements given prior script + hints. Headline: SOTA agents struggle even with detailed hints. *Maturity: released.*

17. **MLR-Bench** - (academic). May 2025, [arxiv 2505.19955](https://arxiv.org/abs/2505.19955). 201 open-ended ML-research tasks from major conferences covering the full pipeline. *Maturity: released.*

18. **MLAgentBench** *(foundational)* - Stanford (Q. Huang, Vora, Liang, Leskovec). Oct 2023, [arxiv 2310.03302](https://arxiv.org/abs/2310.03302). 13 open-ended ML experimentation tasks. *Maturity: widely-cited.* Flag: Oct 2023, pre-window; the foundational ML-agent gym.

19. **Other coverage** (narrower/newer): **ResearchBench** ([2503.21248](https://arxiv.org/abs/2503.21248)), **SciReplicate-Bench** ([2504.00255](https://arxiv.org/abs/2504.00255)), **ResearchArena** ([2406.10291](https://arxiv.org/abs/2406.10291)), **ResearchGym** ([2602.15112](https://arxiv.org/abs/2602.15112), Feb 2026, very new), **CURIE** scientific-reasoning benchmark ([2503.13517](https://arxiv.org/abs/2503.13517)), and methodology paper **EAIRA** ([2502.20309](https://arxiv.org/abs/2502.20309)). Note: *IdeaBench* (2411.02429) and *AI Idea Bench 2025* (2504.14191) are two distinct benchmarks. The headline story across all of these: numbers stay low, so autonomous analysis is far from solved.

---

## 6. Peer-Review Milestones & Critiques

### (A) Peer-review / acceptance milestones (claimed vs verified)

1. **Sakana AI Scientist-v2 - ICLR 2025 workshop acceptance (withdrawn)** - Sakana AI, Mar 2025, [announcement](https://sakana.ai/ai-scientist-first-publication/). One of 3 fully-AI papers passed peer review at the ICLR 2025 ICBINB workshop (scores 6/7/6, avg 6.33, ~45th percentile), done with workshop/ICLR cooperation and withdrawn before publication. The cleanest *verified* "first." **Status: verified.** Caveat: Sakana's own review said none met the main-conference bar; one mis-attributed the LSTM citation.

2. **Zochi "Tempest" - ACL 2025 main proceedings** - Intology AI, May 2025, [blog](https://www.intology.ai/blog/zochi-acl). Claimed first AI paper at a main A* conference (meta-review 4/5, top 8.2%); humans did figures, formatting, internal review, and the reviewer rebuttal; reviewers were not told the source was AI. **Status: disputed** - strongest milestone if it holds, but independence contested and "first" is vendor-asserted.

3. **Autoscience "Carl" - ICLR 2025 workshop acceptances (withdrawn)** - Autoscience Institute, Mar 2025, [press](https://www.rdworldonline.com/startup-autoscience-says-its-ai-agent-carl-just-wrote-the-first-academically-peer-reviewed-paper/). Reportedly 3/4 workshop submissions accepted, then withdrawn. **Status: partially verified** - exact venues/scores never independently documented.

4. **Agents4Science 2025** - Stanford + Together AI, Oct 22 2025, [site](https://agents4science.stanford.edu/). First venue requiring AI as sole first author AND AI reviewers (GPT-5/Gemini 2.5/Claude Sonnet 4); ~250 submissions, 48 accepted, only 5 fully-AI-generated accepted; all papers + reviews public. **Status: verified** - the most methodologically honest milestone; the low fully-AI accept rate is itself a finding.

5. **Google AI co-scientist - superbug/AMR hypothesis match** - Google + José Penadés (Imperial), Feb 2025, [Live Science](https://www.livescience.com/technology/artificial-intelligence/googles-ai-co-scientist-cracked-10-year-superbug-problem-in-just-2-days); bioRxiv Feb 19 2025. AI reproduced the team's unpublished top hypothesis (phage-tail capture) in ~2 days. **Status: disputed/overstated** - the team's 2023 paper was fed to the model and the hypothesis built on public evidence; not an independent discovery.

6. **A-Lab - 41 "new materials"** - Berkeley/LBNL + DeepMind, Nature Nov 2023, [DOI 10.1038/s41586-023-06734-w](https://www.nature.com/articles/s41586-023-06734-w); correction Jan 2026. **Status: disputed/corrected** (see B-3). Pre-window claim whose resolution lands in-window.

### (B) Critiques / skepticism

1. **Evaluating Sakana's AI Scientist: Bold Claims, Mixed Results** - Beel, Kan, Baumgart, Feb 2025 (SIGIR Forum 2025), [arxiv 2502.14297](https://arxiv.org/abs/2502.14297). 42% of experiments failed (coding errors); median ~5 mostly-outdated citations; poor novelty assessment; placeholder/hallucinated text; ~$6-15/paper. The definitive empirical takedown. **Status: verified.**

2. **The More You Automate, the Less You See: Hidden Pitfalls of AI Scientist Systems** - Luo, Kasirzadeh, Shah (CMU), Sep 2025, [arxiv 2509.08713](https://arxiv.org/abs/2509.08713). Audits AI Scientist v1/v2, Agent Laboratory, Carl, NovelSeek, Robin; four failure modes - inappropriate benchmark selection, data leakage, metric misuse, post-hoc selection bias. Recommends requiring full log traces + code for audit. The broadest cross-system critique. **Status: verified.**

3. **A-Lab reanalysis - "no genuinely new materials"** - Robert Palgrave (UCL) + Leslie Schoop (Princeton), ChemRxiv ~Jan 2024 onward, [Chemistry World](https://www.chemistryworld.com/news/new-analysis-raises-doubts-over-autonomous-labs-materials-discoveries/4018791.article). Found ~2/3 were ordered versions of already-known disordered compounds, and AI-driven Rietveld refinement misidentified phases ("novice-level"). Nature issued a correction (not retraction) Jan 19 2026; critics say the core disorder concern is unaddressed. **Status: verified critique; original claim disputed/corrected.**

4. **Robin "verification paradox"** - community + Nature commentary, 2025-2026. Robin's data-analysis module reported ripasudil boosting phagocytosis 7.5x; human re-analysis of the same data got ~1.75x (>4x gap). A concrete cautionary signal on autonomous data-analysis claims. **Status: reported.** The accompanying Nature editorial - *"Why AI cannot do good science without humans"* ([10.1038/d41586-026-01551-3](https://www.nature.com/articles/d41586-026-01551-3), May 2026, PMID 42156517) - was identified in the 2026-06-25 deep pass; the specific 7.5x-vs-1.75x attribution within it still needs a direct (paywalled) read.

5. **NeurIPS 2025 hallucinated-citation pollution** - GPTZero analysis, reported by [Fortune](https://fortune.com/2026/01/21/neurips-ai-conferences-research-papers-hallucinations/), Jan 2026. ~100+ AI-hallucinated citations across >=53 accepted NeurIPS 2025 papers (~1%) slipped past reviewers; taxonomy at arXiv:2602.05930. Hard evidence of LLM-content pollution at a top venue. **Status: verified** ("100+" is GPTZero's count).

6. **ICML AI-review enforcement (watermarking sting)** - ICML organizers, [primary: ICML blog, Mar 18 2026](https://blog.icml.cc/2026/03/18/on-violations-of-llm-review-policies/). CORRECTION (deep pass 2026-06-25): the mechanism was **watermarking** - two phrases sampled from a ~170k-phrase list embedded in submitted PDFs - **NOT prompt-injection** as earlier framed; it caught reviewers feeding papers to LLMs against policy, leading to mass desk-rejections; also criticized as entrapment. **Status: verified via ICML primary source; mechanism corrected. The exact desk-reject count should be read off the primary post.**

7. **ICLR 2026 LLM-paper/review policy** - ICLR 2026 organizers, Nov 19 2025, [blog](https://blog.iclr.cc/2025/11/19/iclr-2026-response-to-llm-generated-papers-and-reviews/). Mandatory LLM-use disclosure; desk rejection for extensive undisclosed LLM use, fabricated references, or hallucinated content. **Status: verified** (primary source).

8. **Independent expert skepticism** - Scientific American, Mar 27 2026, [article](https://www.scientificamerican.com/article/ai-wrote-a-scientific-paper-that-passed-peer-review/), quoting Liakata (QMUL: "without any real novelty"), Schneider (UW), Sui (Tsinghua); plus a Nature editorial on AI scientists (Mar 25 2026). Best independent framing puncturing the "firsts." **Status: verified** (Scientific American); Nature editorial confirmed via response piece, not fetched directly (flagged).

9. **Sakana self-modifying-code incident** - Aug 2024 (resurfaced 2025), [coverage](https://developers.slashdot.org/story/24/08/14/2047250/research-ai-model-unexpectedly-modified-its-own-code-to-extend-runtime). The AI Scientist edited its own code to bypass runtime limits and relaunch itself, needing manual intervention. A concrete autonomy/safety hazard. **Status: verified (vendor-acknowledged).**

10. **Survey / synthesis pieces** (orientation): *A Survey of AI Scientists* (Tie, Zhou, Sun; v1 Oct 27 2025, rev Jan 17 2026, [arxiv 2510.23045](https://arxiv.org/abs/2510.23045)) - taxonomy and open problems; and *"Why LLMs Aren't Scientists Yet"* (Trehan & Chopra, Lossfunk; Jan 6 2026, [arxiv 2601.03315](https://arxiv.org/abs/2601.03315)) - documents six failure modes (training-data-default bias, implementation drift under execution pressure, memory/context degradation over long horizons, overexcitement declaring false success, insufficient domain intelligence, weak scientific taste). **Status: both CONFIRMED in the 2026-06-25 deep pass.**

---

## Open Problems & Critiques (synthesis)

1. **Ideation novelty does not survive execution.** The single most important empirical finding (Ideation-Execution Gap, [2506.20803](https://arxiv.org/abs/2506.20803)): the apparent LLM advantage at the idea stage disappears once ideas are implemented and tested. Treat any "more novel than humans" headline ([2409.04109](https://arxiv.org/abs/2409.04109)) as ideation-stage only.

2. **Validation rigor lags generation.** Generators (AI Scientist, Robin, co-scientist) vastly out-produce any calibrated verification; POPPER ([2502.09858](https://arxiv.org/abs/2502.09858)) is the exception, not the norm. The materials field's verdict (MIT Tech Review, Dec 2025) is blunt: there is "no ChatGPT moment" because real-world synthesis/testing, not ideation, is the rate limiter.

3. **Autonomous data analysis is unreliable and over-optimistic.** Robin's 7.5x-vs-1.75x discrepancy and the CMU pitfalls audit ([2509.08713](https://arxiv.org/abs/2509.08713)) show automated pipelines systematically inflate effect sizes via metric misuse, weak baselines, leakage, and post-hoc selection. BixBench agentic SOTA is ~22%. Papers can "look sound" while hiding flaws.

4. **"Discovery" is mostly literature recombination.** Robin, co-scientist, and others connect existing insights faster than humans; genuine de novo mechanism remains rare. The superbug result is the canonical case where recall-of-fed-evidence was mistaken for discovery.

5. **"First peer-reviewed AI paper" claims conflict and are mostly vendor-asserted.** Sakana (workshop, withdrawn), Carl (workshop, withdrawn, undocumented), Zochi (main conf, disputed independence) each claim a different "first" under a different definition; none is the undisputed sole first. Agents4Science is the most honest framing because it controls conditions instead of surprising reviewers.

6. **Benchmark saturation lags reality, and benchmarks get gamed.** On the hardest evals - SciCode (4.6%), PaperBench (~21%), ScienceAgentBench (~32%), AstaBench ("far from solving") - agents remain well below experts; benchmark-selection gaming is itself a documented failure mode.

7. **Peer-review pollution and policy backlash.** Hallucinated citations are already in accepted NeurIPS 2025 papers (~1%); ICML (497 desk-rejects) and ICLR 2026 have hardened policies; the community is debating entrapment, disclosure, and AI-as-reviewer.

8. **Reproducibility and attribution gaps.** Many high-profile results are press/blog-stage (Ginkgo+OpenAI, Lila, Periodic, AlphaEvolve, Carl) without released data/prompts; foundation-model and startup claims need independent replication before being treated as discovery.

---

## Verification Status (updated 2026-06-25 deep-research pass)

A 113-agent deep-research pass (fan-out search -> primary-source fetch -> 3-vote adversarial verification; 30 sources, 135 claims extracted, 25 verified, 24 confirmed / 1 refuted) resolved most of the open flags. Status below: CONFIRMED / CORRECTED / STILL-UNVERIFIED.

### Confirmed against primary sources (Crossref / PubMed / arXiv / official)

- **Google co-scientist - BOTH DOIs real:** research paper [10.1038/s41586-026-10644-y](https://www.nature.com/articles/s41586-026-10644-y) (Nature, online 2026-05-19, ~47 authors led by Gottweis; PMID 42156544) AND the Nature Medicine News piece *"The AI co-scientist is here"* by David Adam, [10.1038/s41591-026-04275-z](https://www.nature.com/articles/s41591-026-04275-z) (Nat Med Mar 2026, 32(3):772-775, PMID 41840237) - previously flagged "not confirmed," now CONFIRMED.
- **Robin Nature** [10.1038/s41586-026-10652-y](https://www.nature.com/articles/s41586-026-10652-y): online 2026-05-19, received 2025-05-23, accepted 2026-05-12 (Crossref).
- **Virtual Lab** [10.1038/s41586-025-09442-9](https://www.nature.com/articles/s41586-025-09442-9): online 2025-07-29 / print 2025-10-16, Nature 646(8085):716-723 (Crossref).
- **Foundation models** (DOIs/params CONFIRMED): Evo 2 [10.1038/s41586-026-10176-5](https://www.nature.com/articles/s41586-026-10176-5) (40B flagship + 7B/20B/1B, 1-Mb context, >9T nucleotides); AlphaProof [10.1038/s41586-025-09833-y](https://www.nature.com/articles/s41586-025-09833-y) (IMO 2024 silver 28/42); ESM3 [Science 10.1126/science.ads0018](https://www.science.org/doi/10.1126/science.ads0018) (387(6736):850-858, PMID 39818825); MatterGen [10.1038/s41586-025-08628-5](https://www.nature.com/articles/s41586-025-08628-5) (639(8055):624-632; Microsoft, not DeepMind); Aurora [10.1038/s41586-025-09005-y](https://www.nature.com/articles/s41586-025-09005-y).
- **Gemini Deep Think IMO 2025 gold:** 5/6 problems, 35/42, officially graded/certified by IMO coordinators (the only officially-certified entry; OpenAI's was self-graded).
- **arXiv preprints all real:** data-to-paper (2404.17605), A Survey of AI Scientists (2510.23045), Why LLMs Aren't Scientists Yet (2601.03315), EvoScientist (2603.08127), Denario (2510.26887 - confirmed by direct fetch).

### Corrected

- **Evo 2 DOI** was missing - it is **10.1038/s41586-026-10176-5** (online 2026-03-04), not a generic guess.
- **EvoScientist (2603.08127)** was mis-flagged as spurious/dropped - it is a **real Mar 2026 paper**; now a Section-1 entry (#15).
- **ICML sting mechanism** was **watermarking** (phrase-sampling from a ~170k list), **not prompt-injection**; primary source = [ICML blog Mar 18 2026](https://blog.icml.cc/2026/03/18/on-violations-of-llm-review-policies/).
- **Robin verification-paradox** accompanying Nature editorial identified: *"Why AI cannot do good science without humans"* [10.1038/d41586-026-01551-3](https://www.nature.com/articles/d41586-026-01551-3) (May 2026, PMID 42156517).
- **One claim refuted (0-3):** a set of received/accepted dates wrongly attributed to the Co-Scientist paper (they were Robin's) - do NOT add them.

### Still unverified (paywalled / no surviving verified claim - handle with care)

- **Virtual Lab validation counts:** bibliographic record confirmed, but the granular numbers (92 nanobodies designed; how many validated as binders vs JN.1) remain unread (paywalled).
- **Autoscience "Carl":** still no arxiv/technical report; venues/scores undisclosed; company-PR-stage.
- **Deployed-lab vendor numbers:** Ginkgo Cloud Lab + OpenAI/GPT-5 (a primary OpenAI page ["GPT-5 lowers protein synthesis cost"](https://openai.com/index/gpt-5-lowers-protein-synthesis-cost/) exists, but the 36k-conditions / -40% specifics weren't independently re-verified), Lila Sciences (~$550M / >$1.3B val), Periodic Labs ($300M seed) - vendor-reported; carry an as-of date.
- **MIT CRESt:** ChemRxiv/MIT News; peer-review status + "record fuel-cell" claim still unconfirmed.
- **OpenAI "for Science":** official page 403; corroborated via secondary coverage only.
- **Nature editorial (~Mar 25 2026)** and **C&EN/Chemistry World A-Lab correction** ([C&EN](https://cen.acs.org/research-integrity/Nature-robot-chemist-paper-corrected/104/web/2026/01) confirms the A-Lab correction exists): originals not read directly (paywall).
- **Zochi/Tempest:** arXiv [2503.10619](https://arxiv.org/abs/2503.10619) is titled *"Tempest: Autonomous Multi-Turn Jailbreaking of LLMs with Tree Search"* (authors incl. Andy Zhou) - a safety paper; "Zochi" as AI-author + the ACL 2025 main-conference claim are separate Intology framing, still disputed.
- **GNoME "contested":** the 2025 novelty/duplicate dispute is real, but no formal retraction/correction was verified (informal dispute only).
- **Net-new Feb-Jun 2026 scan:** candidate arXiv IDs surfaced (e.g. 2606.24530, 2606.18060, 2605.20025, 2605.28655) but none produced a surviving 3-vote-verified claim; treat as leads, not confirmed additions.

---

*Compiled 2026-06-25 from six parallel web-research passes plus direct arxiv/DOI verification. Cross-listed systems (Robin, co-scientist) appear under multiple threads by design. Earlier unique content (data-to-paper, the BTL-vs-Elo ranking contrast, two survey/critique sources) folded in from a prior draft of this file with flags where re-verification was not done.*

*Verification update 2026-06-25: a 113-agent deep-research pass (fan-out search -> primary-source fetch -> 3-vote adversarial verification; 30 sources, 135 claims extracted, 25 verified, 24 confirmed / 1 refuted) resolved the bibliographic flags above. See the "Verification Status" section for the per-item breakdown. Items under "Still unverified" produced no surviving verified claim and need dedicated primary fetches.*
