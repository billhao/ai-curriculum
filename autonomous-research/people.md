This doc captures major research groups and people in RSI, AI for science etc.

The format is frozen. What exists below — the group sections with their four-column Key
people table and Major papers/systems list, the Other research groups table, the lineage
section, the appendix — is the whole structure. Updates add content into it: new rows, new
people, new papers, new lineage edges. Do not add sections, headings, subsections or table
columns, and do not reorganize. Anything that won't fit an existing slot belongs in a topic
guide, not here.

Affiliations are as printed on the paper (arXiv HTML author block or PDF page 1). Bare
acronyms are kept bare. Blank cells mean not yet researched, not "none". Source maps:
[`rsi-proposer-landscape-2026.md`](rsi-proposer-landscape-2026.md),
[`iclr-2026-rsi-workshop.md`](iclr-2026-rsi-workshop.md), and
[`rsi-catchup-2026-08-to-11-guide.md`](rsi-catchup-2026-08-to-11-guide.md), including its
4 November comparison addendum.

A conference program is not a source. Organizer, keynote and panel rosters are not recorded
here — those are not research output, and a program page gives a current affiliation rather
than the at-publication one every entry below is built from. A program slot never earns
anyone a row, a (PI) mark, or an appendix line. Workshop rosters live in the workshop guide.

**(PI)** marks the group lead, listed first in each table. Assigned only from printed
evidence — the last author or the marked corresponding author of that group's work here. No
mark means the senior slot isn't established from the papers, not that the group has no
lead. Unmarked on purpose: UC Riverside (first author is also corresponding), Edinburgh,
SJTU Zhang/Yu/Lin, Southeast Univ., and the
sole-author works (UCSC, Georgia Tech, Sungmin Lee) where student vs faculty isn't printed.
UCL's Jun Wang is marked from EpiAgent, independently of the earlier Meta-led AIRA paper.
Also unmarked: Recursive Superintelligence and the substrate maintainers, which have no
paper to read a senior slot off.

# Research Groups

## Sakana AI (Tokyo)

### Key people

| Name | Known for | Previous affiliations | Graduate education (school, advisor) |
|---|---|---|---|
| Robert Tjarko Lange (PI) | ShinkaEvolve lead — printed as having initiated and led the project, co-corresponding; Sakana-side author on Darwin Gödel Machine | | |
| Yujin Tang (PI) | AC/DC and MASS last author | | |
| Edoardo Cetin | ShinkaEvolve core contributor, co-corresponding | | |
| Yuki Imajuku | ShinkaEvolve | | |
| Yingtao Tian | AC/DC | | |
| Andrew Dai | AC/DC co-first | | |
| Boris Meinardus | AC/DC co-first | | |
| Ciaran Regan | AC/DC | | |
| Jinglue Xu · Jeffrey Seely | MASS | | |

### Major papers/systems
- ShinkaEvolve (2509.19349) — open Apache-2.0 evolutionary program search, every component ablated; discovered a MoE load-balancing loss
- AC/DC (2604.14969) — ICLR 2026 main track; coevolves model merging with synthetic NL tasks
- Darwin Gödel Machine (2505.22954) — with Clune's UBC group
- MASS (2610.12176) — with UC Berkeley; distills self-generated multi-agent workflows into the model used for subsequent execution, evaluation, and optimization

---

## Weco AI

### Key people

| Name | Known for | Previous affiliations | Graduate education (school, advisor) |
|---|---|---|---|
| Zhengyao Jiang (PI) | Co-founder & CEO; last author on both reward-hacking papers | | |
| Yuxiang Wu (PI) | Co-founder & CTO | | |
| Bingchen Zhao | First author on both reward-hacking papers; co-first author of Meta's Speedrunning Benchmark | Meta + University of Edinburgh (concurrent, printed on 2506.22419) | |
| Dhruv Srikanth | AIDE² first author | | |
| Dixing Xu | AIDE² | | |

### Major papers/systems
- AIDE — the coding-agent scaffold used as reference harness inside OpenAI's MLE-bench, METR's RE-Bench and Meta's aira-dojo
- Reward Hacking in Self-Improving Code Agents — 73.8% / 46.8% proxy-only gains. *Affiliation inferred, not read: OpenReview 403s; poster #58 of 110 at the ICLR 2026 RSI workshop*
- SpecBench (2605.21384) — hacking gap grows +28 pp per 10× code size
- AIDE² (2609.26457) — September technical report on research-harness evolution, external transfer, and an inconclusive same-start improver comparison

---

## Anthropic

### Key people

| Name | Known for | Previous affiliations | Graduate education (school, advisor) |
|---|---|---|---|
| Jan Leike (PI) | Last author, Automated Weak-to-Strong Researcher | | |
| Jan Hendrik Kirchner (PI) | AAR; last author, Automated Researchers Can Mitigate Well-characterized Alignment Failures | | |
| Joe Benton | AAR | | |
| Jiaxin Wen | AAR co-first | | |
| Liang Qiu | AAR co-first | Anthropic Fellows Program (printed, this paper) | |
| Chen Yueh-Han | Automated Researchers Can Mitigate Well-characterized Alignment Failures first author | Anthropic Fellows Program (concurrent, printed on 2608.28945) | |

### Major papers/systems
- Automated Weak-to-Strong Researcher — 9 parallel Claude agents; PGR 0.97 vs human 0.23, plus three self-reported unpredicted reward-hacking exploits and a transfer failure at production scale
- Automated Researchers Can Mitigate Well-characterized Alignment Failures (2608.28945) — August follow-up; automated search for training-based alignment mitigations across ten failure families, with larger-model and production tests

---

## Meta / FAIR at Meta

### Key people

| Name | Known for | Previous affiliations | Graduate education (school, advisor) |
|---|---|---|---|
| Yoram Bachrach (PI) | Last author on both Speedrunning Benchmark and AIRA | | |
| Jakob Nicolaus Foerster | AIRA | | |
| Edan Toledo | AIRA co-first | University College London (concurrent, printed) | |
| Karen Hambardzumyan | AIRA co-first | University College London (concurrent, printed) | |
| Martin Josifoski | AIRA co-first | | |
| Despoina Magka | Speedrunning Benchmark co-first; AIRA | | |
| Minqi Jiang | Speedrunning Benchmark co-first; AIRA | | |
| Roberta Raileanu | AIRA | | |
| Nicola Cancedda | AIRA | | |
| Carole-Jean Wu | AIRA | | |
| Vikas Chandra | Agent-as-a-Judge (printed Meta AI, 2024) | | |
| Yuandong Tian | Agent-as-a-Judge (printed Meta AI, 2024); later a named Recursive Superintelligence founder | | |

### Major papers/systems
- The Automated LLM Speedrunning Benchmark (2506.22419) — 19 nanoGPT records; agents fail to reimplement known improvements even with hints
- AIRA (2507.02554) — agents on MLE-bench; §5.3 measures the proxy/held-out generalization gap at 9–13 absolute points, a year before the 2026 reward-hacking papers
- aira-dojo, llm-speedrunner — the field's two general agentic-research eval harnesses

---

## University of Cambridge — Lane group (+ Flower Labs)

### Key people

| Name | Known for | Previous affiliations | Graduate education (school, advisor) |
|---|---|---|---|
| Nicholas D. Lane (PI) | Last author, Red Queen Gödel Machine | Flower Labs (concurrent, printed) | |
| Alex Iacob | RQGM co-first | | |
| Andrej Jovanović | RQGM co-first | Flower Labs (concurrent, printed) | |
| William F. Shen | RQGM co-first | | |
| Lorenzo Sani | RQGM | Flower Labs (concurrent, printed) | |
| Meghdad Kurmanji · Zeyu Cao · Bill Marino · Xinchi Qiu | RQGM | | |

Co-authors at other institutions: Daniel Burkhardt, Niccolò Alberto Elia Venanzi (NVIDIA);
Nurbek Tastan (MBZUAI); Ambroise Odonnat (Inria).

### Major papers/systems
- Red Queen Gödel Machine (2606.26294) — drops the stationary-evaluator assumption; utilities update at epoch boundaries so agent and evaluator co-adapt

---

## Stanford University

### Key people

| Name | Known for | Previous affiliations | Graduate education (school, advisor) |
|---|---|---|---|
| Tatsunori Hashimoto (PI) | Last author, Execution-Grounded Auto-Research; Ideation-Execution Gap | | |
| James Zou (PI) | Last author, SLDAgent | | |
| Yingcheng Wu (PI) | RSI-Anything corresponding | PhAI Labs (concurrent, printed in Appendix A referenced on the title page) | |
| Chenglei Si | First author, Execution-Grounded Auto-Research | | |
| Haotian Ye | SLDAgent co-first | | |
| Zitong Yang · Yejin Choi · Emmanuel Candès · Diyi Yang | Execution-Grounded Auto-Research | | |
| Fang Wu | Agent0 | | |
| Bo Liu | SPADE co-first | University of Washington (concurrent, printed on 2608.19197) | |
| Xinliang Zhou | CogRSI | | |
| Yoonho Lee · Roshen Nair · Qizheng Zhang · Chelsea Finn | Meta-Harness (Stanford printed on the COLM 2026 title page) | | |
| Ethan Boneh | EvoX (Stanford University printed on the COLM 2026 title page) | | |

### Major papers/systems
- Towards Execution-Grounded Automated AI Research (2601.14525) — evolutionary search beats baselines where RL mode-collapses
- SLDAgent (2507.21184) — senior slot; LLM-discovered scaling laws beat human-derived ones, R² 0.748 vs 0.517
- Ideation-Execution Gap — rank flip between idea scores and executed results
- Agent0 (2511.16043) — co-author
- SPADE (2608.19197) — Bo Liu co-first and Yejin Choi joint last author; shared-weight environment designer and reasoning agent
- RSI-Anything (2610.13912) — Yingcheng Wu corresponding; predictive world models and retained policy–model updates
- CogRSI (2611.01113) — evolving theory-of-mind rules and post hoc semantic auditing
- [Meta-Harness (2603.28052)](https://proceedings.colm.cc/paper_files/paper/2026/file/fde1a69a5b6e554b2f1f727197d2651d-Paper-Conference.pdf) — COLM 2026 main; executable harness optimization using prior source, scores, and execution traces; with MIT and KRAFTON
- [EvoX (2602.23413)](https://proceedings.colm.cc/paper_files/paper/2026/file/fc9f83d9925e6885e8f1ae1e17b3c44b-Paper-Conference.pdf) — COLM 2026 main; Ethan Boneh is the Stanford co-author

---

## Peking University (+ Wizard Quant)

### Key people

| Name | Known for | Previous affiliations | Graduate education (school, advisor) |
|---|---|---|---|
| Yitao Liang (PI) | PKU senior slot on SLDAgent | | |
| Xiaojun Wan (PI) | PKU senior slot on Gödel Agent | | |
| Ling Yang (PI) | RSI-Anything last and corresponding author | PhAI Labs (concurrent, printed in Appendix A referenced on the title page) | |
| Haowei Lin | SLDAgent co-first | Wizard Quant (concurrent, printed) | |
| Xunjian Yin | Gödel Agent first author | Work completed during internship at UC Santa Barbara (printed) | |
| Hubert Lim · Zhengrui Li · Xiangyu Wang | SLDAgent | | |
| Anjie Xu | AutoSOTA | Zhongguancun Academy (concurrent, printed) | |
| Rui Hu | ScienceCUA | | |

Wizard Quant supplies 4 of SLDAgent's 11 authors: Haowei Lin, Wenzheng Feng, Quzhe Huang,
Yujun Li — a quantitative fund entering academic RSI.

### Major papers/systems
- SLDAgent (2507.21184) — lead slot on the map's strongest genuine-discovery claim
- Gödel Agent (2410.04444) — self-referential agent framework (first author, joint with UCSB)
- AutoSOTA (2604.05550) — co-author
- RSI-Anything (2610.13912) — policy/world-model update and retention framework
- ScienceCUA (2611.03238) — source-grounded scientific-software Skills and References, refined through execution feedback

---

## SJTU / SII / GAIR

### Key people

| Name | Known for | Previous affiliations | Graduate education (school, advisor) |
|---|---|---|---|
| Pengfei Liu (PI) | Corresponding author, ASI-Arch; holds SJTU + SII + GAIR jointly | | |
| Yixiu Liu | ASI-Arch co-first (SJTU + SII + GAIR) | | |
| Yang Nan | ASI-Arch co-first (SII + GAIR) | | |
| Weixian Xu | ASI-Arch co-first (SJTU + SII + GAIR) | | |
| Xiangkun Hu | ASI-Arch (SII + GAIR) | | |
| Lyumanshan Ye | ASI-Arch (SJTU + SII + GAIR) | | |
| Zhen Qin | ASI-Arch (Taptap) | | |

Title page prints four separate numbered institutions — 1 Shanghai Jiao Tong University,
2 SII, 3 Taptap, 4 GAIR — under an `SII-GAIR` banner. The expansion of SII and the binding
of GAIR to SJTU are inference, not printed.

### Major papers/systems
- ASI-Arch (2507.18074) — 1,773 architectures, 106 SOTA claims; the canonical example of evaluator failure, since an LLM judge is one third of its fitness function

---

## METR

### Key people

| Name | Known for | Previous affiliations | Graduate education (school, advisor) |
|---|---|---|---|

Time Horizon 1.1 is **corporately authored** — the page's only author field reads `METR`,
with no individual bylines. Individual attribution deliberately left empty.

### Major papers/systems
- Time Horizon 1.1 (Jan 29 2026) — 228-task suite; since-2024 doubling time 88.6 days
- HCAST / time-horizons — the field's longitudinal capability measurement
- RE-Bench — one of the two harnesses Weco's AIDE is measured in

---

## KAUST / IDSIA — Zhuge & Schmidhuber

### Key people

| Name | Known for | Previous affiliations | Graduate education (school, advisor) |
|---|---|---|---|
| Jürgen Schmidhuber (PI) | Gödel machine (2003) — the formal idea the whole field approximates; last author on GPTSwarm and Agent-as-a-Judge; moderates the RSI workshop's Super-Stars panel | KAUST + The Swiss AI Lab IDSIA, USI, SUPSI (concurrent, printed on 2402.16823) | |
| Mingchen Zhuge (PI) | Lead organizer of the ICLR 2026 RSI workshop; first author and co-corresponding on both GPTSwarm and Agent-as-a-Judge; metauto.ai | KAUST AI Initiative (2402.16823) + Meta AI (2410.10934) | |
| Dmitrii Khizbullin | GPTSwarm co-corresponding (KAUST); Agent-as-a-Judge | | |
| Wenyi Wang | GPTSwarm co-first (KAUST); Agent-as-a-Judge (KAUST); OASIS co-corresponding | | |
| Francesco Faccio | GPTSwarm | KAUST + IDSIA (concurrent, printed) | |
| Louis Kirsch | GPTSwarm (IDSIA); meta-RL | IDSIA (printed on 2402.16823) | |
| Dylan R. Ashley | Agent-as-a-Judge (KAUST) | | |
| Yuhui Wang | OASIS co-corresponding (KAUST) | | |
| Firas Laakom · Li Nanbo · Yanning Dai · Yifan Xu · Piotr Piękos · Yilan Zhang · Yimeng Chen | OASIS (KAUST) | | |

### Major papers/systems
- GPTSwarm (2402.16823, ICML 2024) — language agents as optimizable graphs; the "agent topology is a learnable object" primitive
- Agent-as-a-Judge (2410.10934) — with Meta AI; an agentic system scores another's intermediate steps, plus DevAI (55 tasks, 365 hierarchical requirements). This is the automated evaluator that closes a self-improvement loop — and the reason the organizer's own tool sits inside the evaluation critique
- ICLR 2026 RSI workshop (Apr 26 2026, Rio, Room 101-D) — 110 accepted; a dedicated RSI workshop. Sponsors: Tencent, Meta
- [OASIS: Online Anchor-calibrated Self-Improvement System](https://aclanthology.org/2026.findings-emnlp.2150.pdf) — Findings EMNLP 2026; adaptive anchor agents calibrate paired evidence for online branch continuation, reversion, and promotion; Wenyi Wang and Yuhui Wang are marked corresponding

---

## Tübingen — ELLIS Institute / MPI-IS / Univ. Tübingen (+ Thoughtful Lab)

### Key people

| Name | Known for | Previous affiliations | Graduate education (school, advisor) |
|---|---|---|---|
| Maksym Andriushchenko (PI) | PostTrainBench last author and corresponding | ELLIS Institute Tübingen + Max Planck Institute for Intelligent Systems + Tübingen AI Center (concurrent, printed) | |
| Matthias Bethge (PI) | PostTrainBench senior slot | University of Tübingen + Tübingen AI Center (concurrent, printed) | |
| Ben Rank | PostTrainBench co-first | ELLIS Institute Tübingen + MPI-IS + Tübingen AI Center | |
| Hardik Bhatnagar | PostTrainBench co-first | Univ. Tübingen + Tübingen AI Center | |
| Ameya Prabhu | PostTrainBench | Univ. Tübingen + Tübingen AI Center | |
| Shira Eisenberg | PostTrainBench | Thoughtful Lab (work done during an internship there, printed) | |
| Nguyen Karina | PostTrainBench | Thoughtful Lab (printed) | |

### Major papers/systems
- PostTrainBench (2603.08640) — ICLR 2026 RSI workshop oral. Can agents automate post-training under 10 h on one H100? Best agent 23.2% vs 51.1% for official instruction-tuned models. Names three concrete reward-hacking behaviors observed in the wild: training on the test set, downloading an existing instruct checkpoint instead of training one, and using discovered API keys to generate synthetic data without authorization

---

## Recursive Superintelligence

The field's for-profit anchor — out of stealth May 14 2026. Named founders, per the
workshop guide (which sources them from secondary tech press, not a title page): Richard
Socher, Josh Tobin, Tim Rocktäschel, Jeff Clune, Yuandong Tian, Alexey Dosovitskiy, Caiming
Xiong, Tim Shi. No affiliations table yet — no paper has been published under this name, so
nothing here is printed-on-a-paper evidence.

Four of the eight are also on the RSI workshop program: Clune (keynote), Tian (keynote and
panel), Socher (panel); Xiong is a Salesforce co-author on Agent0. The same people running
this dedicated RSI workshop are building the commercial bet on it — worth holding in view when
reading the workshop's own framing.

Funding figures reported in the press (~$650M at ~$4.65B) are inconsistent across outlets
and are deliberately not recorded here.

- [Recursive Flint](https://huggingface.co/RecursiveSI/flint) (October 2026) — Apache-2.0 coding MoE weights and inference code; corporate model-card attribution, with no individual author–affiliation mapping. The release describes contributions from Eureka, the automated research system.
- [First Steps Toward Automated AI Research](https://recursive.com/updates/first-steps-toward-automated-ai-research) — earlier NanoGPT, NanoChat and kernel artifacts; the official updates index dates this report 11 June 2026, despite its migrated October page header.

---

## Other research groups

| Institution | Key people | Key papers |
|---|---|---|
| National University of Singapore (+CUHK-Shenzhen) | **Bingsheng He (PI)**; Nuo Chen, Yicheng Tong, Yufei He; Zichen Liu, Zijian Zhou (SPADE); Guibin Zhang (HarnessForge); Qika Lin, Tiesunlong Shen (R²Flow, printed National University of Singapore) | Diversity Collapse in MAS (2604.18005); SPADE (2608.19197); [R²Flow (2609.33867v3)](https://arxiv.org/pdf/2609.33867v3), under review at ICLR 2027; [HarnessForge](https://aclanthology.org/2026.emnlp-main.1893.pdf), EMNLP 2026 main |
| Tsinghua University — separate collaborations | **Yong Li (PI)**, **Fengli Xu (PI)** (FIB Lab, EE/BNRist, both corresponding on AutoSOTA); Jianzhu Ma; Qi Li, Ke Xu (network security); Caiyin Yang (AEvo); Heng Chang (HarnessForge, marked Project Lead) | AutoSOTA lead; SLDAgent co; Safety in Self-Evolving Agents co; [Harnessing Agentic Evolution (2605.13821)](https://arxiv.org/html/2605.13821v1); [HarnessForge](https://aclanthology.org/2026.emnlp-main.1893.pdf), EMNLP 2026 main |
| UC Santa Barbara — two labs | **Xin Eric Wang (PI)**; **William Yang Wang (PI)** (Heuresis last author); Antonis Antoniades, Deepak Nathani, Zhaotian Weng (GEA and Heuresis); Zhen Zhang, Xiao Pu (GEA, printed arXiv v1); Ritam Saha, Alfonso Amayuelas, Ivan Bercovich (Heuresis); Alfonso Amayuelas (OASIS) | [GEA (2602.04837)](https://arxiv.org/html/2602.04837v1), COLM 2026 main; v1 prints Xiao Pu and Xin Eric Wang, while proceedings metadata uses Sophia Pu and Eric Wang; identities are not equated here; Gödel Agent (2410.04444); [Heuresis (2606.25198)](https://arxiv.org/html/2606.25198v1), NeurIPS 2026 main-track poster under the title Evaluating Search Strategies for Autonomous Machine Learning Research Agents; [OASIS](https://aclanthology.org/2026.findings-emnlp.2150.pdf), Findings EMNLP 2026 |
| Columbia Business School | **Olivier Toubia (PI)**; Melanie Brucks, Yuting Deng | Barriers to Diversity in LLM Ideas (2602.20408) |
| UBC + Vector Institute | **Jeff Clune (PI)** (also Canada CIFAR AI Chair); Jenny Zhang, Shengran Hu, Cong Lu, Yiming Xiong; Xiaoxiao Li (CoEvoSkills, printed University of British Columbia only) | Darwin Gödel Machine (2505.22954); ALMA (2602.07755, workshop oral); AI-GAs (1905.10985); [CoEvoSkills (2604.01687v2)](https://arxiv.org/pdf/2604.01687v2), COLM 2026 main |
| UC Riverside (+AlphaAvatar, Illinois Tech) | Mingguang Chen (first author *and* corresponding — senior slot unclear); Licheng Wang; Bo Qu | RSI survey, 1,250 papers (2607.07663) |
| Google DeepMind | **Matej Balog (PI)** (AlphaEvolve last author, listed as its lead on the RSI-workshop panel page); **Vivek Natarajan (PI)** (AI co-scientist last author); Alexander Novikov, Juraj Gottweis; Zhankui He, Benjamin Coleman, Di Bai, Wang-Cheng Kang (Dream-RSI) | AlphaEvolve (2506.13131); AI co-scientist (2502.18864); Dream-RSI (2609.14858) |
| Zhongguancun Academy | **Tie-Yan Liu (PI)** (AutoSOTA last author); **Zequn Liu (PI)** (ForeSci corresponding); Yingce Xia; Jian Zhao (EpiAgent) | AutoSOTA senior; ForeSci senior/corresponding; EpiAgent (2610.40227) |
| OpenAI | **Aleksander Madry (PI)** (MLE-bench last author); Chan Jun Shern, Neil Chowdhury | MLE-bench (2410.07095) — substrate only; no RSI paper in this map |
| Pangram Labs | **Bradley Emi (PI)** (CTO/co-founder, sole byline) | ICLR 2026 AI-generated-review census, 21% of 75,800 |
| ISCAS + UCAS + Alibaba | **Xianpei Han (PI)**, **Yongbin Li (PI)**, **Hu Wei (PI)** (three marked corresponding authors); Le Sun, Hongyu Lin, Yaojie Lu (ISCAS); Tianyun Zhong (UCAS) | HypoArena / Before the Action (2607.15766) |
| MBZUAI (+CMU) | **Kun Zhang (PI)** (MBZUAI + CMU); Yongqiang Chen, Zhenhao Chen; Bowei He and Xue Liu (Beneath the Diff, joint McGill); Yankai Chen and Xue (Steve) Liu (CoEvoSkills, joint McGill); Zhenting Wang (CoEvoSkills) | CausalEvolve (2603.14575); Red Queen co; [Beneath the Diff](https://aclanthology.org/2026.emnlp-main.1029.pdf), EMNLP 2026 main; [CoEvoSkills (2604.01687v2)](https://arxiv.org/pdf/2604.01687v2), COLM 2026 main |
| UC Santa Cruz | Kargi Chauhan (sole author); Juncheng Wu (Harness Updating co-first), Zijun Wang, Cihang Xie, Yuyin Zhou (Harness Updating, printed UC Santa Cruz) | Dead Science Walking (2606.04220); [Harness Updating Is Not Harness Benefit](https://aclanthology.org/2026.emnlp-main.2326.pdf), EMNLP 2026 main |
| King's College London — Algorithmic Dynamics Lab | **Hector Zenil (PI)** (lab lead, sole author; also Oxford Immune Algorithmics) | On the Limits of Self-Improving (2601.05280) |
| Zhejiang University + Ant Group | **Shouling Ji (PI)**; Ruixiao Lin, Qingming Li (ZJU); Changhua Meng, Shiwen Cui (Ant); Yifei Yao (CoEvoSkills, printed Zhejiang University) | Safety in Self-Evolving LLM Agent Systems (2606.23075); [CoEvoSkills (2604.01687v2)](https://arxiv.org/pdf/2604.01687v2), COLM 2026 main |
| Hong Kong Baptist University — TMLR Group | **Bo Han (PI)** (corresponding); Xuan Li, Zhanke Zhou | Learning to Evolve (OpenReview WnZHbe1Gu0); CausalEvolve co |
| Harvard (+MIT) | **Vijay Janapa Reddi (PI)**; Chenyu Wang, Zishen Wan, Yilun Du | ArchEval (2607.03601) |
| Princeton University — Princeton Language and Intelligence | **Sanjeev Arora (PI)** (last author); Yun Cheng, Xingyu Zhu (both corresponding), Haoyu Zhao | Contextual Drag (2602.04288) |
| University of Edinburgh | Oisin Mac Aodha, Bingchen Zhao (co-first) | Speedrunning Benchmark (2506.22419) |
| Carnegie Mellon University | **Kun Zhang (PI)** (joint MBZUAI); **Aditi Raghunathan** (STV last author); Chen Henry Wu (STV first author); Zeyu Zheng; Yiding Jiang, Seungone Kim (SPADE) | CausalEvolve senior; Denominator Gaming; SPADE (2608.19197); [Self-Trained Verification (2605.30290)](https://arxiv.org/html/2605.30290v1), NeurIPS 2026 |
| SJTU — Zhang/Yu/Lin group | Jianghao Lin (last author), Weinan Zhang, Yong Yu, Rong Shan | Denominator Gaming (2605.09915); Learning to Evolve co |
| University College London | **Jun Wang (PI)** (EpiAgent last and corresponding; printed AI Centre, Department of Computer Science, UCL); Pontus Stenetorp; Edan Toledo, Karen Hambardzumyan (both co-first, joint Meta) | AIRA (2507.02554); EpiAgent (2610.40227) |
| UNC-Chapel Hill (+Salesforce, Stanford) | **Huaxiu Yao (PI)**; Peng Xia; Caiming Xiong, Can Qin (Salesforce) | Agent0 (2511.16043) |
| FutureHouse (+Oxford) | **Samuel G. Rodriques (PI)**; Andrew D. White, Ali E. Ghareeb | Robin (2505.13400) |
| Wizard Quant | Haowei Lin (joint PKU), Wenzheng Feng, Quzhe Huang, Yujun Li | SLDAgent (4 of 11 authors) |
| Microsoft Research (+UC San Diego) | **Weizhu Chen (PI)**; Jianfeng Gao, Chandan Singh; Yufan Zhuang, Jingbo Shang (UCSD); Seungwon Lim (SHOR, printed Microsoft Research Asia + Yonsei University) | Test-time Recursive Thinking (2602.03094); [Towards Direct Evaluation of Harness Optimizers via Priority Ranking / SHOR (2605.22505v1)](https://arxiv.org/pdf/2605.22505v1), NeurIPS 2026 |
| Örebro University — Machine Perception and Interaction Lab | **Andreas Persson (PI)** (last author); Amy Loutfi, Alkis Sygkounas, Rishi Hazra | Evolutionary Discovery of RL Algorithms (2603.28416, GECCO 2026); AIRA co |
| Hexo Labs / Hexo AI (as printed; SIA also prints Oxford) | **Vignesh Baskaran (PI)** (SIA last author; Heuresis); Kunal Bhatia (SIA and Heuresis); Prannay Hebbar, Yogendra Manawat (SIA) | SIA (2605.27276), printed Hexo Labs Palo Alto / Brussels / Toronto; [Heuresis (2606.25198)](https://arxiv.org/html/2606.25198v1), printed Hexo AI, NeurIPS 2026 main-track poster |
| Southeast University (+Duke Kunshan) | Youyong Kong, Qiuyu Tian (first author); Haojie Yin (Duke Kunshan) | ForeSci (2606.00644) |
| University of Würzburg — Computer Vision Lab, CAIDAS & IFI | **Dmitry Ignatov (PI)** (last author); Radu Timofte, Santosh Premi Adhikari | Delta-Based NAS (2605.04903) |
| Nanjing University of Science and Technology | **Chengzhi Zhang (PI)** (corresponding); Shuai Chen | Combinatorial Innovation idea generation (2604.20548) |
| Georgia Institute of Technology | Henry Jiang (sole author); Jiang Zhou (RSI-Env) | DARWIN (2602.05848); RSI-Env (2611.01736) |
| UC Berkeley | Hyunin Lee, Donghyun Lee, Somayeh Sojoudi, Matei Zaharia (MASS); Alexander Krentsel (Exo first and corresponding), Sylvia Ratnasamy, Ion Stoica (Exo); Jiaxin Wen (2608.28945, rather than the earlier Anthropic-affiliated weak-to-strong paper); Shu Liu and Shubham Agarwal (EvoX co-first); Monishwaran Maheswaran, Mert Cemri, Qiuyang Mang, Zhifei Li, Ashwin Naren, Audrey Cheng, Alexander Du, Melissa Pan, Kurt Keutzer, Alvin Cheung, Koushik Sen (EvoX); Alex Dimakis (EvoX, joint Bespoke Labs); Matei Zaharia and Ion Stoica also co-author EvoX | MASS (2610.12176); Exo (2611.00463); Automated Researchers Can Mitigate Well-characterized Alignment Failures (2608.28945); [EvoX (2602.23413)](https://proceedings.colm.cc/paper_files/paper/2026/file/fc9f83d9925e6885e8f1ae1e17b3c44b-Paper-Conference.pdf), COLM 2026 main |
| Google Cloud AI Research / Google Cloud (as printed) | **Jaehyun Nam (PI)**, **Jinsung Yoon (PI)** (ScientistTwo corresponding), **Tomas Pfister (PI)** (last author); Yanzhou Pan, Rui Meng, Parthasarathy Ranganathan (ScientistTwo); Bhavana Dalvi Mishra, Jiefeng Chen, Chun-Liang Li, Palash Goyal, Mihir Parmar, Yiwen Song, Yale Song, Rajarishi Sinha, Burak Gokturk (ScientistOne); Sercan Ö. Arık (MLE-STAR, Google Cloud; Nam also KAIST on that paper) | [ScientistTwo (2609.19644)](https://arxiv.org/pdf/2609.19644), title page; [ScientistOne (2605.26340)](https://arxiv.org/html/2605.26340v1), author block; [MLE-STAR (2506.15692v3)](https://arxiv.org/pdf/2506.15692v3), title page |
| University of Waterloo | Yubo Wang (ScientistTwo, printed on title page) | [ScientistTwo (2609.19644)](https://arxiv.org/pdf/2609.19644) |
| Google | **Xidong Wu (PI)**, **Zheng Zhang (PI)** (Dream-RSI corresponding), **Yunsong Guo (PI)** (last author); Tong Zheng (joint Maryland), Chaoyi Zhang, Ruoqiao Wei, Xue Wang, Yue Zhuan, Renkai Xiang, Xinwu Cheng; Palash Goyal, Mihir Parmar, Yiwen Song, Hamid Palangi, Tomas Pfister, Jinsung Yoon (ScholarPeer); Yale Song (PaperOrchestra) | Dream-RSI (2609.14858); [ScholarPeer (2601.22638v2)](https://arxiv.org/html/2601.22638v2); [PaperOrchestra (2604.05018)](https://arxiv.org/pdf/2604.05018), both print Google |
| University of Maryland, College Park | Heng Huang, Tong Zheng, Rui Liu | Dream-RSI (2609.14858) |
| University of Virginia | Haolin Liu | Dream-RSI (2609.14858) |
| University of Washington | **Natasha Jaques (PI)** (SPADE joint last author); Luke Zettlemoyer, Bo Liu (joint Stanford), Mickel Liu | SPADE (2608.19197) |
| Northeastern University | Simon Yu (co-first), Weiyan Shi; Dakuo Wang (Evo-Harness; Harness Updating Is Not Harness Benefit) | SPADE (2608.19197); [Evo-Harness](https://aclanthology.org/2026.emnlp-main.969.pdf); [Harness Updating Is Not Harness Benefit](https://aclanthology.org/2026.emnlp-main.2326.pdf), both EMNLP 2026 main |
| Massachusetts Institute of Technology | Ao Qu, Paul Pu Liang; Omar Khattab (Meta-Harness last author, printed MIT); Rikiya Takehi, Jayden Teoh (co-first), Alex Zhang, Omar Khattab (Weight-Space Programming, all MIT CSAIL in the original post's author block) | SPADE (2608.19197); [Meta-Harness (2603.28052)](https://proceedings.colm.cc/paper_files/paper/2026/file/fde1a69a5b6e554b2f1f727197d2651d-Paper-Conference.pdf), COLM 2026 main; [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/), November 2026 research post |
| Seoul National University | Seungyeon Jwa (Meta^n), Junsu Kim (SPADE) | [Meta^n (2608.24735)](https://arxiv.org/html/2608.24735v1), NeurIPS 2026; SPADE (2608.19197) |
| University of Minnesota | **Dongyeop Kang (PI)** (Meta^n last author); Zae Myung Kim, Young-Jun Lee | [Meta^n (2608.24735)](https://arxiv.org/html/2608.24735v1), [NeurIPS 2026 poster](https://neurips.cc/virtual/2026/poster/151012) |
| University of Chicago | Zhaorun Chen | SPADE (2608.19197) |
| Stevens Institute of Technology | Zihan Chen, Di Zhu; Hanfei Yu (SPADE) | Auditing the Product, Not the Recursion (2610.27610); SPADE (2608.19197) |
| University of Massachusetts Boston | **Lei Nico Zheng (PI)** (Auditing the Product last author) | Auditing the Product, Not the Recursion (2610.27610) |
| Rutgers University | Meijia Chen (co-first) | False Frontiers (2609.39102) |
| UC San Diego | Yichen Liu, Hanxin Chen | False Frontiers (2609.39102) |
| University of Michigan | Weizhi Du | False Frontiers (2609.39102) |
| McGill University | **Tianyu Shi (PI)** (False Frontiers corresponding); Bowei He, Weixu Zhang, Xue Liu (Beneath the Diff; He and Liu joint MBZUAI); Yankai Chen and Xue (Steve) Liu (CoEvoSkills, joint MBZUAI) | False Frontiers (2609.39102); [Beneath the Diff](https://aclanthology.org/2026.emnlp-main.1029.pdf), EMNLP 2026 main; [CoEvoSkills (2604.01687v2)](https://arxiv.org/pdf/2604.01687v2), COLM 2026 main |
| King Fahd University of Petroleum and Minerals | **Alaa Khamis (PI)** (False Frontiers corresponding and last author) | False Frontiers (2609.39102) |
| Independent researchers (as printed) | Hao Li, Zheng Lu (co-first), Hongshan Lin, Junbai Tian, Zijun Tian, Yufan Zou, Shuhan Sun, Zeyu Zhang, Yueting Li (False Frontiers); Cong Zhang (HEMO; paper prints prior affiliations at Nanyang Technological University, Huawei Noah's Ark Lab, and TikTok Singapore) | False Frontiers (2609.39102); HEMO (2610.40023) |
| ByteDance Seed / M-A-P / TokenWave.AI | S3Gym lists these three affiliations without individual mapping; core contributors include Jiajun Shi, Siyuan Tao, Yuhao Wu, Zexuan Wang, Jingyuan Zhang | S3Gym (2608.31100); individual institutional assignments unresolved |
| Fudan University | Yanwei Fu (joint Shanghai Innovation Institute), Wanying Qu, Jiyao Liu (RSI-Env); Hao Xiao, Qiang Gao (RSI-Anything); Xingjun Ma, Haohang Liu (joint Keeta, Meituan; CARE); Qiang Huang (R²Flow and CollabFlow) | RSI-Env (2611.01736); RSI-Anything (2610.13912); CARE (2611.02839); [R²Flow (2609.33867v3)](https://arxiv.org/pdf/2609.33867v3); [CollabFlow (2609.38662v3)](https://arxiv.org/pdf/2609.38662v3), under review at ICLR 2027 |
| HKUST | Zhiyuan Fan | RSI-Env (2611.01736) |
| Nanjing University / Shanghai Artificial Intelligence Laboratory | **Dongrui Liu (PI)** (Shanghai AI Laboratory, RSI-Env last author); Jianghan Shen (joint Nanjing and Shanghai AI Laboratory) | RSI-Env (2611.01736) |
| PhAI Labs | **Ling Yang (PI)** (joint PKU), **Yingcheng Wu (PI)** (joint Stanford), **Zhenfei Yin (PI)** (joint Oxford), all RSI-Anything corresponding; Taoyong Cui (joint CUHK), Shuhan Xue | RSI-Anything (2610.13912); affiliations in Appendix A referenced by title page |
| Chinese University of Hong Kong | **Pheng Ann Heng (PI)** (RSI-Anything corresponding); Wanli Ouyang, Taoyong Cui (joint PhAI Labs); **Michael R. Lyu (PI)** (RSI-Dojo last author), Shuzheng Gao (corresponding), both Department of Computer Science and Engineering | RSI-Anything (2610.13912); [RSI-Dojo (2611.03698)](https://arxiv.org/pdf/2611.03698), title-page affiliations |
| University of Macau | Pinxian Zeng | RSI-Anything (2610.13912) |
| SJTU — RSI-Anything collaboration | Wanghan Xu | RSI-Anything (2610.13912) |
| ByteDance — RSI-Dojo collaboration | **Shilin He (PI)** (corresponding), Zhaojian Yu, Kai Cai | [RSI-Dojo (2611.03698)](https://arxiv.org/pdf/2611.03698), printed title-page affiliations |
| Shanghai Jiao Tong University — RSI-Dojo collaboration | Yuling Shi | [RSI-Dojo (2611.03698)](https://arxiv.org/pdf/2611.03698), printed title-page affiliations |
| Tsinghua University — RSI-Dojo collaboration | Penghao Yin | [RSI-Dojo (2611.03698)](https://arxiv.org/pdf/2611.03698), printed title-page affiliations |
| Simple Agent Lab / Tsinghua University / The Chinese University of Hong Kong — AutoTrainess collective byline | Zhaojian Yu, Penghao Yin, Shuzheng Gao, Shilin He, Kai Cai, Xiao-Ping Zhang; institutions printed collectively, individual mapping absent | [AutoTrainess (2606.31551)](https://arxiv.org/pdf/2606.31551), title page |
| Shanghai Jiao Tong University / Theseus Labs | **Xuanhe Zhou (PI)** (corresponding); Fan Wu; Bangrui Xu and Sidi Chen (co-first, joint Humanlaya); Jie Yang (co-first, joint Shanghai AI Lab) | SelfImprove-Bench (2611.02797) |
| Humanlaya | **Zhoufutu Wen (PI)** (SelfImprove-Bench last and corresponding); Qianyu Yang, Zijie Wang; Bangrui Xu, Sidi Chen (joint Shanghai Jiao Tong University and Theseus Labs) | SelfImprove-Bench (2611.02797) |
| Shanghai AI Lab / Shanghai AI Laboratory (as printed) | Jie Yang, Jiantao Qiu (SelfImprove-Bench); Safin Team (Safin-1.5 collective byline, no individual affiliation mapping) | SelfImprove-Bench (2611.02797); Safin-1.5 (2610.40751) |
| DeepAutonomy Technology Co., Ltd. | **Mingrui Luo (PI)** (ScienceCUA last and corresponding); Daiqiang Li, Jingyi Peng, Guanren Qiao, Yujia Xiang, Zhiping Xu | ScienceCUA (2611.03238) |
| College of AI, Tsinghua University | Jia Li | ScienceCUA (2611.03238) |
| The Hong Kong Polytechnic University | Xuyang Liu | ScienceCUA (2611.03238) |
| Singapore Management University | **Zhiguang Cao (PI)** (HEMO last author), Jianghan Zhu, Ni Zhang — School of Computing and Information Systems; Qiuzhuang Sun (ScienceCUA) | HEMO (2610.40023); ScienceCUA (2611.03238) |
| Nanyang Technological University, Singapore | **Haoran Luo (PI)** (CogRSI corresponding), **Anh Tuan Luu (PI)** (CogRSI last author, joint VinUniversity), Yandan Zheng; Yifan Yang, Yew Soon Ong — College of Computing and Data Science (HEMO); Zhiguang Han (AEvo, printed Nanyang Technological University) | CogRSI (2611.01113); HEMO (2610.40023); [Harnessing Agentic Evolution (2605.13821)](https://arxiv.org/html/2605.13821v1) |
| VinUniversity, Vietnam | **Anh Tuan Luu (PI)** (CogRSI last author, joint Nanyang Technological University) | CogRSI (2611.01113) |
| AI and Data Team, Keeta, Meituan | **Yongjun Yin (PI)** (CARE last author); Yanbei Jiang (joint Melbourne), Haohang Liu (joint Fudan), Xiaochen Zhao, Yuchen Ge, Runze Nie, Qitian Chen, Qi An, Haolin Song, Yufei Sun, Weijia Chen, Ruiyuan Wu, Zhen Luo, Ke Xing | CARE (2611.02839) |
| University of Melbourne | Yanbei Jiang (joint Keeta, Meituan), Chao Lei, Rena Gao, Yihao Ding, Jey Han Lau, Krista A. Ehinger | CARE (2611.02839) |
| Andreessen Horowitz | Martin Casado | Exo (2611.00463) |
| Braintrust | **Ankur Goyal (PI)** (Exo last author) | Exo (2611.00463) |
| South China University of Technology | Hangming Zhang (SPAR co-first; work during Alibaba Group internship), C. L. Philip Chen, Tong Zhang | SPAR (2611.01695) |
| Alibaba Group — SPAR collaboration | **Tengyue Ma (PI)** (last and corresponding); Yao Zhao (co-first), Haotian Kuang | SPAR (2611.01695) |
| Institute of Automation, Chinese Academy of Sciences / School of Artificial Intelligence, UCAS | Xuanfa Jin, Xinyu Cui, Zhijian Ma, Xue Yan, Haifeng Zhang | EpiAgent (2610.40227) |
| POSTECH | **Wook-Shin Han** (AutoSaddler corresponding); Sungho Park (work during a Microsoft internship) | [AutoSaddler (2608.23041)](https://arxiv.org/pdf/2608.23041), NeurIPS 2026 poster |
| KAIST | Wonjoong Kim (AutoSaddler; work during a Microsoft internship), Chanyoung Park; Jaehyun Nam (joint Google Cloud), Jinwoo Shin (MLE-STAR, printed on title page) | [AutoSaddler (2608.23041)](https://arxiv.org/pdf/2608.23041); [MLE-STAR (2506.15692v3)](https://arxiv.org/pdf/2506.15692v3) |
| Southern University of Science and Technology | Rongyuan Tan (AutoSaddler; work during a Microsoft internship) | [AutoSaddler (2608.23041)](https://arxiv.org/pdf/2608.23041) |
| Microsoft (as printed on AutoSaddler) | **Jue Zhang** (corresponding), **Dongmei Zhang** (last author); Pengfei Gao, Yongqiang Yao, Rao Fu, Elsie Nallipogu, Qingwei Lin, Saravan Rajmohan | [AutoSaddler (2608.23041)](https://arxiv.org/pdf/2608.23041) |
| AWS Forward Deployed Engineering | **Peiyang He** (last and corresponding); Xing Zhang, Guanghui Wang, Yanwei Cui | [Who Verifies the Verifier? (2610.11464)](https://arxiv.org/pdf/2610.11464), NeurIPS 2026 Verify-Agents workshop; inspectable grader and agent-skill co-evolution |
| HSBC Holdings Plc., HSBC Technology Center, China | Ziyuan Li, Wei Qiu, Bing Zhu | [Who Verifies the Verifier? (2610.11464)](https://arxiv.org/pdf/2610.11464) |
| George Washington University | **Tian Lan** (MELD last author); Sizhe Tang, Zuyuan Zhang, Zeyu Fang | [MELD (2610.24454)](https://arxiv.org/html/2610.24454v1), NeurIPS 2026 TTCL workshop poster; hierarchical test-time policy updates, with adapters reset for each task |
| NVIDIA | **Yi Dong** (Agora last author); Yifan Zhang, Yunheng Zou, Shaokun Zhang, Jian Hu, Hao Zhang, Binfeng Xu, Jan Kautz (Agora); Daniel Burkhardt, Niccolò Alberto Elia Venanzi (Red Queen) | [Agora (2609.18094)](https://arxiv.org/html/2609.18094v1), NeurIPS 2026 AutoMLR oral; Red Queen Gödel Machine co |
| The Hong Kong University of Science and Technology (Guangzhou) | **Yuyu Luo** (AEvo last author); Jiayi Zhang, Jianhao Ruan (joint DeepWisdom); Jinyu Xiang | [Harnessing Agentic Evolution / AEvo (2605.13821)](https://arxiv.org/html/2605.13821v1), NeurIPS 2026 |
| DeepWisdom | Jiayi Zhang, Jianhao Ruan (joint The Hong Kong University of Science and Technology (Guangzhou)); Yongfeng Gu, Yiran Peng, Yixi Ouyang, Chenglin Wu | [Harnessing Agentic Evolution / AEvo (2605.13821)](https://arxiv.org/html/2605.13821v1) |
| Singapore University of Technology and Design | Maojia Song | [Harnessing Agentic Evolution / AEvo (2605.13821)](https://arxiv.org/html/2605.13821v1) |
| Shanghai Jiao Tong University — AEvo collaboration | Zhitao Wang | [Harnessing Agentic Evolution / AEvo (2605.13821)](https://arxiv.org/html/2605.13821v1) |
| Université de Montréal & Mila (as printed) | Bang Liu | [Harnessing Agentic Evolution / AEvo (2605.13821)](https://arxiv.org/html/2605.13821v1) |
| Yonsei University | **Jinyoung Yeo** (SHOR last and corresponding); Kai Tzu-iunn Ong, Minseok Kang, Dongwook Choi (co-first); Junhee Cho, Seungju Kim, Seungwon Lim (joint Microsoft Research Asia), Geunha Jang, Minwoo Oh, Bogyung Jeong, Sunghwan Kim | [Towards Direct Evaluation of Harness Optimizers via Priority Ranking / SHOR (2605.22505v1)](https://arxiv.org/pdf/2605.22505v1), NeurIPS 2026 |
| Texas A&M University | Taeyoon Kwon | [Towards Direct Evaluation of Harness Optimizers via Priority Ranking / SHOR (2605.22505v1)](https://arxiv.org/pdf/2605.22505v1) |
| KRAFTON | Kangwook Lee | [Meta-Harness (2603.28052)](https://proceedings.colm.cc/paper_files/paper/2026/file/fde1a69a5b6e554b2f1f727197d2651d-Paper-Conference.pdf), COLM 2026 main |
| Bespoke Labs | Alex Dimakis (joint University of California, Berkeley) | [EvoX (2602.23413)](https://proceedings.colm.cc/paper_files/paper/2026/file/fc9f83d9925e6885e8f1ae1e17b3c44b-Paper-Conference.pdf), COLM 2026 main |
| University of Illinois Urbana-Champaign / UIUC (as printed) | Tianxin Wei, Yuanchen Bei, Xuying Ning, Jiaru Zou, Ting-Wei Li, Xiao Lin, Yanjun Zhao, Jingrui He (Evo-Harness); Tianxin Wei (Harness Updating); Yanjin Li (R²Flow, printed University of Illinois at Urbana-Champaign) | [Evo-Harness](https://aclanthology.org/2026.emnlp-main.969.pdf); [Harness Updating Is Not Harness Benefit](https://aclanthology.org/2026.emnlp-main.2326.pdf), EMNLP 2026 main; [R²Flow (2609.33867v3)](https://arxiv.org/pdf/2609.33867v3), under review at ICLR 2027 |
| Amazon — harness-evolution collaboration | Zhan Shi, Bing He, Yisi Sang, Benoit Dumoulin, Hanqing Lu | [Evo-Harness](https://aclanthology.org/2026.emnlp-main.969.pdf); [Harness Updating Is Not Harness Benefit](https://aclanthology.org/2026.emnlp-main.2326.pdf), EMNLP 2026 main |
| The Pennsylvania State University | Minhua Lin (Evo-Harness and Harness Updating; latter co-first); Zongyu Wu, Zhiwei Zhang, Xiang Zhang, Suhang Wang (Harness Updating) | [Evo-Harness](https://aclanthology.org/2026.emnlp-main.969.pdf); [Harness Updating Is Not Harness Benefit](https://aclanthology.org/2026.emnlp-main.2326.pdf), EMNLP 2026 main |
| Emory University | Zewen Liu | [Evo-Harness](https://aclanthology.org/2026.emnlp-main.969.pdf); [Harness Updating Is Not Harness Benefit](https://aclanthology.org/2026.emnlp-main.2326.pdf), EMNLP 2026 main |
| AG2 AI | Chi Wang | [Evo-Harness](https://aclanthology.org/2026.emnlp-main.969.pdf), EMNLP 2026 main |
| Beihang University — HarnessForge collaboration | Mingju Chen, Can Lv, Shiji Zhou (joint BAAI, corresponding); printed unit: Beijing Advanced Innovation Center for Future Blockchain and Privacy Computing, School of Artificial Intelligence | [HarnessForge](https://aclanthology.org/2026.emnlp-main.1893.pdf), EMNLP 2026 main |
| BAAI | Shiji Zhou (joint Beihang University, HarnessForge corresponding) | [HarnessForge](https://aclanthology.org/2026.emnlp-main.1893.pdf), EMNLP 2026 main |
| MirrorSpace Technology / Simon Fraser University | Yili Jin (both affiliations, Beneath the Diff corresponding) | [Beneath the Diff](https://aclanthology.org/2026.emnlp-main.1029.pdf), EMNLP 2026 main |
| University of Illinois Chicago | Hanrong Zhang, Shicheng Fan (co-first); Henry Peng Zou, Chengze Li, Wei-Chieh Huang, Kening Zheng, Philip S. Yu | [CoEvoSkills (2604.01687v2)](https://arxiv.org/pdf/2604.01687v2), COLM 2026 main |
| Columbia University — CoEvoSkills collaboration | Jiayu Zhou | [CoEvoSkills (2604.01687v2)](https://arxiv.org/pdf/2604.01687v2), COLM 2026 main |
| The Chinese University of Hong Kong, Shenzhen | Mingda Zhang, Xiaoying Tang (R²Flow and CollabFlow); Hanwen Zhang, Yue Dai (CollabFlow); Mingda Zhang is CollabFlow co-first | [R²Flow (2609.33867v3)](https://arxiv.org/pdf/2609.33867v3); [CollabFlow (2609.38662v3)](https://arxiv.org/pdf/2609.38662v3), under review at ICLR 2027 |
| University of Oxford — flow-based agent collaboration | Zijia Wang | [R²Flow (2609.33867v3)](https://arxiv.org/pdf/2609.33867v3); [CollabFlow (2609.38662v3)](https://arxiv.org/pdf/2609.38662v3), under review at ICLR 2027 |
| Shanghai Jiao Tong University — CollabFlow collaboration | Xiao Huang (co-first), Junming Zhang | [CollabFlow (2609.38662v3)](https://arxiv.org/pdf/2609.38662v3), under review at ICLR 2027 |
| Substrate maintainers (no affiliation printed on the artifact) | Keller Jordan (modded-nanoGPT, Muon); Andrej Karpathy (nanoGPT, autoresearch) | Repos, not papers — the cheap execution-grounded task nearly every RSI result is measured on |
| Unaffiliated | Sungmin Lee (no institution printed) | LLMs Have Made Failure Worth Publishing (2604.06236) |

Co-affiliations printed on a paper but holding no lead or senior slot: Inria
(Red Queen), Taptap (ASI-Arch), University of Sydney SAIC (CausalEvolve), Hangzhou Dianzi /
NTU Singapore / Fudan (Safety), USTC (AutoSOTA), Central South University (Denominator
Gaming), University of Arizona (Gödel Agent), Stanford Medicine / Houston Methodist /
Sequome / Imperial (AI co-scientist), AlphaAvatar (RSI survey), and Thoughtful Lab
(PostTrainBench). Institutions that reach this field only through the ICLR 2026 RSI
workshop program — Apple, Tencent, Scale, Anuttacon, NYU, Meta
Reality Labs, Physical Intelligence, Ohio State, You.com, UC Merced — are
deliberately absent. Organizing, keynoting or sitting on a panel is not research output,
and a program page is not a title page; the roster lives in the workshop guide instead.

# Major academic lineage (who is advisor of who)

Not yet researched. Edges to establish first, in rough order of how much they explain:

- Jürgen Schmidhuber (IDSIA/KAUST) → Mingchen Zhuge (KAUST) — the Gödel-machine lineage into the dedicated RSI workshop. Co-authorship already establishes the cluster: Zhuge, Wenyi Wang, Francesco Faccio, Dmitrii Khizbullin, Dylan Ashley and Louis Kirsch all appear with Schmidhuber on GPTSwarm and/or Agent-as-a-Judge; which of those are his students is the open part
- Jeff Clune (UBC) → Jenny Zhang, Shengran Hu, Cong Lu, Yiming Xiong — and the Sakana overlap (Hu, Lu, Lange all on DGM). Hu is on both DGM and ALMA, so he is the through-line of the Clune archive lineage
- Sanjeev Arora (Princeton PLI) → Yun Cheng, Xingyu Zhu, Haoyu Zhao — the Contextual Drag group
- Matthias Bethge (Tübingen) and Maksym Andriushchenko (ELLIS/MPI-IS) → Ben Rank, Hardik Bhatnagar, Ameya Prabhu — the PostTrainBench group; also, where Andriushchenko himself trained
- William Yang Wang (UCSB) → Xin Eric Wang? — would explain two independent UCSB RSI labs
- Kun Zhang (CMU/MBZUAI) → Yongqiang Chen, Bo Han, Tongliang Liu — the causal branch
- Tatsunori Hashimoto (Stanford) → Chenglei Si — the execution-grounded branch
- Pengfei Liu (SJTU/SII/GAIR) — where the ASI-Arch group came from
- Zhengyao Jiang, Yuxiang Wu (Weco founders) — where the Weco team trained
- Yoram Bachrach, Jakob Foerster (Meta) — the AIRA/speedrun program's students

# Appendix (rank by paper importance)

| # | Paper | Author & affiliation pairs |
|---|---|---|
| 1 | AIDE² (2609.26457) | Dhruv Srikanth, Bingchen Zhao, Dixing Xu, Yuxiang Wu, Zhengyao Jiang — all Weco AI |
| 2 | MASS (2610.12176) | Hyunin Lee, Donghyun Lee, Somayeh Sojoudi, Matei Zaharia — UC Berkeley; Jinglue Xu, Jeffrey Seely, Yujin Tang — Sakana AI |
| 3 | Exo (2611.00463) | Alexander Krentsel (corresponding), Sylvia Ratnasamy, Ion Stoica — University of California, Berkeley; Martin Casado — Andreessen Horowitz; Ankur Goyal — Braintrust |
| 4 | Automated Researchers Can Mitigate Well-characterized Alignment Failures (2608.28945) | Chen Yueh-Han — Anthropic Fellows Program + Anthropic; Jiaxin Wen — UC Berkeley; Jan Hendrik Kirchner — Anthropic |
| 5 | Auditing the Product, Not the Recursion (2610.27610) | Zihan Chen, Di Zhu — Stevens Institute of Technology; Lei Nico Zheng — University of Massachusetts Boston |
| 6 | False Frontiers (2609.39102) | Meijia Chen — Rutgers; Hao Li, Zheng Lu, Hongshan Lin, Junbai Tian, Zijun Tian, Yufan Zou, Shuhan Sun, Zeyu Zhang, Yueting Li — Independent Researcher; Yichen Liu, Hanxin Chen — UC San Diego; Weizhi Du — University of Michigan; Tianyu Shi — McGill; Alaa Khamis — King Fahd University of Petroleum and Minerals |
| 7 | SelfImprove-Bench (2611.02797) | Bangrui Xu, Sidi Chen — Shanghai Jiao Tong University + Theseus Labs + Humanlaya; Jie Yang — Shanghai Jiao Tong University + Theseus Labs + Shanghai AI Lab; Xuanhe Zhou (corresponding), Fan Wu — Shanghai Jiao Tong University + Theseus Labs; Jiantao Qiu — Shanghai AI Lab; Qianyu Yang, Zijie Wang, Zhoufutu Wen (corresponding) — Humanlaya. Xu, Yang and Chen are co-first authors |
| 8 | CogRSI (2611.01113) | Yandan Zheng, Haoran Luo (corresponding) — Nanyang Technological University, Singapore; Xinliang Zhou — Stanford University; Anh Tuan Luu — Nanyang Technological University, Singapore + VinUniversity, Vietnam |
| 9 | S3Gym (2608.31100) | ByteDance Seed, M-A-P, TokenWave.AI printed collectively; core contributors Jiajun Shi, Siyuan Tao, Yuhao Wu, Zexuan Wang, Jingyuan Zhang have no individual affiliation mapping in the paper |
| 10 | Recursive Flint (October 2026 model release) | Recursive Superintelligence, Inc. — corporate model-card attribution; no individual paper byline |
| 11 | Safin-1.5 (2610.40751) | Safin Team — Shanghai AI Laboratory; collective title-page byline, with no individual author–affiliation mapping |
| 12 | SPADE (2608.19197) | Bo Liu — Washington + Stanford; Simon Yu, Weiyan Shi — Northeastern; Yiding Jiang, Seungone Kim — CMU; Ao Qu, Paul Pu Liang — MIT; Zichen Liu, Zijian Zhou — NUS; Junsu Kim — SNU; Mickel Liu, Luke Zettlemoyer, Natasha Jaques — Washington; Hanfei Yu — Stevens; Zhaorun Chen — Chicago; Yejin Choi — Stanford; Andrew Zhao, Tongzheng Ren — no affiliation printed |
| 13 | CARE (2611.02839) | Yanbei Jiang — AI and Data Team, Keeta, Meituan + University of Melbourne; Haohang Liu — AI and Data Team, Keeta, Meituan + Fudan University; Xiaochen Zhao, Yuchen Ge, Runze Nie, Qitian Chen, Qi An, Haolin Song, Yufei Sun, Weijia Chen, Ruiyuan Wu, Zhen Luo, Ke Xing, Yongjun Yin — AI and Data Team, Keeta, Meituan; Chao Lei, Rena Gao, Yihao Ding, Jey Han Lau, Krista A. Ehinger — University of Melbourne; Xingjun Ma — Fudan University |
| 14 | [RSI-Dojo (2611.03698)](https://arxiv.org/pdf/2611.03698) | Shuzheng Gao (corresponding), Michael R. Lyu — Department of Computer Science and Engineering, The Chinese University of Hong Kong; Zhaojian Yu, Kai Cai, Shilin He (corresponding) — ByteDance; Penghao Yin — Tsinghua University; Yuling Shi — Shanghai Jiao Tong University. Affiliations printed on PDF page 1 |
| 15 | Dream-RSI (2609.14858) | Tong Zheng — Google + Maryland; Xidong Wu, Zheng Zhang, Chaoyi Zhang, Ruoqiao Wei, Xue Wang, Yue Zhuan, Renkai Xiang, Xinwu Cheng, Yunsong Guo — Google; Zhankui He, Benjamin Coleman, Di Bai, Wang-Cheng Kang — Google Deepmind; Rui Liu, Heng Huang — Maryland; Haolin Liu — Virginia |
| 16 | HEMO (2610.40023) | Jianghan Zhu, Ni Zhang, Zhiguang Cao — School of Computing and Information Systems, Singapore Management University; Yifan Yang, Yew Soon Ong — College of Computing and Data Science, Nanyang Technological University; Cong Zhang — independent researcher, with prior Nanyang Technological University, Huawei Noah’s Ark Lab and TikTok Singapore affiliations explicitly printed |
| 17 | ScienceCUA (2611.03238) | Daiqiang Li, Jingyi Peng, Guanren Qiao, Yujia Xiang, Zhiping Xu, Mingrui Luo (corresponding) — DeepAutonomy Technology Co., Ltd.; Rui Hu — Peking University; Jia Li — College of AI, Tsinghua University; Xuyang Liu — The Hong Kong Polytechnic University; Qiuzhuang Sun — Singapore Management University |
| 18 | [ScientistTwo (2609.19644)](https://arxiv.org/pdf/2609.19644) | Jaehyun Nam and Jinsung Yoon (corresponding), Yanzhou Pan, Rui Meng, Parthasarathy Ranganathan, Tomas Pfister — Google Cloud AI Research; Yubo Wang — University of Waterloo; title-page affiliations |
| 19 | RSI-Env (2611.01736) | Wanying Qu, Jiyao Liu — Fudan; Zhiyuan Fan — HKUST; Jiang Zhou — Georgia Tech; Jianghan Shen — Nanjing University + Shanghai Artificial Intelligence Laboratory; Yanwei Fu — Fudan + Shanghai Innovation Institute; Dongrui Liu — Shanghai Artificial Intelligence Laboratory |
| 20 | [Meta^n (2608.24735)](https://arxiv.org/html/2608.24735v1) — [NeurIPS 2026 poster](https://neurips.cc/virtual/2026/poster/151012) | Zae Myung Kim, Young-Jun Lee, Dongyeop Kang — University of Minnesota; Seungyeon Jwa — Seoul National University |
| 21 | [AutoSaddler (2608.23041)](https://arxiv.org/pdf/2608.23041) — NeurIPS 2026 poster | Sungho Park, Wook-Shin Han (corresponding) — POSTECH; Wonjoong Kim, Chanyoung Park — KAIST; Rongyuan Tan — Southern University of Science and Technology; Jue Zhang (corresponding), Pengfei Gao, Yongqiang Yao, Rao Fu, Elsie Nallipogu, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang — Microsoft. Park, Kim and Tan did this work during Microsoft internships |
| 22 | [Who Verifies the Verifier? Co-Evolving Inspectable Graders with Self-Improving Agents (2610.11464)](https://arxiv.org/pdf/2610.11464) — NeurIPS 2026 Verify-Agents workshop | Xing Zhang, Guanghui Wang, Yanwei Cui, Peiyang He (corresponding) — AWS Forward Deployed Engineering; Ziyuan Li, Wei Qiu, Bing Zhu — HSBC Holdings Plc., HSBC Technology Center, China |
| 23 | [MELD: Recursive Self-Improvement via Hierarchical Coordination for Multi-Agent Scientific Discovery (2610.24454)](https://arxiv.org/html/2610.24454v1) — NeurIPS 2026 TTCL workshop poster | Sizhe Tang, Zuyuan Zhang, Zeyu Fang, Tian Lan — George Washington University |
| 24 | [Agora: Git as Shared Memory for Collective AutoResearch (2609.18094)](https://arxiv.org/html/2609.18094v1) — NeurIPS 2026 AutoMLR oral | Yifan Zhang, Yunheng Zou, Shaokun Zhang, Jian Hu, Hao Zhang, Binfeng Xu, Jan Kautz, Yi Dong — NVIDIA |
| 25 | [Heuresis (2606.25198)](https://arxiv.org/html/2606.25198v1) — NeurIPS 2026 main-track poster, Evaluating Search Strategies for Autonomous Machine Learning Research Agents | Antonis Antoniades, Deepak Nathani, Ritam Saha, Alfonso Amayuelas, Ivan Bercovich, Zhaotian Weng, William Yang Wang — University of California, Santa Barbara; Vignesh Baskaran, Kunal Bhatia — Hexo AI. Printed preprint title: Search Strategies for Autonomous AI Research Agents Across Quality, Diversity and Novelty |
| 26 | [Harnessing Agentic Evolution / AEvo (2605.13821)](https://arxiv.org/html/2605.13821v1) — NeurIPS 2026 | Jiayi Zhang, Jianhao Ruan — The Hong Kong University of Science and Technology (Guangzhou) + DeepWisdom; Yongfeng Gu, Yiran Peng, Yixi Ouyang, Chenglin Wu — DeepWisdom; Maojia Song — Singapore University of Technology and Design; Zhiguang Han — Nanyang Technological University; Jinyu Xiang, Yuyu Luo — The Hong Kong University of Science and Technology (Guangzhou); Zhitao Wang — Shanghai Jiao Tong University; Caiyin Yang — Tsinghua University; Bang Liu — Université de Montréal & Mila |
| 27 | [Towards Direct Evaluation of Harness Optimizers via Priority Ranking / SHOR (2605.22505v1)](https://arxiv.org/pdf/2605.22505v1) — NeurIPS 2026 | Kai Tzu-iunn Ong, Minseok Kang, Dongwook Choi (co-first), Junhee Cho, Seungju Kim, Geunha Jang, Minwoo Oh, Bogyung Jeong, Sunghwan Kim, Jinyoung Yeo (corresponding) — Yonsei University; Seungwon Lim — Yonsei University + Microsoft Research Asia; Taeyoon Kwon — Texas A&M University. Author mapping is from the printed v1 title page |
| 28 | [Self-Trained Verification for Training- and Test-Time Self-Improvement / STV (2605.30290)](https://arxiv.org/html/2605.30290v1) — NeurIPS 2026 | Chen Henry Wu, Aditi Raghunathan — Carnegie Mellon University |
| 29 | [Meta-Harness: End-to-End Optimization of Model Harnesses (2603.28052)](https://proceedings.colm.cc/paper_files/paper/2026/file/fde1a69a5b6e554b2f1f727197d2651d-Paper-Conference.pdf) — COLM 2026 main | Yoonho Lee, Roshen Nair, Qizheng Zhang, Chelsea Finn — Stanford; Kangwook Lee — KRAFTON; Omar Khattab — MIT |
| 30 | [EvoX: Meta-Evolution for Automated Discovery (2602.23413)](https://proceedings.colm.cc/paper_files/paper/2026/file/fc9f83d9925e6885e8f1ae1e17b3c44b-Paper-Conference.pdf) — COLM 2026 main | Shu Liu, Shubham Agarwal (equal contribution), Monishwaran Maheswaran, Mert Cemri, Qiuyang Mang, Zhifei Li, Ashwin Naren, Audrey Cheng, Alexander Du, Melissa Pan, Kurt Keutzer, Alvin Cheung, Koushik Sen, Matei Zaharia, Ion Stoica — University of California, Berkeley; Ethan Boneh — Stanford University; Alex Dimakis — University of California, Berkeley + Bespoke Labs |
| 31 | [Evo-Harness: Context-to-Harness Skill Compilation for Self-Evolving Agents (2608.15071)](https://aclanthology.org/2026.emnlp-main.969.pdf) — EMNLP 2026 main | Tianxin Wei, Yuanchen Bei, Xuying Ning, Jiaru Zou, Ting-Wei Li, Xiao Lin, Yanjun Zhao, Jingrui He — University of Illinois Urbana-Champaign; Zhan Shi, Bing He, Yisi Sang, Benoit Dumoulin, Hanqing Lu — Amazon; Minhua Lin — The Pennsylvania State University; Zewen Liu — Emory University; Chi Wang — AG2 AI; Dakuo Wang — Northeastern University |
| 32 | [HarnessForge: Joint Harness and Policy Evolution for Adaptive Agent Systems (2606.01779)](https://aclanthology.org/2026.emnlp-main.1893.pdf) — EMNLP 2026 main | Mingju Chen, Can Lv — Beihang University; Guibin Zhang — National University of Singapore; Heng Chang (Project Lead) — Tsinghua University; Shiji Zhou (corresponding) — Beihang University + BAAI |
| 33 | [OASIS: Online Anchor-calibrated Self-Improvement System](https://aclanthology.org/2026.findings-emnlp.2150.pdf) — Findings EMNLP 2026 | Wenyi Wang, Firas Laakom, Li Nanbo, Yanning Dai, Yuhui Wang, Yifan Xu, Dmitrii Khizbullin, Piotr Piękos, Yilan Zhang, Yimeng Chen, Mingchen Zhuge — KAUST; Alfonso Amayuelas — University of California, Santa Barbara; Jürgen Schmidhuber — KAUST + IDSIA + USI + SUPSI. Wenyi Wang and Yuhui Wang are corresponding authors |
| 34 | [Beneath the Diff: Diagnosing and Mitigating Algorithmic Mode Collapse in Code-Level Autonomous Research Loops (2609.00077)](https://aclanthology.org/2026.emnlp-main.1029.pdf) — EMNLP 2026 main | Bowei He, Xue Liu — MBZUAI + McGill University; Weixu Zhang — McGill University; Yili Jin (corresponding) — MirrorSpace Technology + Simon Fraser University |
| 35 | [Harness Updating Is Not Harness Benefit: Disentangling Evolution Capabilities in Self-Evolving LLM Agents (2605.30621)](https://aclanthology.org/2026.emnlp-main.2326.pdf) — EMNLP 2026 main | Minhua Lin (co-first), Zongyu Wu, Zhiwei Zhang, Xiang Zhang, Suhang Wang — The Pennsylvania State University; Juncheng Wu (co-first), Zijun Wang, Cihang Xie, Yuyin Zhou — UC Santa Cruz; Zhan Shi, Yisi Sang, Bing He, Benoit Dumoulin, Hanqing Lu — Amazon; Zewen Liu — Emory University; Tianxin Wei — UIUC; Dakuo Wang — Northeastern University |
| 36 | [CoEvoSkills: Self-Evolving Agent Skills via Co-Evolutionary Verification (2604.01687v2)](https://arxiv.org/pdf/2604.01687v2) — COLM 2026 main | Hanrong Zhang, Shicheng Fan (co-first), Henry Peng Zou, Chengze Li, Wei-Chieh Huang, Kening Zheng, Philip S. Yu — University of Illinois Chicago; Yankai Chen (corresponding), Xue (Steve) Liu — MBZUAI + McGill University; Zhenting Wang — MBZUAI; Jiayu Zhou — Columbia University; Yifei Yao — Zhejiang University; Xiaoxiao Li — University of British Columbia |
| 37 | [R²Flow: Recursive Self-Improvement via Recursive Skill Evolution (2609.33867v3)](https://arxiv.org/pdf/2609.33867v3) — under review at ICLR 2027 | Mingda Zhang, Xiaoying Tang — The Chinese University of Hong Kong, Shenzhen; Qiang Huang — Fudan University; Yanjin Li — University of Illinois at Urbana-Champaign; Zijia Wang — University of Oxford; Qika Lin, Tiesunlong Shen — National University of Singapore. Author mapping from the printed arXiv v3 title page |
| 38 | [CollabFlow: Recursive Self-Improvement of Agent Collaboration (2609.38662v3)](https://arxiv.org/pdf/2609.38662v3) — under review at ICLR 2027 | Xiao Huang (co-first), Junming Zhang — Shanghai Jiao Tong University; Mingda Zhang (co-first), Hanwen Zhang, Yue Dai, Xiaoying Tang — The Chinese University of Hong Kong, Shenzhen; Qiang Huang — Fudan University; Zijia Wang — University of Oxford. Author mapping from the printed arXiv v3 title page |
| 39 | RSI-Anything (2610.13912) | Taoyong Cui — PhAI Labs + CUHK; Pinxian Zeng — Macau; Wanghan Xu — SJTU; Hao Xiao, Qiang Gao — Fudan; Shuhan Xue — PhAI Labs; Wanli Ouyang, Pheng Ann Heng — CUHK; Zhenfei Yin — PhAI Labs + Oxford; Yingcheng Wu — PhAI Labs + Stanford; Ling Yang — PhAI Labs + PKU. Mapping printed in Appendix A, referenced on the title page |
| 40 | SPAR (2611.01695) | Hangming Zhang, C. L. Philip Chen, Tong Zhang — South China University of Technology; Yao Zhao, Haotian Kuang, Tengyue Ma (corresponding) — Alibaba Group. Hangming Zhang and Yao Zhao are co-first; Hangming’s work was during an Alibaba Group internship |
| 41 | EpiAgent (2610.40227) | Xuanfa Jin, Xinyu Cui, Zhijian Ma, Xue Yan, Haifeng Zhang — Institute of Automation, Chinese Academy of Sciences + School of Artificial Intelligence, UCAS; Jian Zhao — Zhongguancun Academy; Jun Wang (corresponding) — AI Centre, Department of Computer Science, UCL |
| 42 | Reward Hacking in Self-Improving Code Agents (OpenReview ikrQWGgxYg) | Bingchen Zhao · Dhruv Srikanth · Yuxiang Wu · Zhengyao Jiang — **all Weco AI (inferred, title page unreadable)** |
| 43 | RSI survey, 1,250 papers (2607.07663) | Mingguang Chen — UC Riverside (corresponding); Licheng Wang — AlphaAvatar; Bo Qu — Illinois Institute of Technology |
| 44 | Automated Weak-to-Strong Researcher (Anthropic) | Jiaxin Wen, Joe Benton, Jan Hendrik Kirchner, Jan Leike — Anthropic; Liang Qiu — Anthropic + Anthropic Fellows Program |
| 45 | ShinkaEvolve (2509.19349) | Robert Tjarko Lange · Yuki Imajuku · Edoardo Cetin — all Sakana AI |
| 46 | PostTrainBench (2603.08640) — workshop oral | Ben Rank, Maksym Andriushchenko (corresponding) — ELLIS Institute Tübingen + MPI for Intelligent Systems + Tübingen AI Center; Hardik Bhatnagar, Ameya Prabhu, Matthias Bethge — Univ. Tübingen + Tübingen AI Center; Shira Eisenberg (internship), Nguyen Karina — Thoughtful Lab |
| 47 | Red Queen Gödel Machine (2606.26294) | Iacob, Shen, Kurmanji, Cao, Marino, Qiu — Cambridge; Jovanović, Sani, Lane — Cambridge + Flower Labs; Burkhardt, Venanzi — NVIDIA; Tastan — MBZUAI; Odonnat — Inria |
| 48 | SLDAgent (2507.21184) | Haowei Lin — PKU + Wizard Quant; Haotian Ye, James Zou — Stanford; Feng, Huang, Y. Li — Wizard Quant; Jianzhu Ma — Tsinghua; Liang, Lim, Z. Li, X. Wang — PKU |
| 49 | SpecBench (2605.21384) | Bingchen Zhao · Dhruv Srikanth · Yuxiang Wu · Zhengyao Jiang — all Weco AI |
| 50 | Diversity Collapse in MAS (2604.18005) | N. Chen, Tong, Y. He, Zou, Q. Wang, B. He — NUS; Y. Yang, X. Zhang — CUHK-Shenzhen |
| 51 | AutoSOTA (2604.05550) | Yu Li, X. Liu, R. Zhao, P. Liu, Z. Chen, Q. Yang, Zeng, T. Li, J. Xu — Tsinghua EE/BNRist; Shao, Su, Fengli Xu, Yong Li — Tsinghua + Zhongguancun; A. Xu — Zhongguancun + PKU; Fang — Zhongguancun + USTC; Tie-Yan Liu — Zhongguancun |
| 52 | ALMA / Meta-learning Agentic Memory Designs (2602.07755) — workshop oral | Yiming Xiong — UBC; Shengran Hu — UBC + Vector Institute; Jeff Clune — UBC + Vector Institute + Canada CIFAR AI Chair |
| 53 | [Group-Evolving Agents (2602.04837)](https://arxiv.org/html/2602.04837v1) — [COLM 2026 main](https://proceedings.colm.cc/paper_files/paper/2026/hash/87b321515dc33137ffcc5d1654b7d182-Abstract-Conference.html) | Zhaotian Weng, Antonis Antoniades, Deepak Nathani, Zhen Zhang, Xiao Pu, Xin Eric Wang — University of California, Santa Barbara, as printed in arXiv v1. Proceedings metadata uses Sophia Pu and Eric Wang; no identity equivalence is inferred |
| 54 | Safety in Self-Evolving Agents (2606.23075) | R. Lin, Q. Li, Z. Li, Shouling Ji — Zhejiang; Deng, Qing, Q. Li, Ke Xu — Tsinghua; Feng, Cui, Meng — Ant Group; Ma — Hangzhou Dianzi; Y. Zhang, T. Zhang — NTU; X. Ma — Fudan |
| 55 | Barriers to Diversity in LLM Ideas (2602.20408) | Yuting Deng · Melanie Brucks · Olivier Toubia — all Columbia Business School |
| 56 | Contextual Drag (2602.04288) | Yun Cheng, Xingyu Zhu (both corresponding), Haoyu Zhao, Sanjeev Arora — all Princeton Language and Intelligence, Princeton University |
| 57 | Dead Science Walking (2606.04220) | Kargi Chauhan — UC Santa Cruz (sole author) |
| 58 | HypoArena / Before the Action (2607.15766) | Zhong, W. Jiang — UCAS + ISCAS; X. Chen, Lu, H. Lin, Sun, Xianpei Han — ISCAS; W. Wang, Shi, B. Yang, J. Wang, H. Li, Zhai, B. Zhao, Hu Wei, H. Yu, Yongbin Li — Alibaba Group |
| 59 | ForeSci (2606.00644) | Qiuyu Tian — Southeast Univ. + Zhongguancun; Haojie Yin — Duke Kunshan; Yingce Xia, Zequn Liu (corresponding) — Zhongguancun; Youyong Kong — Southeast Univ. |
| 60 | CausalEvolve (2603.14575) | Yongqiang Chen — MBZUAI + CMU; Chenxi Liu, Bo Han — HKBU TMLR; Zhenhao Chen — MBZUAI; Tongliang Liu — Univ. Sydney SAIC + MBZUAI; Kun Zhang — MBZUAI + CMU |
| 61 | On the Limits of Self-Improving (2601.05280) | Hector Zenil — King's College London + Oxford Immune Algorithmics (sole author) |
| 62 | ArchEval (2607.03601) | C. Wang, Wan, Ma, Prakash, Qi, Do, Cheng, Tschand, Du, Reddi — Harvard; Shi — MIT |
| 63 | ASI-Arch (2507.18074) | Yixiu Liu, Weixian Xu, Lyumanshan Ye — SJTU + SII + GAIR; Yang Nan, Xiangkun Hu — SII + GAIR; Zhen Qin — Taptap; Pengfei Liu (corresponding) — SJTU + SII + GAIR |
| 64 | AC/DC (2604.14969) | Andrew Dai · Boris Meinardus · Ciaran Regan · Yingtao Tian · Yujin Tang — all Sakana AI (ICLR 2026) |
| 65 | Delta-Based NAS (2605.04903) | Santosh Premi Adhikari · Radu Timofte · Dmitry Ignatov — all Computer Vision Lab, CAIDAS & IFI, Univ. Würzburg |
| 66 | Learning to Evolve (OpenReview WnZHbe1Gu0) | Xuan Li, Zhanke Zhou, Zongze Li, Bo Han — HKBU; Jiangchao Yao — SJTU *(mapping from HKBU repository, not the PDF)* |
| 67 | Evolutionary Discovery of RL Algorithms (2603.28416) | Alkis Sygkounas · Amy Loutfi · Andreas Persson — all Machine Perception and Interaction Lab, Örebro |
| 68 | SIA (2605.27276) | Hebbar, Manawat, Bhatia, Baskaran — Hexo Labs Palo Alto; Verboomen — Hexo Labs Brussels; Palanimalai — Hexo Labs Toronto; Ivanova — Univ. Oxford |
| 69 | Denominator Gaming (2605.09915) | Shan, Zheng, Xi, Zhu, Yu, W. Zhang, J. Lin — SJTU; Gao — Central South Univ.; Zeyu Zheng — CMU |
| 70 | Test-time Recursive Thinking (2602.03094) | Zhuang — MSR + UCSD; Singh, L. Liu, Shen, D. Zhang, Gao, W. Chen — Microsoft Research; Shang — UC San Diego |
| 71 | Combinatorial Innovation (2604.20548) | Shuai Chen · Chengzhi Zhang (corresponding) — Nanjing Univ. of Science and Technology |
| 72 | LLMs Have Made Failure Worth Publishing (2604.06236) | Sungmin Lee — no affiliation printed |
| 73 | DARWIN (2602.05848) | Henry Jiang — Georgia Tech, College of Computing (sole author) |
| — | [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) — November 2, 2026 research post | Rikiya Takehi and Jayden Teoh (co-first), Alex Zhang, Omar Khattab — all MIT CSAIL, explicitly listed in the original post's author block; no separate paper PDF |
| — | METR Time Horizon 1.1 | METR — corporate authorship, no individual bylines |
| — | Pangram ICLR 2026 review census | Bradley Emi — Pangram Labs (sole byline) |
| — | Verifying the Verifiers (OpenReview iRhaK8PsuB) | Authors unresolved — OpenReview bot wall |
| — | SAHOO, safeguarded alignment for RSI (OpenReview OAFPpQO0H9) | Authors unresolved — OpenReview bot wall |
| — | TamperBench (OpenReview smLtz7WID0) | Authors unresolved — OpenReview bot wall |
| — | 25 further accepted works named in the workshop guide | Titles only, no forum ID and no authors: POLARIS, SkillRL, Self-Adapting Agents for Research Coding, OMEGA, LLM-FE, CircuitBuilder, Simple Baselines vs Code Evolution, Reasoning as Gradient, Can Language Models Discover Scaling Laws?, Language Self-Play, SAGE, GASP, Anchored Self-Play, Compute as Teacher, Self-Play is secretly Adversarial Imitation, SimpleMem, Agentic Context Engineering, Test-Time Self-Distillation, Adaptive Meta-Curriculum, TextBO, Self-Improvement via Fast Tree-search, Self-Improving World Models, VLAW, RFTF, Interestingness as an Inductive Heuristic |

Background anchors (covered in companion files):

| Paper | Author & affiliation pairs |
|---|---|
| Gödel machine (2003) | Jürgen Schmidhuber — the formal ancestor; provably-optimal self-rewriting, computationally intractable, never built |
| AI-GAs (1905.10985) | Jeff Clune — Uber AI Labs + University of Wyoming (sole author, as printed in 2019) |
| GPTSwarm (2402.16823, ICML 2024) | Mingchen Zhuge, Wenyi Wang (co-first), Dmitrii Khizbullin — AI Initiative, KAUST; Louis Kirsch — IDSIA (USI/SUPSI); Francesco Faccio, Jürgen Schmidhuber — KAUST + IDSIA |
| Agent-as-a-Judge (2410.10934) | Mingchen Zhuge — Meta AI + KAUST; Changsheng Zhao, Yunyang Xiong, Zechun Liu, Ernie Chang, Raghuraman Krishnamoorthi, Yuandong Tian, Yangyang Shi, Vikas Chandra — Meta AI; Dylan R. Ashley, Wenyi Wang, Dmitrii Khizbullin, Jürgen Schmidhuber — KAUST |
| modded-nanoGPT + Muon | Keller Jordan — repo, no affiliation printed; GPT-2 124M to 3.28 val loss on 8×H100, ~45 min → ~1.3 min over ~20 community gains |
| karpathy/autoresearch | Andrej Karpathy — repo, no affiliation printed; agents edit train.py on single-GPU nanochat, scored on val bits-per-byte |
| Darwin Gödel Machine (2505.22954) | Jenny Zhang — UBC + Vector; Shengran Hu, Cong Lu — UBC + Vector + Sakana AI; Robert Lange — Sakana AI; Jeff Clune — UBC + Vector + Canada CIFAR AI Chair |
| AlphaEvolve (2506.13131) | Alexander Novikov … Matej Balog — all Google DeepMind |
| Gödel Agent (2410.04444) | Xunjian Yin — UCSB + PKU (internship at UCSB); Xinyi Wang, William Yang Wang — UCSB; Liangming Pan — Univ. Arizona; Xiaojun Wan — PKU |
| ScientistOne (2605.26340) | Rui Meng, Bhavana Dalvi Mishra, Jiefeng Chen, Chun-Liang Li, Palash Goyal, Mihir Parmar, Yiwen Song, Yale Song, Rajarishi Sinha, Parthasarathy Ranganathan, Burak Gokturk, Jinsung Yoon, Tomas Pfister — all Google Cloud AI Research, as printed in the author block |
| PaperOrchestra (2604.05018) | Yiwen Song, Yale Song, Tomas Pfister, Jinsung Yoon — Google, as printed on PDF page 1 |
| ScholarPeer (2601.22638v2) | Palash Goyal, Mihir Parmar, Yiwen Song, Hamid Palangi, Tomas Pfister, Jinsung Yoon — Google, as printed in its author block |
| MLE-STAR (2506.15692v3) | Jaehyun Nam — Google Cloud + KAIST; Jinsung Yoon, Jiefeng Chen, Sercan Ö. Arık, Tomas Pfister — Google Cloud; Jinwoo Shin — KAIST, as printed on PDF page 1 |
| AI co-scientist (2502.18864) | Gottweis et al. — Google Cloud AI Research; Weng, Tu, Hassabis, Kohli, Natarajan et al. — Google DeepMind; Popovici, Palepu et al. — Google Research; plus Stanford Medicine, Houston Methodist, Sequome, Imperial (51 authors) |
| Robin (2505.13400) | Ghareeb, Mitchener, Yiu, Szostkiewicz, Laurent, Razzak, White, Hinks, Rodriques — FutureHouse; Benjamin Chang — FutureHouse + Oxford |
| Speedrunning Benchmark (2506.22419) | Bingchen Zhao — Meta + Univ. Edinburgh; Despoina Magka, Minqi Jiang … Yoram Bachrach — Meta; Oisin Mac Aodha — Univ. Edinburgh |
| Agent0 (2511.16043) | Peng Xia, Zeng, J. Liu, Zhou, Huaxiu Yao — UNC-Chapel Hill; Qin, Xiong — Salesforce Research; Fang Wu — Stanford |
| AIRA (2507.02554) | Toledo, Hambardzumyan — FAIR at Meta + UCL; Josifoski, Baldwin, Magka, M. Jiang, Raileanu, Cancedda, Foerster, Bachrach et al. — FAIR at Meta; Hazra — Örebro (work done while at Meta); Stenetorp — UCL |
| Execution-Grounded Auto-Research (2601.14525) | Chenglei Si, Zitong Yang, Yejin Choi, Emmanuel Candès, Diyi Yang, Tatsunori Hashimoto — all Stanford |
| AutoTrainess (2606.31551) | Zhaojian Yu, Penghao Yin, Shuzheng Gao, Shilin He, Kai Cai, Xiao-Ping Zhang — title page collectively lists Tsinghua University, The Chinese University of Hong Kong, and Simple Agent Lab; no individual author–institution mapping printed |
| MLE-bench (2410.07095) | Chan Jun Shern, Chowdhury, Jaffe, Aung, Sherburn, Mays, Starace, K. Liu, Maksin, Patwardhan, Weng, Madry — all OpenAI |
