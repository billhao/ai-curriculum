This doc captures major research groups and people in RSI, AI for science etc.

Affiliations are as printed on the paper (arXiv HTML author block or PDF page 1). Bare
acronyms are kept bare. Blank cells mean not yet researched, not "none". Source maps:
[`rsi-proposer-landscape-2026.md`](rsi-proposer-landscape-2026.md) and
[`iclr-2026-rsi-workshop.md`](iclr-2026-rsi-workshop.md).

One exception to the printed-on-the-paper rule, marked wherever it applies: the workshop
program roster (organizers, keynotes, panel) has no title page. Those affiliations come
from the workshop program page as relayed by the guide — one source removed from a paper,
and a *current* affiliation rather than an at-publication one. Treated as weaker evidence.

**(PI)** marks the group lead, listed first in each table. Assigned only from printed
evidence — the last author or the marked corresponding author of that group's work here. No
mark means the senior slot isn't established from the papers, not that the group has no
lead. Unmarked on purpose: UC Riverside (first author is also corresponding), Edinburgh and
UCL (senior only within a Meta-led paper), SJTU Zhang/Yu/Lin, Southeast Univ., and the
sole-author works (UCSC, Georgia Tech, Sungmin Lee) where student vs faculty isn't printed.
Also unmarked: Recursive Superintelligence and the substrate maintainers, which have no
paper to read a senior slot off. A keynote or panel slot is not evidence of a PI role and
never produces a mark on its own.

# Research Groups

## Sakana AI (Tokyo)

### Key people

| Name | Known for | Previous affiliations | Graduate education (school, advisor) |
|---|---|---|---|
| Robert Tjarko Lange (PI) | ShinkaEvolve lead — printed as having initiated and led the project, co-corresponding; Sakana-side author on Darwin Gödel Machine | | |
| Yujin Tang (PI) | AC/DC last author | | |
| Edoardo Cetin | ShinkaEvolve core contributor, co-corresponding | | |
| Yuki Imajuku | ShinkaEvolve | | |
| Yingtao Tian | AC/DC | | |
| Andrew Dai | AC/DC co-first | | |
| Boris Meinardus | AC/DC co-first | | |
| Ciaran Regan | AC/DC | | |

### Major papers/systems
- ShinkaEvolve (2509.19349) — open Apache-2.0 evolutionary program search, every component ablated; discovered a MoE load-balancing loss
- AC/DC (2604.14969) — ICLR 2026 main track; coevolves model merging with synthetic NL tasks
- Darwin Gödel Machine (2505.22954) — with Clune's UBC group

---

## Weco AI

### Key people

| Name | Known for | Previous affiliations | Graduate education (school, advisor) |
|---|---|---|---|
| Zhengyao Jiang (PI) | Co-founder & CEO; last author on both reward-hacking papers | | |
| Yuxiang Wu (PI) | Co-founder & CTO | | |
| Bingchen Zhao | First author on both reward-hacking papers; co-first author of Meta's Speedrunning Benchmark | Meta + University of Edinburgh (concurrent, printed on 2506.22419) | |
| Dhruv Srikanth | Member of technical staff | | |

### Major papers/systems
- AIDE — the coding-agent scaffold used as reference harness inside OpenAI's MLE-bench, METR's RE-Bench and Meta's aira-dojo
- Reward Hacking in Self-Improving Code Agents — 73.8% / 46.8% proxy-only gains. *Affiliation inferred, not read: OpenReview 403s; poster #58 of 110 at the ICLR 2026 RSI workshop*
- SpecBench (2605.21384) — hacking gap grows +28 pp per 10× code size

---

## Anthropic

### Key people

| Name | Known for | Previous affiliations | Graduate education (school, advisor) |
|---|---|---|---|
| Jan Leike (PI) | Last author, Automated Weak-to-Strong Researcher | | |
| Jan Hendrik Kirchner | AAR | | |
| Joe Benton | AAR | | |
| Jiaxin Wen | AAR co-first | | |
| Liang Qiu | AAR co-first | Anthropic Fellows Program (printed, this paper) | |
| Julian Schrittwieser | AlphaZero / MuZero; on the RSI workshop's Super-Stars panel *(program-listed affiliation, not a title page)* | | |

### Major papers/systems
- Automated Weak-to-Strong Researcher — 9 parallel Claude agents; PGR 0.97 vs human 0.23, plus three self-reported unpredicted reward-hacking exploits and a transfer failure at production scale

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
| Vikas Chandra | Agent-as-a-Judge (printed Meta AI, 2024); RSI workshop organizer, where the program lists him at Meta Reality Labs | | |
| Yuandong Tian | Agent-as-a-Judge (printed Meta AI, 2024); RSI workshop keynote *and* Super-Stars panelist, where the program lists him as stealth — and a named Recursive Superintelligence founder | | |

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
- Red Queen Gödel Machine (2606.26294) — drops the stationary-evaluator assumption; utilities update at epoch boundaries so agent and evaluator co-adapt. The map's only structural fix for proposer-games-the-metric

---

## Stanford University

### Key people

| Name | Known for | Previous affiliations | Graduate education (school, advisor) |
|---|---|---|---|
| Tatsunori Hashimoto (PI) | Last author, Execution-Grounded Auto-Research; Ideation-Execution Gap | | |
| James Zou (PI) | Last author, SLDAgent | | |
| Chenglei Si | First author, Execution-Grounded Auto-Research | | |
| Haotian Ye | SLDAgent co-first | | |
| Zitong Yang · Yejin Choi · Emmanuel Candès · Diyi Yang | Execution-Grounded Auto-Research | | |
| Fang Wu | Agent0 | | |
| Chelsea Finn | MAML; RSI workshop keynote *(program-listed affiliation, not a title page)* | | |

### Major papers/systems
- Towards Execution-Grounded Automated AI Research (2601.14525) — evolutionary search beats baselines where RL mode-collapses
- SLDAgent (2507.21184) — senior slot; LLM-discovered scaling laws beat human-derived ones, R² 0.748 vs 0.517
- Ideation-Execution Gap — rank flip between idea scores and executed results
- Agent0 (2511.16043) — co-author

---

## Peking University (+ Wizard Quant)

### Key people

| Name | Known for | Previous affiliations | Graduate education (school, advisor) |
|---|---|---|---|
| Yitao Liang (PI) | PKU senior slot on SLDAgent | | |
| Xiaojun Wan (PI) | PKU senior slot on Gödel Agent | | |
| Haowei Lin | SLDAgent co-first | Wizard Quant (concurrent, printed) | |
| Xunjian Yin | Gödel Agent first author | Work completed during internship at UC Santa Barbara (printed) | |
| Hubert Lim · Zhengrui Li · Xiangyu Wang | SLDAgent | | |
| Anjie Xu | AutoSOTA | Zhongguancun Academy (concurrent, printed) | |

Wizard Quant supplies 4 of SLDAgent's 11 authors: Haowei Lin, Wenzheng Feng, Quzhe Huang,
Yujun Li — a quantitative fund entering academic RSI.

### Major papers/systems
- SLDAgent (2507.21184) — lead slot on the map's strongest genuine-discovery claim
- Gödel Agent (2410.04444) — self-referential agent framework (first author, joint with UCSB)
- AutoSOTA (2604.05550) — co-author

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
| Wenyi Wang | GPTSwarm co-first (KAUST); Agent-as-a-Judge (KAUST) | | |
| Francesco Faccio | GPTSwarm | KAUST + IDSIA (concurrent, printed) | |
| Louis Kirsch | GPTSwarm (IDSIA); RSI workshop keynote on meta-RL, where the program lists him as stealth | IDSIA (printed on 2402.16823) | |
| Dylan R. Ashley | Agent-as-a-Judge (KAUST) | | |

### Major papers/systems
- GPTSwarm (2402.16823, ICML 2024) — language agents as optimizable graphs; the "agent topology is a learnable object" primitive
- Agent-as-a-Judge (2410.10934) — with Meta AI; an agentic system scores another's intermediate steps, plus DevAI (55 tasks, 365 hierarchical requirements). This is the automated evaluator that closes a self-improvement loop — and the reason the organizer's own tool sits inside the evaluation critique
- ICLR 2026 RSI workshop (Apr 26 2026, Rio, Room 101-D) — 110 accepted; the field's only dedicated venue. Sponsors: Tencent, Meta

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
the field's only venue are building the commercial bet on it — worth holding in view when
reading the workshop's own framing.

Funding figures reported in the press (~$650M at ~$4.65B) are inconsistent across outlets
and are deliberately not recorded here.

---

## Other research groups

| Institution | Key people | Key papers |
|---|---|---|
| National University of Singapore (+CUHK-Shenzhen) | **Bingsheng He (PI)**; Nuo Chen, Yicheng Tong, Yufei He | Diversity Collapse in MAS (2604.18005) |
| Tsinghua University — three separate labs | **Yong Li (PI)**, **Fengli Xu (PI)** (FIB Lab, EE/BNRist, both corresponding on AutoSOTA); Jianzhu Ma; Qi Li, Ke Xu (network security) | AutoSOTA lead; SLDAgent co; Safety in Self-Evolving Agents co |
| UC Santa Barbara — two labs | **Xin Eric Wang (PI)**; **William Yang Wang (PI)** | GEA (2602.04837); Gödel Agent (2410.04444) |
| Columbia Business School | **Olivier Toubia (PI)**; Melanie Brucks, Yuting Deng | Barriers to Diversity in LLM Ideas (2602.20408) |
| UBC + Vector Institute | **Jeff Clune (PI)** (also Canada CIFAR AI Chair; the workshop program lists him as UBC/DeepMind and he keynotes); Jenny Zhang, Shengran Hu, Cong Lu, Yiming Xiong | Darwin Gödel Machine (2505.22954); ALMA (2602.07755, workshop oral); AI-GAs (1905.10985) |
| UC Riverside (+AlphaAvatar, Illinois Tech) | Mingguang Chen (first author *and* corresponding — senior slot unclear); Licheng Wang; Bo Qu | RSI survey, 1,250 papers (2607.07663) |
| Google DeepMind | **Matej Balog (PI)** (AlphaEvolve last author, listed as its lead on the RSI-workshop panel page); **Vivek Natarajan (PI)** (AI co-scientist last author); Alexander Novikov, Juraj Gottweis | AlphaEvolve (2506.13131); AI co-scientist (2502.18864) |
| Zhongguancun Academy | **Tie-Yan Liu (PI)** (AutoSOTA last author); **Zequn Liu (PI)** (ForeSci corresponding); Yingce Xia | AutoSOTA senior; ForeSci senior/corresponding |
| OpenAI | **Aleksander Madry (PI)** (MLE-bench last author); Chan Jun Shern, Neil Chowdhury | MLE-bench (2410.07095) — substrate only; no RSI paper in this map |
| Pangram Labs | **Bradley Emi (PI)** (CTO/co-founder, sole byline) | ICLR 2026 AI-generated-review census, 21% of 75,800 |
| ISCAS + UCAS + Alibaba | **Xianpei Han (PI)**, **Yongbin Li (PI)**, **Hu Wei (PI)** (three marked corresponding authors); Le Sun, Hongyu Lin, Yaojie Lu (ISCAS); Tianyun Zhong (UCAS) | HypoArena / Before the Action (2607.15766) |
| MBZUAI (+CMU) | **Kun Zhang (PI)** (MBZUAI + CMU); Yongqiang Chen, Zhenhao Chen | CausalEvolve (2603.14575); Red Queen co |
| UC Santa Cruz | Kargi Chauhan (sole author) | Dead Science Walking (2606.04220) |
| King's College London — Algorithmic Dynamics Lab | **Hector Zenil (PI)** (lab lead, sole author; also Oxford Immune Algorithmics) | On the Limits of Self-Improving (2601.05280) |
| Zhejiang University + Ant Group | **Shouling Ji (PI)**; Ruixiao Lin, Qingming Li (ZJU); Changhua Meng, Shiwen Cui (Ant) | Safety in Self-Evolving LLM Agent Systems (2606.23075) |
| Hong Kong Baptist University — TMLR Group | **Bo Han (PI)** (corresponding); Xuan Li, Zhanke Zhou | Learning to Evolve (OpenReview WnZHbe1Gu0); CausalEvolve co |
| Harvard (+MIT) | **Vijay Janapa Reddi (PI)**; Chenyu Wang, Zishen Wan, Yilun Du | ArchEval (2607.03601) |
| Princeton University — Princeton Language and Intelligence | **Sanjeev Arora (PI)** (last author); Yun Cheng, Xingyu Zhu (both corresponding), Haoyu Zhao | Contextual Drag (2602.04288) |
| University of Edinburgh | Oisin Mac Aodha, Bingchen Zhao (co-first) | Speedrunning Benchmark (2506.22419) |
| Carnegie Mellon University | **Kun Zhang (PI)** (joint MBZUAI); Zeyu Zheng | CausalEvolve senior; Denominator Gaming |
| SJTU — Zhang/Yu/Lin group | Jianghao Lin (last author), Weinan Zhang, Yong Yu, Rong Shan | Denominator Gaming (2605.09915); Learning to Evolve co |
| University College London | Pontus Stenetorp; Edan Toledo, Karen Hambardzumyan (both co-first, joint Meta) | AIRA (2507.02554) |
| UNC-Chapel Hill (+Salesforce, Stanford) | **Huaxiu Yao (PI)**; Peng Xia; Caiming Xiong, Can Qin (Salesforce) | Agent0 (2511.16043) |
| FutureHouse (+Oxford) | **Samuel G. Rodriques (PI)**; Andrew D. White, Ali E. Ghareeb | Robin (2505.13400) |
| Wizard Quant | Haowei Lin (joint PKU), Wenzheng Feng, Quzhe Huang, Yujun Li | SLDAgent (4 of 11 authors) |
| Microsoft Research (+UC San Diego) | **Weizhu Chen (PI)**; Jianfeng Gao, Chandan Singh; Yufan Zhuang, Jingbo Shang (UCSD) | Test-time Recursive Thinking (2602.03094) |
| Örebro University — Machine Perception and Interaction Lab | **Andreas Persson (PI)** (last author); Amy Loutfi, Alkis Sygkounas, Rishi Hazra | Evolutionary Discovery of RL Algorithms (2603.28416, GECCO 2026); AIRA co |
| Hexo Labs (Palo Alto / Brussels / Toronto, +Oxford) | **Vignesh Baskaran (PI)** (last author); Prannay Hebbar, Yogendra Manawat | SIA (2605.27276) |
| Southeast University (+Duke Kunshan) | Youyong Kong, Qiuyu Tian (first author); Haojie Yin (Duke Kunshan) | ForeSci (2606.00644) |
| University of Würzburg — Computer Vision Lab, CAIDAS & IFI | **Dmitry Ignatov (PI)** (last author); Radu Timofte, Santosh Premi Adhikari | Delta-Based NAS (2605.04903) |
| Nanjing University of Science and Technology | **Chengzhi Zhang (PI)** (corresponding); Shuai Chen | Combinatorial Innovation idea generation (2604.20548) |
| Georgia Institute of Technology | Henry Jiang (sole author) | DARWIN (2602.05848) |
| Substrate maintainers (no affiliation printed on the artifact) | Keller Jordan (modded-nanoGPT, Muon); Andrej Karpathy (nanoGPT, autoresearch) | Repos, not papers — the cheap execution-grounded task nearly every RSI result is measured on |
| Unaffiliated | Sungmin Lee (no institution printed) | LLMs Have Made Failure Worth Publishing (2604.06236) |

Co-affiliations printed on a paper but holding no lead or senior slot: NVIDIA and Inria
(Red Queen), Taptap (ASI-Arch), University of Sydney SAIC (CausalEvolve), Hangzhou Dianzi /
NTU Singapore / Fudan (Safety), USTC (AutoSOTA), Central South University (Denominator
Gaming), University of Arizona (Gödel Agent), Stanford Medicine / Houston Methodist /
Sequome / Imperial (AI co-scientist), AlphaAvatar (RSI survey), and Thoughtful Lab
(PostTrainBench). Institutions that appear only through the workshop program — Apple,
ByteDance, Tencent, BAAI, Scale, Anuttacon, NYU, CUHK, Meta Reality Labs, Berkeley,
Physical Intelligence, Mila, Ohio State, You.com, UC Merced — are listed in the program
section below rather than given rows here, since a program listing is weaker evidence than
a title page.

# ICLR 2026 RSI workshop program (Apr 26 2026, Rio de Janeiro)

The single densest who's-who in the field: 110 accepted papers (4 oral · 21 spotlight ·
75 poster · 10 short), sponsored by Tencent and Meta. Affiliations below are as listed on
the workshop program, *not* read off a title page — one source removed, and current rather
than at-publication. Names already covered in a group section above are cross-referenced,
not duplicated.

### Organizers (11)

| Name | Program-listed affiliation | Known for |
|---|---|---|
| Mingchen Zhuge (lead) | KAUST | GPTSwarm, Agent-as-a-Judge; metauto.ai — see KAUST / IDSIA above |
| Jürgen Schmidhuber | KAUST / IDSIA | Gödel machine (2003); moderates the Super-Stars panel |
| Sherry Yang | NYU / DeepMind | |
| Vikas Chandra | Meta Reality Labs | Agent-as-a-Judge — see Meta above |
| Ailing Zeng | Anuttacon | |
| Deyao Zhu | ByteDance | |
| Rong Zou | Apple | |
| Yan Hu | CUHK | |
| Mengjia Li | BAAI | |
| Yunzhong He | Scale | |
| Levi Li | Tencent | |

### Keynotes (8 × 30 min)

| Name | Program-listed affiliation | Known for |
|---|---|---|
| Jeff Clune | UBC / DeepMind | AI-GAs, Darwin Gödel Machine, ALMA — see UBC + Vector above |
| Chelsea Finn | Stanford | MAML |
| Sergey Levine | Berkeley / Physical Intelligence | |
| Yuandong Tian | stealth | Agent-as-a-Judge; Recursive Superintelligence founder |
| Louis Kirsch | stealth | meta-RL; GPTSwarm (IDSIA) — see KAUST / IDSIA above |
| Bing Liu | Scale | |
| Bang Liu | Mila | |
| Yu Su | Ohio State / NeoCognition | |

### Super-Stars panel (moderated by Schmidhuber)

| Name | Program-listed affiliation | Known for |
|---|---|---|
| Julian Schrittwieser | Anthropic | AlphaZero / MuZero |
| Richard Socher | You.com | Recursive Superintelligence founder |
| Yuandong Tian | stealth | also a keynote |
| Matej Balog | DeepMind | AlphaEvolve lead |
| Ming-Hsuan Yang | UC Merced / DeepMind | |
| Vladlen Koltun | Apple | |

### Accepted works named in the guide with authors not yet resolved

OpenReview sat behind a bot-challenge wall, so these have titles and forum IDs but no
author list. Resolving them is the highest-yield next pass on this file.

| Work | Track | OpenReview ID |
|---|---|---|
| Reward Hacking in Self-Improving Code Agents | poster | ikrQWGgxYg — attributed to Weco AI above, inferred |
| Verifying the Verifiers | | iRhaK8PsuB |
| SAHOO (safeguarded alignment for RSI) | | OAFPpQO0H9 |
| TamperBench | | smLtz7WID0 |

Titled but with neither ID nor authors in the guide: POLARIS, SkillRL, Self-Adapting Agents
for Research Coding, OMEGA, LLM-FE, CircuitBuilder, "Simple Baselines vs Code Evolution",
Reasoning as Gradient, "Can Language Models Discover Scaling Laws?", Language Self-Play,
SAGE, GASP, Anchored Self-Play, Compute as Teacher, "Self-Play is secretly Adversarial
Imitation", SimpleMem, Agentic Context Engineering, Test-Time Self-Distillation, Adaptive
Meta-Curriculum, TextBO, Self-Improvement via Fast Tree-search, Self-Improving World Models,
VLAW, RFTF, "Interestingness as an Inductive Heuristic".

# Major academic lineage (who is advisor of who)

Not yet researched. Edges to establish first, in rough order of how much they explain:

- Jürgen Schmidhuber (IDSIA/KAUST) → Mingchen Zhuge (KAUST) — the Gödel-machine lineage into the field's only venue. Co-authorship already establishes the cluster: Zhuge, Wenyi Wang, Francesco Faccio, Dmitrii Khizbullin, Dylan Ashley and Louis Kirsch all appear with Schmidhuber on GPTSwarm and/or Agent-as-a-Judge; which of those are his students is the open part
- Jeff Clune (UBC) → Jenny Zhang, Shengran Hu, Cong Lu, Yiming Xiong — and the Sakana overlap (Hu, Lu, Lange all on DGM). Hu is on both DGM and ALMA, so he is the through-line of the Clune archive lineage
- Sanjeev Arora (Princeton PLI) → Yun Cheng, Xingyu Zhu, Haoyu Zhao — the Contextual Drag group
- Matthias Bethge (Tübingen) and Maksym Andriushchenko (ELLIS/MPI-IS) → Ben Rank, Hardik Bhatnagar, Ameya Prabhu — the PostTrainBench group; also, where Andriushchenko himself trained
- Sergey Levine (Berkeley) ↔ Chelsea Finn (Stanford) — both keynote; the MAML/meta-RL branch feeding RSI's "learn to learn" framing
- William Yang Wang (UCSB) → Xin Eric Wang? — would explain two independent UCSB RSI labs
- Kun Zhang (CMU/MBZUAI) → Yongqiang Chen, Bo Han, Tongliang Liu — the causal branch
- Tatsunori Hashimoto (Stanford) → Chenglei Si — the execution-grounded branch
- Pengfei Liu (SJTU/SII/GAIR) — where the ASI-Arch group came from
- Zhengyao Jiang, Yuxiang Wu (Weco founders) — where the Weco team trained
- Yoram Bachrach, Jakob Foerster (Meta) — the AIRA/speedrun program's students

# Appendix (rank by paper importance)

| # | Paper | Author & affiliation pairs |
|---|---|---|
| 1 | Reward Hacking in Self-Improving Code Agents (OpenReview ikrQWGgxYg) | Bingchen Zhao · Dhruv Srikanth · Yuxiang Wu · Zhengyao Jiang — **all Weco AI (inferred, title page unreadable)** |
| 2 | RSI survey, 1,250 papers (2607.07663) | Mingguang Chen — UC Riverside (corresponding); Licheng Wang — AlphaAvatar; Bo Qu — Illinois Institute of Technology |
| 3 | Automated Weak-to-Strong Researcher (Anthropic) | Jiaxin Wen, Joe Benton, Jan Hendrik Kirchner, Jan Leike — Anthropic; Liang Qiu — Anthropic + Anthropic Fellows Program |
| 4 | ShinkaEvolve (2509.19349) | Robert Tjarko Lange · Yuki Imajuku · Edoardo Cetin — all Sakana AI |
| 5 | PostTrainBench (2603.08640) — workshop oral | Ben Rank, Maksym Andriushchenko (corresponding) — ELLIS Institute Tübingen + MPI for Intelligent Systems + Tübingen AI Center; Hardik Bhatnagar, Ameya Prabhu, Matthias Bethge — Univ. Tübingen + Tübingen AI Center; Shira Eisenberg (internship), Nguyen Karina — Thoughtful Lab |
| 6 | Red Queen Gödel Machine (2606.26294) | Iacob, Shen, Kurmanji, Cao, Marino, Qiu — Cambridge; Jovanović, Sani, Lane — Cambridge + Flower Labs; Burkhardt, Venanzi — NVIDIA; Tastan — MBZUAI; Odonnat — Inria |
| 7 | SLDAgent (2507.21184) | Haowei Lin — PKU + Wizard Quant; Haotian Ye, James Zou — Stanford; Feng, Huang, Y. Li — Wizard Quant; Jianzhu Ma — Tsinghua; Liang, Lim, Z. Li, X. Wang — PKU |
| 8 | SpecBench (2605.21384) | Bingchen Zhao · Dhruv Srikanth · Yuxiang Wu · Zhengyao Jiang — all Weco AI |
| 9 | Diversity Collapse in MAS (2604.18005) | N. Chen, Tong, Y. He, Zou, Q. Wang, B. He — NUS; Y. Yang, X. Zhang — CUHK-Shenzhen |
| 10 | AutoSOTA (2604.05550) | Yu Li, X. Liu, R. Zhao, P. Liu, Z. Chen, Q. Yang, Zeng, T. Li, J. Xu — Tsinghua EE/BNRist; Shao, Su, Fengli Xu, Yong Li — Tsinghua + Zhongguancun; A. Xu — Zhongguancun + PKU; Fang — Zhongguancun + USTC; Tie-Yan Liu — Zhongguancun |
| 11 | ALMA / Meta-learning Agentic Memory Designs (2602.07755) — workshop oral | Yiming Xiong — UBC; Shengran Hu — UBC + Vector Institute; Jeff Clune — UBC + Vector Institute + Canada CIFAR AI Chair |
| 12 | Group-Evolving Agents (2602.04837) | Weng, Antoniades, Nathani, Z. Zhang, Pu, Xin Eric Wang — all UC Santa Barbara |
| 13 | Safety in Self-Evolving Agents (2606.23075) | R. Lin, Q. Li, Z. Li, Shouling Ji — Zhejiang; Deng, Qing, Q. Li, Ke Xu — Tsinghua; Feng, Cui, Meng — Ant Group; Ma — Hangzhou Dianzi; Y. Zhang, T. Zhang — NTU; X. Ma — Fudan |
| 14 | Barriers to Diversity in LLM Ideas (2602.20408) | Yuting Deng · Melanie Brucks · Olivier Toubia — all Columbia Business School |
| 15 | Contextual Drag (2602.04288) | Yun Cheng, Xingyu Zhu (both corresponding), Haoyu Zhao, Sanjeev Arora — all Princeton Language and Intelligence, Princeton University |
| 16 | Dead Science Walking (2606.04220) | Kargi Chauhan — UC Santa Cruz (sole author) |
| 17 | HypoArena / Before the Action (2607.15766) | Zhong, W. Jiang — UCAS + ISCAS; X. Chen, Lu, H. Lin, Sun, Xianpei Han — ISCAS; W. Wang, Shi, B. Yang, J. Wang, H. Li, Zhai, B. Zhao, Hu Wei, H. Yu, Yongbin Li — Alibaba Group |
| 18 | ForeSci (2606.00644) | Qiuyu Tian — Southeast Univ. + Zhongguancun; Haojie Yin — Duke Kunshan; Yingce Xia, Zequn Liu (corresponding) — Zhongguancun; Youyong Kong — Southeast Univ. |
| 19 | CausalEvolve (2603.14575) | Yongqiang Chen — MBZUAI + CMU; Chenxi Liu, Bo Han — HKBU TMLR; Zhenhao Chen — MBZUAI; Tongliang Liu — Univ. Sydney SAIC + MBZUAI; Kun Zhang — MBZUAI + CMU |
| 20 | On the Limits of Self-Improving (2601.05280) | Hector Zenil — King's College London + Oxford Immune Algorithmics (sole author) |
| 21 | ArchEval (2607.03601) | C. Wang, Wan, Ma, Prakash, Qi, Do, Cheng, Tschand, Du, Reddi — Harvard; Shi — MIT |
| 22 | ASI-Arch (2507.18074) | Yixiu Liu, Weixian Xu, Lyumanshan Ye — SJTU + SII + GAIR; Yang Nan, Xiangkun Hu — SII + GAIR; Zhen Qin — Taptap; Pengfei Liu (corresponding) — SJTU + SII + GAIR |
| 23 | AC/DC (2604.14969) | Andrew Dai · Boris Meinardus · Ciaran Regan · Yingtao Tian · Yujin Tang — all Sakana AI (ICLR 2026) |
| 24 | Delta-Based NAS (2605.04903) | Santosh Premi Adhikari · Radu Timofte · Dmitry Ignatov — all Computer Vision Lab, CAIDAS & IFI, Univ. Würzburg |
| 25 | Learning to Evolve (OpenReview WnZHbe1Gu0) | Xuan Li, Zhanke Zhou, Zongze Li, Bo Han — HKBU; Jiangchao Yao — SJTU *(mapping from HKBU repository, not the PDF)* |
| 26 | Evolutionary Discovery of RL Algorithms (2603.28416) | Alkis Sygkounas · Amy Loutfi · Andreas Persson — all Machine Perception and Interaction Lab, Örebro |
| 27 | SIA (2605.27276) | Hebbar, Manawat, Bhatia, Baskaran — Hexo Labs Palo Alto; Verboomen — Hexo Labs Brussels; Palanimalai — Hexo Labs Toronto; Ivanova — Univ. Oxford |
| 28 | Denominator Gaming (2605.09915) | Shan, Zheng, Xi, Zhu, Yu, W. Zhang, J. Lin — SJTU; Gao — Central South Univ.; Zeyu Zheng — CMU |
| 29 | Test-time Recursive Thinking (2602.03094) | Zhuang — MSR + UCSD; Singh, L. Liu, Shen, D. Zhang, Gao, W. Chen — Microsoft Research; Shang — UC San Diego |
| 30 | Combinatorial Innovation (2604.20548) | Shuai Chen · Chengzhi Zhang (corresponding) — Nanjing Univ. of Science and Technology |
| 31 | LLMs Have Made Failure Worth Publishing (2604.06236) | Sungmin Lee — no affiliation printed |
| 32 | DARWIN (2602.05848) | Henry Jiang — Georgia Tech, College of Computing (sole author) |
| — | METR Time Horizon 1.1 | METR — corporate authorship, no individual bylines |
| — | Pangram ICLR 2026 review census | Bradley Emi — Pangram Labs (sole byline) |
| — | ICLR 2026 RSI workshop | 110 accepted (4 oral · 21 spotlight · 75 poster · 10 short). Full organizer / keynote / panel roster with affiliations in the program section above |

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
| AI co-scientist (2502.18864) | Gottweis et al. — Google Cloud AI Research; Weng, Tu, Hassabis, Kohli, Natarajan et al. — Google DeepMind; Popovici, Palepu et al. — Google Research; plus Stanford Medicine, Houston Methodist, Sequome, Imperial (51 authors) |
| Robin (2505.13400) | Ghareeb, Mitchener, Yiu, Szostkiewicz, Laurent, Razzak, White, Hinks, Rodriques — FutureHouse; Benjamin Chang — FutureHouse + Oxford |
| Speedrunning Benchmark (2506.22419) | Bingchen Zhao — Meta + Univ. Edinburgh; Despoina Magka, Minqi Jiang … Yoram Bachrach — Meta; Oisin Mac Aodha — Univ. Edinburgh |
| Agent0 (2511.16043) | Peng Xia, Zeng, J. Liu, Zhou, Huaxiu Yao — UNC-Chapel Hill; Qin, Xiong — Salesforce Research; Fang Wu — Stanford |
| AIRA (2507.02554) | Toledo, Hambardzumyan — FAIR at Meta + UCL; Josifoski, Baldwin, Magka, M. Jiang, Raileanu, Cancedda, Foerster, Bachrach et al. — FAIR at Meta; Hazra — Örebro (work done while at Meta); Stenetorp — UCL |
| Execution-Grounded Auto-Research (2601.14525) | Chenglei Si, Zitong Yang, Yejin Choi, Emmanuel Candès, Diyi Yang, Tatsunori Hashimoto — all Stanford |
| MLE-bench (2410.07095) | Chan Jun Shern, Chowdhury, Jaffe, Aung, Sherburn, Mays, Starace, K. Liu, Maksin, Patwardhan, Weng, Madry — all OpenAI |
