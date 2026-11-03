This doc captures major research groups and people in RSI, AI for science etc.

The format is frozen. What exists below — the group sections with their four-column Key
people table and Major papers/systems list, the Other research groups table, the lineage
section, the appendix — is the whole structure. Updates add content into it: new rows, new
people, new papers, new lineage edges. Do not add sections, headings, subsections or table
columns, and do not reorganize. Anything that won't fit an existing slot belongs in a topic
guide, not here.

Affiliations are as printed on the paper (arXiv HTML author block or PDF page 1). Bare
acronyms are kept bare. Blank cells mean not yet researched, not "none". Source maps:
[`rsi-proposer-landscape-2026.md`](rsi-proposer-landscape-2026.md) and
[`iclr-2026-rsi-workshop.md`](iclr-2026-rsi-workshop.md), and
[`rsi-catchup-2026-08-to-11-guide.md`](rsi-catchup-2026-08-to-11-guide.md).

A conference program is not a source. Organizer, keynote and panel rosters are not recorded
here — those are not research output, and a program page gives a current affiliation rather
than the at-publication one every entry below is built from. A program slot never earns
anyone a row, a (PI) mark, or an appendix line. Workshop rosters live in the workshop guide.

**(PI)** marks the group lead, listed first in each table. Assigned only from printed
evidence — the last author or the marked corresponding author of that group's work here. No
mark means the senior slot isn't established from the papers, not that the group has no
lead. Unmarked on purpose: UC Riverside (first author is also corresponding), Edinburgh and
UCL (senior only within a Meta-led paper), SJTU Zhang/Yu/Lin, Southeast Univ., and the
sole-author works (UCSC, Georgia Tech, Sungmin Lee) where student vs faculty isn't printed.
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
| Jan Hendrik Kirchner | AAR | | |
| Joe Benton | AAR | | |
| Jiaxin Wen | AAR co-first | | |
| Liang Qiu | AAR co-first | Anthropic Fellows Program (printed, this paper) | |

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
- Red Queen Gödel Machine (2606.26294) — drops the stationary-evaluator assumption; utilities update at epoch boundaries so agent and evaluator co-adapt. The map's only structural fix for proposer-games-the-metric

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

### Major papers/systems
- Towards Execution-Grounded Automated AI Research (2601.14525) — evolutionary search beats baselines where RL mode-collapses
- SLDAgent (2507.21184) — senior slot; LLM-discovered scaling laws beat human-derived ones, R² 0.748 vs 0.517
- Ideation-Execution Gap — rank flip between idea scores and executed results
- Agent0 (2511.16043) — co-author
- SPADE (2608.19197) — Bo Liu co-first and Yejin Choi joint last author; shared-weight environment designer and reasoning agent
- RSI-Anything (2610.13912) — Yingcheng Wu corresponding; predictive world models and retained policy–model updates

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

Wizard Quant supplies 4 of SLDAgent's 11 authors: Haowei Lin, Wenzheng Feng, Quzhe Huang,
Yujun Li — a quantitative fund entering academic RSI.

### Major papers/systems
- SLDAgent (2507.21184) — lead slot on the map's strongest genuine-discovery claim
- Gödel Agent (2410.04444) — self-referential agent framework (first author, joint with UCSB)
- AutoSOTA (2604.05550) — co-author
- RSI-Anything (2610.13912) — policy/world-model update and retention framework

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
| Louis Kirsch | GPTSwarm (IDSIA); meta-RL | IDSIA (printed on 2402.16823) | |
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

- [Recursive Flint](https://huggingface.co/RecursiveSI/flint) (October 2026) — Apache-2.0 coding MoE weights and inference code; corporate model-card attribution, with no individual author–affiliation mapping. The release describes contributions from Eureka, the automated research system.
- [First Steps Toward Automated AI Research](https://recursive.com/updates/first-steps-toward-automated-ai-research) — earlier NanoGPT, NanoChat and kernel artifacts; the official updates index dates this report 11 June 2026, despite its migrated October page header.

---

## Other research groups

| Institution | Key people | Key papers |
|---|---|---|
| National University of Singapore (+CUHK-Shenzhen) | **Bingsheng He (PI)**; Nuo Chen, Yicheng Tong, Yufei He; Zichen Liu, Zijian Zhou (SPADE) | Diversity Collapse in MAS (2604.18005); SPADE (2608.19197) |
| Tsinghua University — three separate labs | **Yong Li (PI)**, **Fengli Xu (PI)** (FIB Lab, EE/BNRist, both corresponding on AutoSOTA); Jianzhu Ma; Qi Li, Ke Xu (network security) | AutoSOTA lead; SLDAgent co; Safety in Self-Evolving Agents co |
| UC Santa Barbara — two labs | **Xin Eric Wang (PI)**; **William Yang Wang (PI)** | GEA (2602.04837); Gödel Agent (2410.04444) |
| Columbia Business School | **Olivier Toubia (PI)**; Melanie Brucks, Yuting Deng | Barriers to Diversity in LLM Ideas (2602.20408) |
| UBC + Vector Institute | **Jeff Clune (PI)** (also Canada CIFAR AI Chair); Jenny Zhang, Shengran Hu, Cong Lu, Yiming Xiong | Darwin Gödel Machine (2505.22954); ALMA (2602.07755, workshop oral); AI-GAs (1905.10985) |
| UC Riverside (+AlphaAvatar, Illinois Tech) | Mingguang Chen (first author *and* corresponding — senior slot unclear); Licheng Wang; Bo Qu | RSI survey, 1,250 papers (2607.07663) |
| Google DeepMind | **Matej Balog (PI)** (AlphaEvolve last author, listed as its lead on the RSI-workshop panel page); **Vivek Natarajan (PI)** (AI co-scientist last author); Alexander Novikov, Juraj Gottweis; Zhankui He, Benjamin Coleman, Di Bai, Wang-Cheng Kang (Dream-RSI) | AlphaEvolve (2506.13131); AI co-scientist (2502.18864); Dream-RSI (2609.14858) |
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
| Carnegie Mellon University | **Kun Zhang (PI)** (joint MBZUAI); Zeyu Zheng; Yiding Jiang, Seungone Kim (SPADE) | CausalEvolve senior; Denominator Gaming; SPADE (2608.19197) |
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
| Georgia Institute of Technology | Henry Jiang (sole author); Jiang Zhou (RSI-Env) | DARWIN (2602.05848); RSI-Env (2611.01736) |
| UC Berkeley | Hyunin Lee, Donghyun Lee, Somayeh Sojoudi, Matei Zaharia | MASS (2610.12176), with Sakana AI |
| Google | **Xidong Wu (PI)**, **Zheng Zhang (PI)** (Dream-RSI corresponding), **Yunsong Guo (PI)** (last author); Tong Zheng (joint Maryland), Chaoyi Zhang, Ruoqiao Wei, Xue Wang, Yue Zhuan, Renkai Xiang, Xinwu Cheng | Dream-RSI (2609.14858) |
| University of Maryland, College Park | Heng Huang, Tong Zheng, Rui Liu | Dream-RSI (2609.14858) |
| University of Virginia | Haolin Liu | Dream-RSI (2609.14858) |
| University of Washington | **Natasha Jaques (PI)** (SPADE joint last author); Luke Zettlemoyer, Bo Liu (joint Stanford), Mickel Liu | SPADE (2608.19197) |
| Northeastern University | Simon Yu (co-first), Weiyan Shi | SPADE (2608.19197) |
| Massachusetts Institute of Technology | Ao Qu, Paul Pu Liang | SPADE (2608.19197) |
| Seoul National University | Seungyeon Jwa (Meta^n), Junsu Kim (SPADE) | Meta^n (2608.24735); SPADE (2608.19197) |
| University of Minnesota | **Dongyeop Kang (PI)** (Meta^n last author); Zae Myung Kim, Young-Jun Lee | Meta^n (2608.24735) |
| University of Chicago | Zhaorun Chen | SPADE (2608.19197) |
| Stevens Institute of Technology | Zihan Chen, Di Zhu; Hanfei Yu (SPADE) | Auditing the Product, Not the Recursion (2610.27610); SPADE (2608.19197) |
| University of Massachusetts Boston | **Lei Nico Zheng (PI)** (Auditing the Product last author) | Auditing the Product, Not the Recursion (2610.27610) |
| Rutgers University | Meijia Chen (co-first) | False Frontiers (2609.39102) |
| UC San Diego | Yichen Liu, Hanxin Chen | False Frontiers (2609.39102) |
| University of Michigan | Weizhi Du | False Frontiers (2609.39102) |
| McGill University | **Tianyu Shi (PI)** (False Frontiers corresponding) | False Frontiers (2609.39102) |
| King Fahd University of Petroleum and Minerals | **Alaa Khamis (PI)** (False Frontiers corresponding and last author) | False Frontiers (2609.39102) |
| Independent researchers (as printed) | Hao Li, Zheng Lu (co-first), Hongshan Lin, Junbai Tian, Zijun Tian, Yufan Zou, Shuhan Sun, Zeyu Zhang, Yueting Li | False Frontiers (2609.39102) |
| ByteDance Seed / M-A-P / TokenWave.AI | S3Gym lists these three affiliations without individual mapping; core contributors include Jiajun Shi, Siyuan Tao, Yuhao Wu, Zexuan Wang, Jingyuan Zhang | S3Gym (2608.31100); individual institutional assignments unresolved |
| Fudan University | Yanwei Fu (joint Shanghai Innovation Institute), Wanying Qu, Jiyao Liu (RSI-Env); Hao Xiao, Qiang Gao (RSI-Anything) | RSI-Env (2611.01736); RSI-Anything (2610.13912) |
| HKUST | Zhiyuan Fan | RSI-Env (2611.01736) |
| Nanjing University / Shanghai Artificial Intelligence Laboratory | **Dongrui Liu (PI)** (Shanghai AI Laboratory, RSI-Env last author); Jianghan Shen (joint Nanjing and Shanghai AI Laboratory) | RSI-Env (2611.01736) |
| PhAI Labs | **Ling Yang (PI)** (joint PKU), **Yingcheng Wu (PI)** (joint Stanford), **Zhenfei Yin (PI)** (joint Oxford), all RSI-Anything corresponding; Taoyong Cui (joint CUHK), Shuhan Xue | RSI-Anything (2610.13912); affiliations in Appendix A referenced by title page |
| Chinese University of Hong Kong | **Pheng Ann Heng (PI)** (RSI-Anything corresponding); Wanli Ouyang, Taoyong Cui (joint PhAI Labs) | RSI-Anything (2610.13912) |
| University of Macau | Pinxian Zeng | RSI-Anything (2610.13912) |
| SJTU — RSI-Anything collaboration | Wanghan Xu | RSI-Anything (2610.13912) |
| Substrate maintainers (no affiliation printed on the artifact) | Keller Jordan (modded-nanoGPT, Muon); Andrej Karpathy (nanoGPT, autoresearch) | Repos, not papers — the cheap execution-grounded task nearly every RSI result is measured on |
| Unaffiliated | Sungmin Lee (no institution printed) | LLMs Have Made Failure Worth Publishing (2604.06236) |

Co-affiliations printed on a paper but holding no lead or senior slot: NVIDIA and Inria
(Red Queen), Taptap (ASI-Arch), University of Sydney SAIC (CausalEvolve), Hangzhou Dianzi /
NTU Singapore / Fudan (Safety), USTC (AutoSOTA), Central South University (Denominator
Gaming), University of Arizona (Gödel Agent), Stanford Medicine / Houston Methodist /
Sequome / Imperial (AI co-scientist), AlphaAvatar (RSI survey), and Thoughtful Lab
(PostTrainBench). Institutions that reach this field only through the ICLR 2026 RSI
workshop program — Apple, Tencent, BAAI, Scale, Anuttacon, NYU, Meta
Reality Labs, Physical Intelligence, Mila, Ohio State, You.com, UC Merced — are
deliberately absent. Organizing, keynoting or sitting on a panel is not research output,
and a program page is not a title page; the roster lives in the workshop guide instead.

# Major academic lineage (who is advisor of who)

Not yet researched. Edges to establish first, in rough order of how much they explain:

- Jürgen Schmidhuber (IDSIA/KAUST) → Mingchen Zhuge (KAUST) — the Gödel-machine lineage into the field's only venue. Co-authorship already establishes the cluster: Zhuge, Wenyi Wang, Francesco Faccio, Dmitrii Khizbullin, Dylan Ashley and Louis Kirsch all appear with Schmidhuber on GPTSwarm and/or Agent-as-a-Judge; which of those are his students is the open part
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
| 3 | Auditing the Product, Not the Recursion (2610.27610) | Zihan Chen, Di Zhu — Stevens Institute of Technology; Lei Nico Zheng — University of Massachusetts Boston |
| 4 | False Frontiers (2609.39102) | Meijia Chen — Rutgers; Hao Li, Zheng Lu, Hongshan Lin, Junbai Tian, Zijun Tian, Yufan Zou, Shuhan Sun, Zeyu Zhang, Yueting Li — Independent Researcher; Yichen Liu, Hanxin Chen — UC San Diego; Weizhi Du — University of Michigan; Tianyu Shi — McGill; Alaa Khamis — King Fahd University of Petroleum and Minerals |
| 5 | S3Gym (2608.31100) | ByteDance Seed, M-A-P, TokenWave.AI printed collectively; core contributors Jiajun Shi, Siyuan Tao, Yuhao Wu, Zexuan Wang, Jingyuan Zhang have no individual affiliation mapping in the paper |
| 6 | Recursive Flint (October 2026 model release) | Recursive Superintelligence, Inc. — corporate model-card attribution; no individual paper byline |
| 7 | SPADE (2608.19197) | Bo Liu — Washington + Stanford; Simon Yu, Weiyan Shi — Northeastern; Yiding Jiang, Seungone Kim — CMU; Ao Qu, Paul Pu Liang — MIT; Zichen Liu, Zijian Zhou — NUS; Junsu Kim — SNU; Mickel Liu, Luke Zettlemoyer, Natasha Jaques — Washington; Hanfei Yu — Stevens; Zhaorun Chen — Chicago; Yejin Choi — Stanford; Andrew Zhao, Tongzheng Ren — no affiliation printed |
| 8 | Dream-RSI (2609.14858) | Tong Zheng — Google + Maryland; Xidong Wu, Zheng Zhang, Chaoyi Zhang, Ruoqiao Wei, Xue Wang, Yue Zhuan, Renkai Xiang, Xinwu Cheng, Yunsong Guo — Google; Zhankui He, Benjamin Coleman, Di Bai, Wang-Cheng Kang — Google Deepmind; Rui Liu, Heng Huang — Maryland; Haolin Liu — Virginia |
| 9 | RSI-Env (2611.01736) | Wanying Qu, Jiyao Liu — Fudan; Zhiyuan Fan — HKUST; Jiang Zhou — Georgia Tech; Jianghan Shen — Nanjing University + Shanghai Artificial Intelligence Laboratory; Yanwei Fu — Fudan + Shanghai Innovation Institute; Dongrui Liu — Shanghai Artificial Intelligence Laboratory |
| 10 | Meta^n (2608.24735) | Zae Myung Kim, Young-Jun Lee, Dongyeop Kang — University of Minnesota; Seungyeon Jwa — Seoul National University |
| 11 | RSI-Anything (2610.13912) | Taoyong Cui — PhAI Labs + CUHK; Pinxian Zeng — Macau; Wanghan Xu — SJTU; Hao Xiao, Qiang Gao — Fudan; Shuhan Xue — PhAI Labs; Wanli Ouyang, Pheng Ann Heng — CUHK; Zhenfei Yin — PhAI Labs + Oxford; Yingcheng Wu — PhAI Labs + Stanford; Ling Yang — PhAI Labs + PKU. Mapping printed in Appendix A, referenced on the title page |
| 12 | Reward Hacking in Self-Improving Code Agents (OpenReview ikrQWGgxYg) | Bingchen Zhao · Dhruv Srikanth · Yuxiang Wu · Zhengyao Jiang — **all Weco AI (inferred, title page unreadable)** |
| 13 | RSI survey, 1,250 papers (2607.07663) | Mingguang Chen — UC Riverside (corresponding); Licheng Wang — AlphaAvatar; Bo Qu — Illinois Institute of Technology |
| 14 | Automated Weak-to-Strong Researcher (Anthropic) | Jiaxin Wen, Joe Benton, Jan Hendrik Kirchner, Jan Leike — Anthropic; Liang Qiu — Anthropic + Anthropic Fellows Program |
| 15 | ShinkaEvolve (2509.19349) | Robert Tjarko Lange · Yuki Imajuku · Edoardo Cetin — all Sakana AI |
| 16 | PostTrainBench (2603.08640) — workshop oral | Ben Rank, Maksym Andriushchenko (corresponding) — ELLIS Institute Tübingen + MPI for Intelligent Systems + Tübingen AI Center; Hardik Bhatnagar, Ameya Prabhu, Matthias Bethge — Univ. Tübingen + Tübingen AI Center; Shira Eisenberg (internship), Nguyen Karina — Thoughtful Lab |
| 17 | Red Queen Gödel Machine (2606.26294) | Iacob, Shen, Kurmanji, Cao, Marino, Qiu — Cambridge; Jovanović, Sani, Lane — Cambridge + Flower Labs; Burkhardt, Venanzi — NVIDIA; Tastan — MBZUAI; Odonnat — Inria |
| 18 | SLDAgent (2507.21184) | Haowei Lin — PKU + Wizard Quant; Haotian Ye, James Zou — Stanford; Feng, Huang, Y. Li — Wizard Quant; Jianzhu Ma — Tsinghua; Liang, Lim, Z. Li, X. Wang — PKU |
| 19 | SpecBench (2605.21384) | Bingchen Zhao · Dhruv Srikanth · Yuxiang Wu · Zhengyao Jiang — all Weco AI |
| 20 | Diversity Collapse in MAS (2604.18005) | N. Chen, Tong, Y. He, Zou, Q. Wang, B. He — NUS; Y. Yang, X. Zhang — CUHK-Shenzhen |
| 21 | AutoSOTA (2604.05550) | Yu Li, X. Liu, R. Zhao, P. Liu, Z. Chen, Q. Yang, Zeng, T. Li, J. Xu — Tsinghua EE/BNRist; Shao, Su, Fengli Xu, Yong Li — Tsinghua + Zhongguancun; A. Xu — Zhongguancun + PKU; Fang — Zhongguancun + USTC; Tie-Yan Liu — Zhongguancun |
| 22 | ALMA / Meta-learning Agentic Memory Designs (2602.07755) — workshop oral | Yiming Xiong — UBC; Shengran Hu — UBC + Vector Institute; Jeff Clune — UBC + Vector Institute + Canada CIFAR AI Chair |
| 23 | Group-Evolving Agents (2602.04837) | Weng, Antoniades, Nathani, Z. Zhang, Pu, Xin Eric Wang — all UC Santa Barbara |
| 24 | Safety in Self-Evolving Agents (2606.23075) | R. Lin, Q. Li, Z. Li, Shouling Ji — Zhejiang; Deng, Qing, Q. Li, Ke Xu — Tsinghua; Feng, Cui, Meng — Ant Group; Ma — Hangzhou Dianzi; Y. Zhang, T. Zhang — NTU; X. Ma — Fudan |
| 25 | Barriers to Diversity in LLM Ideas (2602.20408) | Yuting Deng · Melanie Brucks · Olivier Toubia — all Columbia Business School |
| 26 | Contextual Drag (2602.04288) | Yun Cheng, Xingyu Zhu (both corresponding), Haoyu Zhao, Sanjeev Arora — all Princeton Language and Intelligence, Princeton University |
| 27 | Dead Science Walking (2606.04220) | Kargi Chauhan — UC Santa Cruz (sole author) |
| 28 | HypoArena / Before the Action (2607.15766) | Zhong, W. Jiang — UCAS + ISCAS; X. Chen, Lu, H. Lin, Sun, Xianpei Han — ISCAS; W. Wang, Shi, B. Yang, J. Wang, H. Li, Zhai, B. Zhao, Hu Wei, H. Yu, Yongbin Li — Alibaba Group |
| 29 | ForeSci (2606.00644) | Qiuyu Tian — Southeast Univ. + Zhongguancun; Haojie Yin — Duke Kunshan; Yingce Xia, Zequn Liu (corresponding) — Zhongguancun; Youyong Kong — Southeast Univ. |
| 30 | CausalEvolve (2603.14575) | Yongqiang Chen — MBZUAI + CMU; Chenxi Liu, Bo Han — HKBU TMLR; Zhenhao Chen — MBZUAI; Tongliang Liu — Univ. Sydney SAIC + MBZUAI; Kun Zhang — MBZUAI + CMU |
| 31 | On the Limits of Self-Improving (2601.05280) | Hector Zenil — King's College London + Oxford Immune Algorithmics (sole author) |
| 32 | ArchEval (2607.03601) | C. Wang, Wan, Ma, Prakash, Qi, Do, Cheng, Tschand, Du, Reddi — Harvard; Shi — MIT |
| 33 | ASI-Arch (2507.18074) | Yixiu Liu, Weixian Xu, Lyumanshan Ye — SJTU + SII + GAIR; Yang Nan, Xiangkun Hu — SII + GAIR; Zhen Qin — Taptap; Pengfei Liu (corresponding) — SJTU + SII + GAIR |
| 34 | AC/DC (2604.14969) | Andrew Dai · Boris Meinardus · Ciaran Regan · Yingtao Tian · Yujin Tang — all Sakana AI (ICLR 2026) |
| 35 | Delta-Based NAS (2605.04903) | Santosh Premi Adhikari · Radu Timofte · Dmitry Ignatov — all Computer Vision Lab, CAIDAS & IFI, Univ. Würzburg |
| 36 | Learning to Evolve (OpenReview WnZHbe1Gu0) | Xuan Li, Zhanke Zhou, Zongze Li, Bo Han — HKBU; Jiangchao Yao — SJTU *(mapping from HKBU repository, not the PDF)* |
| 37 | Evolutionary Discovery of RL Algorithms (2603.28416) | Alkis Sygkounas · Amy Loutfi · Andreas Persson — all Machine Perception and Interaction Lab, Örebro |
| 38 | SIA (2605.27276) | Hebbar, Manawat, Bhatia, Baskaran — Hexo Labs Palo Alto; Verboomen — Hexo Labs Brussels; Palanimalai — Hexo Labs Toronto; Ivanova — Univ. Oxford |
| 39 | Denominator Gaming (2605.09915) | Shan, Zheng, Xi, Zhu, Yu, W. Zhang, J. Lin — SJTU; Gao — Central South Univ.; Zeyu Zheng — CMU |
| 40 | Test-time Recursive Thinking (2602.03094) | Zhuang — MSR + UCSD; Singh, L. Liu, Shen, D. Zhang, Gao, W. Chen — Microsoft Research; Shang — UC San Diego |
| 41 | Combinatorial Innovation (2604.20548) | Shuai Chen · Chengzhi Zhang (corresponding) — Nanjing Univ. of Science and Technology |
| 42 | LLMs Have Made Failure Worth Publishing (2604.06236) | Sungmin Lee — no affiliation printed |
| 43 | DARWIN (2602.05848) | Henry Jiang — Georgia Tech, College of Computing (sole author) |
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
| AI co-scientist (2502.18864) | Gottweis et al. — Google Cloud AI Research; Weng, Tu, Hassabis, Kohli, Natarajan et al. — Google DeepMind; Popovici, Palepu et al. — Google Research; plus Stanford Medicine, Houston Methodist, Sequome, Imperial (51 authors) |
| Robin (2505.13400) | Ghareeb, Mitchener, Yiu, Szostkiewicz, Laurent, Razzak, White, Hinks, Rodriques — FutureHouse; Benjamin Chang — FutureHouse + Oxford |
| Speedrunning Benchmark (2506.22419) | Bingchen Zhao — Meta + Univ. Edinburgh; Despoina Magka, Minqi Jiang … Yoram Bachrach — Meta; Oisin Mac Aodha — Univ. Edinburgh |
| Agent0 (2511.16043) | Peng Xia, Zeng, J. Liu, Zhou, Huaxiu Yao — UNC-Chapel Hill; Qin, Xiong — Salesforce Research; Fang Wu — Stanford |
| AIRA (2507.02554) | Toledo, Hambardzumyan — FAIR at Meta + UCL; Josifoski, Baldwin, Magka, M. Jiang, Raileanu, Cancedda, Foerster, Bachrach et al. — FAIR at Meta; Hazra — Örebro (work done while at Meta); Stenetorp — UCL |
| Execution-Grounded Auto-Research (2601.14525) | Chenglei Si, Zitong Yang, Yejin Choi, Emmanuel Candès, Diyi Yang, Tatsunori Hashimoto — all Stanford |
| MLE-bench (2410.07095) | Chan Jun Shern, Chowdhury, Jaffe, Aung, Sherburn, Mays, Starace, K. Liu, Maksin, Patwardhan, Weng, Madry — all OpenAI |
