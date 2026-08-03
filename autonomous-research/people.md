This doc captures major research groups and people in RSI, AI for science etc.

Affiliations are as printed on the paper (arXiv HTML author block or PDF page 1). Bare
acronyms are kept bare. Blank cells mean not yet researched, not "none". Source map:
[`rsi-proposer-landscape-2026.md`](rsi-proposer-landscape-2026.md).

**(PI)** marks the group lead, listed first in each table. Assigned only from printed
evidence — the last author or the marked corresponding author of that group's work here. No
mark means the senior slot isn't established from the papers, not that the group has no
lead. Unmarked on purpose: UC Riverside (first author is also corresponding), Edinburgh and
UCL (senior only within a Meta-led paper), SJTU Zhang/Yu/Lin, Southeast Univ., and the
sole-author works (UCSC, Georgia Tech, Sungmin Lee) where student vs faculty isn't printed.

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

## Other research groups

| Institution | Key people | Key papers |
|---|---|---|
| National University of Singapore (+CUHK-Shenzhen) | **Bingsheng He (PI)**; Nuo Chen, Yicheng Tong, Yufei He | Diversity Collapse in MAS (2604.18005) |
| Tsinghua University — three separate labs | **Yong Li (PI)**, **Fengli Xu (PI)** (FIB Lab, EE/BNRist, both corresponding on AutoSOTA); Jianzhu Ma; Qi Li, Ke Xu (network security) | AutoSOTA lead; SLDAgent co; Safety in Self-Evolving Agents co |
| UC Santa Barbara — two labs | **Xin Eric Wang (PI)**; **William Yang Wang (PI)** | GEA (2602.04837); Gödel Agent (2410.04444) |
| Columbia Business School | **Olivier Toubia (PI)**; Melanie Brucks, Yuting Deng | Barriers to Diversity in LLM Ideas (2602.20408) |
| UBC + Vector Institute | **Jeff Clune (PI)**; Jenny Zhang, Shengran Hu, Cong Lu | Darwin Gödel Machine (2505.22954) |
| UC Riverside (+AlphaAvatar, Illinois Tech) | Mingguang Chen (first author *and* corresponding — senior slot unclear); Licheng Wang; Bo Qu | RSI survey, 1,250 papers (2607.07663) |
| Google DeepMind | **Matej Balog (PI)** (AlphaEvolve last author, listed as its lead on the RSI-workshop panel page); **Vivek Natarajan (PI)** (AI co-scientist last author); Alexander Novikov, Juraj Gottweis | AlphaEvolve (2506.13131); AI co-scientist (2502.18864) |
| KAUST + IDSIA | **Mingchen Zhuge (PI)** (lead organizer); **Jürgen Schmidhuber (PI)** | ICLR 2026 RSI workshop — the field's only dedicated venue |
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
| Unaffiliated | Sungmin Lee (no institution printed) | LLMs Have Made Failure Worth Publishing (2604.06236) |

Co-affiliations printed on a paper but holding no lead or senior slot: NVIDIA and Inria
(Red Queen), Taptap (ASI-Arch), University of Sydney SAIC (CausalEvolve), Hangzhou Dianzi /
NTU Singapore / Fudan (Safety), USTC (AutoSOTA), Central South University (Denominator
Gaming), University of Arizona (Gödel Agent), Stanford Medicine / Houston Methodist /
Sequome / Imperial (AI co-scientist), AlphaAvatar (RSI survey), and the RSI-workshop
committee-only affiliations (Apple, ByteDance, Tencent, BAAI, Scale, Anuttacon, NYU, CUHK,
Meta Reality Labs).

# Major academic lineage (who is advisor of who)

Not yet researched. Edges to establish first, in rough order of how much they explain:

- Jürgen Schmidhuber (IDSIA/KAUST) → Mingchen Zhuge (KAUST) — the Gödel-machine lineage into the field's only venue
- Jeff Clune (UBC) → Jenny Zhang, Shengran Hu, Cong Lu — and the Sakana overlap (Hu, Lu, Lange all on DGM)
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
| 5 | Red Queen Gödel Machine (2606.26294) | Iacob, Shen, Kurmanji, Cao, Marino, Qiu — Cambridge; Jovanović, Sani, Lane — Cambridge + Flower Labs; Burkhardt, Venanzi — NVIDIA; Tastan — MBZUAI; Odonnat — Inria |
| 6 | SLDAgent (2507.21184) | Haowei Lin — PKU + Wizard Quant; Haotian Ye, James Zou — Stanford; Feng, Huang, Y. Li — Wizard Quant; Jianzhu Ma — Tsinghua; Liang, Lim, Z. Li, X. Wang — PKU |
| 7 | SpecBench (2605.21384) | Bingchen Zhao · Dhruv Srikanth · Yuxiang Wu · Zhengyao Jiang — all Weco AI |
| 8 | Diversity Collapse in MAS (2604.18005) | N. Chen, Tong, Y. He, Zou, Q. Wang, B. He — NUS; Y. Yang, X. Zhang — CUHK-Shenzhen |
| 9 | AutoSOTA (2604.05550) | Yu Li, X. Liu, R. Zhao, P. Liu, Z. Chen, Q. Yang, Zeng, T. Li, J. Xu — Tsinghua EE/BNRist; Shao, Su, Fengli Xu, Yong Li — Tsinghua + Zhongguancun; A. Xu — Zhongguancun + PKU; Fang — Zhongguancun + USTC; Tie-Yan Liu — Zhongguancun |
| 10 | Group-Evolving Agents (2602.04837) | Weng, Antoniades, Nathani, Z. Zhang, Pu, Xin Eric Wang — all UC Santa Barbara |
| 11 | Safety in Self-Evolving Agents (2606.23075) | R. Lin, Q. Li, Z. Li, Shouling Ji — Zhejiang; Deng, Qing, Q. Li, Ke Xu — Tsinghua; Feng, Cui, Meng — Ant Group; Ma — Hangzhou Dianzi; Y. Zhang, T. Zhang — NTU; X. Ma — Fudan |
| 12 | Barriers to Diversity in LLM Ideas (2602.20408) | Yuting Deng · Melanie Brucks · Olivier Toubia — all Columbia Business School |
| 13 | Dead Science Walking (2606.04220) | Kargi Chauhan — UC Santa Cruz (sole author) |
| 14 | HypoArena / Before the Action (2607.15766) | Zhong, W. Jiang — UCAS + ISCAS; X. Chen, Lu, H. Lin, Sun, Xianpei Han — ISCAS; W. Wang, Shi, B. Yang, J. Wang, H. Li, Zhai, B. Zhao, Hu Wei, H. Yu, Yongbin Li — Alibaba Group |
| 15 | ForeSci (2606.00644) | Qiuyu Tian — Southeast Univ. + Zhongguancun; Haojie Yin — Duke Kunshan; Yingce Xia, Zequn Liu (corresponding) — Zhongguancun; Youyong Kong — Southeast Univ. |
| 16 | CausalEvolve (2603.14575) | Yongqiang Chen — MBZUAI + CMU; Chenxi Liu, Bo Han — HKBU TMLR; Zhenhao Chen — MBZUAI; Tongliang Liu — Univ. Sydney SAIC + MBZUAI; Kun Zhang — MBZUAI + CMU |
| 17 | On the Limits of Self-Improving (2601.05280) | Hector Zenil — King's College London + Oxford Immune Algorithmics (sole author) |
| 18 | ArchEval (2607.03601) | C. Wang, Wan, Ma, Prakash, Qi, Do, Cheng, Tschand, Du, Reddi — Harvard; Shi — MIT |
| 19 | ASI-Arch (2507.18074) | Yixiu Liu, Weixian Xu, Lyumanshan Ye — SJTU + SII + GAIR; Yang Nan, Xiangkun Hu — SII + GAIR; Zhen Qin — Taptap; Pengfei Liu (corresponding) — SJTU + SII + GAIR |
| 20 | AC/DC (2604.14969) | Andrew Dai · Boris Meinardus · Ciaran Regan · Yingtao Tian · Yujin Tang — all Sakana AI (ICLR 2026) |
| 21 | Delta-Based NAS (2605.04903) | Santosh Premi Adhikari · Radu Timofte · Dmitry Ignatov — all Computer Vision Lab, CAIDAS & IFI, Univ. Würzburg |
| 22 | Learning to Evolve (OpenReview WnZHbe1Gu0) | Xuan Li, Zhanke Zhou, Zongze Li, Bo Han — HKBU; Jiangchao Yao — SJTU *(mapping from HKBU repository, not the PDF)* |
| 23 | Evolutionary Discovery of RL Algorithms (2603.28416) | Alkis Sygkounas · Amy Loutfi · Andreas Persson — all Machine Perception and Interaction Lab, Örebro |
| 24 | SIA (2605.27276) | Hebbar, Manawat, Bhatia, Baskaran — Hexo Labs Palo Alto; Verboomen — Hexo Labs Brussels; Palanimalai — Hexo Labs Toronto; Ivanova — Univ. Oxford |
| 25 | Denominator Gaming (2605.09915) | Shan, Zheng, Xi, Zhu, Yu, W. Zhang, J. Lin — SJTU; Gao — Central South Univ.; Zeyu Zheng — CMU |
| 26 | Test-time Recursive Thinking (2602.03094) | Zhuang — MSR + UCSD; Singh, L. Liu, Shen, D. Zhang, Gao, W. Chen — Microsoft Research; Shang — UC San Diego |
| 27 | Combinatorial Innovation (2604.20548) | Shuai Chen · Chengzhi Zhang (corresponding) — Nanjing Univ. of Science and Technology |
| 28 | LLMs Have Made Failure Worth Publishing (2604.06236) | Sungmin Lee — no affiliation printed |
| 29 | DARWIN (2602.05848) | Henry Jiang — Georgia Tech, College of Computing (sole author) |
| — | METR Time Horizon 1.1 | METR — corporate authorship, no individual bylines |
| — | Pangram ICLR 2026 review census | Bradley Emi — Pangram Labs (sole byline) |
| — | ICLR 2026 RSI workshop | Zhuge — KAUST; Schmidhuber — KAUST/IDSIA; Zeng — Anuttacon; Zhu — ByteDance; Zou — Apple; S. Yang — NYU/DeepMind; Hu — CUHK; M. Li — BAAI; He — Scale; L. Li — Tencent; Chandra — Meta Reality Labs |

Background anchors (covered in companion files):

| Paper | Author & affiliation pairs |
|---|---|
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
