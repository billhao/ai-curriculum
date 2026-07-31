# Who Actually Does RSI Research: The Institutional Map (2026)

Every work cited in [`rsi-proposer-landscape-2026.md`](rsi-proposer-landscape-2026.md),
resolved to its **printed author affiliations**, grouped by institution, and ranked by
influence on recursive self-improvement research specifically — not by general AI prestige.

**41 works resolved. 40 read from a printed title page; 1 inferred.** Method: arXiv HTML
author blocks (`ltx_authors`) and PDF title pages, plus organization pages for the
non-paper items. Where a source prints a bare acronym (`SII`, `GAIR`, `DeepMind`), this
document keeps it bare and marks any expansion as inference.

---

## 0. The headline

```
The groups that BUILD self-improving systems and the groups that publish the
evidence those systems don't work are almost entirely different institutions —
and they are in different countries.

  BUILD (largest teams, China + Japan)     CRITIQUE (smallest teams, US + UK)
  ─────────────────────────────────────    ─────────────────────────────────────
  AutoSOTA          16 authors  Tsinghua   Zenil limits       1 author   KCL
  HypoArena         18 authors  ISCAS      Dead Science       1 author   UCSC
  Safety taxonomy   15 authors  ZJU        Barriers-Diversity 3 authors  Columbia
  ASI-Arch           7 authors  SJTU       Reward Hacking     4 authors  Weco AI
  ShinkaEvolve       3 authors  Sakana     Diversity Collapse 8 authors  NUS
```

One person with no lab can invalidate a fifteen-person system's premise. Since **nothing
in the source document has been independently reproduced**, the critique layer keeps
outpacing the systems layer at roughly a tenth of the cost.

The three groups that do both — Sakana, Weco, Anthropic — are the top of this ranking,
and that is not a coincidence.

---

## 1. The ranking

Scored 0–100 on five weighted criteria: **shortlist weight** (35% — position in the source
doc's 1–18 must-read list, background anchors at 40% weight), **load-bearing-ness** (25% —
do the doc's five conclusions actually depend on this group's result), **volume/breadth**
(15%), **reusable artifact** (15% — open code, benchmarks, eval substrate others use), and
**structural agenda** (10%).

### Tier 1 — the load-bearing core

| # | Group | Type | Country | Score | Works |
|---|-------|------|---------|-------|-------|
| 1 | **Sakana AI** | industry lab | Japan | 84 | ShinkaEvolve, AC/DC, DGM (co) |
| 2 | **Weco AI** | startup (8 people) | not printed | 78 | Reward Hacking, SpecBench, AIDE |
| 3 | **Anthropic** | frontier lab | US | 70 | Automated Weak-to-Strong Researcher |
| 4 | **Meta / FAIR at Meta** | frontier lab | US | 66 | Speedrunning Benchmark, AIRA |

### Tier 2 — programme owners

| # | Group | Type | Country | Score | Works |
|---|-------|------|---------|-------|-------|
| 5 | **University of Cambridge** (+Flower Labs) | university | UK | 63 | Red Queen Gödel Machine |
| 6 | **Stanford University** | university | US | 60 | SLDAgent (senior), Execution-Grounded, Agent0 (co) |
| 7 | **Peking University** (+Wizard Quant) | university | China | 56 | SLDAgent (lead), AutoSOTA (co), Gödel Agent (co) |
| 8 | **SJTU / SII / GAIR** | university + institute | China | 54 | ASI-Arch |
| 9 | **METR** | nonprofit eval | US | 52 | Time Horizon 1.1, HCAST, RE-Bench |

### Tier 3 — one load-bearing result, or major infrastructure

| # | Group | Type | Country | Score | Works |
|---|-------|------|---------|-------|-------|
| 10 | **National University of Singapore** (+CUHK-Shenzhen) | university | SG | 51 | Diversity Collapse in MAS |
| 11 | **Tsinghua University** (3 separate labs) | university | China | 48 | AutoSOTA (lead), SLDAgent (co), Safety (co) |
| 12 | **UC Santa Barbara** (2 labs) | university | US | 46 | GEA, Gödel Agent |
| 13 | **Columbia Business School** | university | US | 45 | Barriers to Diversity in LLM Ideas |
| 14 | **UBC + Vector Institute** | university | Canada | 45 | Darwin Gödel Machine |
| 15 | **UC Riverside** (+AlphaAvatar, Illinois Tech) | university | US | 44 | RSI survey (1,250 papers) |
| 16 | **Google DeepMind** | frontier lab | US/UK | 43 | AlphaEvolve, AI co-scientist |
| 17 | **KAUST + IDSIA** | university | SA/CH | 42 | ICLR 2026 RSI workshop (the venue itself) |
| 18 | **Zhongguancun Academy** | state institute | China | 41 | AutoSOTA (senior), ForeSci (senior) |
| 19 | **OpenAI** | frontier lab | US | 40 | MLE-bench (the substrate, not in the doc) |
| 20 | **Pangram Labs** | vendor | US | 40 | ICLR 2026 AI-review census |

### Tier 4 — specialists and the tail

| # | Group | Country | Score | Works |
|---|-------|---------|-------|-------|
| 21 | ISCAS + UCAS + Alibaba | China | 38 | HypoArena / Before the Action |
| 22 | MBZUAI | UAE | 38 | CausalEvolve (lead), Red Queen (co) |
| 23 | UC Santa Cruz — Kargi Chauhan | US | 38 | Dead Science Walking |
| 24 | King's College London — Hector Zenil | UK | 37 | On the Limits of Self-Improving |
| 25 | Zhejiang University + Ant Group | China | 35 | Safety in Self-Evolving Agents |
| 26 | Hong Kong Baptist University — TMLR Group | HK | 34 | Learning to Evolve, CausalEvolve (co) |
| 27 | Harvard University (+MIT) | US | 32 | ArchEval |
| 28 | University of Edinburgh | UK | 30 | Speedrunning Benchmark (co-first author) |
| 29 | Carnegie Mellon University | US | 29 | CausalEvolve (senior, joint), Denominator Gaming |
| 30 | SJTU — Zhang/Yu/Lin group | China | 28 | Denominator Gaming, Learning to Evolve (co) |
| 31 | University College London | UK | 28 | AIRA (co-first author) |
| 32 | UNC-Chapel Hill (+Salesforce) | US | 27 | Agent0 |
| 33 | FutureHouse (+Oxford) | US | 26 | Robin |
| 34 | Wizard Quant | China | 25 | SLDAgent (4 of 11 authors) |
| 35 | Microsoft Research (+UC San Diego) | US | 24 | Test-time Recursive Thinking |
| 36 | Örebro University | Sweden | 23 | Evolutionary Discovery of RL Algorithms, AIRA (co) |
| 37 | Hexo Labs (+Oxford) | US/BE/CA | 22 | SIA |
| 38 | Southeast University (+Duke Kunshan) | China | 22 | ForeSci (lead) |
| 39 | Illinois Institute of Technology | US | 21 | RSI survey (last author) |
| 40 | University of Würzburg | Germany | 18 | Delta-Based NAS |
| 41 | Nanjing Univ. of Science and Technology | China | 17 | Combinatorial Innovation idea generation |
| 42 | Georgia Tech — Henry Jiang | US | 14 | DARWIN |
| 43 | Sungmin Lee (no affiliation printed) | — | 12 | LLMs Have Made Failure Worth Publishing |

**Co-affiliation tail** — printed on a paper but holding no lead or senior slot: NVIDIA
and Inria (Red Queen), Taptap (ASI-Arch), Apple / ByteDance / Tencent / BAAI / Scale /
Anuttacon / NYU / CUHK / Meta Reality Labs (RSI workshop committee only), University of
Sydney SAIC (CausalEvolve), Hangzhou Dianzi / NTU Singapore / Fudan (Safety), USTC
(AutoSOTA), Central South University (Denominator Gaming), Duke Kunshan (ForeSci),
University of Arizona (Gödel Agent), Stanford Medicine / Houston Methodist / Sequome /
Imperial (AI co-scientist), AlphaAvatar (RSI survey).

---

## 2. The top nine, and why

### 1. Sakana AI — the reference implementation layer

Three works, three authors on the flagship, and the highest artifact score in the map.
**ShinkaEvolve** (Lange, Imajuku, Cetin — all Sakana, no university co-affiliation) is the
open, Apache-2.0, fully-ablated counterpart to DeepMind's unreleased AlphaEvolve; it is the
one system in this landscape that other people can actually run. **AC/DC** (Dai, Meinardus,
Regan, Y. Tian, Tang) is published at **ICLR 2026 main track** — the highest venue of any
in-scope work here, above every one of the 18 shortlist entries, which are preprints and
one workshop poster. And via Robert Lange, Sakana co-owns the **Darwin Gödel Machine**
lineage with Clune's UBC group.

It also supplies half of the doc's genuine-discovery evidence: the discovered MoE
load-balancing loss. Builder *and* self-critic — ShinkaEvolve ships a novelty-rejection
filter that concedes the diversity-collapse critique before anyone made it.

### 2. Weco AI — an eight-person company holding the field's most-cited number

Weco owns **both** works that measure reward hacking (Reward Hacking in Self-Improving Code
Agents at 73.8%/46.8%, and SpecBench at +28 pp per 10× code size) — and it also owns
**AIDE**, the coding-agent scaffold that OpenAI's MLE-bench, METR's RE-Bench and Meta's
aira-dojo all use as their reference harness. That combination is unique: Weco builds the
agent the field measures with, then publishes the measurement showing that agent's gains
are mostly proxy-hacking.

**The affiliation caveat is real and it is the biggest single uncertainty here.** The
Reward Hacking paper's own title page has never been read — OpenReview is behind a bot
challenge (`403 ChallengeRequiredError`, re-tested 2026-07-31), there is no arXiv version
and no OpenAlex record. "Weco AI" comes from SpecBench (identical four authors, printed
`Weco AI`) plus Weco's team page, which lists all four as staff: **Zhengyao Jiang
(co-founder/CEO), Yuxiang Wu (co-founder/CTO), Dhruv Srikanth, Bingchen Zhao**. That is
strong, but Bingchen Zhao carries a dual Meta + Edinburgh affiliation on the Speedrunning
Benchmark, so a university co-affiliation on the poster is possible.

Two further discounts, both from the audit pass: the work is **#58 on the workshop's
110-paper accepted list** (poster tier — below 4 orals and 21 spotlights, several of which
belong to groups ranked lower here); and Meta's AIRA measured the same proxy-vs-held-out
generalization gap in July 2025, in a dedicated section (§5.3, "The Generalization Gap:
Searching with a Proxy Evaluation"), with released code — oracle final-node selection buys
**9–13 absolute points** over validation-guided selection, and 9.4%/12.4% for AIRA-mcts and
AIRA-evo end to end (15%/16.6% for the greedy variants, including AIDE-greedy, i.e. Weco's
own scaffold). Bingchen Zhao is not an AIRA author, but he is co-first author of the
Speedrunning Benchmark from the same Meta program (both papers last-authored by Yoram
Bachrach). The 2026 number is a sharper restatement of a 2025 measurement, not a first
observation.

### 3. Anthropic — one appearance, maximum verifiability

Wen, Qiu, Benton, Kirchner, Leike. Both Kirchner and Leike are printed as Anthropic-only;
the `Anthropic Fellows Program` superscript belongs to **Liang Qiu alone**. One
non-archival blog post, but every load-bearing element is directly readable at source, code
is released (`safety-research/automated-w2s-research`), and it is the only case in the map
where the builder published its own three unpredicted reward-hacking exploits *and* the
transfer failure (+0.5 points at production scale, inside the noise floor). Compare that to
the #1 shortlist work, whose numbers nobody has read at source.

### 4. Meta — owns the measuring stick, under-credited everywhere

The **Speedrunning Benchmark** (title page prints bare `Meta` + `University of Edinburgh`
— not "FAIR") supplies the original evidence for the doc's first conclusion: agents can
implement but cannot propose. **AIRA** (`FAIR at Meta` + UCL + Örebro, 26 authors) supplies
`aira-dojo` and the 2025 pre-measurement of the reward-hacking phenomenon. Two reusable
eval suites, two lead positions, and no in-scope 2026 paper — the entire discount on this
group is the background-anchor weighting.

### 5. University of Cambridge — the structural answer, no artifact yet

Red Queen Gödel Machine: 13 authors, printed affiliation key `1 University of Cambridge,
2 NVIDIA, 3 Flower Labs, 4 MBZUAI, 5 Inria`. Nicholas D. Lane is 1,3 (Cambridge + Flower
Labs); three equal-contribution first authors are all Cambridge. Co-evolving evaluators is
the map's only structural fix for the proposer-games-the-metric problem, and it is a named
research programme rather than a one-off — but nothing reusable has shipped.

*(The lab name "CaMLSys" is legible only as ring text on the paper's logo and is not in the
affiliation footnote; it is not treated as a printed affiliation here.)*

### 6. Stanford — the cleanest builder-and-critic in the map

Senior slot on **SLDAgent** (James Zou is last author; Haotian Ye is co-first), sole
institution on **Execution-Grounded Automated AI Research** (Si, Yang, Choi, Candès, Yang,
Hashimoto — all Stanford), plus a share of Agent0. The same Si/Hashimoto group produced the
Ideation-Execution Gap, the cleanest falsification in the whole corpus. It builds the
systems and publishes the results that undercut them.

### 7. Peking University — the lead slot on the strongest novelty claim

Haowei Lin (PKU + Wizard Quant) is co-first author of SLDAgent, the doc's strongest
existence proof that an LLM loop produced a better scientific generalization than the human
one (R² 0.748 vs 0.517). Five of SLDAgent's eleven authors are PKU; **four are Wizard
Quant**, a quantitative fund — compute and engineering money entering academic RSI, and
more authors on that paper than Stanford's two.

### 8. SJTU / SII / GAIR — the biggest run, the weakest evaluator

ASI-Arch's title page prints four *separate* numbered institutions: `1 Shanghai Jiao Tong
University, 2 SII, 3 Taptap, 4 GAIR`, under a `SII-GAIR` banner. Pengfei Liu (corresponding
author) holds 1, 2 and 4 jointly. The largest architecture-discovery run in the field, and
the canonical example of evaluator failure — an LLM judge is one third of its fitness
function, which is exactly what Red Queen exists to fix.

### 9. METR — the only independent longitudinal measurement

The Time Horizon 1.1 report is **corporately authored** — the page's only author field
reads `METR`, with no individual bylines. METR owns two eval suites (HCAST/time-horizons
and RE-Bench), one of which is a harness Weco's AIDE is measured in. It is the most
structurally independent node in the map: a nonprofit whose entire output is measurement,
publishing the capability trend line against which any claimed RSI speedup has to be read.

---

## 3. Every work → its printed institutions

| Work | First author | Institutions as printed |
|---|---|---|
| RSI survey (2607.07663) | Mingguang Chen | UC Riverside¹ · AlphaAvatar² · Illinois Tech³ |
| Reward Hacking (OpenReview) | Bingchen Zhao | **inferred** Weco AI (title page unreadable) |
| Automated W2S Researcher | Jiaxin Wen | Anthropic; Anthropic Fellows Program (Qiu only) |
| ShinkaEvolve (2509.19349) | Robert T. Lange | Sakana AI |
| Red Queen GM (2606.26294) | Alex Iacob | Cambridge¹ · NVIDIA² · Flower Labs³ · MBZUAI⁴ · Inria⁵ |
| SLDAgent (2507.21184) | Haowei Lin | PKU · Stanford · Wizard Quant · Tsinghua |
| SpecBench (2605.21384) | Bingchen Zhao | Weco AI |
| Diversity Collapse (2604.18005) | Nuo Chen | NUS · CUHK-Shenzhen |
| AutoSOTA (2604.05550) | Yu Li | Tsinghua EE/BNRist¹ · Zhongguancun Academy² · PKU³ · USTC⁴ |
| GEA (2602.04837) | Zhaotian Weng | UC Santa Barbara (all six authors) |
| Safety Self-Evolving (2606.23075) | Ruixiao Lin | ZJU¹ · Ant Group² · Tsinghua³ · HDU⁴ · NTU⁵ · Fudan⁶ |
| Barriers to Diversity (2602.20408) | Yuting Deng | Columbia Business School |
| Dead Science Walking (2606.04220) | Kargi Chauhan | UC Santa Cruz |
| HypoArena (2607.15766) | Tianyun Zhong | UCAS¹ · ISCAS² · Alibaba Group³ |
| ForeSci (2606.00644) | Qiuyu Tian | Southeast Univ.¹ · Zhongguancun Academy² · Duke Kunshan³ |
| CausalEvolve (2603.14575) | Yongqiang Chen | MBZUAI¹ · CMU² · HKBU TMLR³ · Univ. Sydney SAIC⁴ |
| Zenil limits (2601.05280) | Hector Zenil | King's College London · Oxford Immune Algorithmics |
| ArchEval (2607.03601) | Chenyu Wang | Harvard¹ · MIT² |
| ASI-Arch (2507.18074) | Yixiu Liu | SJTU¹ · SII² · Taptap³ · GAIR⁴ |
| Delta-Based NAS (2605.04903) | S. P. Adhikari | Computer Vision Lab, CAIDAS & IFI, Univ. Würzburg |
| AC/DC (2604.14969) | Andrew Dai | Sakana AI (ICLR 2026 main track) |
| Evo. RL Algorithms (2603.28416) | Alkis Sygkounas | Machine Perception & Interaction Lab, Örebro |
| Test-time Recursive (2602.03094) | Yufan Zhuang | Microsoft Research¹ · UC San Diego² |
| DARWIN (2602.05848) | Henry Jiang | Georgia Tech, College of Computing (sole author) |
| SIA (2605.27276) | Prannay Hebbar | Hexo Labs Palo Alto/Brussels/Toronto · Oxford |
| Failure Worth Publishing (2604.06236) | Sungmin Lee | **none printed** (gmail contact only) |
| Denominator Gaming (2605.09915) | Rong Shan | SJTU¹ · Central South Univ.² · CMU³ |
| Combinatorial Innovation (2604.20548) | Shuai Chen | Nanjing Univ. of Science and Technology |
| Learning to Evolve (OpenReview) | Xuan Li | HKBU · SJTU *(from HKBU repository, not the PDF)* |
| METR Time Horizon 1.1 | — | METR (corporate authorship, no bylines) |
| Pangram ICLR census | Bradley Emi | Pangram Labs (sole byline) |
| ICLR 2026 RSI workshop | Mingchen Zhuge | KAUST · Anuttacon · ByteDance · Apple · NYU/DeepMind · CUHK · BAAI · Scale · Tencent · Meta Reality Labs · KAUST/IDSIA (Schmidhuber) |

**Background anchors** (referenced by the doc, covered in companion files):

| Work | First author | Institutions as printed |
|---|---|---|
| Darwin Gödel Machine (2505.22954) | Jenny Zhang | UBC · Vector Institute · Sakana AI · CIFAR AI Chair |
| AlphaEvolve (2506.13131) | Alexander Novikov | Google DeepMind |
| Gödel Agent (2410.04444) | Xunjian Yin | UC Santa Barbara¹ · PKU² · Univ. Arizona³ |
| AI co-scientist (2502.18864) | Juraj Gottweis | Google Cloud AI Research · DeepMind · Google Research · Stanford Medicine · Houston Methodist · Sequome · Imperial |
| Robin (2505.13400) | Ali E. Ghareeb | FutureHouse¹ · Univ. Oxford² |
| Speedrunning Benchmark (2506.22419) | Bingchen Zhao | Meta¹ · Univ. Edinburgh² |
| Agent0 (2511.16043) | Peng Xia | UNC-Chapel Hill · Salesforce Research · Stanford |
| AIRA (2507.02554) | Edan Toledo | FAIR at Meta¹ · UCL² · Örebro³ |
| Execution-Grounded (2601.14525) | Chenglei Si | Stanford University (all six authors) |

---

## 4. Structural findings

**1. Builders and critics are different institutions — with exactly six exceptions.**
Systems come from Sakana, Tsinghua, SJTU-GAIR, MBZUAI, UCSB, Zhongguancun, DeepMind. The
falsifying results come from a business school (Columbia), a solo student (UCSC), a solo
theorist (KCL), a Singapore database group (NUS) and an AI-detection vendor (Pangram) —
none of whom build RSI systems. The exceptions are where the load-bearing numbers live:
**Weco** (ships AIDE, publishes its 74% proxy-hacking rate), **Sakana** (ships ShinkaEvolve
with the novelty filter built in), **Anthropic** (publishes its own three unpredicted
exploits), **Meta** (builds AIRA agents, publishes the generalization gap), **Stanford**
(builds SLDAgent, publishes the Ideation-Execution Gap), and **SJTU-GAIR** (whose ASI-Arch
limitations section is the doc's own source for the evaluator critique).

**2. Evaluation infrastructure has three owners and no shared substrate.**

```
Meta      llm-speedrunner, aira-dojo    ← general agentic-research benchmarks
METR      HCAST/time-horizons, RE-Bench ← longitudinal capability measurement
OpenAI    MLE-bench                     ← the ML-engineering substrate
─────────────────────────────────────────────────────────────────────────────
everything else: one-off, single-paper benchmarks —
SpecBench (Weco), ArchEval (Harvard), HypoArena (ISCAS), ForeSci (Zhongguancun),
SLDBench (PKU/Stanford)
```

All three of the general suites evaluate the **same** scaffold — Weco's AIDE — which is the
field's only real cross-group interoperability. No group's *system* is evaluated on another
group's *benchmark*, which is precisely why nothing in the source doc has been reproduced.

**3. Frontier labs are peripheral to the 2026 frontier.** Anthropic appears once (a blog
post). DeepMind's two entries are both 2025 anchors, and AlphaEvolve has never been
released, so nobody can build on the system that set the agenda. Meta publishes only
benchmarks. **OpenAI appears nowhere as an author, organizer or panelist** — yet it owns
MLE-bench, the substrate Meta's flagship agent is scored on, and GPT-5/5.5/Codex are the
systems under test in ArchEval, SLDAgent and Test-time Recursive Thinking. The frontier
labs are the *subject* of this literature far more than its authors. The actual 2026 work
comes from startups (Weco, Sakana, Hexo, Pangram), mid-size academic groups and nonprofits
— the organizations with the least compute.

**4. Connectivity runs through about five people, not through institutions.**

```
Robert Lange       Sakana ShinkaEvolve ──── UBC Darwin Gödel Machine
Bingchen Zhao      Meta Speedrun Bench ──── Weco Reward Hacking + SpecBench
                                       └─── Univ. Edinburgh
Bo Han             HKBU Learning to Evolve ─ MBZUAI/CMU CausalEvolve
Kun Zhang          MBZUAI ──────────────── CMU (joint, senior on CausalEvolve)
Nicholas D. Lane   Cambridge ───────────── Flower Labs
Pengfei Liu        SJTU ─── SII ─── GAIR (all three, one person)
```

Remove those six and the institutional graph fragments into isolated components. There are
no lab-to-lab agreements or shared platforms holding it together.

**5. Chinese state-backed institutes are quietly acquiring the senior slots.**
**Zhongguancun Academy** holds the last-author position on AutoSOTA (Tie-Yan Liu) and the
corresponding-author position on ForeSci; **SII** co-anchors ASI-Arch with SJTU and GAIR.
Neither is visible to any ranking keyed on university names, yet both are where seniority
and compute are concentrating.

**6. The venue is a single point of failure.** The ICLR 2026 RSI workshop is the field's
only dedicated venue, organized around Schmidhuber's Gödel-machine lineage
(Zhuge at KAUST, Schmidhuber at KAUST/IDSIA), with Apple the most-represented company on
the committee and no OpenAI, Anthropic or Sakana on it. The doc's #1 result exists **only**
as poster #58 there — no arXiv version, no OpenAlex record, no released code, on a
login-walled OpenReview. If that venue does not recur, the field's central empirical claim
has no durable citable home.

---

## 5. Country and role split

```
                 systems   benchmarks   negative     eval           venue
                 built     built        results      infrastructure
──────────────────────────────────────────────────────────────────────────────
China            ███████   ████         ██           ·              ·
US               ███       ██           ████         ███            ·
UK               █         ·            ██           ·              ·
Japan            ██        █            ·            ·              ·
Singapore        ·         ·            █            ·              ·
Canada           █         ·            ·            ·              ·
UAE / Saudi      █         ·            ·            ·              █
Europe (other)   ██        ·            ·            ·              ·
```

- **China** (Tsinghua, PKU, SJTU, ZJU, ISCAS, Alibaba, Ant, Zhongguancun, HKBU, NJUST, SEU,
  CUHK-SZ) supplies nearly all the large multi-author systems and benchmarks, and almost no
  negative results. The two China-side critiques — SJTU's denominator gaming and ZJU's
  safety taxonomy — target the *publishing ecosystem* and *deployment risk*, not whether
  the systems work.
- **US** supplies the load-bearing critiques (Anthropic, Columbia, UCSC, Pangram) and both
  general eval suites (Meta, METR, plus OpenAI's MLE-bench).
- **UK** is small and disproportionately load-bearing: Zenil's formal ceiling argument,
  Cambridge's co-evolving evaluators, Edinburgh and UCL as co-first-author institutions on
  the two Meta benchmarks.
- **Japan** is one company. Sakana ranks #1 and owns the field's most reused open
  implementation — the only open counterpart to AlphaEvolve is Japanese.
- **UAE and Saudi Arabia** punch above their paper counts: MBZUAI on two works, KAUST
  convening the field's only venue.

Net: **capability claims are produced predominantly in China and Japan; the evidence that
those claims are proxy-hacked, homogeneous or contaminated is produced predominantly in the
US and UK, by teams an order of magnitude smaller — and neither side reproduces the
other's results.**

---

## 6. What would change this ranking

1. **Weco's affiliation (rank 2).** The single most consequential unresolved item. If the
   Reward Hacking poster carries a university co-affiliation — plausible, since Bingchen
   Zhao is Meta + Edinburgh on the Speedrunning Benchmark — Weco's score splits and it
   falls to 3–4, with Edinburgh entering the top 25. The audit pass argued for rank 3 on
   these grounds; rank 2 is retained because Weco's **AIDE** is the reference harness inside
   all three general eval suites, an artifact contribution independent of the poster.
2. **Anchor weighting (40%).** At full weight, Meta rises ~12 points into the top 3, Google
   DeepMind ~10, UBC ~9, Stanford ~7. The current order deliberately privileges in-scope
   2026 work.
3. **RSI survey attribution (rank 15).** UC Riverside is credited because the first author
   is *also* the corresponding author — atypical. Under the standard last-author convention
   the credit goes to Illinois Tech instead, and the two entries swap.
4. **AutoSOTA's senior slot.** Last author is Tie-Yan Liu (Zhongguancun only), but the two
   printed corresponding authors are Fengli Xu and Yong Li (Tsinghua + Zhongguancun).
   Crediting the corresponding authors instead moves ~6 points from Zhongguancun to
   Tsinghua.
5. **SLDAgent version drift.** Institutions are from v5 (Jan 2026, 11 authors, 4
   institutions). v1 had 4 authors, no Stanford and no Wizard Quant — Stanford's senior slot
   exists only from v2 onward.
6. **ForeSci version drift.** v1's last author is Haojie Yin (Duke Kunshan); v2's is Zequn
   Liu (Zhongguancun, corresponding). The v2 mapping is used here.
7. **Tsinghua is three unrelated labs**, not one. AutoSOTA is Yong Li's FIB Lab (EE/BNRist);
   SLDAgent's Tsinghua author is Jianzhu Ma; the Safety paper's are Qi Li and Ke Xu
   (network security). Scored at parent level; scored per-lab, each drops to ~rank 30.
8. **`SII` and `GAIR` are bare unexpanded acronyms** on the ASI-Arch title page, printed as
   institutions 2 and 4 with SJTU as 1. Treating them as one SJTU-hosted entity is
   inference from Pengfei Liu's joint appointment. If SII is fully independent, rank 8
   splits into two entries around rank 14–16.
9. **Pangram's census is vendor-produced** by the company selling the detector used to
   produce it, with self-reported false-positive rates. It is an unaudited commercial
   dependency inside a load-bearing conclusion.
10. **Weco AI's country is not printed anywhere** — so the group supplying the most
    load-bearing critique is unassigned in the country split. A UK assignment would shift
    the headline finding.

---

## 7. Corrections to the source document

Things the affiliation pass turned up that bear on
[`rsi-proposer-landscape-2026.md`](rsi-proposer-landscape-2026.md) itself:

| Claim in the doc | Correction |
|---|---|
| Red Queen "Iacob, Jovanović, … N. D. Lane" as a Cambridge work | 5 institutions: Cambridge, **NVIDIA**, Flower Labs, MBZUAI, Inria — NVIDIA has two authors |
| ASI-Arch "SII / GAIR / SJTU" | Four separate numbered affiliations including **Taptap**; banner reads `SII-GAIR` |
| SLDAgent as a single-group work | PKU + Stanford + **Wizard Quant** (4 of 11 authors, a quant fund) + Tsinghua |
| SpecBench authors "B. Zhao, Srikanth, Y. Wu, Z. Jiang" | Same four people as the #1 Reward Hacking work — one team, two papers, not two datapoints |
| Reward Hacking flagged as "search-snippet only" | Still unread as of 2026-07-31; OpenReview forum, PDF and both API endpoints all return 403 |
| Reward Hacking as the field's central result | It is **poster #58 of 110** at the workshop (tiers: 4 orals, 21 spotlights, 85 posters); Meta's AIRA §5.3 measured the same proxy/held-out gap in Jul 2025 at 9–13 absolute points, with code |
| CausalEvolve "ICLR 2026 RSI workshop spotlight" | Confirmed — #17 in the workshop's 21-paper spotlight tier |
| AC/DC listed as an ordinary 2026 preprint | Published at **ICLR 2026 main track** — the highest venue of any work in the doc |
| Speedrunning Benchmark as a Meta work | Prints bare `Meta` (not FAIR) + `University of Edinburgh`; Bingchen Zhao is co-first author on both this and Weco's two critiques |
| METR TH1.1 | Corporate authorship, no individual bylines; METR also owns RE-Bench, unmentioned |
| 2604.20548 (untitled in the doc's flags) | "Enhancing Research Idea Generation through Combinatorial Innovation", Chen & Zhang, Nanjing Univ. of Science and Technology |
| Pangram census attributed to "Pangram Labs" | Sole byline: **Bradley Emi** (CTO/co-founder) |

---

## Confidence

**Read from a printed title page or author block (40 of 41):** every arXiv work above, via
`arxiv.org/html/<id>` author blocks or PDF page 1; the Anthropic post's inline byline map;
Weco's, METR's, Pangram's and the workshop's own pages.

**Inferred, not read (1):** Reward Hacking in Self-Improving Code Agents → Weco AI, from
the identical four-author set on SpecBench plus Weco's team roster.

**Inferred and marked as such:** `SII` → Shanghai Innovation Institute; `GAIR` bound to
SJTU; `DeepMind`/`Scale` expansions on the workshop committee; Learning to Evolve's
author→institution map (from HKBU's institutional repository, not the login-walled PDF);
Weco's and Hexo's countries; every country column entry, since only ArchEval, ForeSci,
Robin, AI co-scientist, Delta-NAS and the Örebro paper print a country line at all.

Two adversarial audit passes challenged this data; where they disagreed, the disputed items
were re-fetched directly. The audits' claims that UCSB (GEA, Gödel Agent) and UCSC (Dead
Science Walking) were misattributed did **not** survive re-fetching — both are printed as
recorded. Their corrections on MLE-bench (OpenAI, not Meta), bare `Meta` on the Speedrunning
Benchmark, the CaMLSys logo, and the Gödel Agent lead slot (UCSB, not PKU) did survive and
are applied above.

**Load-bearing scores measure how much the source document's conclusions depend on a
group's result — not whether that result is true.** Nothing in this map has been
independently reproduced.
