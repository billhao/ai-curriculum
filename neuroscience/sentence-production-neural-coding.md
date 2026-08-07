# Sentence Production: Hybrid Layout and Magnitude-Independent Coding

Morgan, Devinsky, Doyle, Dugan, Friedman, Flinker (NYU Grossman School of
Medicine / NYU Tandon), *Science Advances* 12(32), eaec0518, 5 August 2026.
[doi:10.1126/sciadv.aec0518](https://doi.org/10.1126/sciadv.aec0518)

## Takeaways

- In 10 epilepsy patients (1256 electrodes), two supposedly equivalent probes
  of "higher-order language" barely agree: sentence-vs-list flagged 60
  electrodes, active-vs-passive 125, only 6 both.
- Measuring *information* instead of *activity* found sentence structure at 133
  electrodes, event meaning at 92, word form at 126 — mostly different ones.
- Headline: structure and meaning content were unrelated to how hard an
  electrode fired (structure $P = 0.987$, semantics $P = 0.248$); word-form
  content tracked firing strength as expected ($P < 0.001$).
- Clustering 741 electrodes gave 5 networks: three quiet, broadly distributed,
  high-information ones plus two loud ones locked to the picture and to speech.
- Layout is "hybrid," distributed *and* focal, so activity-thresholded
  localizers miss the sites carrying higher-order linguistic information.

## Background

Almost all we know about language in the brain comes from *comprehension*
studies in scanners: speaking creates motion and muscle artifacts that wreck
fMRI and EEG. Production is poorly mapped, and the field is split between
"syntax lives in specific hubs" and "syntax is spread across the language
network." Lineage:
(1) [Sahin 2009](https://doi.org/10.1126/science.1174481) — word identity,
grammar, and sound processed in that order within millimeters of Broca's area;
(2) [Pallier 2011](https://doi.org/10.1073/pnas.1018711108) — response magnitude
scaling with amount of structure, the logic this paper breaks;
(3) [Morgan 2025](https://doi.org/10.1038/s44271-025-00270-1) — same lab and
paradigm, decoded the planned word and its subject/object role.

## What They Did

Ten patients (3 female, mean age 30, range 20–45) had electrode grids placed on
the left hemisphere before epilepsy surgery — **ECoG** (electrocorticography):
voltage read off the cortical surface, immune to speech motion artifacts, with
millimeter and millisecond resolution. The measure is **high-gamma** power, the
strength of 70–150 Hz fluctuations, tracking local firing and the fMRI signal.

```
Sentence (60): question -> cartoon -> [PLAN ~1141 ms] -> speech
  active  "Who sprayed whom?"        -> "The chicken sprayed Dracula"
  passive "Who was sprayed by whom?" -> "Dracula was sprayed by the chicken"
List (60): arrow -> same cartoon -> "chicken Dracula"  Naming (96): "chicken"
MAGNITUDE  sentence > list -> 60 elec;  active != passive -> 125;  both: 6
INFORMATION (RSA)  |dY| = b0 + b1*D_struct + b2*D_seman + b3*D_lex + b4*D_RT
                                    133         92          126  electrodes
```

Same picture, same meaning, same content words, different grammar. Analysis
targets the **planning period**, picture onset to speech onset; passives took
longer (median 1424 vs 1165 ms), so trials were time-warped to a common
duration. **RSA** (representational similarity analysis) is pairwise: for every
two trials, compute how different the neural activity was, then regress that on
how different the trials were in structure (active vs. passive), meaning
(distance between GPT-2 layer-8 sentence embeddings), and word form (same vs.
different first word), with reaction-time difference as a covariate.

## What They Found

**1. Hybrid spatial organization.** Neither side of the localization debate is
right: information sits across broad swaths of left peri-Sylvian cortex *and*
concentrates in specific regions.

| Information | Term | Focal regions | Tracks activity? |
|---|---|---|---|
| Sentence structure | D_struct | IFG, MFG | No ($P = 0.987$) |
| Event meaning | D_seman | MTG, IPL | No ($P = 0.248$) |
| Word form / sounds | D_lex | STG, SMC | Yes ($P < 0.001$) |

IFG/MFG = inferior/middle frontal gyrus; STG/MTG = superior/middle temporal
gyrus; IPL = inferior parietal lobule; SMC = sensorimotor cortex. Most
electrodes carried only one of the three, arguing against a single interwoven
system. Non-negative matrix factorization over 741 electrodes recovered the
split with no anatomy given: three clusters are broadly distributed and near
baseline in activity, each specialized for one information type, while the two
loud clusters — frontal at picture onset, sensorimotor at speech onset — carry
no structure or meaning information ($P > 0.05$).

**2. Magnitude-independent coding.** The field assumes — the authors call it
*elevation-processing equivalence* (EPE) — that a region working harder shows
more activity. That fails for higher-order language. Mechanically: an electrode
can sit at its resting mean all task long, yet its small trial-to-trial
deviations around that mean are reliably larger between an active and a
passive sentence than between two actives. The signal lives in the
*pattern of differences across trials*, not the *loudness*. Deep-learning
analogue: a hidden unit whose mean activation is identical across classes but
whose activation *direction* is class-informative — a linear probe reads it
out, a mean-activation heatmap shows nothing. Word form is the control that
behaves the old way, making this a dissociation, not a null result.

## Why It Matters

Much of language neuroscience finds regions by thresholding on activity, then
studies only those voxels or channels. If higher-order information is
magnitude-independent, that pipeline discards the relevant sites before
analysis begins. For speech brain-computer interfaces this is a
hypothesis, not a result — the paper neither cites nor tests BCI work. Current
neuroprostheses decode from loud sensorimotor sites, exactly where word-form
information lives here, so that choice looks well-founded; but structure and
meaning at quiet distributed sites would be invisible to activity-driven
channel selection.

On the AI parallel, stay honest. GPT-2 embeddings appear only as an
off-the-shelf meaning space, built from active-syntax sentences so the
regressor stayed orthogonal to the structure term. Nothing here says the brain
is a transformer. The real overlap is methodological — both fields moved from
"which unit lights up" to "what can be linearly read out." The divergence is as
real: a transformer carries syntax and semantics in one residual stream at the
same layers; this study finds them at non-overlapping cortical sites.

## Key Papers

1. Morgan 2026, *Sci. Adv.* — this paper.
   [doi](https://doi.org/10.1126/sciadv.aec0518) ·
   [preprint](https://doi.org/10.1101/2024.06.20.599931) ·
   [data](https://doi.org/10.5281/zenodo.20543385)
2. Sahin 2009, *Science* — staged lexical/grammatical/phonological processing in
   Broca's area. [doi](https://doi.org/10.1126/science.1174481) ·
   Morgan 2025, *Comms Psych* — decoding words during sentence production.
   [doi](https://doi.org/10.1038/s44271-025-00270-1)
3. Pallier 2011, *PNAS* — activity scaling with constituent structure.
   [doi](https://doi.org/10.1073/pnas.1018711108) · Matchin & Hickok 2020 —
   syntax-hub position. [doi](https://doi.org/10.1093/cercor/bhz180) ·
   Kriegeskorte 2008 — RSA method.
   [doi](https://doi.org/10.3389/neuro.06.004.2008) · Grill-Spector 2006 —
   repetition suppression. [doi](https://doi.org/10.1016/j.tics.2005.11.006)
