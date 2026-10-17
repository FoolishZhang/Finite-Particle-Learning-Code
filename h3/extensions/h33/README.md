# Budget sensitivity and learned local proposal evidence

This additive v1.2.0 package supplies the evidence used in Appendices D.5 and
D.6 of the h3.3 research manuscript and subsequent wording revisions. It is
code and saved evidence, not a new experiment, a manuscript publication, an
anonymous supplement, or a claim of acceptance. The original v1.1.0 `h3/`
scientific files and their manifest remain byte-for-byte unchanged.

## Supported commands

Use Python 3.11 and the versions in [requirements.txt](requirements.txt).
From the repository root, choose fresh output paths outside this checkout:

```text
python -I -m pip --isolated install -r h3/extensions/h33/requirements.txt
python -I -B tools/verify_release.py
python -I -B h3/extensions/h33/tools/check.py --mode all --output ../h33-check-001.json
python -I -B h3/extensions/h33/tools/plot.py --output-dir ../h33-figures-001
```

The check also supports `--mode identity`, `saved`, and `imports`. Saved checks
use independent arithmetic and do not import the original model or analyzers.
Import mode loads the unchanged source closures in separate processes. None
of these commands samples particles, trains densities, evaluates likelihoods,
computes quadrature, or runs a scientific RNG. Plotting calls the original
plotters on published aggregates. It reports PNG byte comparisons; fonts and
library versions can change image bytes on another platform.

The older checks in [h3/README.md](../../README.md) remain available for the
original reversal and risk experiments. Their passing does not by itself
check this extension. Each file is bound by [MANIFEST](MANIFEST.json), with
selected original identities and transformations in [SOURCE_MAP](SOURCE_MAP.json).
Original source identifiers in data fields are provenance, not download paths.
The public [index](data/INDEX.json) identifies the material actually included.

## D.5: retrospective budget comparison

[Paired cases](data/budget/PAIRED_CASES.csv), [risks](data/budget/RISK_ANALYSIS.json)
and [diagnostics](data/budget/DIAGNOSTICS.json) retain all 24 original strong-signal
datasets, both starting points and all outputs. These are the same datasets
already published under `h3/data/observations/case_02000.csv` through
`case_02023.csv`; no new observed datasets or teachers were generated.

The reported quantity is 2048 times the GROUP-minus-integrated-ML joint KL
difference after averaging the two start-specific risks within each dataset.
The higher-budget comparison and its paired change from the original program
form a separate family of two, with 24 dataset units and 23 degrees of freedom.
Their selection followed the earlier results: this is retrospective numerical
sensitivity, not fresh confirmation or an equivalence test. The original
family of four is not replaced.

The original outputs passed 12/48 original diagnostics and 36/48 under the
new reference; higher-budget outputs passed 48/48. Diagnostic units are start
outputs, not independent datasets. The second numerical diagnostic is the
maximum coordinatewise replicate SE of the reference gradient, not the SE of
the mapping norm. Failures are retained and never filter risks. Passing this
diagnostic does not certify a global empirical minimizer.

[Scientific settings](data/budget/SCIENTIFIC_CONFIG.json) and the saved
[policy](data/budget/ANALYSIS_POLICY.json) identify budgets, unchanged solver
policy and common risk rules. Order comparisons are ordinary quadrature
qualifications, not rigorous integration-error bounds. The checker recomposes
saved risks and diagnostics; it does not reevaluate their integrals or rerun
the optimizer. The optional budget plot displays every paired case.

## D.6: frozen learned local proposals

Three independent local simulator training pools each supply 65,536 training
labels and an independent 32,768 validation labels. The smaller 16,384 training
sample is a prefix, so there are six fits from three pools, not six independent
training repetitions. [Frozen proposals](data/training/FROZEN_PROPOSALS.json),
the six checkpoint selections and all twelve raw/sym density parameter files
are included. They use a known local channel and simulated latent-observation
labels, not inference from joint observations alone or an amortized encoder.

The raw proposal is a two-component Gaussian mixture with defensive mass
0.05 on the standard normal. The sym proposal averages it with its reflected
local-state counterpart, producing four tied components. The same checkpoint
minimizes average raw/sym validation NLL over epochs 0, 5, 20 and 80. No candidate
contrast or target correlation selects the density. The checker reconstructs
this selection from saved scores and verifies the reflection parameter algebra;
it does not retrain or reevaluate validation NLL.

The later [precision study](data/precision/ANALYSIS_001.json) retains all twelve
chain comparisons (three pools, two nested sizes, two amplitudes). Fixed K=8,
direction (1,1,0)/sqrt(2) and alpha=0.3201223434446316 define the candidates.
It uses the actual nonconstant proposal weights and complete g/H/B matrices.
Independent coefficient estimates are frozen before a fresh validation stream.
The same-bank prediction uses origin derivatives from that validation bank.
Both teacher weights and origin derivatives refer to the same finite GH
probability function, rather than silently substituting exact-posterior scores.

The independent error is W_ind=(D-P_ind)/t^2; the same-bank Taylor remainder
is W_same=(D-P_same)/t^2. Their SEs use the 16 complete replicate differences,
retaining covariance. The approximate simultaneous 95% family contains 12 D,
12 W_ind and 12 W_same quantities, with df=15. Every precision interval lies
within the fixed +/-0.01 tolerance; all same-bank intervals also exclude zero.
An accurate approximation and a small resolved remainder therefore coexist.
Uncertainty is conditional on the six frozen proposals, not uncertainty over
training populations, finite-N learned estimators, or generative risk.

The full GH28/GH36 saved comparison concerns only the preselected pool-0,
16,384-training proposal. It is not a certificate for all six proposals or the
continuous integral. The published R-level records suffice to reconstruct its
complete-replicate differences, but not the original model scores.

The [earlier study](data/previous/ANALYSIS.json) remains a separate family24,
with its original 1/12 precision passes, full raw controls and unresolved
measurements. The other eleven do not become scientific counterexamples, and
the later family36 does not rewrite or pool them. The old controls and all
matrices remain inspectable; no new raw-control significance claims are made.

## Included scope and limits

|Level|Support|
|---|---|
|File identity|All explicitly listed extension files; original v1.1.0 payload preserved.|
|Saved arithmetic|Budget risks/diagnostics; checkpoint selection; original family24; complete new family36 from R-level matrices and contrasts; single-proposal ordinary rule comparison.|
|Scientific source|Unchanged source closures and original analyzers; isolated imports.|
|Figures|Original budget and precision plotters, saved inputs, archived PNG/SVG.|
|Not bundled|Raw proposal samples, local weights, uniform arrays, physical training arrays, full per-B derivative arrays, complete optimization traces and internal management contracts.|
|Not supported|Full original sampling, quadrature, optimizer or training replay; original per-B independent verifier; mathematical-proof certification.|

Historical runners and analyzers retain their original source/contract locks.
Their public scientific projections are deliberately not accepted as original
management contracts; do not disable those checks. No new large release asset
or historical data-asset replacement is needed for the supported commands.

Author: **ZHANG** (provisional display name). Original software: MIT. Original
documents, figures and synthetic data: CC BY 4.0. Existing third-party notices
remain unchanged. NumPy, SciPy, Matplotlib and psutil are installed dependencies.
