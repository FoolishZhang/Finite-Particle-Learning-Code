# Anonymous supplementary material: saved evidence and inexpensive checks

This directory is self-contained. It can be copied outside the source repository;
no repository checkout, Git history, private arrays, user configuration, or
original delivery scripts are required. It contains the public scalar and matrix
endpoints used by the paper, selected arithmetic routines, and the two displayed
figures. The package manifest binds these derived files to their own bytes.

The checks reconstruct arithmetic from **saved finite sums and analytic error
bounds**. They do not recompute production quadrature, generate new samples,
optimize parameters, or automatically prove the analytic error inequalities.
Successful package hashes alone establish file identity, not a mathematical
proof. Consult the retained proof appendices of the paper for the analytic
assumptions and error arguments. No new scientific experiment was performed to
prepare this supplement.

## Start here

Use CPython 3.11 with assertions enabled; do not use `-O` or `-OO`.
From this directory, the standard-library check needs no package installation:

```text
python -B check.py --level fraction
```

It checks exact rational composition for the angle witness, all four R02
remainder budgets, R03 teacher/root/risk/spectral arithmetic, and the figure
data schema. R03 square roots use integer square comparisons and outward
dyadic rounding. It does not import Arb, NumPy, or the scientific producers.

For the additional Arb checks, create a fresh virtual environment outside this
directory. On Linux/macOS use its `bin/python`; on Windows use
`Scripts/python.exe`:

```text
python -I -m venv ../review-env
../review-env/bin/python -I -m pip --isolated install -r requirements.txt
../review-env/bin/python -B check.py --level all
```

The full check adds the saved Hessian class aggregation, shifted matrix LDL,
closed-form Hessian error budget, root/minimum witness, KL root transfer,
R03 Arb spectral refinement, and Section 4.3's exact-integer sample-moment and
probability-bound composition. It explicitly verifies the paper's probability
lower bound of 0.7035, joint KL about 0.01043 and the certified curvature signs.
NumPy is required only by the extracted saved-matrix input guard in that route.
Matplotlib is required only for optional figure regeneration.

Both the full and standard-library checks took about 55 seconds in the tested
fresh Windows environment; most of the cost is exact R02 rational arithmetic.
Actual measured costs, interpreter, dependency versions and
tested platform are recorded in `VALIDATION.json`. Platform-independent command
syntax is provided; only the platform listed in that record was executed.

## What each result can be checked against

| Paper result | Saved inputs under `artifacts/` | Rebuilt layer and remaining premise |
|---|---|---|
| Theorem 3: root, local minimum modulo common rotation | `certified_f8_hessian_v1_20261008`, `local_witness_feasibility_v1_20261008`, `certified_f8_gradient_v1_20261008` | Sum all 41 saved class matrices; exact saved-matrix floors; elementary analytic Hessian error budget; fixed-radius root/minimum gates. Saved finite gradient and its certified error remain inputs. |
| Theorem 3: joint KL about 0.01043 | `certified_teacher_v1_20261008`, `submission_candidate_v1_8_20261008/RISK_RANGE_RESULT.json` | Sum all 256 probability-box KL and chi-square terms; Fisher/root-distance transfer to the same root. Teacher and candidate integral enclosures remain inputs, without renormalization. |
| Same-root fixed-length angle curvature | `angular_witness_v1_20261010` and teacher/candidate inputs above | 1,280 label interval containments; two numerator expressions and denominator agreement; four strictly negative summaries; rational transfer cap and exact algebra fixtures. Saved alias/tail bounds remain inputs. |
| R03 continuous teacher family, absolute parameter at most 0.001 | `r03_teacher_family_v1_20261011` | All 256 teacher/risk labels and 128 conditional candidate rows at the claimed box; signed matrix combinations; root inclusion/contraction; risk transfer; all three saved spectral boxes and quantitative consequences. Global derivative and integration bounds remain inputs. The wider 0.01 box remains unresolved. |
| R02 finite remainder | `r02_remainder_v1_20261010` | Sixteen saved Gaussian comparison sums, twenty error formulas and four final enclosures. The sign is certified at 4096; 64, 256 and 1024 remain unresolved. Saved noise and active-model certificates remain inputs. |
| Section 4.3 grid program | `failure_probability_v1_20261006` | Restore exact binary64 integer moments; confidence allowances and conditional Cantelli bound. Ideal iid sampling and numerical allowances remain conditions; banks and every score node are not replayed. The future native program's uniform error is not certified. |
| Figures 1 and 2 | `nuisance_contrast_v1_20261006`, `local_geometry_v1_20261006`, `asymptotic_bridge_v1_20261008` | Inspect plot endpoints and matched-selection record; regenerate plots from saved summaries. Statistical intervals are not reestimated and the matched selection is not rerun. |

The directory names label historical scientific objects, not a requirement to
run the original repository. The selected arithmetic routines in `scripts/`
have no producer entry points or original manuscript-delivery dependencies.
They retain their original mathematical operations; the package runner supplies
the selected local objects and verifies its own manifest. Provenance metadata
was removed from derived copies; the original evidence was not modified.

## Figures

`figures/finite_instance.png` is Figure 1 and
`figures/bridge_summary_en.png` is Figure 2. Their current bytes were matched to
the original plot-generation receipts before packaging. They are included as
the paper figures. To generate additional PNGs from the supplied data:

```text
python -B plot.py --output ../review-plots
```

The plotting command writes only to the supplied output directory. Rendering
may vary with fonts and Matplotlib; scientific coordinates and bars come from
the saved summaries. Byte-identical re-rendering is not a proof gate.

## Production replay boundary

This is an inexpensive verification package, not a complete production replay
environment. The original quadrature kernels, training experiments and large
sample banks are not included. Full independent production requires implementing
the quadrature/error formulas in the paper and separately budgeting computation;
no supplied command pretends to perform that work. Historical production
dimensions are visible in the saved configurations and work counters: 41
Hessian classes each using 881 by 769 factors; the R03 candidate box uses 128
labels each with 641 by 337 factors; the probability audit uses 262,144 banks.
These counts are not a runtime guarantee for a new implementation.

Omitting production kernels is an explicit choice of this package's low-cost
scope. Supplying a few kernels without their complete numerical qualification
and input dependencies would not establish independent production replay. The
package therefore resolves access to saved public evidence and its arithmetic,
while full production reproducibility remains a separate task.

This package offers selective scientific endpoint and arithmetic review.
It does not check typesetting or the manuscript delivery, establish a universal
optimizer result, resolve the unreported finite-step radius/gain, or replace an
independent review of the analytic proof.
