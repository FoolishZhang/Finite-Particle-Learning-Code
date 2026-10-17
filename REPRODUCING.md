## v1.2.0: compact h3.3 evidence

New D.5 budget and D.6 learned-proposal material is in [h3/extensions/h33](h3/extensions/h33/README.md). It adds saved-value checks, frozen densities and deterministic plots. The original v1.1.0 h3 scientific payload remains unchanged. Full sampling, integration, fitting and training replay are not claimed. Fresh credential-disabled retrieval and the supported checks have passed; see the [v1.2.0 network record](release/V1_2_0_NETWORK.json).

# Reproduction entrypoints

For the current proposal/conditional-posterior/local-risk evidence, use [h3/README.md](h3/README.md):

    python -I -B h3/tools/check.py --mode all --output ../h3-check-001.json
    python -I -B h3/tools/plot.py --output-dir ../h3-figures-001

These consume the compact data committed in h3. They do not require the historical large data assets or execute scientific fits. Full scientific replay is explicitly outside the supported h3 commands.

# Historical h2.2 production and verification levels

The original scientific modules are provided under `script/` with their original
paths and bytes. `tools/production_replay.py` is a thin caller of their functions;
it does not carry a second implementation of the quadrature or scoring kernels.
It avoids the historical authors' one-run Git, feedback and manuscript-delivery
locks by supplying the same scientific arguments directly to those functions.
It does not alter the scientific modules, frozen inputs, or reference results.

Use Python 3.11 and the release's pinned scientific dependencies. Do not disable
assertions with `-O` or `-OO`. The release was tested with an isolated interpreter
in a new environment with no system or user site packages. Windows was tested;
Linux and macOS were not tested in this release.

## Three different operations

1. Read the source, exact configurations and saved reference objects. The result
   map points to the production roots and their recursive local dependencies.
2. Rebuild saved arithmetic with `supplementary/h2_2/check.py --level all`. This
   checks interval/matrix/rational composition; it does not regenerate the
   finite sums, statistical banks or optimizer trajectories.
3. Run small manufactured production fixtures through the original routines:

```text
python -I -B tools/production_replay.py --mode small --target all --output-root ../runs/core-small-001
```

The output must be a new directory outside the source checkout. The runner
rejects overwrite, source-tree writes, historical scientific-array reads and
subprocess launches. It records the bytes of scientific modules actually loaded
and input files actually used. Temporary fixtures and outputs also live in this
new directory. It does not mutate module output paths or call historical
`freeze`, `run` or `delivery` wrappers.

Small mode covers 16 targets. Theorem 3's teacher, candidate-probability, gradient
and Hessian routes run original manufactured tests, including direct-label
Hessian comparisons, symmetry, saturation and partial-budget handling. Root/KL
is explicitly a saved-arithmetic target. R01 calls the original pair finite sum
on an artificial eight-dimensional three-node grid, plus the original independent
path test. R02 runs small active-moment, nuisance-CF and Gaussian-comparison
grids. R03 reuses original derivative/candidate self-tests and runs a manufactured
five-node, 256-label risk grid. Figure 1's score and curvature kernels are compared
with original Arb expressions on manufactured inputs. The finite-program route
generates four manufactured banks with a distinct fixture seed and forms exact
integer sample moments; it makes no population confidence claim. Figure 2 reuses
the original manufactured score/generator qualification and calls the original
block worker once with one bank, two particles and a distinct fixture seed;
its raw array and receipt are written only into the new output directory.
Small-grid error bounds
may be wide or unresolved; passing the interface test does not certify the paper
result on that manufactured grid.

The original optional Torch/autograd curvature qualification is source-readable
but was not run. The default small route uses the original NumPy/Arb production
curvature functions and does not require Torch.

## Paper-scale regeneration commands: provided, not executed in this release

Each command below selects a single target and a new output directory:

```text
python -I -B tools/production_replay.py --mode paper --target teacher --output-root ../runs/teacher-001
```

Replace `teacher` by a target listed below. `--mode paper --target all` is rejected
to avoid silently starting the entire expensive production suite. Paper mode
regenerates the stated finite objects from the saved exact scientific inputs;
it does not by itself recreate every historical management receipt or assert
that all newly generated numerical qualifications passed.

| Target | Original production functions and supplied input | Paper dimensions / original budget |
|---|---|---|
| `teacher` | `certified_teacher_compute.finite_grid_block`, `aggregate_blocks`; teacher `INPUTS.json` | 161 by 161 Gaussian nodes; all 256 labels; 11 blocks; 192 bits; 600 s |
| `candidate_probability` | `certified_teacher_risk.finite_candidate`; exact `a_exact` and `h_exact` in `CANDIDATE_RISK_INTEGRALS.json` | 177 one-dimensional nodes; 256 labels; 192 bits; 60 s |
| `gradient` | `certified_f8_gradient_compute.precompute_inner`, `finite_representative`; gradient `INPUTS.json` | 41 classes, 881 inner by 769 outer factors each; 192 bits; 1,800 s |
| `hessian` | `certified_f8_hessian_compute.precompute_inner`, `restore_moment`, `finite_representative`, `aggregate_records`; Hessian `INPUTS.json` | 41 classes, 881 by 769 factors, 35 moment columns; 192 bits; 3,600 s |
| `root_kl_saved` | `supplementary/h2_2/check.theorem3` | Saved matrix floors, gradient cap, error budget, root/minimum gates and KL transfer; always saved arithmetic |
| `angular_pair`, `angular_audit` | `angular_witness_integrals.finite_pair_sum`, `angular_witness_path_audit.finite_path_audit`; angle `INPUTS.json` | 177 nodes, 256 labels; independent numerator expressions; 192 bits; 600 s each |
| `r02_active` | `r02_finite_scale_constants.compute_constants`; active `INPUTS.json.parameters` | Step 1/16, N=160, 321-node outer/inner rules; 192 bits; 600 s |
| `r02_noise` | `r02_remainder_noise.finite_noise`; remainder `INPUTS.json.parameters`, saved certified active constants | Step 1/16, N=192; 273 Fourier frequencies; 192 bits; 600 s |
| `r02_weights` | `r02_remainder_propagate.make_weight_grid`; remainder `INPUTS.json.weight_parameters` | Step 1/128, N=1536; 3,073 Gaussian nodes; 192 bits; 120 s |
| `r03_teacher` | `r03_teacher_derivative.finite_grid_block`, `aggregate_blocks`, `compose`; `BOX_INPUTS.json.teacher_config` | 25,921 Gaussian points, 512 scalar sums; 192 bits; 120 s |
| `r03_candidate` | `r03_candidate_interval.precompute`, `finite_label_block`, `analytic_error_bounds`; all three `BOX_INPUTS.json` boxes | Each of h01/h001/h0001: 128 labels, 641 by 337 factors, 99 moments; 900 s per box |
| `r03_risk` | `r03_risk_interval.finite_risk`; all three boxes and exact direction in `BOX_INPUTS.json` | Each box: 177 nodes, 256 labels, 512 sums; 192 bits; 120 s per box |
| `probability_audit` | Original `failure_probability_compute.point_differences`; original `failure_probability_audit.exact_stats`, `infer`; same PCG64DXSM seed/role/block rule | 262,144 banks, K=100, n=256; about 6.79 billion normal scalars; original 1,200 s scoring budget; raw outputs require substantial memory/storage |
| `figure2` | Original `asymptotic_bridge_run.worker_block` calls `generate_block`/`score_block`; original `asymptotic_bridge_statistics.analyse`; `REFERENCE.json` | 131,072 banks, K=100; 2,048 blocks; shared n=64/256/1024/4096; fresh raw blocks and 26-target conditional statistics; sequential thin scheduler, runtime not tested |

The listed budgets are original execution caps, not promises of runtime on a
new computer. Functions can return a partial object or reject a budget stop;
the thin caller retains that output and does not promote it to a successful
paper certificate. The statistical programs retain ideal-iid and numerical
allowance conditions. Regenerating random samples does not certify a Gaussian
generator or a future native floating-point algorithm.

The finite-program thin route forms the original exact sample moments and
conditional inference but does not run the full saved-score error replay or all
historical bank diagnostics. Those original routines are source-readable in
`failure_probability_interval.py` and `failure_probability_audit.py`. This route
is therefore not a new unconditional native-program certification.

## Remaining orchestration boundaries

The full historical Figure 1 matched selection is source-readable in
`nuisance_contrast_compute.run` and `finite_selection_compute`, while its
population curvature is in `local_geometry_compute`. The matched selection
requires the prior training-label count asset; its original wrapper also binds
historical freeze/governance files. The thin `figure1_kernels` route supports
small scoring/curvature production only; paper mode for that target explicitly
rejects rather than pretending those management dependencies are satisfied.
The paper figure itself and saved plot-data regeneration remain available from
the anonymous supplement. This release does not claim that the old selection
wrapper runs unchanged in a clean export.

Likewise, finite-sum regeneration and complete certificate composition are
different steps. The original synthesis sources are provided:
`certified_teacher_finalize`, `certified_f8_gradient_finalize`,
`certified_f8_hessian_finalize`, `candidate_v18_risk_range`,
`r02_remainder_noise.compose_bounds`, `r02_remainder_propagate.propagate`,
`r03_continuous_compose.compute`, `r03_spectral_refinement.compose` and
`r03_quantitative_consequences.compute_from_saved`. Their mathematical functions
can consume correctly assembled fresh objects. Their historical file drivers
are preserved as source, not relabeled as tested clean-export CLIs. The release
tested the original production kernels on small fixtures and the existing
saved certificate composition; it did not run a new full certificate pipeline.

Source closure, scientific inputs and historical source-freeze matches are
recorded internally before export. The public release binds its own final bytes
and reports the actual tested levels in its validation record. Missing raw
assets and source-only historical routes have their own status; a hash record
is not a downloaded asset or a completed production replay.
