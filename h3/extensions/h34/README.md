# h3.4 compact saved evidence

This additive v1.3.0 material accompanies the h3.4 research draft. It contains 30 homogeneous and heterogeneous branch points and their precomputed analytic responses, all six retained unknown-channel population points, six full-coordinate local-risk records, two 18-state moment tables, and the compact execution diagnostics used in the evidence appendix. Author: ZHANG. No acceptance or independent external review is claimed.

The adaptation figure distinguishes the proposal inputs: F has no free tilt, L uses its own group's observation, and A uses other groups' observations while the proposal remains factorized. The base proposal still changes with the channel. Analytic response lines were computed before these numerical points; no coefficient is fitted here. Numerical branch points are deterministic qualified stationary points, not risk samples or a continuous-branch or global-optimum certificate.

## Supported checks

Use Python 3.11 and install `requirements.txt` in a separate environment. The tested versions are NumPy 2.3.5 and Matplotlib 3.11.2. From the repository root, choose fresh outputs outside this checkout:

```text
python -B h3/extensions/h34/tools/check.py --output ../h34_saved_checks.json
python -B h3/extensions/h34/tools/plot.py --output-dir ../h34_redraw
```

The checker verifies the extension's file identities, independently factorizes the saved one-dimensional moments to reconstruct the full 4/58/64-coordinate Hessian, score covariance, Schur complement, influence covariance and risk, and checks saved coordinate/difference arithmetic and curve data. The numerical risk functions are unchanged extracts from the frozen internal saved-arithmetic audit; only imports, constants and public file wrappers are projected. This is an internal arithmetic check, not an external scientific review. The figure tool redraws the supplied PNG from saved points and analytic constants. Neither command runs model integration, random sampling, training, fitting or optimization; raw PMF integration is not replayed.

## Scope and statistical units

- The six local-risk records are F/L/A at two deterministic moment rules. They are not six statistical repetitions. The local expected-risk theorem concerns specified compact-domain exact empirical global minimizers under root-N teachers eta_N = h/sqrt(N), with fixed dimension and particle count, exact particle expectation and its stated regularity/localization conditions. The same-data plug-in uses an efficient marginal first-stage channel estimate; no finite-N experiment for the unknown channel is included. Joint and same-data plug-in have the same leading covariance. The channel adds the common `1/(2N)` leading term under those assumptions.
- All six unknown-channel population points use fixed-teacher comparisons and full main/high stationarity qualification. Positive marginal KL sums and negative joint-KL differences are different readouts; neither is a universal risk improvement. The saved `lambda` keys denote the channel called alpha in the manuscript. Pointwise stationary comparisons do not certify globality at t=.1 or .2, or exact equality with a leading asymptotic coefficient.
- The old nine-endpoint decomposition is the prespecified selected panel. Its two terms compare local fixed-theta profiles and joint references; they are not global regret. High-order coverage is limited to the stated r0 subset. The matched A-minus-L statistics use four complete numerical repetitions and sample SD; fixed-point joint-A noise summaries use three complete repetitions and sample SD. These units are not finite-N validation datasets or paired standard errors.
- Earlier risk, budget and learned-proposal materials remain under [h3](../../README.md) and [h33](../h33/README.md) with their original statistical units and bytes.

`SOURCE_MAP.json` lists opaque historical source identities and projection counts; it does not advertise private source paths as public downloads. `MANIFEST.json` covers every extension file except itself. Full original quadrature, training arrays, checkpoint logs, manuscript PDFs and styles are not distributed by this extension. Final English technical notes are included: [A: model and localization](notes/APPENDIX_MODEL_EN.md#model-proof), [B: adaptation](notes/APPENDIX_ADAPT_EN.md#adapt-proof), [C: common channel](notes/APPENDIX_CHANNEL_EN.md#channel-proof), and [D: local sampling risk](notes/APPENDIX_RISK_EN.md#risk-proof), with [references](notes/references.bib). Only Markdown anchor/link syntax is projected for GitHub; the narrative and mathematics retain the approved manuscript content. Citation keys in the notes refer to the supplied BibTeX entries. This source material is separate from publication of a manuscript PDF.

Code is MIT licensed; authored text, selected numerical material and figures are CC BY 4.0, attributed to ZHANG, under the repository's license files. Historical payloads retain their original licensing. See [the Chinese entry](README_ZH.md).
