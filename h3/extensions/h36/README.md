# Input access and joint outputs in finite-particle learning

This compact extension supports the h3.6 study of how input access and output factorization selectively change correlation learning at fixed particle count. It contains the general structural proof, English technical appendices A--H and references, saved neural endpoint matrices and paired statistics, same-TRAIN reference summaries, and a separate finite-sample risk experiment. The manifest identifies every delivered file.

Install the small checking/plotting dependencies and choose fresh outputs outside the source checkout:

```text
python -m pip install -r h3/extensions/h36/requirements.txt
python -B h3/extensions/h36/tools/check.py --output ../h36-saved-check.json
python -B h3/extensions/h36/tools/plot.py --output-dir ../h36-neural-redraw
```

The checker verifies declared repository file hashes and reaggregates the following saved evidence. It does not require optional historical large assets.

| Material | Saved object and statistical unit | Supported check |
|---|---|---|
| General structure | Gaussian correlation scores, independent posterior anchor, groupwise fixed observation kernels | Read the [structural proof](notes/APPENDIX_STRUCTURE_EN.md); regular local profiles and projection identities have explicit distinct conditions |
| Fresh neural confirmation | d8/K4, four proposal structures, two teacher signs, one initialization, 96 endpoint matrices; 12 independent paired datasets | Reconstruct weighted odd responses, four paired means/SDs and Bonferroni--Student approximate simultaneous 95% intervals |
| Old same-update reassessment | Original 12 datasets after completing the same fixed update count | Reaggregate saved paired statistics, separately from fresh confirmation |
| Same-TRAIN diagnostics | Saved likelihood-reference, posterior-KL and odd/even matrix summaries | File identity and availability of the saved descriptions; no repeated likelihood fitting or posterior computation |
| Old R64 risk | Three-group Gaussian--logistic model, K8, fixed local/all-input exponential-tilt proposals; 64 independent datasets per (N,S), 256 total | Reaggregate the beta=1 controlled and raw paired risk differences, theoretical leading comparison, parameter covariance and finite-R bias--variance identity |

The fresh neural program compares local input/product output (L), all input/product output (A), all input/four pair blocks (B), and all input/full joint output (JNT). All 28 candidate correlations are learned. The noise entries in `neural_design.json` are **variances**. Positive and negative teachers share underlying draws within a dataset, and methods share the signed dataset and training seed. Different dataset IDs are independent. The final primary output is the last legal endpoint at 28000 updates, without best-start selection. These are fixed-program outputs; they are not certified neural class minima.

The confirmation used a resource amendment frozen after data generation and before access to the fresh primary effects. Predetermined resource-stopped trajectories were recovered at the same ID/seed or from complete state, producing the 96 full-update endpoints. Earlier resource-limited results and the old post hoc reassessment retain their original identities. The new R12 is not pooled with old cohorts. Cross-block B-minus-A equivalence concerns the prespecified weighted response within ±.05, rather than exact zero or coordinatewise equivalence of all correlations. Full design and diagnostic definitions are in [Appendix H](notes/APPENDIX_NEURAL_EN.md).

The R64 experiment is newly included in this extension; the earlier v1.3.0 public materials did not contain these confirmatory results. Its reported risk is true joint observation KL under the specified numerical generative distribution, evaluated offline. The fixed beta=1 score control changes the risk-difference estimator, never the fitted outputs. More complete correlation recovery lowers squared attenuation bias and raises variance, so its risk advantage changes with signal strength. This risk evidence belongs to the old three-group model and is distinct from the neural structural confirmation. The compact extracted numbers were later accepted as mathematically identical to the formally verified final result; pending-verification metadata is not a second analysis. See [risk theory](notes/APPENDIX_RISK_EN.md) and [numerical details](notes/APPENDIX_EVIDENCE_EN.md).

The two main figures are the [neural structure comparison](notes/figures/neural_structure.png) and the [old R64 finite-sample risk comparison](notes/figures/local_risk.png). The R64 figure is supplied in its original manuscript bytes. The plot command supports only the neural figure: it reuses the manuscript drawing function to redraw `neural_structure.png` from the saved matrices and verifies byte identity with the delivered PNG. Its SVG is a convenience output; only PNG byte identity is checked. Recomputed arithmetic is low-cost verification of saved outputs, not a replay of training, original data generation, model integration or optimization. Raw arrays and checkpoints are not included. Earlier h3, h33 and h34 scientific files remain byte-for-byte unchanged.

Self-authored code: MIT. Self-authored technical text, figures and synthetic numerical data: CC BY 4.0. Author: ZHANG. This named source release does not imply paper publication, acceptance, a new method, or application validation.
