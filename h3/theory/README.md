# Proof sources and model scope

The three English appendix sources are byte-preserved from the h3.2 manuscript. The linked reading page concatenates them and replaces Pandoc heading attributes with explicit HTML anchors for GitHub navigation; scientific prose and display mathematics are unchanged:

- [A: conditional proposal limit and same-model ordering](ALL_PROOFS.md#proposal-proof)
- [B: group-posterior response, global localization and second-order structure](ALL_PROOFS.md#group-proof)
- [C: exact-estimator risk, fixed-teacher limit and numerical-output scope](ALL_PROOFS.md#risk-proof)

The scientific model is specified in A.1 and the package README. B and C use
that known finite group channel and three physical Gaussian correlations.
Cross-appendix references refer to the corresponding files here. These proofs
do not depend on the separate historical h2.2 results.

The Gaussian ordering theorem uses sequential choices of a fixed finite K,
then a finite noise level, then sufficiently large M. Its unknown thresholds do
not certify the finite K8/M32/M128 experiment. The group risk theorem instead
uses exact empirical global minimization with particle randomness averaged,
along local alternatives. The fixed-program evidence has its own numerical
qualifications. Both limitations remain in the original text.

These are author proof sources, not a formal theorem-prover artifact. Publishing
them or passing the data checker does not amount to independent proof review.
