# Learning compensation under finite-particle objectives

This is a short research statement, not a new full manuscript. The frozen v3.1 manuscript and PDF are unchanged. The empirical results below have been sealed and independently audited.

A low-risk representation can exist without a particle objective selecting it. A weak positive common-factor Gaussian-logistic teacher can be approximated, in complete observation KL, by a model whose latent marginal scales all equal a fixed positive floor. Adjusted factor loadings give an \(O(d^{-2})\) approximation while preserving bounded, strictly positive covariance spectra. The analytically matched loadings use teacher parameters. The present question is whether observations alone can identify a useful compensating representation, and whether continuous likelihood-based learning chooses one.

**Compensation is learnable in a known-group model.** Consider two known equal-size groups, zero means, positive common loadings, and teacher correlation \(\rho_d=c/(d-1)\). The two teacher scales and c are unknown but belong to fixed ranges, with c bounded away from zero. The candidate model fixes every latent scale at a and learns two effective amplitudes
\[
\lambda_g=w_g/\sqrt{d-1}.
\]
A simple moment estimator uses the empirical squared spin sum within each group, rescales it by the known floor response \(\alpha(a)=E[Z\tanh(aZ/2)]\), and applies a predeclared projection and square root. It does not use the teacher PMF, teacher scales, actual c, or oracle loadings.

For N independent complete observation vectors, the estimator satisfies
\[
\sup_{\text{allowed teachers}}
E\,D(P_d^\star\Vert Q_{\widehat w,d})
\le C\bigl(d^{-2}+N^{-1}\bigr).
\]
The proof controls within-vector dependence through a dimension-uniform fourth moment of each normalized group sum, retains the finite-d Hermite remainder, and transfers loading error directly to full observation forward KL. It does not use a KL triangle inequality. The sample unit is a d-bit vector, so the observation cost is \(Nd\) bits. The result estimates effective amplitudes, not the original scales and c separately.

Under the stated scale-separation and dimension conditions, a [split extension](UNKNOWN_GROUP_THEORY.md) can learn the group identities with \(n_1=O(d\log d)\) vectors and then attain \(O(d^{-2})\) expected joint KL with \(n_2=d^2\) independent vectors; this is a theoretical extension, while the present experiment keeps the groups known.

**Marginalization has a guarantee that allows representation error.** We do not assume that the finite-d teacher observation law is exactly contained in the fixed-floor candidate class; we also do not prove that no alternative parameter could be observationally equivalent. Let a feasible comparison model \(Q_0\) have
\(r_d=D(P_d^\star\Vert Q_0)\). For the private-noise-marginalized particle objective, a pointwise bias bound \(\mathcal B_d=O(d)\), a diagonal bias bound \(C\omega_K\to0\), and a weighted Hellinger inequality yield
\[
D(P_d^\star\Vert Q_{\widehat w})
\le \varepsilon+r_d+
\left(\sqrt{C\omega_K}+\sqrt{\mathcal B_dr_d}\right)^2
\]
whenever its population objective is within \(\varepsilon\) of the comparison model. A compensating \(r_d=O(d^{-2})\), K=d, and vanishing total optimization error therefore suffice for vanishing joint KL.

The population statement also has a continuous empirical bridge. With shared B banks and N labels, a deterministic parameter net plus a Gaussian-maximum bound gives uniform objective control. Along \(K=d,N=B=d^2\), a progressively refined two-dimensional net and controlled numerical error support consistency in probability and expectation. This theorem applies to global approximate empirical minimizers or complete net minimization. It does not certify a fixed multistart optimizer.

**The learned representation need not be the one preferred by raw scoring.** The experiment used d=16/64, eight independent streams per dimension, N=256/4096, B=64, K=d, and the same candidate box \([.5,2.5]^2\). Raw and col shared labels and public-factor banks; MOM and numerical quasiMLE used the same labels. Matching K does not match computational cost. All 128 method outputs were sealed before risk evaluation and independent-bank score confirmation. The 96 selected continuous solutions passed their projected-gradient, finite-difference and grid diagnostics after nine fixed starts; these are not global-optimality certificates.

At N=4096, mean complete observation KL across eight streams was:

|d|raw|MOM|col|numerical quasiMLE|
|---:|---:|---:|---:|---:|
|16|0.030105|0.000291|0.001237|0.000178|
|64|0.033366|0.000267|0.000257|0.000161|

At d64, the registered raw–MOM and raw–col risk gates passed. Nevertheless, independent banks estimated the raw-trained output to have a **lower raw objective** than MOM: the mean difference was −0.04647 nats, with the original approximate simultaneous interval [−0.05740,−0.03555]. Thus a low-risk representation was learned from observations, yet the tested raw procedure selected a predictor with substantially higher observation risk despite a lower independently estimated raw objective.

The limits matter. Five of eight original risk gates and both independent raw–MOM score gates passed. At d16, raw–MOM and raw–col failed the registered SE≤.002 rule despite positive interval lower bounds; col–quasiMLE also failed its .001 noninferiority margin. Risk precision passed in 18/24 comparisons and score precision in 12/16. All failures remain, with separate approximate t families and numerically qualified rather than rigorously certified integrals.

All candidates have balanced one-bit marginals. Descriptively, for the prespecified event \(|\sum_iY_i|\le\sqrt d\) at d64/N4096, the true probability was .6527; mean absolute probability error was .0917 for raw and .00572 for MOM. This event summary adds no confidence interval or success gate.

The current uniform-objective bound alone is about 7.65 and 17.15 nats at d16/d64, before numerical and global-optimization errors. It supplies an asymptotic path, not a practical certificate for these small measured KL values.

The moment estimator is a classical finite-information baseline. Low-order binary moments already support factor and association estimation in [Wu and Bentler (2013)](https://doi.org/10.1016/j.csda.2012.06.022) and [Shi et al. (2024)](https://arxiv.org/abs/2404.08457v1). Our proposed contribution lies in the connection among a complete observation-risk guarantee, finite-particle objective preferences, and the actual selection of compensating representations in this restricted model. Group identities are given in the experiment. In the split theorem they are unknown; group size, the floor response and parameter bounds including scale separation remain known, and the two data batches are independent. Positive loadings and balanced binary marginals remain model assumptions, with numerical and optimization qualifications stated separately.

Proofs and evidence: moment learning (historical reference not distributed), [continuous objectives and risk transfer](CONTINUOUS_THEORY.md), [complete results and failures](RESULTS.md), independent run-audit receipt (historical reference not distributed), and direct literature comparison (historical reference not distributed).
