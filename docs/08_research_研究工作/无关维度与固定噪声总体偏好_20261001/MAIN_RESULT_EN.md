# Independent nuisance dimensions induce spurious correlation

**Core claim.** In a correctly specified Gaussian–logistic family with a fixed positive logit variance, appending independent observations whose distribution is known and unrelated to the learned parameter can make the raw prior-particle population objective prefer an incorrect boundary correlation. This occurs even under a fully independent observed teacher and with particle counts growing polynomially in dimension, including one particle per observed coordinate.

Only a fixed active covariance block is learned. The n additional coordinates have the same independent fair-bit observed distribution under the teacher and every candidate. Thus the exact observed likelihood gains a parameter-independent term and total observed KL remains unchanged. Their per-particle conditional likelihoods are nevertheless random and enter the same log-mean, changing the surrogate's parameter preference. If the added factor is constant across particles, the effect disappears.

For two active coordinates, let the teacher's base correlation satisfy 0≤theta<R and let rho range over [-R,R], with 0<R<1. Let a>0 be the fixed logit scale, r_a(rho) the observed spin correlation, v_a=Var(log sigma(aZ)), and J_K=E max_{k≤K}Z_k. For any fixed b,C>0, uniformly in rho and integer 2≤K≤Cn^b,

\[
\frac{\sqrt{nv_a}}{J_K}\,\partial_\rho F_{n,K}(\rho)
\longrightarrow-\frac{a^2}{4}\{r_a(\theta)+r_a(\rho)\}.
\]

The sum, rather than a teacher–candidate difference, yields a strictly concave limiting potential. Uniform convergence of values and first derivatives shows that, for all sufficiently large n, the actual unique global minimum is exactly R when theta>0. For theta=0, the only actual global minima are ±R. A separate fixed-K result gives a uniform C² limit for general fixed active covariance blocks: the objective favors larger conditional active log-likelihood variance. Exact Gumbel pair integrals and relative local-density and upper-tail estimates resolve the nearly tied particles that determine this vanishing parameter-dependent term.

For a=1, R=.75 and the independent teacher, the eventual minimizers have total integrated-model KL at least 1/128 and an active same-label event probability error at least 1/16. The theorem provides no computable finite-n threshold. A sufficient condition for every epsilon-near minimizer to approach the incorrect endpoint set is epsilon=o(J_K/sqrt(n)); for K=n+2 this becomes o(sqrt(log(n)/n)). The truth itself has o(1) surrogate regret, so an unrestricted o(1)-near-optimality claim would be false.

A separate fixed-dimensional weak-emission calculation identifies a minimal transition. With an independent teacher and K=2, the leading correlation-dependent term is (10−3D)a^6 rho²/512. For sufficiently small positive a, the actual minimizer is zero in D=2,3 and the two boundary points in D≥4. Adding only two irrelevant coordinates can therefore reverse the preference. This small-a remainder is not uniform in growing D.

At fixed a=1 and K=4, 65,536 independent banks change the same spurious candidate's score gap from +.003350 without padding to −.000772 with 14 nuisance coordinates and −.000379 with 4094. All six prespecified sign gates pass using a simultaneous empirical-Bernstein family of 18 quantities. Deterministic K2 profiles pass 3593 numerical qualification checks and ten prespecified scientific readouts. These are not finite-n continuous-optimum certificates; no growing-K simulation or new model training was performed.

The risk contract matters: F evaluates the expected log risk of the random K-particle predictor, whereas the stated KL evaluates the integrated observed model Q. Exactly marginalizing the nuisance factors preserves Q and restores F_active,K+n log2, but does not remove the active block's own finite-K bias. Total KL and a fixed active event retain the effect; per-coordinate averages, uniformly sampled full-dimensional queries and total-count summaries may dilute it.

The proposed contribution is the failure of parameter-selection invariance under parameter-independent observation extensions, with actual-minimizer and observed-risk consequences. Weight collapse, Gumbel identities, Price differentiation and marginalization are established tools. The result does not cover unknown active locations, unrestricted full-dimensional covariance learning, SGD, or natural-data benefits. See proofs and evidence (historical reference not distributed), scope (historical reference not distributed), and direct literature neighbors (historical reference not distributed). This is a core statement, not a new complete manuscript; v3.1 remains unchanged.
