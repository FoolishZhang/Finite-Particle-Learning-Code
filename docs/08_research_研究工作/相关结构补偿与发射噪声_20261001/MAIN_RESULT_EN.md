# Structural compensation and emission-noise limits in particle training

Working core for the next manuscript. This is not a complete submission; the frozen v3.1 paper remains unchanged. Full proofs, numerical contracts and retained failures are linked from the research report.

## The learning question

In a Gaussian-logistic model, logits eta follow N(mu,Sigma), and binary spins Y_i are conditionally independent with probability sigma(Y_i eta_i). The integrated observation model is Q_theta. A standard prior-particle objective replaces its likelihood by an average of K conditional likelihoods, then takes the negative logarithm and averages over labels and particles. Denote this population objective by F_K. Its optimizer need not minimize the exact risk H(P)+D(P||Q_theta).

Previous results established this discrepancy with known correlation structure, including fixed positive scale floors, population near-minimizers, and finite-grid empirical learning. The next question is whether allowing correlation to change removes the prediction error. Scale shrinkage alone cannot answer it.

## A noise barrier with trainable full covariance

Fix 0<nu<=a²<=Lambda, independently of dimension, and allow all mu in R^d and all covariance entries to vary subject to nu I<=Sigma<=Lambda I and Sigma_ii>=a². The teacher has balanced observation marginals. These assumptions permit full covariance learning under spectral control; they do not describe an unconstrained Cholesky implementation.

Write l(x)=log[2cosh(x/2)], g(x)=sech²(x/2), and Psi=sum_i E l(eta_i). The raw objective satisfies 0<=Psi−F_K<=sqrt(2 Lambda d log K). A feasible independent floor model provides the comparison value d ell(a), where ell(a)=E l(aZ).

The key channel comparison is

$$
(g+2l)''=\tfrac32g(1-g)\ge0.
$$

Even convexity therefore gives, for every real mean and every s>=a,

$$
E g(\mu+sZ)\ge\tau_a^2-2\{E l(\mu+sZ)-\ell(a)\},
\qquad\tau_a^2=E g(aZ).
$$

Consequently, every epsilon-near population minimizer retains conditional Bernoulli variance at least tau_a²−2 delta on average, with delta=sqrt(2 Lambda log K/d)+epsilon/d. Gaussian concentration makes this variance concentrate. Conditional Berry–Esseen then bounds interval probability for the observed normalized sum, uniformly over its arbitrary latent drift. No central limit theorem for that drift is required.

If log K=o(d) and epsilon=o(d), every interval of fixed half-width h has asymptotic candidate probability at most 2Phi(h/tau_a)−1. A teacher assigning strictly more probability to one such fixed interval therefore yields a positive event-risk and joint-KL lower bound over the entire near-optimal set. Population minimizers exist even with unbounded means, because F_K>=||mu||_1/2−sqrt(2 Lambda d log K).

For the original negative-correlation two-scale teacher, its normalized-sum variance is V_star=1−c baralpha_star². The sufficient condition is V_star<tau_a². With c=.9, floor1 and scales(6,2), the qualified numerical values are .577571 and .826484. The result is conditional on this signal separation and on spectral control; it is not a universal claim about all negative dependence. [Strongest proof](CHANNEL_THEORY.md), review (historical reference not distributed).

## A strictly positive-definite compensation example

For a positive common-factor teacher with fixed 0<c<1, a<=s_i_star<=b and rho=c/(d−1), define alpha(s)=E[Z tanh(sZ/2)]. Set every candidate scale to a, but change its common-factor loading to

$$
\lambda_i=\sqrt\rho\,\alpha(s_i^\star)/\alpha(a).
$$

The candidate covariance is a²{diag(1−lambda_i²)+lambda lambda^T}. It has a fixed positive lower spectral bound for sufficiently large d. Matching the first response coefficient, bounding the full Gaussian-tail remainder and the Bernoulli KL denominator, and marginalizing the common factor give

$$
D(P_d\Vert Q_d^{\rm comp})\le C d\rho^3=O(d^{-2}).
$$

This is full observation KL, not an inference from moment matching or a count CLT. The construction uses teacher parameters and establishes representational capability, not learned recovery. For the fixed interior two-scale teacher, in a common class with sufficiently permissive fixed spectral bounds, K=1 has both asymptotically correct and predictively wrong exact minimizers: the objective depends only on means and marginal scales. For general subexponential K, this construction has only an o(d)-regret guarantee, not o(1) or exact optimality. Proof (historical reference not distributed), review (historical reference not distributed).

Independent reference calculations for d16/64/256 give compensated KL about 7.38e−8, 4.45e−9 and 2.74e−10, while retaining the old correlation at the same floor leaves approximately .01390, .01291 and .01269. These are convergence-qualified values, not certified integration intervals.

## Capability and objective preference are separate

A fixed-candidate population-score experiment compares the compensating model with a model using 1.5 times its loadings. At d16/K10,K16 and d64/K10,K64, the raw objective favors this overcompensated model although its joint KL is higher by roughly .028 or .024. All four prespecified pairwise criteria passed their original approximate intervals. No covariance optimizer was fitted, and the comparison does not exclude other low-risk representations.

A separately frozen, independent larger-K follow-up reverses this pairwise ordering at d16. At d64/K4096 the interval still crosses zero and the recovery criterion fails. The two confidence families remain separate. The population target is estimated using group permutation symmetry and a known single-particle control variate; the finite-bank table is not a new normalized deployment distribution. [Results](RESULTS.md).

## Role of the earlier recovery results

Private-noise marginalization preserves each candidate's Q and, on the stated positive family, gives near-optimizer joint-risk recovery. The earlier zero-mean fixed-grid theorem further separates N, K and shared-bank budget B. These remain computational controls. The new compensation example instead changes the candidate covariance; its role is to determine when parameter shrinkage need not damage observations.

The proposed contribution is the complete connection between objective-induced parameter preference, observable consequences and the limits of compensation. Generic Monte Carlo objectives and estimator-dependent learning are established prior work. The present results do not establish unconstrained covariance SGD, arbitrary nonbalanced teachers, practical optimal budgets or natural-task gains. Scope (historical reference not distributed), direct-neighbor note (historical reference not distributed).
