# Equal statistical tasks and log-weight variance can yield different learning limits

We construct Gaussian-logistic nuisance representations with the same integrated observation law, nuisance dimension, total log-weight variance, and particle count, but different raw likelihood-learning limits. The active pair and its emission scale remain fixed; only known parameter-independent nuisance scales change. No objective-correction pool is used.

Write v(a)=Var(logσ(aG)). The diffuse representation uses n scales equal to one. A bounded heterogeneous representation alternates two fixed positive scales with the same total variance. The concentrated representation uses n−1 scales 1/n and one scale A_n chosen so that v(A_n)+(n−1)v(1/n)=n v(1); A_n is of order √n. Every representation integrates to the same independent fair-bit nuisance law. Centering each nuisance log score also matches its mean at zero without changing normalized weights or parameter comparisons.

The wrong subleading correlation direction extends uniformly to deterministic nuisance scale arrays in any fixed positive compact interval. The required relative density, upper-tail and near-tie estimates hold for the nonidentically distributed triangular array; writing s_n²=Σv(a_i), the range 2≤K≤C n^p retains endpoint convergence for population near minimizers with regret o(J_K/s_n), and positive integrated risk for fixed interior true correlations.

In contrast, the concentrated normalized weight W_n satisfies EW_n=1, bounded EW_n², and ||logW_n||₂=O(√n). A positive-average log-error bound implies uniform convergence of the raw population criterion, after subtracting n log2, to exact active cross-entropy when K=n+2. On refining predetermined grids of size O(n^1/4) that include both endpoints, actual complete-row grid ERM is consistent for any predetermined N→∞, even with one shared training bank. With the same data, K=n+2, B=n and candidate grid, concentrated and diffuse exact grid ERM therefore have respectively zero and strictly positive limiting expected joint KL at a fixed interior true parameter.

The weight geometry explains the distinction: under a suitable coupling W_n converges in L² to 2 times a Gaussian half-space indicator. As K also grows, noise-only ESS/K converges in probability to 1/2. Diffuse and bounded heterogeneous noise instead have ESS→1 and maximum normalized weight→1 in probability for polynomial K. These are not finite-n formulas, claims about the complete active-plus-nuisance ESS, or expected-ESS convergence claims.

The recovery criterion extends to a fixed finite active observation alphabet with uniform positive integrated probabilities and relative log-moment control, including compact finite-dimensional Gaussian-logistic classes. A separate four-variable, six-correlation box supplies a multivariate negative population example. We do not combine that population result with the positive learner guarantee into an unproved multivariate shared-bank learning separation.

All twelve prespecified population-direction comparisons passed on 65,536 new banks. Sixteen new paired complete-data streams at n=254, K=256, B=128 and N=4096 gave:

|True correlation|Mean diffuse joint KL|Mean concentrated joint KL|Diffuse boundary selections|Concentrated boundary selections|
|---:|---:|---:|---:|---:|
|0|.00841757|.000126907|16/16|0/16|
|.6|.000347575|.000149114|16/16|1/16|

Bounded heterogeneity also selected a boundary in all sixteen runs at each true parameter. At θ=.6, concentrated mean correlation was .590273, compared with .584866 for the same-data numerical MLE. All three prespecified training comparisons passed their separate development t intervals. No additional noninferiority claim against MLE is inferred.

The concentrated sequence deliberately leaves the original uniform nuisance-scale window: its small scales vanish and its large scale diverges. The construction exposes a limitation of total log variance as a summary; it does not refute the fixed-noise theorem or provide an unknown-structure correction algorithm. Numerical matching, reference integration, floating-point errors and finite-grid scope remain distinct from the analytic limits. The v5.1 manuscripts remain frozen.
