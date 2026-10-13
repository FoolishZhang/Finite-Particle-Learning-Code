# Joint scale freedom: objective elimination and the remaining learning boundary

This study tests a qualification of the preceding active-scale result. With a fixed, diffuse nuisance representation, freeing the active means and scales can turn incorrect correlation preference into loss of dependence. Once nuisance scales are also free, that fixed-representation theorem no longer determines the optimum of the joint parameter class. The relevant question becomes whether a specified finite-budget learning procedure can use the available nuisance descent direction while preserving useful active dependence.

The mathematical analysis is complete in [THEORY.md](THEORY.md). It retains known independent blocks, zero nuisance means, fair nuisance observations, and a Gaussian-logistic active model. It does not cover free cross-covariances, task heads, regularizers, or full neural-network training. The v8 manuscripts and prior scientific packages remain unchanged.

## What the joint objective permits

For every fixed active parameter and every finite n and K, the population raw objective is strictly increasing in a positive shared nuisance scale τ. This directly inherits the previous coordinatewise monotonicity result. If τ=0 is feasible, every joint global minimum uses τ=0, and the remaining objective is exactly the active K-particle score plus n log 2. With a positive floor ε, the optimum is instead at the floor, and the remaining nuisance score bias is at most nε²/8. A finite softplus parameter generally only approaches this mathematical boundary.

Thus the fixed-positive-nuisance collapse result is not an unavoidable optimum of the jointly free class. At any positive τ, even a state stable against active perturbations still has a population descent direction in τ. This is an objective statement, not an optimizer convergence claim. Eliminating nuisance variation also does not eliminate the active finite-K bias.

A standard Jensen comparison makes the latter distinction explicit. If the teacher is representable, an output with joint population regret r satisfies

\[
\mathrm{KL}(P\Vert Q_\theta)
\le r+\{F_{A,K}(\theta_*)-H(P)\}.
\]

The term in braces is the active particle bias at a true parameter. It tends to zero as K grows under the existing positive-kernel and log-moment conditions. Consequently, growing K and vanishing joint population regret suffice for risk recovery when zero nuisance scale is allowed. A positive floor adds at most nε²/8 to this bound. These are direct consequences of established tools, not a new optimization algorithm or a guarantee for the present training budget.

## What the local boundary does—and does not—predict

At the product-marginal state with zero nuisance scale, the joint Hessian in active means, active physical scales, and shared nuisance scale has diagonal blocks

\[
D,\qquad D-(1-1/K)(\Gamma\circ C),\qquad n/(4K),
\]

and zero cross-blocks. Here D is the diagonal matrix of teacher Bernoulli variances and Γ is the teacher covariance. Correlation C remains unidentified at zero active scale.

Writing B=δ+(1−δ)λ_max(D^(−1/2)ΓD^(−1/2)), the condition B>K/(K−1) guarantees a constructed feasible correlation and nonnegative scale direction that lowers both the raw objective and observation risk. It does not guarantee such a direction at every current C. In particular, C=I has positive active curvature D/K in the explicit scale–correlation coordinates. That coordinate-level local stability is not a parameterization-independent trap: a free Gaussian factor can approach the same product model along the constructed dependence direction. The zero-scale identification of all C values changes the relevant local topology.

For finite n and K, nuisance collision χ(τ) is continuous and χ(0)=1/K. Its large-τ limit is exactly E[1/M; M≥1]+P(M=0), where M is binomial with parameters K and 2^(−n). These endpoints can guarantee a small-τ window where the constructed activation improves both criteria and a large-τ window where their preferences oppose. The argument neither assumes nor proves global monotonicity of χ or a unique crossing.

The threshold concerns population curvature at the product state. Collision measurements from adaptive training banks are descriptive diagnostics only. They do not statistically confirm entry into a population window, and crossing the threshold for the entire correlation class does not establish a descent direction at the currently learned C. The empirical main conclusion must therefore use the prespecified observation-risk outputs.

## Empirical result and audit

The prespecified study trained 96 private controlled programs on 16 new datasets from the same four-active-variable generating law, using K=100, batches of 64 and 512 Adam updates. All programs learned active means, scales and six correlations from identity correlation. NEAR and UNIT differed only in initial active scale (.1 versus 1), using the same smoothed empirical-logit mean initialization. FIXED held nuisance scale at 1, FREE learned one shared physical scale, and ZERO evaluated the active finite-K score plus n log 2. ZERO is a known-structure quality reference, not exact MLE or an equal-cost competitor.

The sole primary output averages the final 64 post-update means and full physical covariance matrices, yielding one Gaussian model. Mean KL for FIXED/FREE/ZERO is .003401367/.003323737/.002049900 under NEAR and .003403310/.003040216/.001564023 under UNIT. The true product-marginal risk is .0027110543. FREE does not confirm improvement over that baseline under either start: its paired difference interval is [.000250534,.000974831] under NEAR and [−.000141695,.000800018] under UNIT. ZERO improves over the product baseline under both starts, with intervals [−.001165817,−.000156492] and [−.001577875,−.000716186]. FREE improves over FIXED only under UNIT. The original shared q=6, alpha=.05, df=15 development-t family passes 3/6 effect gates and 6/6 precision gates; all failures remain. Numerical integration error is excluded from these intervals.

FREE's mean nuisance scale falls to .045799/.045229, but shrinking that scale does not by itself establish net observation-risk recovery. On average, their joint predictions still outperform their own product marginals: a posthoc identity computed from existing PMFs gives positive mean logscore gains over each output's own product marginals (.000083894/.000371060), without new tests or intervals. Thus failing the true product baseline must not be called complete loss of dependence. No initial or intermediate risks were integrated, and parameter paths do not identify causal shares of residual nuisance noise, parameterization or finite-budget optimization.

All 288 main quadrature PMFs and 32 independently integrated GH44 PMFs pass their fixed numerical qualifications. All 384 prespecified training-state native32/math64 controls pass; the earlier deterministic fixture remains at 5/6 because one shared-scale gradient exceeds the unchanged absolute tolerance. The study concerns the actual native32 program, not an all-states mathematical-gradient certificate. The independent audit replays all input hashes and all 49,152 saved updates, but reevaluates bottom-level gradients at only 384 states; it is not complete independent retraining. There are 65,536 new observations and no neural-network fits; paired arms share 13,631,488,000 training prior Gaussian scalars, counted once. See [the full results](RESULTS.md) and independent review (historical reference not distributed).

## Authoring milestone

The active-scale study and this joint-scale follow-up are bounded scientific enhancements: they address whether an important model freedom absorbs the reported failure and identify the remaining boundary. They are not prerequisites for an unlimited sequence of new experiments.

After this prespecified study is closed and independently checked, the next authoring stage should integrate only the evidence needed for one defensible central claim. The necessary work is to settle the contribution hierarchy and theorem-level comparison with the closest literature, lead with the strongest completed evidence, compress historical experiments into supporting material, and align the abstract, figures, limitations, and main-text length. Finite-K theory, population near-optimality, and actual optimizer outcomes must remain separate. Additional research should reopen only for a specific unresolved objection that could change the central claim, rather than postponing authoring until every possible extension is explored.
