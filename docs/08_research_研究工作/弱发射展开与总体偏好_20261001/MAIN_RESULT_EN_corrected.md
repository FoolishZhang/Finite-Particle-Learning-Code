# Covariance inflation from a finite inner-particle budget

An unbiased likelihood estimate can induce a biased population training objective after taking its logarithm. We quantify this effect in a correctly specified Gaussian–logistic model: for a fixed number K of prior particles, the population minimizer follows an explicit correlation-inflation and constraint-projection map in a weak-emission limit. The result concerns the actual expected objective over both labels and particles. Its bias therefore persists, when the projected point differs from the teacher, as more labels and independent banks estimate that same fixed-K objective.

## Model and population minimizer

Fix the dimension d and a centered Gaussian teacher with unit-diagonal covariance T. Conditional on its Gaussian coordinates, the binary spins $Y_i\in\{-1,1\}$ are independent with success probabilities $\sigma(aX_i)$, where a is the emission scale. Candidates use the same a and a covariance C in a compact convex class

$$
\mathcal C=\{C:\operatorname{diag}C=\mathbf1,\ \nu I\preceq C\preceq\Lambda I\},
\qquad T\in\mathcal C,\quad 0<\nu<1<\Lambda.
$$

The true observation law $P_a$ is exactly the candidate law $Q_{a,T}$. The raw objective is

$$
F_{a,K}(C)=E_{Y\sim P_a}\,E_{X_{1:K}\stackrel{iid}{\sim}N(0,C)}
\left[-\log\left\{\frac1K\sum_k\prod_i\sigma(aY_iX_{ki})\right\}\right].
$$

Let $c=(C_{ij})_{i<j}$, $t=(T_{ij})_{i<j}$, and let $\Gamma$ be the feasible set of these non-diagonal coordinates. A fourth-order expansion gives

$$
F_{a,K}(C)=h_{d,K}(a)+a^4\left[
\frac{(K-1)(K-2)}{32K^2}\|c\|^2
-\frac{K-1}{16K}c^\top t\right]+O(a^6),
$$

where h does not depend on C. The remainder is uniform in C together with its first two covariance derivatives. The teacher distribution also depends on a; its second-order term is essential to the displayed cross term.

For $K>2$, define $c_0=\Pi_\Gamma(Kt/(K-2))$ and let $C_0$ have unit diagonal and non-diagonal coordinates $c_0$. For sufficiently small positive a, the actual population objective has a **unique global minimizer** satisfying

$$
C_a=C_0+O(a^2),\qquad
D(P_a\Vert Q_{a,C_a})=\frac{a^4}{32}\|c_0-t\|^2+O(a^6).
$$

Uniform second-derivative control establishes strict convexity; two variational inequalities establish the parameter rate even when the projection lies on the boundary. If the inflated covariance $I+K\operatorname{off}(T)/(K-2)$ is feasible, the projection does not truncate it, and the risk coefficient becomes $\sum_{i<j}T_{ij}^2/[8(K-2)^2]$. A sufficient condition for strictly positive risk at all sufficiently small positive a is $c_0\ne t$; when $c_0=t$, the fourth-order formula alone does not determine whether higher-order risk is positive.

For fixed positive a small enough to make this coefficient decisive, uniform strong laws on the compact class imply that ideal global empirical near-minimizers converge to $C_a$ as both the number N of labels and the number B of independent banks tend to infinity. All labels may share the same B banks. Thus increasing N or log-outside B does not replace increasing log-inside K. The limiting risk is positive under the stated condition. In a subsequent limit $a\to0$, that positive value itself vanishes as $a^4$.

## Conditional correction and its boundary

When inflation is not truncated and $K>2$, the non-oracle transformation

$$
C_a^{\rm corr}=I+\frac{K-2}{K}(C_a-I)
$$

is feasible because it is a convex combination with I. It satisfies $C_a^{corr}-T=O(a^2)$. A direct Price bound, $|\partial_{C_{ij}}Q_{a,C}(y)|\le a^2/16$, turns this into an $O(a^4)$ probability error and hence an $O(a^8)$ upper bound on observed joint KL. This is an upper rate, not a proved sharp rate. At a boundary teacher whose raw optimum is already correct, the same shrinkage can increase risk; it is not a universal repair and provides no K2 correction.

## Deterministic numerical evidence

For d=2, teacher correlation $\theta=.15$, and candidate correlation $\rho\in[-.75,.75]$, the weak-emission targets are .45, .30 and .20 for K3, K4 and K8. K2 has a vanishing fourth-order quadratic penalty; for sufficiently small a its exact population minimum is the upper endpoint .75. K1 is exactly flat over the whole correlation interval.

At the predeclared smallest scale a=.05, the numerical outputs are:

|K|Raw located rho|Raw observed joint KL|After fixed correction|
|---|---:|---:|---:|
|2|.75000|7.014e−8|Not applied|
|3|.44963|1.749e−8|2.988e−15|
|4|.30000|4.384e−9|5.729e−21|
|8|.20004|4.879e−10|1.888e−16|

The frozen programme covers five a values, 20 primary location procedures, 60 quadrature-variant executions and 15 fixed correction evaluations. All 8 prespecified weak-emission prediction readings and all numerical qualification gates passed. An independent audit checked the saved calculations, independently rebuilt 704 moment/KL comparisons, and compared nine K2 points with direct four-dimensional integration at two orders. No labels, random particle banks or data fits were generated. The finite quadrature and root search are not strict integration or continuous-global-optimum certificates. K4's corrected risk decreases faster in these outputs; no sharp $a^8$ or $a^{12}$ law is asserted.

## Scope of the contribution

The narrow result is an explicit fixed-K covariance map, uniform control connecting it to actual population minima, and the corresponding observed-risk scale. Gaussian covariance differentiation, log-mean bias, convex shrinkage and M-estimation consistency are established tools and phenomena; historical priority for this specialization is not certified. The theorem fixes d and K, fixes unit-diagonal base covariance, and takes a to zero. The actual logit spectral floor is $a^2\nu$, so this is distinct from the project's fixed-positive-floor high-dimensional results. It does not certify SGD or a finite multistart solver, and the numerical behavior at a=1 does not extend the small-a theorem.

Proofs: weak-emission and learning theorem (historical reference not distributed), [Price identities](PRICE_THEORY.md), [conditional correction and counterexample](CORRECTION_THEORY.md). Evidence: [complete results](RESULTS.md), independent numerical audit (historical reference not distributed), literature boundary (historical reference not distributed).
