# Appendix B. Positioning and the conditional-variance interpretation

## B.1. An exact interpretation of the limiting potential

Keep the fixed Gaussian marginals and teacher of Section 2. Write

$$
A_C(y,Z)=\sum_{i=1}^m\log\sigma(y_iX_i),\qquad
T(C)=E_{Y\sim Q_{C_*}^A}\!\left[\operatorname{Var}_Z A_C(Y,Z)\right].
\tag{B.1}
$$

For each fixed $y$, the mean $E_ZA_C(y,Z)$ depends only on the fixed one-dimensional marginals. Thus its derivative with respect to a free off-diagonal covariance vanishes. The mixed derivative of $A_C^2$ is $2(\partial_{x_i}A_C)(\partial_{x_j}A_C)$, since $A_C$ is a coordinatewise sum. Gaussian integration by parts gives Price's identity [@price1958]. The square and derivative terms are dominated by Gaussian-integrable envelopes uniformly on the compact class. Applying (6),

$$
\partial_{c_{ij}}T(C)=2EH_{ij}
=\frac{\kappa_{ij}(C_*)+\kappa_{ij}(C)}2,
\qquad V(C)=-\frac12\{T(C)-T(I)\}.
\tag{B.2}
$$

The second identity follows by integrating over the feasible segment from $I$ and using $V(I)=0$. It is an interpretation of the existing potential, not an independent discovery of variance-driven selection. It concerns conditional *active log likelihood*, followed by a true-observation average. It is distinct from nuisance log-weight variance, the variance of a stochastic gradient, and the variance of a predictive probability. A separate numerical check uses means $(-.6,.4)$, teacher correlation $.25$ and candidate correlations $(-.8,-.3,0,.4,.8)$. Direct conditional-variance quadrature at orders 32/48/64 agrees with the integrated covariance expression (64-point Legendre) to at most $5.85\times10^{-14}$; the identity is justified analytically above.

## B.2. What the nearest asymptotics predict, and what they do not

Daudel--Roueff's v2 Proposition 1 proof, equation (208), has a subleading mean-gradient term $\sqrt{2\log K}\,\partial_\psi B$ in addition to $\partial_\psi E\Lambda$ [@daudel2026]. That mean-gradient formula takes particle count and $B$ to infinity with $\log K=o(B^2)$. The relevant locator is the Gaussian log-weight random-field condition and Section D.4.1, equation (208), of Daudel--Roueff. Formally matching conditional log-weight means and variances in this model gives

$$
\begin{gathered}
B_C(y)^2=nv+\operatorname{Var}_Z A_C(y,Z),\\
-\sqrt{2\log K}\,E_Y\partial_{c_{ij}}B_C(Y)
=-\frac{\sqrt{2\log K}}{2\sqrt{nv}}\partial_{c_{ij}}T(C)+O\!\left(\frac{\sqrt{\log K}}{n^{3/2}}\right).
\end{gathered}
\tag{B.3}
$$

As $n\to\infty$ and $K\to\infty$ within the polynomial range, this predicts the coefficient in (8). It is a mechanism connection, not a direct application to the actual model: logistic log weights are not a Gaussian random field, and an ordinary CLT does not give relative pair-kernel accuracy near extremes or permit differentiation of an uncontrolled remainder. The precise Gaussian step is in Section D.4.1, proof of (208): normalized weights are independent of the centered derivatives of the standardized log-weight field, so the derivative contribution has zero expectation. Section D.3's Gaussian-field property supplies this independence; Section 3.4, after (49), explains that it is lost under merely approximate Gaussianity. We locate the condition by its content and these sections rather than its version-dependent label. Uniformity over the full correlation class and fixed $K\ge2$ also need to be established. Moreover, replacing a log weight by a moment-matched Gaussian changes its exact exponential integral in general: the latter is $\log E e^\Lambda=E\Lambda+\operatorname{Var}(\Lambda)/2$. Thus the formal replacement need not preserve the statistical model.

Non-Gaussian log-weight sums already occur in Daudel et al., Theorem 6 (A2 and equations (37)--(38)), with $\log K=o(d^{1/3})$ [@daudel2023]. The scalar expansion is not a uniform $o(w)$ parameter-difference or gradient theorem. Likewise, Bickel et al., Proposition 3.1, assumes the relative tail/density conditions of its Lemmas A.1--A.2 and $(\log K\log d)/d\to0$ [@bickel2008]. On identifying their particle count with $K$ and score standard deviation with $\sqrt{nv}$, their expected maximum-weight excess has the same large-$K$ leading scale $\sqrt{2\log K}/\sqrt{nv}$ as $w_{n,K}$; that scale is not new. It does not directly provide the marked conditional pair estimate (A.13). Earlier collapse results [@bengtsson2008; @snyder2008] are therefore background, not novelty here. These comparisons do not rule out future extensions of those frameworks. We make no exponential-budget necessity claim for recovering this specific parameter choice.

## B.3. Theorem-level comparison with nearby results

The distinction is uniform parameter-selection-scale analysis for the actual non-Gaussian model and its observation-risk consequence. The conditional-variance identity in B.1 connects the mechanism to established high-mismatch gradient theory. General Jensen bias, estimator-dependent learning, collapse, and correlation-matrix extreme-point geometry are established tools or phenomena. The following comparison concerns the stated analysis objects; it does not claim that earlier frameworks could never be extended to this model.

|Result|Key assumptions / object|Established conclusion and distinction here|
|---|---|---|
|Monte Carlo objectives / FIVO [@maddison2017]|Positive unbiased likelihood estimators; expected logarithmic objectives, including particle filtering estimators.|General objective construction and lower-bound properties. The present question identifies the signed correlation field induced by an observation-equivalent nuisance representation.|
|AESMC, Claim 1 [@le2018aesmc]|Nonnegative unbiased normalizing-constant estimator under a proposal on an extended sampling space.|The expected-log gap is an extended-space KL to the estimator-induced target, not observation KL of the integrated candidate. It does not identify our uniform signed correlation field.|
|Daudel et al., Theorem 6 [@daudel2023]|High-dimensional VR-IWAE bounds and weight models, including non-Gaussian log-weight sums under specified assumptions.|A2 and equations (37)--(38), with $\log K=o(d^{1/3})$, yield scalar objective-gap asymptotics beyond an exactly log-normal model. Scalar flattening alone does not supply the full-correlation signed field and risk statement proved here.|
|Daudel and Roueff, v2 [@daudel2026]|Gaussian log-weight random-field condition; derivative/expectation interchange; Section D.4.1 of Daudel--Roueff, Proposition 1, equation (208).|The standard-deviation derivative formally predicts our large-K coefficient via B.1. Actual non-Gaussian conditional pair control, uniformity over C and fixed K require additional analysis; an uncontrolled CLT remainder cannot be differentiated.|
|Particle-filter collapse [@bengtsson2008; @bickel2008; @snyder2008]|Bickel et al. Proposition 3.1: appendix relative tail/density conditions and $(\log K\log d)/d$ tending to zero.|Maximum-weight excess and extreme-value scale are established; this does not directly control marked overlaps conditional on active-score shifts.|
|Simulated likelihood [@lee1995]|Discrete-choice simulated maximum likelihood and logarithmic derivative bias.|Classical bias background; not a theorem about this growing-dimension uniform field.|
|Multiperiod probit simulations [@geweke1997]|GHK simulated likelihood for multinomial multiperiod probit models.|The published abstract reports downward bias in serial-correlation estimates. We do not infer a universal sign or a nuisance-growth mechanism from that report; the full experimental analysis was not rechecked here.|
|Extreme correlation matrices [@litam1994]|Real correlation matrices and their extreme-point geometry.|The rank restriction is existing geometry. Our model-specific step places the minima of the particle-induced strictly concave potential in that extreme set, then transfers the separation to observation KL.|
|Latent-variable collapse [@lucas2019; @wangziyin2022]|Analyzed linear latent-variable models and their likelihood/variational objectives.|Collapse and limits of variance learning predate this work. Our scale boundary uses independent nuisance collision, the constrained correlation class, and a Gaussian-logistic observation-KL limit. The uniform-weight Hessian is a specialization, not a separate novelty claim.|
|Present fixed-scale theorem|Fixed active dimension and Gaussian marginals; independent positive-compact nuisance scales; polynomial K; population regret $o(w)$.|Uniform first effective signed field, strictly concave matrix potential, floor-eigenvalue geometry and positive observation-risk bound. It does not establish a finite-optimizer limit.|
|Present free-active-scale boundary|Compact active means/scales, fixed unit nuisance scale, polynomial K; population regret $o(1)$.|Uniform leading-order convergence to a marginal objective and risk tending to teacher total correlation. Jointly free nuisance scales fall under A.9 instead; finite-program outcomes are separate evidence.|
