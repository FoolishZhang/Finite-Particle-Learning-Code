# Free active scales change the failure mode, without guaranteeing recovery

Releasing active means and variances does not universally remove the distortion caused by a fixed nuisance representation. It can change the population failure mode from a wrong correlation preference to a loss of observable dependence. The analysis keeps the independent nuisance scales fixed at one and releases the bounded active means, nonnegative active scales, and full active correlation matrix. It does not cover jointly shrinking the nuisance scales or an end-to-end neural model.

At the product distribution with the correct observation marginals, the finite-particle objective has the exact active-scale Hessian

\[
H_s=D-(1-\chi)(\Sigma_Y\circ C),\qquad
\chi=E\sum_k\pi_k^2,
\]

where \(\pi\) contains only the independent nuisance weights, \(D\) contains the Bernoulli marginal variances, and \(\Sigma_Y\) is the true observation covariance. The mean Hessian is D and the expected mean–scale cross block is zero. This extends the project's earlier uniform-particle local result, replacing \(1/K\) by the nuisance-weight collision probability; free-mean local stability itself is not a new finding of this package.

For \(C\succeq\delta I\) with unit diagonal, let

\[
B=\delta+(1-\delta)\lambda_{\max}
   (D^{-1/2}\Sigma_YD^{-1/2}).
\]

The sharp condition \((1-\chi)B<1\) makes the product-marginal parameter manifold locally stable for the original score over all allowed active means, scales, and correlations. If \(B>1\), a specified feasible nonnegative scale activation nevertheless decreases integrated observation risk. Thus mean compensation and variance freedom need not remove the local discrepancy. This is an exact population statement at the product boundary, not a direction rule at arbitrary trained states. The equality case is unresolved by second-order information; pairwise-independent higher-order dependence is also invisible to this local curvature test.

A separate global result uses the existing non-Gaussian nuisance near-tie bound. With fixed active dimension and at least two but at most polynomially many particles, the nuisance overlap \(1-\chi\) vanishes. A weighted Jensen-gap inequality and the uniform inverse moments of the compact Gaussian-logistic family then give uniform convergence of the centered particle objective to

\[
L(\mu,s)=\sum_i\{E\operatorname{softplus}(\mu_i+s_iG)-p_i\mu_i\}.
\]

Its unique mean–scale minimizer has \(\mu_i=\operatorname{logit}(p_i)\) and \(s_i=0\); C is unidentifiable there. Consequently, population outputs with \(o(1)\) regret converge in observation law to the product of the true marginals. Their KL risk converges to the true distribution's total correlation, positive for every nonindependent teacher. A claim of failure despite zero achievable model risk additionally requires that the teacher be representable in the released family. This is a population near-optimality result, not a prediction of an Adam endpoint.

With a fixed positive active-scale floor \(\varepsilon\), the limiting optimum instead uses that floor and the marginal-compensating means. Its pairwise spin covariance is at most \(\varepsilon^2(1-\delta)/4\) in absolute value. If a true spin covariance exceeds this bound, Pinsker's inequality gives a positive observation-risk lower bound. This separates a zero-scale closure from what a strictly positive-scale implementation can attain.

The theoretical increment is the weighted, correlation-constrained local criterion and the global free-scale dependence-collapse boundary. Jensen, Schur-product, Price/Stein, and Pinsker arguments are established tools, not proposed algorithms. Finite numerical evidence is reported separately; none is implied by these analytic conditions. See [the complete assumptions and proofs](THEORY.md).

In the prespecified K=100 finite check, a scale activation of amplitude t=.2 lowers integrated KL by .000201177, .000846886 and .001842596 in three dependent teachers. Its raw-score changes are negative without nuisance coordinates but become +.003195826, +.003023317 and +.002686154 with 256 nuisance coordinates. All six directional and eight precision gates pass. The independent null is retained: its product model is correct, and the same activation increases KL by .000014367. This is a finite candidate comparison, not a fitted free-scale endpoint.

The study uses 32,768 independent prior banks and no new observations or fitted models. Raw-score intervals are Bonferroni development-t approximations over 16 independent prior streams, conditional on fixed numerical teachers; collision bounds use the bounded-variable Hoeffding inequality. Reference integration error is qualified separately and excluded from both intervals. The independent audit replays all input hashes and saved arithmetic, rescoring only 32 prefix banks, checks 4096 calls to the unchanged native concept-loss API, and integrates 16 additional reference PMFs. All checks pass; neither a complete native32 certificate nor a complete bottom-level score rerun is claimed.
