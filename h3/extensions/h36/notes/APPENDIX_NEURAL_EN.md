<a id="neural-details"></a>

# H. Neural structure experiment

## H.1. Model, candidates and proposal classes

The experiment tests input access and joint output structure in a known Gaussian observation model. The latent dimension is eight and the particle count is four:

$$
X\sim N(0,C),\qquad Y\mid X\sim N(X,T),\qquad
T=\operatorname{diag}(.25,1,4,.25,1,4,.25,1),\qquad
C_{\mathrm{teacher},\pm}=I\pm .1G.
\tag{H.1}
$$

Here the diagonal entries of T are noise **variances**. G is the symmetric adjacency matrix of the eight-cycle with the additional edge {1,5}, divided by its maximum degree, three. The teacher has nine edges. Every fitted C has unit diagonal and all 28 correlation coordinates free, including edges absent from the teacher; a row-normalized Cholesky parameterization ensures positive definiteness. The decoder and T are fixed and identifiable.

All four proposals are Gaussian. L has eight separate scalar-input networks, with factor i reading only $Y_i$. A reads all eight observations and outputs a diagonal covariance. B reads all observations and outputs four independent bivariate blocks, {1,2}, {3,4}, {5,6}, {7,8}. JNT reads all observations and outputs a full Cholesky factor. Thus “within” and “cross” always refer to B's partition: four teacher edges lie within its blocks and five cross them. L uses a two-layer tanh network of width 64 for each factor; A, B and JNT each use a shared two-layer tanh network of width 64. All include an affine skip and output the mean and permitted Cholesky entries. Diagonal scales are exponentiated log scales with protective interval [−6,3]; proposal condition numbers are limited to $10^{10}$ and candidate correlation eigenvalues must exceed $10^{-4}$. These numerical safeguards delimit the implemented families.

Training uses observations and known noise variances, without true latent values, teacher correlations, analytic posterior targets or theoretical response values. The networks' affine skips and constant covariance outputs can express the independent posterior anchor. JNT can also express the exact Gaussian posterior at a fitted candidate when the stated safeguards permit it. These are representational facts, rather than claims that a trained endpoint attains the ideal profile.

## H.2. Fixed training program and data roles

The fresh confirmation uses 12 independent dataset IDs, 300–311. Each has independent TRAIN, SELECT and EVALUATE roles of 32768, 8192 and 32768 observations. The two teacher signs use the same underlying standard normal draws; methods share each signed dataset and the training seed. Different IDs are independent. The data seed is $273100+80000+100r+j$, where r is the dataset ID and $j=0,1,2$ indexes TRAIN, SELECT and EVALUATE. The training seed is $274200+1000r$, shared across signs and structures.

There is one predeclared initialization per method and sign. All candidates start at $C=I$. Each trajectory has 8000 proposal-only updates using the analytic negative ELBO with C fixed at I, followed by 20000 joint updates of the original negative importance-weighted objective. Adam uses batches of 1024. The initial learning rate is .001 for pretraining and for L/A/B joint training, and .0003 for JNT joint training; each phase has a cosine schedule ending at one percent of its initial rate. Gradient norms are clipped at 10. Computation uses CPU double precision and one native thread per trajectory. The endpoint is the last legal state at the fixed update budget, with no early stopping or best-start choice. SELECT supplies independent objective diagnostics, never a replacement primary checkpoint. EVALUATE is reserved and is not used to choose the program, form the primary correlation contrasts, or fit the same-TRAIN references below.

For joint training the criterion and its two expected gradient blocks are

$$
\begin{gathered}
F_K(C,\phi;y)=-E\log\!\left[\frac1K\sum_{k=1}^K
  \frac{p_C(X_k,y)}{q_\phi(X_k\mid y)}\right],\\
X_k=T_\phi(\epsilon_k,y),\qquad
\alpha_k=\frac{p_C(X_k,y)/q_\phi(X_k\mid y)}
 {\sum_\ell p_C(X_\ell,y)/q_\phi(X_\ell\mid y)},\\
\nabla_C F_K=-E\sum_k\alpha_k\nabla_C\log p_C(X_k,y),\\
\nabla_\phi F_K=-E\sum_k\alpha_k^2
 \nabla_{x_k}\log\frac{p_C(X_k,y)}{q_\phi(X_k\mid y)}
 \partial_\phi T_\phi(\epsilon_k,y).
\end{gathered}
\tag{H.2}
$$

The proposal block uses IWAE-DReG [@tucker2019]. It applies to this continuous single-layer latent model with separate generating and proposal parameters, common positive support and the usual differentiation and tail conditions. Squared normalized weights are not normalized again. The implementation stops explicit density parameters while retaining the proposal's sampling path and the density's input derivative; the generating path uses first-power weights. DReG changes the gradient estimator, not the criterion: reported objectives use the original log mean weight. Its proposal gradient agrees with the raw reparameterized gradient in particle expectation, not necessarily on an individual particle bank. No claim of a new estimator or uniform variance improvement is needed here.

## H.3. Paired responses, predictions and uncertainty

Write $\widehat C_{r,Q,+}$ and $\widehat C_{r,Q,-}$ for the two fitted correlation matrices. With $S$ equal to all, within-block, or cross-block teacher edges, define

$$
\begin{gathered}
D_{r,Q}=\frac{\widehat C_{r,Q,+}-\widehat C_{r,Q,-}}{.2},\qquad
E_{r,Q}=\frac{\widehat C_{r,Q,+}+\widehat C_{r,Q,-}}2-I,\\
\rho_{r,Q,S}=\frac{\sum_{ij\in S}J_{ij}G_{ij}(D_{r,Q})_{ij}}
 {\sum_{ij\in S}J_{ij}G_{ij}^2},\qquad
J_{ij}=v_iv_j,\quad v_i=(1+\tau_i)^{-1}.
\end{gathered}
\tag{H.3}
$$

D is the odd response to the paired teachers; E retains the common even displacement. The projection uses only teacher edges, although all candidate edges are learned. E is a matrix-valued descriptive quantity, not the EVALUATE data role. A unit response corresponds to the teacher direction under this normalization.

Let B(i) denote the output block containing i. The structural calculation gives diagonal residuals and leading edge responses

$$
\begin{gathered}
R_{L,ij}=1-v_iv_j,\qquad R_{A,ij}=(1-v_i)(1-v_j),\\
R_{B,ij}=\begin{cases}0,&B(i)=B(j),\\
R_{A,ij},&B(i)\ne B(j),\end{cases}\qquad R_{\mathrm{JNT},ij}=0,\\
a_{Q,ij}=\frac{J_{ij}}{J_{ij}+R_{Q,ij}/4},\qquad
\rho^{(0)}_{Q,S}=\frac{\sum_{ij\in S}J_{ij}G_{ij}^2a_{Q,ij}}
 {\sum_{ij\in S}J_{ij}G_{ij}^2}.
\end{gathered}
\tag{H.4}
$$

These predictions follow from the independent anchor and the regular local profile calculation. They are leading local coefficients, compared here with a finite-amplitude response at ±.1 from a specified neural training program. They are not fitted to the observed contrasts.

The four predeclared contrasts are A−L on all teacher edges, B−A within blocks, B−A across blocks, and JNT−B across blocks. For each contrast, compute its paired value $z_r$ within dataset r. With $R=12$ independent datasets, sample standard deviation s and four comparisons, the two-sided Bonferroni–Student intervals are

$$
\overline z\ \pm\ t_{11,\,1-.05/(2\cdot4)}\frac{s}{\sqrt{12}}.
\tag{H.5}
$$

The critical value is 2.9808724301. These are approximate simultaneous 95% intervals under the Student approximation across datasets. The 96 fitted endpoints are not 96 independent replicates. A directional effect requires both $\overline z\ge .05$ and a strictly positive lower bound. Cross-block B−A equivalence requires its entire interval to lie in [−.05,.05]. This is a tolerance for a weighted response, rather than exact zero or coordinatewise equivalence of all 28 correlations.

| Contrast | Leading prediction | Mean | Paired SD | Approx. simultaneous 95% CI |
|---|---:|---:|---:|---:|
| A − L, all | 0.231703 | 0.231493 | 0.001275 | [0.230396, 0.232590] |
| B − A, within | 0.121754 | 0.122326 | 0.001166 | [0.121323, 0.123329] |
| B − A, cross | 0.000000 | 0.000103 | 0.000248 | [-0.000111, 0.000317] |
| JNT − B, cross | 0.104512 | 0.104921 | 0.001099 | [0.103975, 0.105867] |

All three directional rules and the one equivalence rule are met; all four leading predictions lie inside their corresponding intervals. The main figure uses separate horizontal scales, including a magnified zero neighborhood for the cross-block B−A contrast. Its equivalence bounds ±.05 lie outside that magnified panel. The table and figure are generated from the saved 96 matrices, with the 12 per-dataset contrast values and the means, standard deviations and intervals checked against the saved analysis.

## H.4. Resource revision and the status of earlier results

The mathematical program, seeds, new dataset roster and four statistical rules were fixed before generating the fresh data. A resource amendment was frozen after generation and before inspection of the fresh primary effects. It allowed predetermined resource-stopped trajectories to be restarted with the same ID and seed, or resumed from complete optimizer and random-number state; it did not change the model, learning rates, sample sizes, gradient estimator or endpoint rule. The resulting confirmation has all 96 endpoints at 28000 updates. It is a resource-amended fixed-update confirmation, rather than completion under an unchanged initial resource protocol.

The earlier resource-limited primary analysis met 1 of the four rules and remains part of the record. Completing that earlier cohort to equal updates yields a **post hoc** reassessment meeting 4 of four; its auxiliary initialization and the fresh confirmation retain their separate identities. Neither is substituted for the earlier primary result, and their dataset counts are not pooled with the fresh R12. Fixed-update completion establishes the program comparison, without certifying neural class optima or optimizer convergence to a regular profile.

## H.5. Same-TRAIN likelihood, posterior KL and even displacement

For each of the 24 fresh signed TRAIN datasets, integrate the Gaussian model analytically. The mean is fixed at zero, so the sufficient statistic is the uncentered moment $S_y=Y^\top Y/N$. Let $V_C=C+T$ and let $L_{\rm TRAIN}$ be the exact empirical observation NLL. For a fixed proposal, the particle expectation obeys

$$
\begin{gathered}
L_{\rm TRAIN}(C)=\tfrac12\left[8\log(2\pi)+\log\det V_C+
 \operatorname{tr}(V_C^{-1}S_y)\right],\\
0\le F_{K,\rm TRAIN}(C,q)-L_{\rm TRAIN}(C)
 \le \frac1N\sum_{n=1}^N
 \operatorname{KL}\!\left(q(\cdot\mid y_n)\Vert p_C(\cdot\mid y_n)\right).
\end{gathered}
\tag{H.6}
$$

The reference fits the same 28 free correlations on TRAIN, using two deterministic starts (I and a small sinusoidal perturbation), L-BFGS followed by at most 12 Newton refinements. Each start must have correlation-gradient infinity norm at most $10^{-9}$, candidate minimum eigenvalue above $10^{-4}$ and local correlation-Hessian minimum eigenvalue above $10^{-8}$. The two fitted matrices must agree within $10^{-6}$ elementwise and their objectives within $10^{-10}$ nat/observation. All 24 fresh references meet these local numerical conditions. “Exact” describes analytic likelihood and derivatives; two starts and positive local curvature do not provide a global MLE certificate.

The analytic posterior has covariance $P_C=(C^{-1}+T^{-1})^{-1}$ and mean $P_CT^{-1}y$. The Gaussian reverse KL from the frozen network proposal is evaluated analytically and averaged over the same TRAIN observations. The bound in (H.6) concerns fresh particle expectation, not a particular Monte Carlo readout, and does not by itself bound parameter or gradient error. The fixed JNT endpoints have the following descriptive ranges:

| Fixed JNT endpoint diagnostic | Range across 24 endpoints |
|---|---:|
| Same-TRAIN reference NLL difference, nat/observation | 3.234e-06–5.510e-06 |
| Empirical correlation gradient $\ell_\infty$ norm | 4.831e-04–1.128e-03 |
| Mean analytic TRAIN $\mathrm{KL}(q\Vert p_C)$ | 5.730e-06–7.415e-06 |

Their empirical gradients are nonzero despite small likelihood gaps and KL values. JNT's exact posterior is expressible at these candidates within the implementation safeguards; this does not assert that the learned proposal equals it. No additional finite-K integration was performed for these diagnostics.

The following saved-matrix ranges compare each endpoint and its odd/even parts with the locally qualified same-TRAIN MLE. Frobenius norms use the full symmetric matrices; D and E for the reference are defined by the same signed pairing as (H.3).

| Structure | $\|C-C_{\rm MLE}\|_F$ | $\|D-D_{\rm MLE}\|_F$ | $\|E-E_{\rm MLE}\|_F$ |
|---|---:|---:|---:|
| L | 0.06760–0.11770 | 0.65663–0.67423 | 0.04564–0.08664 |
| A | 0.04389–0.08040 | 0.35884–0.37038 | 0.02897–0.06289 |
| B | 0.03264–0.06868 | 0.25261–0.26401 | 0.02803–0.05942 |
| JNT | 0.00848–0.01387 | 0.00260–0.00445 | 0.00848–0.01382 |

The reference itself has $\|E_{\rm MLE}\|_F$ between 0.07743 and 0.13458. Thus a precise odd response can coexist with appreciable common displacement from the teacher. For L, A and B, differences from the empirical likelihood reference also include the restricted-proposal finite-K target shift; they are not pure optimization errors. These descriptions distinguish competing explanations without a causal decomposition of all errors. Closeness to a TRAIN MLE does not establish lower population or held-out risk. The neural experiment supplies selective structural evidence for this fixed d8/K4 program; the separate finite-data risk experiment has its own model and estimands.
