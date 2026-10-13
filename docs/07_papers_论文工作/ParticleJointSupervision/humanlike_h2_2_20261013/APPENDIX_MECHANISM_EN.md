# Appendix I. Joint learning and local stability without a designated removable block

The main argument certifies a local minimum with every bias and loading free, then shows that a dependence change preserving candidate logit marginals lowers risk from that same point. Read the [model](#dense-model), [full derivatives](#full-derivatives) and [Laplace representation](#laplace-representation) in I.1–I.3, followed by the [root certificate](#proof-theorem-3) in I.8 and the [angular-path proof](#same-root-angle) in I.10. The [explicit teacher-family extension](#quantitative-teacher-family) in I.11 proves the quantitative angular-persistence corollary; the [qualitative continuation](#local-persistence) in I.9 concerns the original radial risk direction. Sections I.4–I.7 give the numerical discovery context and its limits, including the empirical fits and K11/K12 comparisons, which retain their numerical status.

## I.1. Model, targets, and the weak-loading interface {#dense-model}

This appendix studies a finite-dimensional model distinct from the nuisance-growth setting of Theorems 1–2. Let $m=8$, $Z\sim N(0,I_2)$, and allow every bias and loading to vary:

$$
\begin{aligned}
A_y(z;b,W)&=\prod_{j=1}^m\sigma(b_j+W_jz)^{y_j}
[1-\sigma(b_j+W_jz)]^{1-y_j},\qquad y\in\{0,1\}^m,\\
p_\theta(y)&=E_ZA_y(Z;\theta),\qquad
F_K^q(\theta)=\sum_yq_yE\left[-\log\left\{K^{-1}\sum_{k=1}^KA_y(Z_k;\theta)\right\}\right],\\
F_\infty^q(\theta)&=-\sum_yq_y\log p_\theta(y),\qquad
R(\theta)=D_{\mathrm{KL}}(q_*\Vert p_\theta).
\end{aligned}
\tag{I.1}
$$

The target $q$ is fixed during candidate differentiation: either empirical pattern frequencies or the teacher distribution $q_*$. The teacher has $b_*=0$ and rows $W_{*,j}=.8(\cos\phi_{*,j},\sin\phi_{*,j})$, with $\phi_*=(-.7,-.5,-.3,-.1,.1,.3,.5,.7)$. All rows are nonzero and every off-diagonal loading inner product is positive; Gaussian logistic monotonicity also gives nonzero pairwise observation dependence. There is no designated independent observation block to delete. Bernoulli latent-variable modelling is established [@tipping1998]; the contribution here concerns a particular objective geometry, not a new model or generic latent-collapse phenomenon [@burda2016; @lucas2019; @urban2021].

Since $\operatorname{rank}(WW^\top)\le2$, this class is not the positive-spectral-floor full-correlation class of Theorems 1–2. Neither their spectral conclusion nor their nuisance-growth rate is transferred to this model.

For the weak-loading calculation only, use spins $Y_j\in\{-1,1\}$, zero biases, $W_j^\top=\varepsilon v_j(\phi)$, and $v_j=s_j(\cos\phi_j,\sin\phi_j)^\top$ with fixed positive $s_j$. Let $c=(v_i^\top v_j)_{i<j}$ and let the teacher have the same $\varepsilon,s$ and angle $\phi_*$. For fixed $m,K$, uniformly with two angle derivatives on a compact angle neighbourhood,

$$
\begin{aligned}
F_{K,\varepsilon}(\phi)-F_{K,\varepsilon}(\phi_*)
&=\varepsilon^4\{a_K(\|c\|^2-\|c_*\|^2)-b_Kc_*^\top(c-c_*)\}+O(\varepsilon^6),\\
a_K&=\frac{(K-1)(K-2)}{32K^2},\qquad b_K=\frac{K-1}{16K}.
\end{aligned}
\tag{I.2}
$$

For a smooth path through the teacher, write $\dot c=dc/dt|_0$. The corresponding risk and directional derivatives are

$$
\begin{aligned}
R_\varepsilon(\phi)&=\frac{\varepsilon^4}{32}\|c-c_*\|^2+O(\varepsilon^6),\\
F_{K,\varepsilon}'(0)&=-\frac{K-1}{8K^2}\varepsilon^4c_*^\top\dot c+O(\varepsilon^6),\\
R_\varepsilon'(0)&=0,\qquad
R_\varepsilon''(0)=\frac{\varepsilon^4}{16}\|\dot c\|^2+O(\varepsilon^6).
\end{aligned}
\tag{I.3}
$$

Here is a coefficient derivation. Put $x_j=v_j^\top Z$ and expand $2^mA_y=1+\varepsilon h_1+\varepsilon^2h_2+\varepsilon^3h_3+\cdots$:

$$
\begin{aligned}
h_1&=\tfrac12\sum_jy_jx_j,\qquad h_2=\tfrac14\sum_{i<j}y_iy_jx_ix_j,\\
h_3&=\tfrac18\sum_{i<j<\ell}y_iy_jy_\ell x_ix_jx_\ell-\tfrac1{24}\sum_jy_jx_j^3,\\
[-\log(1+\textstyle\sum_{r\ge1}\varepsilon^r\bar h_r)]_{\varepsilon^4}
&=-\bar h_4+\bar h_1\bar h_3+\tfrac12\bar h_2^2-\bar h_1^2\bar h_2+\tfrac14\bar h_1^4.
\end{aligned}
\tag{I.4}
$$

Bars average the $K$ particles. Uniform-spin orthogonality and Gaussian fourth moments give the following coefficients of each $c_{ij}^2$; terms depending only on the fixed row norms cancel in differences.

|Term in (I.4)|Coefficient of $c_{ij}^2$|
|---|---:|
|$-E\bar h_4$|$0$|
|$E\bar h_1\bar h_3$|$0$|
|$E\bar h_2^2/2$|$(K+1)/(32K)$|
|$-E\bar h_1^2\bar h_2$|$-(K+1)/(8K^2)$|
|$E\bar h_1^4/4$|$3/(16K^2)$|

In particular,

$$
\begin{aligned}
E(\overline{x_ix_j})^2&=(1+K^{-1})c_{ij}^2+K^{-1}s_i^2s_j^2,\\
E(\bar x_i\bar x_j\overline{x_ix_j})&=K^{-2}\{s_i^2s_j^2+(K+1)c_{ij}^2\},\\
E_Z[-\bar h_2+\bar h_1^2/2]
&=\frac{\sum_js_j^2}{8K}-\frac{K-1}{4K}\sum_{i<j}y_iy_jc_{ij}.
\end{aligned}
\tag{I.5}
$$

The teacher moment $E(Y_iY_j)=\varepsilon^2c_{*,ij}/4+O(\varepsilon^4)$ supplies the $b_K$ term. Expanding the finite-alphabet KL and using the same spin orthogonality gives (I.3). At $K=2$, the exact two-particle log-cosh identity independently yields cancellation of the two $\|c\|^2$ coefficients, confirming $a_2=0$ and $b_2=1/32$.

The remainder is not a pointwise uniform Taylor bound on unbounded Gaussian inputs. Real derivatives of log-sigmoid are bounded; derivatives of log-sum-exp are weighted cumulants with weights in $[0,1]$. Through the required sixth loading derivative and two angle derivatives, a polynomial in $\max_k\|Z_k\|$ dominates on compact parameter sets. Gaussian moments justify differentiation under expectation. Simultaneously reversing all latent particles makes the objective even in $\varepsilon$. These facts give the stated remainder; no explicit sixth-order constant, growing-$K$ uniformity, or threshold for $\varepsilon=.8$ is supplied.

For $K>2$, completing the square in (I.2) gives, up to candidate-angle-independent terms, $\varepsilon^4a_K\|c-\{K/(K-2)\}c_*\|^2+O(\varepsilon^6)$: this fixed-row-length weak-loading approximation favours amplified pairwise inner products. For the stated teacher angles and $s_j=1$, let $L_*$ be the Laplacian with off-diagonal edge weights $\cos(\phi_{*,i}-\phi_{*,j})$. At the collinear candidate, the leading angle Hessian on $\mathbf1^\perp$ is $\varepsilon^4\{b_KL_*-2a_K(8)I\}$. Its minimum changes sign formally at $K=16/(8-\lambda_{\min}(L_*|_{\mathbf1^\perp}))\simeq4.32690638$, obtained from one fixed eigendecomposition, without a $K$ scan. This is a real-valued boundary of the leading coefficient for fixed equal row lengths and sufficiently small common loading, not a quantitative prediction at loading $.8$. The qualified 11/12 separation in I.6 instead concerns the complete finite-loading target with biases and row lengths refit at each $K$. The weak expansion gives preference intuition but does not quantitatively predict that refitted transition; neither number identifies a universal critical particle count.

For $s_j=1$ and $\phi(t)=(1-t)\phi_*$, every $c_{*,ij}>0$ and $\dot c_{ij}=(\phi_{*,i}-\phi_{*,j})\sin(\phi_{*,i}-\phi_{*,j})>0$. Thus sufficiently weak loadings give finite-objective descent and local risk increase. $K=1$ is exactly angle invariant. The finite-$\varepsilon$ check retains both unsuccessful precision cases:

|$\varepsilon$|$F_2(.05)-F_2(0)$|$R(.05)$|$F_2'(0)$|GH16/20 derivative difference|Qualification|
|---:|----------:|----------:|----------:|----------:|--------|
|.125|$-2.46194897\,10^{-6}$|$1.53201221\,10^{-7}$|$-5.01126089\,10^{-5}$|$9.92045\,10^{-18}$|Passed|
|.25|$-3.78093733\,10^{-5}$|$2.20782999\,10^{-6}$|$-7.68065055\,10^{-4}$|$1.01103\,10^{-12}$|Passed|
|.5|$-4.60084001\,10^{-4}$|$2.51342565\,10^{-5}$|$-.00932588949$|$1.40079\,10^{-7}$|Failed|
|.8|$-.00187535140$|$1.00148195\,10^{-4}$|$-.0379671263$|$2.04217\,10^{-5}$|Failed|

These use complete four-dimensional $K=2$ banks at GH orders 8/12/16/20; derivative and paired-loss qualification requires successive 16/20 differences at most $10^{-7}$. The separate two-dimensional teacher-PMF check passed at $.8$, permitting learning, without qualifying its four-dimensional angular calculation.

## I.2. Full derivatives and the conditional local-type criterion {#full-derivatives}

::: {#full-hessian-criterion}

**Proposition 3 (rank-one Hessian structure).** Fix $q$ and $K$, allowing the integrated objective, and let $W=(a,0)$ with $a_j>0$. In coordinates $(b,u,v)$, write $H_{ba}$ for the bias/first-column block, $H_\perp$ for the second-column block, $g_a=\partial_a F_K^q$ and $D_a=\operatorname{diag}(a)$. Then

$$
\begin{aligned}
H_{\rm full}&=\operatorname{diag}(H_{ba},H_\perp),\\
H_\phi&=D_aH_\perp D_a-\operatorname{diag}(a\odot g_a),\\
H_\perp a&=g_a,\qquad H_\phi\mathbf1=0.
\end{aligned}
$$

:::

Return to bits. At $W_j=(a_j,0)$, with $a_j>0$, write

$$
\begin{aligned}
p_k&=\sigma(b+az_k),\quad r_k=y-p_k,\quad D_k=\operatorname{diag}\{p_{kj}(1-p_{kj})\},
\quad\pi_k=\frac{A_y(z_k)}{\sum_\ell A_y(z_\ell)},\\
G_k&=\binom{r_k}{z_kr_k},\qquad
B_k=\begin{pmatrix}D_k&z_kD_k\\z_kD_k&z_k^2D_k\end{pmatrix}.
\end{aligned}
\tag{I.6}
$$

Expectations below include $y\sim q$ and iid particles. Differentiating log-sum-exp in the coordinate order $(b,a)$ gives

$$
\begin{aligned}
g_{ba}&=-E\sum_k\pi_kG_k,\\
H_{ba}&=E\sum_k\pi_kB_k-E\sum_k\pi_k(1-\pi_k)G_kG_k^\top\\
&\quad+E\sum_{k\ne\ell}\pi_k\pi_\ell G_kG_\ell^\top.
\end{aligned}
\tag{I.7}
$$

The last term is essential: the negative Hessian is the weighted conditional negative Hessian minus the particle-score covariance. Cross-particle outer products are formed before averaging over labels.

In full coordinates $(b,u,v)$, where $W_j=(u_j,v_j)$, reflection of the second Gaussian coordinate gives $F(b,u,v)=F(b,u,-v)$. At $(u,v)=(a,0)$,

$$
\begin{aligned}
g_v&=0,\qquad H_{(b,u),v}=0,\qquad
H_{\rm full}=\operatorname{diag}(H_{ba},H_\perp),\\
H_\perp&=E\left[\sum_k\pi_kD_k-\sum_k\pi_k(1-\pi_k)r_kr_k^\top\right].
\end{aligned}
\tag{I.8}
$$

Independent, centred second particle coordinates remove cross-particle terms only from the transverse block. A norm-preserving angle path also has second variation $\ddot W_{j1}=-a_j\dot\phi_j^2$, so

$$
H_\phi=\operatorname{diag}(a)H_\perp\operatorname{diag}(a)
-\operatorname{diag}(a\odot g_a),\qquad H_\perp a=g_a,\qquad H_\phi\mathbf1=0.
\tag{I.9}
$$

For completeness, Gaussian integration by parts applies to

$$
\begin{aligned}
\partial_{z_k}(\pi_kr_{ki})
&=\pi_k(1-\pi_k)(a^\top r_k)r_{ki}-\pi_ka_i(D_k)_{ii},\\
E[z_k\pi_kr_{ki}]&=E[\partial_{z_k}(\pi_kr_{ki})].
\end{aligned}
\tag{I.10}
$$

This proves the Ward relation in (I.9). The equivalent signed Laplacian is

$$
\begin{aligned}
w_{ij}&=a_ia_jE\sum_k\pi_k(1-\pi_k)r_{ki}r_{kj},\qquad i\ne j,\\
(H_\phi)_{ij}&=-w_{ij},\qquad (H_\phi)_{ii}=\sum_{j\ne i}w_{ij},\\
d^\top H_\phi d&=\sum_{i<j}w_{ij}(d_i-d_j)^2.
\end{aligned}
\tag{I.11}
$$

Weights may be negative; a structural zero row sum alone does not validate quadrature. At $K=1$, $H_\phi=0$ exactly.

At a **genuine** stationary point $g_b=g_a=0$, let $V$ have orthonormal columns spanning $a^\perp$. The sufficient condition is

$$
\lambda_{\min}(H_{ba})>0,\qquad
\lambda_{\min}(V^\top H_\perp V)>0.
\tag{I.12}
$$

The slice $a^\top v=0$ is transverse to the common-rotation orbit. By (I.8), (I.12) makes its Hessian positive definite, yielding a strict local minimum modulo rotation; for $m=8$ there are 23 positive nonrotation directions. A negative transverse direction gives local descent. Discrete reflection adds no continuous zero mode. These conditional statements do not turn a small floating-point gradient into a certified stationary point. Section I.8 supplies the additional error bounds and root inclusion for the population K8 problem; the empirical fits and the K11/K12 continuation retain their numerical status. Compact-neighbourhood Gaussian polynomial domination justifies the derivatives, and positivity of each integrated label probability gives the analogous local argument for $F_\infty^q$.

## I.3. One-particle Laplace computation and paired differences {#laplace-representation}

For each label choose any $c_y>0$, put $x=A_y/c_y$, $u=e^v$, and define

$$
\Psi_y(u)=E e^{-ux},\qquad B_{0,y}(u)=E[xe^{-ux}],\qquad M_{G,y}(u)=E[xe^{-ux}G].
\tag{I.13}
$$

The identities $S^{-1}=\int_0^\infty e^{-uS}du$ and $S^{-2}=\int_0^\infty ue^{-uS}du$, with particle exchangeability, reduce the bank expectation to a single latent integral and one auxiliary integral. For discrete normal nodes $(z_\ell,\omega_\ell)$, replace the inner expectations by weighted sums:

$$
\begin{aligned}
d_{y\ell}&=\omega_\ell x_\ell K\int_{\mathbb R}u e^{-ux_\ell}\Psi_y(u)^{K-1}\,dv,\\
c_{y\ell}&=\omega_\ell x_\ell K(K-1)\int_{\mathbb R}u^2e^{-ux_\ell}B_{0,y}(u)\Psi_y(u)^{K-2}\,dv.
\end{aligned}
\tag{I.14}
$$

The labelwise derivatives are then

$$
\begin{aligned}
g_{ba,y}&=-\sum_\ell d_{y\ell}G_\ell,\\
H_{ba,y}&=\sum_\ell d_{y\ell}B_\ell-\sum_\ell c_{y\ell}G_\ell G_\ell^\top\\
&\quad+K(K-1)\int_{\mathbb R}u^2M_{G,y}(u)M_{G,y}(u)^\top\Psi_y(u)^{K-2}\,dv.
\end{aligned}
\tag{I.15}
$$

Average these quantities with $q_y$ afterwards. Signed-term Fubini is justified by $\sum_k\pi_k(1-\pi_k)\le1-1/K$ and integrable Gaussian score moments. At $K=1$, $g=-EG$ and $H=EB$. For integrated likelihood, with posterior $\pi_y(dz)\propto A_y(z)\varphi(z)dz$,

$$
\begin{aligned}
g_{ba,\infty,y}&=-E_{\pi_y}G,\\
H_{ba,\infty,y}&=E_{\pi_y}B-E_{\pi_y}GG^\top
+(E_{\pi_y}G)(E_{\pi_y}G)^\top,\\
H_{\perp,\infty,y}&=E_{\pi_y}D-E_{\pi_y}rr^\top.
\end{aligned}
\tag{I.16}
$$

Reflection and the Ward relations still apply after target averaging. At an integrated model's own target, its collinear angle Hessian vanishes; strict positivity is not imposed on this check. For a numerical normal measure $\nu$ of raw mass $M$, retain its weights and the finite-bank mass $M^K$:

$$
\begin{aligned}
F_{K,y}^{(\nu)}&=-M^K\log c_y+
\int_{\mathbb R}\{\Psi_y^{(\nu)}(e^v)^K-M^Ke^{-Ke^v}\}\,dv,\\
\sum_\ell d_{y\ell}&=M^K,\qquad
0\le\sum_\ell c_{y\ell}\le M^K(1-1/K).
\end{aligned}
\tag{I.17}
$$

The equalities refer to the untruncated auxiliary integral; their numerical residuals are checked. Integrated posterior weights instead have mass one. No normal or target weight is silently renormalized.

For matched noncollinear paths with an angle direction $d_{\mathrm{ang}}^{(12)}\in\mathbb R^8$, the common scale is fixed at the baseline:

$$
W_j(t)=a_j(\cos(td_{\mathrm{ang},j}^{(12)}),\sin(td_{\mathrm{ang},j}^{(12)})),\qquad
c_y=\frac{\int A_{y,0}\,d\nu}{M},\qquad
\Psi_{y,t}^{(\nu)}(u)=\int e^{-uA_{y,t}/c_y}\,d\nu.
\tag{I.18}
$$

The primary changes are evaluated directly as

$$
\begin{aligned}
\Delta F_K^{(\nu)}(t)&=\sum_yq_y\int_{\mathbb R}
\{\Psi_{y,t}^{(\nu)}(e^v)^K-\Psi_{y,0}^{(\nu)}(e^v)^K\}\,dv,\\
\Delta R^{(\nu)}(t)&=-\sum_yq_y\{\log p_t^{(\nu)}(y)-\log p_0^{(\nu)}(y)\}.
\end{aligned}
\tag{I.19}
$$

Common constants cancel by the Frullani identity. Only the inner integral becomes two-dimensional, not $2K$-dimensional. Exponent differences use sign-preserving `expm1`; absolute-loss subtraction is an interface check. Small-$u$ evaluation uses the original mass with `log1p`/`expm1`; very small sums use log-sum-exp. Under the true Gaussian law, fixed row norms preserve every univariate observation marginal, and reflection makes both changes even in $t$. Local smoothness gives

$$
F_K(t)-F_K(0)=\tfrac12t^2(d_{\mathrm{ang}}^{(12)})^\top H_\phi d_{\mathrm{ang}}^{(12)}+O(t^4),\qquad
R(t)-R(0)=\tfrac12t^2(d_{\mathrm{ang}}^{(12)})^\top H_{\phi,\infty}d_{\mathrm{ang}}^{(12)}+O(t^4).
\tag{I.20}
$$

The computed finite differences below are not generated from (I.20).

## I.4. Full learning and the mixed fixed-marginal control

Four exploratory streams each contain 4096 complete observations from the $.8$ teacher. All methods share data and initialization: Jeffreys-smoothed marginal logits and iid $N(0,.25^2)$ loading entries. Biases and loading entries are constrained to $[-2,2]$. Projected Adam takes 1200 updates, learning rates $.02$ for 800 steps and $.005$ thereafter, $(\beta_1,\beta_2)=(.9,.999)$, and stabilizer $10^{-8}$; the output is LAST. Each step shares 128 fresh two-dimensional prior vectors between $K=8$ and $64$, arranged as 16 or 2 banks. Each bank takes its own log-mean before averaging; pooling banks would optimize a different objective. GH20 supplies the integrated empirical comparator. Equal particle counts do not equalize gradient noise or total cost.

REF starts at GH-Adam LAST and uses GH40 L-BFGS-B, at most 300 iterations/450 calls, gradient tolerance $10^{-7}$ and function tolerance $10^{-12}$. All four already met the gradient criterion: one call, zero further iterations. IND fits independent Bernoulli margins with Jeffreys smoothing. The teacher enters only locked-output population evaluation, not training. Data/prior streams use PCG64DXSM with SeedSequence $[202610070100,r,\mathrm{role}]$; roles 0–4 specify latent draws, emission uniforms, initialization, training priors and independent evaluation.

|Stream|K8 KL|K64 KL|REF KL|IND KL|
|---|---:|---:|---:|---:|
|0|.010714173|.000882779|.001013446|.106851719|
|1|.013518917|.005017770|.004563076|.108354984|
|2|.008767421|.003466091|.002591933|.107382320|
|3|.011357337|.002199853|.002356989|.107318617|
|Mean|.011089462|.002891623|.002631361|.107476910|

K64 has lower risk than K8 in every stream, but beats REF in two streams: more accurate integration is not uniformly best in finite data. K8 unit-row Gram matrices have second eigenvalues $1.20\,10^{-5}$–$2.30\,10^{-4}$, versus 1.3403–1.5988 for REF and 1.0785–1.4567 for K64. This is near-one-factor **loading structure**, not rank-one observation covariance.

For each stream, 4096 new 64-particle banks evaluate fixed models jointly; $F_8$ uses the first eight particles. Entries below are K8 minus REF, with conditional MCSE in parentheses, in nats per eight-bit observation.

|Stream|Empirical $F_8$ difference|Population $F_8$ difference|
|---|---:|---:|
|0|−.017111839 (.000420685)|−.019030398 (.000421690)|
|1|−.017389742 (.000861898)|−.017994448 (.000870445)|
|2|−.016952623 (.000519001)|−.019887290 (.000506188)|
|3|−.017387109 (.000383751)|−.018698806 (.000387336)|

Both empirical integrated NLL and population KL prefer REF; $F_{64}$ evaluation also prefers REF in all streams and both targets. These are fixed-output ordering comparisons, not global-optimality or training-population confidence claims.

The staged profile control fixes REF biases and row norms but replaces unit directions by K8 or K64 directions, producing H8/H64 without teacher alignment or refitting. A separately frozen bank family $[202610070300,r,0]$ uses the same evaluation budget. All candidates remain feasible, and the maximum marginal discrepancy is $6.67\,10^{-16}$.

|Stream|H8−REF KL|Empirical $F_8$ difference (MCSE)|Population $F_8$ difference (MCSE)|
|---|---:|---:|---:|
|0|+.015649258|−.000384622 (.000474800)|−.001914842 (.000475939)|
|1|+.015691309|−.000283583 (.000921181)|+.000805674 (.000943760)|
|2|+.015529662|−.002605471 (.000575264)|−.000903313 (.000572604)|
|3|+.016110010|+.001209140 (.000399319)|−.000149413 (.000400167)|

Thus H8 worsens risk throughout, but its $F_8$ ordering is mixed and some differences are comparable to MCSE. $F_{64}$ consistently penalizes H8. H64−REF KL is $(−.000009563,+.000896783,+.000587691,+.000160217)$; its mean KL is $.00304014$, versus H8's $.01837642$. Neither this improvement control nor later curvature results erase the mixed H8 comparison. No banks were added to obtain a preferred sign.

## I.5. Row-length controls and full stationary signatures

Canonical collinear anchors use positive row norms, not the original first loading column. Each stream combines K8/REF biases with K8/REF row norms; at fixed K8 bias, row norms also interpolate at $\lambda=.25,.5,.75$. At K8 biases and row norms, the minimum eigenvalues of the angle Hessian projected onto $\mathbf1^\perp$ are:

|Stream|Empirical $F_8$|Population $F_8$|Population $F_{64}$|Risk|
|---|---:|---:|---:|---:|
|0|+.002854452|+.003744359|−.004021564|−.005337262|
|1|+.004466493|+.002889122|−.006012170|−.007533965|
|2|+.004168796|+.002341562|−.008817385|−.010706974|
|3|+.003203522|+.003711700|−.003994969|−.005288578|

Changing only to REF row norms makes empirical $F_8$ minima range from $−.01958$ to $−.01101$, and population minima from $−.02006$ to $−.01332$. Changing only biases does not change endpoint types in this panel. Interpolation gives numerical sign brackets $(0,.25)$ or $(.25,.5)$, without uniqueness or monotonicity. These anchors have nonzero bias/radial gradients; angle stability alone is not a full stationary-point claim.

To check that distinction, four empirical $F_8$ problems E0–E3 and one population problem P optimize all $(b,a)$, with $b\in[-2,2]^8$, $a\in[.05,2]^8$. They start from the corresponding K8 canonical anchor; P uses stream 0. P is an oracle diagnostic, not a fifth independent dataset.

|Endpoint|$\|g_{ba}\|_\infty$|$\lambda_{\min}(H_{ba})$|$\lambda_{\min}(V^\top H_\perp V)$|KL|
|---|---:|---:|---:|---:|
|E0|$9.93946\,10^{-11}$|.05029121|.01062942|.01110942|
|E1|$4.53156\,10^{-11}$|.05191026|.01891921|.01373345|
|E2|$5.02400\,10^{-14}$|.05465116|.01576894|.00908874|
|E3|$2.96443\,10^{-12}$|.05410205|.01389994|.01229271|
|P|$2.93578\,10^{-12}$|.05396628|.01485717|.01043357|

All pass the numerical full-local-minimum signature. The displayed values remain readouts at the original approximate endpoints. The population P objective additionally has the nearby exact local minimum certified in I.8; this certification does not extend to E0–E3. At P, the integrated-risk gradient has norm $.02456458$. Along its fixed negative unit $(b,a)$ direction, all five prespecified finite steps pass their qualification:

|Step|$\Delta F_8$|$\Delta R$|
|---:|---:|---:|
|−.02|$+1.49974827\,10^{-5}$|$+5.09524051\,10^{-4}$|
|−.01|$+3.74871128\,10^{-6}$|$+2.50202674\,10^{-4}$|
|0|0|0|
|+.01|$+3.74715392\,10^{-6}$|$-2.41091767\,10^{-4}$|
|+.02|$+1.49850334\,10^{-5}$|$-4.73081829\,10^{-4}$|

The positive $.02$ step reduces KL from $.01043357299$ to $.00996049116$ while increasing $F_8$. At the same five endpoints, population $F_{64}$ and risk angle blocks have negative directions; they were not optimized to stationarity there.

## I.6. Refitting each K and the common-direction comparison

Each population problem is then refit, starting at the preceding feasible LAST, for the complete coarse sequence $9,10,12,16,24,32,64,128,\infty$. Only afterwards is the first adjacent qualified positive-to-negative finite pair refined, with at most six integer midpoints and interpolated starts. One refinement at 11 yields the retained pair $[11,12]$. The table uses Cartesian transverse eigenvalues, not angle-direction curvatures.

|K|$\|g_{ba}\|_\infty$|$\lambda_{\min}(H_{ba})$|Transverse minimum|Negative modes|KL|
|---:|---:|---:|---:|---:|---:|
|8, reused|$2.936\,10^{-12}$|.0539663|+.01485717|0|.010433573|
|9|$4.910\,10^{-14}$|.0530561|+.00959400|0|.009461482|
|10|$5.902\,10^{-11}$|.0524036|+.00530421|0|.008842072|
|11|$2.513\,10^{-12}$|.0519187|+.00174653|0|.008426627|
|12|$4.369\,10^{-14}$|.0515476|−.00124838|1|.008136147|
|16|$3.982\,10^{-14}$|.0506760|−.00960756|1|.007557032|
|24|$3.617\,10^{-14}$|.0500081|−.01810993|1|.007228793|
|32|$3.951\,10^{-11}$|.0497429|−.02239912|1|.007134125|
|64|$3.200\,10^{-14}$|.0494177|−.02886195|1|.007055936|
|128|$9.837\,10^{-12}$|.0492813|−.03210040|1|.007039262|
|$\infty$|$7.083\,10^{-12}$|.0491572|−.03533998|2|.007034194|

All new points pass radial stationarity qualification. The integrated limit remains constrained to the collinear submodel; its positive risk is not a limit on the full two-factor model. The second negative mode at infinity was not pursued. Integer type separation is not a theorem about a unique branch, monotone transition, or universal critical K.

The direction $d_{\mathrm{ang}}^{(12)}\in\mathbb R^8$ is the K12 endpoint's minimum angle eigenvector, mapped from a fixed Helmert basis, with the largest-magnitude component positive. Its rounded entries are $(-.5316182,-.3864170,-.2460486,-.0867412,.0867412,.2460486,.3864170,.5316182)$; the full-precision unit vector, orthogonal to $\mathbf1$, is used throughout. The angle eigengap is $.0122054$. This angle direction differs from the physical risk direction $d_{\mathrm{risk}}$ in I.8.5. Using the same $(d_{\mathrm{ang}}^{(12)})^\top H_\phi d_{\mathrm{ang}}^{(12)}$ in all four cells gives:

|Parameters|$F_{11}$|$F_{12}$|
|---|---:|---:|
|$\theta_{11}$|+.000542300179|−.000056581709|
|$\theta_{12}$|+.000226560907|−.000394546595|

Changing K first gives contributions $−.000598881888$ and $−.000337964886$; changing parameters first gives $−.000315739271$ and $−.000621107502$. Both total $−.000936846774$, with interaction $−.000022225614$. This is order-dependent four-cell algebra, not a unique causal decomposition. Cross cells retain the radial compensation because they are not stationary for their cross objective.

## I.7. Direct noncollinear behaviour, numerical qualification, and limits

The two locked endpoints use (I.18) with the same $d_{\mathrm{ang}}^{(12)}$. Primary amplitude is $|t|=.02$; $.05,.10$ are fixed sensitivities. Every negative and positive side is evaluated independently, with a shared zero baseline. Main-rule values are:

|K|t|$\Delta F_K$|$\Delta R$|$2\Delta F_K/t^2$|$2\Delta R/t^2$|
|---:|-----:|-------------:|-------------:|-------------:|-------------:|
|11|−.10|$+2.90960527\,10^{-6}$|$-3.62277988\,10^{-5}$|+.000581921053|−.00724555975|
|11|−.05|$+6.90272432\,10^{-7}$|$-9.10994254\,10^{-6}$|+.000552217946|−.00728795403|
|11|−.02|$+1.08777517\,10^{-7}$|$-1.45996822\,10^{-6}$|+.000543887585|−.00729984112|
|11|+.02|$+1.08777517\,10^{-7}$|$-1.45996822\,10^{-6}$|+.000543887585|−.00729984112|
|11|+.05|$+6.90272432\,10^{-7}$|$-9.10994254\,10^{-6}$|+.000552217946|−.00728795403|
|11|+.10|$+2.90960527\,10^{-6}$|$-3.62277988\,10^{-5}$|+.000581921053|−.00724555975|
|12|−.10|$-1.76055405\,10^{-6}$|$-3.91948957\,10^{-5}$|−.000352110811|−.00783897915|
|12|−.05|$-4.79905287\,10^{-7}$|$-9.85357108\,10^{-6}$|−.000383924230|−.00788285687|
|12|−.02|$-7.85692828\,10^{-8}$|$-1.57903202\,10^{-6}$|−.000392846414|−.00789516009|
|12|+.02|$-7.85692828\,10^{-8}$|$-1.57903202\,10^{-6}$|−.000392846414|−.00789516009|
|12|+.05|$-4.79905287\,10^{-7}$|$-9.85357108\,10^{-6}$|−.000383924230|−.00788285687|
|12|+.10|$-1.76055405\,10^{-6}$|$-3.91948957\,10^{-5}$|−.000352110811|−.00783897915|

Both zero rows have differences zero by definition. At $.02$, $2\Delta R/t^2$ is $−.00729984112$ and $−.00789516009$. The finite-objective quadratic readouts differ from the saved curvatures by .293%/.431%; at $.10$ the deviations become 7.31%/10.76%. Taylor agreement is not a qualification gate. The primary risk improvements are only about .017%/.019% of baseline KL: this is a mechanism check, not substantial recovery or a newly optimized two-factor solution.

**Numerical protocols.** PMFs and learned endpoints use GH40/60/80 and a GH72 tensor rule rotated by $\pi/7$: mass tolerance $2\,10^{-12}$, PMF difference $2\,10^{-7}$, risk difference $2\,10^{-6}$, and NLL/gradient tolerances $10^{-5}/10^{-4}$. Fixed-margin curvature controls compare normal-weighted Legendre384 on $[-10,10]$ with Legendre512 on $[-11,11]$, using log-u grids $[-24,56]/.125$ and $[-28,64]/.125$. Additional Hermite96 and Legendre256/$[-10,10]$ checks both use $[-24,48]/.25$. The matrix, spectrum, Ward and mass gates are the corresponding ones specified below. All fixed-anchor curvature qualifications passed.

Stationary solves use two fixed phases: Legendre128/$[-10,10]$, log-u $[-24,48]/.25$, at most 12 accepted steps/80 calls; then Legendre256/$[-10,10]$, $[-28,56]/.25$, at most 4/40. Gradient stops are $10^{-9}/10^{-10}$. Newton directions use $H_s=(H+H^\top)/2$ and shift $\max(0,10^{-4}-\lambda_{\min}(H_s))I$, with infinity norm capped at $.25$. The initial multiplier is $\min(1,.99\alpha_{\rm box})$, where $\alpha_{\rm box}$ is the largest feasible multiplier, followed by at most 12 halvings. Armijo uses $10^{-4}$ and fixed loss slack $5\,10^{-13}$. Qualification uses the **unshifted** Hessian; symmetrization removes arithmetic skew without clipping eigenvalues.

Final stationary/continuation comparisons use Legendre256/$[-10,10]$, log-u $[-28,56]/.25$, against Legendre384/$[-11,11]$, $[-32,64]/.125$. Gates are $10^{-7}$ for loss, $2\,10^{-7}$ for gradients/matrices/Ward and raw coefficient-mass residuals, $2\,10^{-6}$ for spectra, and $2\,10^{-12}$ for normal mass. Own-objective signatures additionally require gradient infinity norm at most $10^{-8}$, box margin at least $10^{-3}$, positive radial minimum above $10^{-5}$, and Cartesian Ward/rotation residuals at most $2\,10^{-7}$. Transverse signs use thresholds $\pm10^{-5}$. All reported stationary, continuation and four-cell qualifications passed.

For direct paths the three independently specified rules are:

|Rule|First Gaussian axis|Second axis|log-u interval / step|
|---|---|---|---|
|A|GH96|GH24|$[-28,56]/.25$|
|B, main|GH144|GH32|$[-32,64]/.125$|
|C, check|Normal-weighted Legendre384, $[-11,11]$|GH40|$[-36,72]/.125$|

Paired $\Delta F$ requires B/C absolute difference $\le10^{-10}$ and relative difference $\le2\%$, and A/C $\le10^{-9}$ and $\le10\%$; corresponding risk limits are $10^{-11}/2\%$ and $10^{-10}/10\%$. Relative comparisons use the maximum of both magnitudes and $10^{-12}$. Signs additionally require same B/C sign and minimum magnitude exceeding $\max(10^{-9},20|\Delta_B-\Delta_C|)$. Every rule checks norm error $\le5\,10^{-14}$, marginal change $\le2\,10^{-10}$, both effect-evenness residuals $\le10^{-11}$, and PMF evenness $\le2\,10^{-12}$.

Direct paths additionally require normal/model mass error $\le2\,10^{-12}$, unchanged biases, and nonnegative PMFs. A/B-to-C absolute-loss, raw-KL and PMF limits are respectively $10^{-7},10^{-10},2\,10^{-10}$. Two-dimensional baselines are compared with the saved one-dimensional reference at loss/PMF tolerances $10^{-7}/2\,10^{-10}$. Direct-versus-subtracted interfaces have tolerance $5\,10^{-12}$. These prespecified screens are neither statistical confidence bounds nor rigorous integration bounds; the earlier $10^{-7}$ absolute-loss gate alone cannot resolve the primary effects.

All six baseline rules and twelve nonzero parameter points passed. Maximum B/C paired discrepancies were $5.620\,10^{-17}$ for $\Delta F$ and $1.551\,10^{-16}$ for $\Delta R$; maximum marginal change was $3.886\,10^{-16}$. Independent execution checks reconstruct saved aggregations, solver decisions and qualification, with separately implemented selected-label integrals. For the direct paths, all fourteen C-rule PMFs were independently reconstructed; adaptive auxiliary integration checked labels 0,85,255 at positive $.02,.10$ for both anchors, with maximum label-difference discrepancy $2.776\,10^{-16}$. Negative-side integrals were not independently repeated, although every saved negative-side result entered aggregation and evenness checks. This is not full independent reoptimization or a rigorous Gaussian/auxiliary remainder certificate.

A retained training-record erratum concerns sixteen intermediate parameter fields aliased to GH-LAST. Contemporaneous path files and saved trajectories supplied the correct parameters; losses, gradients, qualification flags, updates, locked outputs and profile construction were unaffected. Original records and the failed audit attempt were preserved, with additive corrected metadata rather than rewritten science.

The four training streams remain the only independent learning replicates. Deterministic follow-ups do not create new statistical confidence families or inherit the older appendix coverage guarantees. The fixed finite-path panel closes this local mechanism test without additional two-dimensional optimization, second-mode searches or extra K/teacher/seed scans. Strict stationary existence remains unestablished for the four empirical fits and the K11/K12 continuation. The population K8 problem is certified below; global recovery, Adam attraction and independent-condition prediction remain unestablished.

![Learning and local geometry. (a) Observation KL for four independent streams on a logarithmic scale; black bars are means. (b) Minimum Cartesian transverse curvature, excluding common rotation, after refitting at each particle count; these are not angular eigenvalues or a continuous bifurcation. (c,d) K11/K12 objective and risk changes in nats per observation along $d_{\mathrm{ang}}^{(12)}$, with biases and row lengths fixed. These small numerical effects differ from the same-root K8 certificate (14)--(15).](figures/joint_mechanism.png){width=100%}

## I.8. Certified population local minimum and risk conflict {#proof-theorem-3}

This section proves Theorem 3 for the exact population objective (I.1), at K8. The witness is a genuine stationary point of the expectation objective, located by inclusion rather than by treating the printed numerical endpoint as an exact root. The teacher, the centre and every norm in the argument refer to the same model.

::: {#full-local-statement}

**Full quantitative statement of Theorem 3.**

**(i) Full-space local minimum and positive risk.**

For the exact teacher in (13), let $q_*$ be its observation law and $\mathcal R(\theta)=D(q_*\Vert Q_\theta)$. There exist a collinear interior parameter $\theta_*=(0,a_*,0)$ with $a_{*,i}>0$ and a fixed physical unit direction $d_{\mathrm{risk}}\in\mathbb R^{24}$ such that $\theta_*$ is a strict local minimum of $F_8^{q_*}$ in the full 24-dimensional parameter space modulo common latent rotations. For the exact dyadic center $\theta_0$ specified in [Appendix I.8](#proof-theorem-3),

$$
\begin{gathered}
\|\theta_*-\theta_0\|_2<1.805\times10^{-8},\\
\nabla F_8^{q_*}(\theta_*)=0,\qquad
D\mathcal R(\theta_*)[d_{\mathrm{risk}}]<-.0201723.
\end{gathered}
$$

Every observation marginal is correct, $Q_{\theta_*}(Y_j=1)=q_*(Y_j=1)=1/2$, while $0.0104335716\le\mathcal R(\theta_*)\le0.0104335744$. The fixed $10^{-7}$ step along $d_{\mathrm{risk}}$ raises $F_8$ and strictly lowers risk while preserving these observation marginals; row lengths and Gaussian logit variances change.

```{=latex}
\Needspace{5\baselineskip}
```

**(ii) Same-root angular conflict.**

There is also a fixed $d=d_{\mathrm{ang}}^{(8)}\in\mathbf1^\perp\subset\mathbb R^8$, $\|d\|_2=1$, such that the path $\theta_{\mathrm{ang}}(t)=(0,W(t))$ satisfies

$$
\begin{aligned}
W_i(t)&=a_{*,i}\bigl(\cos(td_i),\sin(td_i)\bigr),\\
r(t)&=\mathcal R(\theta_{\mathrm{ang}}(t)),\\
r'(0)&=0,\qquad r''(0)<-.00450377.
\end{aligned}
$$

There exists $\varepsilon>0$ such that, for every $0<|t|<\varepsilon$,

$$
\begin{gathered}
F_8^{q_*}(\theta_{\mathrm{ang}}(t))>F_8^{q_*}(\theta_*),\\
\mathcal R(\theta_{\mathrm{ang}}(t))<\mathcal R(\theta_*),\\
\operatorname{rank}W(t)=2.
\end{gathered}
$$

The path preserves all row lengths, logit marginals $N(0,a_{*,i}^2)$ and fair observation marginals. Its unit normalization is in angle space, not in Cartesian path velocity. The preserved candidate logit variances differ from the teacher's $.8^2$.

:::

### I.8.1. Exact inputs, symmetry and the full parameter space

The teacher length and angles are exact rationals. The candidate centre is an exact dyadic interpretation of the paired symmetrization of the old population endpoint. Its positive first four row lengths are specified below, and the remaining four are their reversal. The centre is not reoptimized. Write the loading columns as u and v; vectors use physical Euclidean norms and matrices use operator norms.

$$
\begin{aligned}
W_{*,j}&=\tfrac45(\cos\phi_{*,j},\sin\phi_{*,j}),\quad b_*=0,\quad
\phi_*=(-\tfrac7{10},-\tfrac12,-\tfrac3{10},-\tfrac1{10},\tfrac1{10},\tfrac3{10},\tfrac12,\tfrac7{10}),\\
\theta_0&=(0,a_0,0),\qquad a_{0,j}=a_{0,9-j},\\
(a_{0,1},\ldots,a_{0,4})&=
\left(\frac{4236614515730759}{9007199254740992},
\frac{1274451243427741}{2251799813685248},
\frac{181129089615167}{281474976710656},
\frac{775431105319425}{1125899906842624}\right).
\end{aligned}
\tag{I.21}
$$

Let P reverse the eight observations. The teacher satisfies label-complement symmetry and reversal symmetry: the latter reverses the sign of its second latent coordinate. Simultaneously complementing labels and all latent particles proves whole-vector bias evenness, not separate evenness of each bias. Reflection of the second latent coordinate is valid for every target. Consequently, with four isometric coordinates,

$$
\begin{aligned}
q_*(\mathbf1-y)&=q_*(y)=q_*(Py),\qquad
F_8^{q_*}(-b,W)=F_8^{q_*}(b,W),\\
F_8^q(b,u,-v)&=F_8^q(b,u,v),\qquad
U_+=\frac1{\sqrt2}(e_1+e_8,e_2+e_7,e_3+e_6,e_4+e_5),\\
a&=U_+\xi,\qquad f(\xi)=F_8^{q_*}(0,U_+\xi,0),\qquad
\nabla_bF_8^{q_*}=\nabla_vF_8^{q_*}=0,\quad \nabla_aF_8^{q_*}=U_+\nabla f.
\end{aligned}
\tag{I.22}
$$

The last line holds on the paired zero-bias, collinear subspace. It gives equality of the four-coordinate and full 24-coordinate gradient norms. It does not reduce the local-type test to four directions: the test retains eight biases, four symmetric and four antisymmetric row-length directions, and seven nonrotation transverse directions. All Hessians below are differentiated in the unconstrained full model before evaluation at the paired centre.

For computation only, a label has four paired counts. Gaussian particle reflection identifies each count with its complement, leaving 41 lexicographically chosen representatives. Original teacher probabilities are summed into these classes. The teacher is not assumed exchangeable within individual pairs.

$$
\begin{aligned}
c_j&=y_j+y_{9-j}\in\{0,1,2\},\qquad
A_c(z)=\prod_{j=1}^4\sigma(a_{0,j}z)^{c_j}[1-\sigma(a_{0,j}z)]^{2-c_j},\\
G_{\xi,c}(z)&=\frac z{\sqrt2}\{c-2\sigma(a_{0,1:4}z)\},\qquad
A_{2-c}(-z)=A_c(z),\quad G_{\xi,2-c}(-z)=G_{\xi,c}(z),\\
\mu_c&=-E\sum_{k=1}^8\pi_kG_{\xi,c}(Z_k)=\mu_{2-c},\qquad
\nabla f=\sum_{r\in\mathcal C}w_r\mu_r,\\
\mathcal C&=\{\min(c,2-c):c\in\{0,1,2\}^4\},\quad|\mathcal C|=41,\quad
w_r=\sum_{[author-local path omitted]
\end{aligned}
\tag{I.23}
$$

The equality of class gradients uses the sign-reflected Gaussian bank, not an unchanged fixed bank. For Hessians the full label residual must be retained.

### I.8.2. The Hessian density that is actually integrated

Set the likelihood scale to one and use the physical Laplace variable. With the full 16-coordinate score and conditional negative Hessian from (I.6), define the following inner moments; all expectations are one-dimensional standard Gaussian expectations at the collinear point. Here $B=B_y(z)$ is the single-particle conditional negative Hessian, whereas $\mathcal B=\mathcal B_y(v)$ is its $t$-weighted Gaussian moment; the label and auxiliary-coordinate dependence is suppressed in (I.24).

$$
\begin{aligned}
u&=e^v,\quad t=uA_y(z)e^{-uA_y(z)},\quad\Psi=E e^{-uA_y(Z)},\quad T_0=Et,\\
\mathcal R&=E[tG],\quad\mathcal Q=E[tGG^\top],\quad\mathcal B=E[tB],\quad
\mathcal Q_p=E[trr^\top],\quad\mathcal B_p=E[t\operatorname{diag}\sigma'(a_0Z)],\\
I_{ba,y}(v)&=K\Psi^{K-1}\mathcal B-K(K-1)\Psi^{K-2}(T_0\mathcal Q-\mathcal R\mathcal R^\top),\\
I_{p,y}(v)&=K\Psi^{K-1}\mathcal B_p-K(K-1)\Psi^{K-2}T_0\mathcal Q_p,\qquad K=8.
\end{aligned}
\tag{I.24}
$$

Integrating these densities over the real auxiliary axis and averaging over labels gives the complete radial and transverse blocks. The first term is the conditional negative Hessian. The second term contains the same-particle contribution minus the distinct-particle score outer product. In the transverse block the latter vanishes because different particles have independent, centred second coordinates. The identities for the reciprocal and squared reciprocal in I.3 supply the displayed powers and factors; there is no additional scaling by K or u.

One must distinguish this same/distinct-particle density from direct parameter differentiation of the gradient density. For a real bank put the sum and likelihood weights below; the resulting covariance is positive semidefinite.

$$
\begin{aligned}
S&=\sum_kA_k,\quad w_k=A_k/S,\quad s=e^vS,\\
\bar B&=\sum_kw_kB_k,\quad\bar G=\sum_kw_kG_k,\quad\bar Q=\sum_kw_kG_kG_k^\top,\\
I_{ba}(v)&=E[e^{-s}\{s\bar B-s^2(\bar Q-\bar G\bar G^\top)\}],\\
I_{\rm dir}(v)&=E[e^{-s}\{s(\bar B-\bar Q)+s^2\bar G\bar G^\top\}],\\
I_{ba}(v)-I_{\rm dir}(v)&=E[\partial_v(se^{-s})\bar Q],\\
\int_{\mathbb R}(I_{ba}-I_{\rm dir})\,dv&=0.
\end{aligned}
\tag{I.25}
$$

The boundary term is zero for each positive bank sum, with Gaussian score moments justifying exchange of expectation and integration. The densities have the same full real integral but different finite-grid values. The error envelopes below apply to the first density, which is the one used for the certificate.

For a complement-side label use its effective label complement, while the self-complement count is included once. This preserves the three diagonal blocks; the bias–radial mixed block changes sign and is zero only after using exact teacher symmetry. For each representative form weighted label moments and inner probability moments:

$$
\begin{aligned}
W&=\sum q_y,\quad M_i=\sum q_y\widetilde y_i,\quad S_{ij}=\sum q_y\widetilde y_i\widetilde y_j,\\
T_l&=E[tz^l],\quad P_{l,i}=E[tz^lp_i(z)],\quad PP_{l,ij}=E[tz^lp_i(z)p_j(z)],\quad p_i(z)=\sigma(a_{0,i}z),\\
D_l&=W\operatorname{diag}(P_l-\operatorname{diag}PP_l),\quad l=0,2,\\
J_{bb}&=W(T_0PP_0-P_0P_0^\top),\\
J_{aa}&=(T_0T_2-T_1^2)S-T_0(MP_2^\top+P_2M^\top)
+T_1(MP_1^\top+P_1M^\top)+W(T_0PP_2-P_1P_1^\top),\\
J_p&=T_0(T_0S-MP_0^\top-P_0M^\top+WPP_0),\\
I_{bb}&=K\Psi^{K-1}D_0-K(K-1)\Psi^{K-2}J_{bb},\quad
I_{aa}=K\Psi^{K-1}D_2-K(K-1)\Psi^{K-2}J_{aa},\\
I_p&=K\Psi^{K-1}D_0-K(K-1)\Psi^{K-2}J_p.
\end{aligned}
\tag{I.26}
$$

The label moments cancel algebraically in the first covariance difference, but remain in the other two. They are eight-coordinate vectors and matrices; no division by class mass is used. Four distinct probabilities reduce the inner scalar moment count to 36 including the Laplace transform, without reducing any Hessian dimension. This retains the information needed for broken-pair and antisymmetric row-length directions.

### I.8.3. Interval quadrature, discrete tails and composition

The computation uses 192-bit midpoint-radius arithmetic [@johansson2017]. Finite sums are enclosed by their exact dyadic endpoints. They are then enlarged by analytic infinite-grid and omitted-grid errors. Agreement between floating-point rules is not used as an error bound. For an analytic strip with vanishing contour edges and integrable boundary envelopes, Fourier contour shift followed by Poisson periodization gives [@trefethen2014]

$$
\left\|\int_{\mathbb R}f(x)\,dx-h\sum_{k\in\mathbb Z}f(kh)\right\|
\le\frac{M_++M_-}{e^{2\pi d/h}-1},\qquad
M_\pm\ge\int_{\mathbb R}\|f(x\pm id)\|\,dx.
\tag{I.27}
$$

The same argument holds for vector Euclidean and matrix operator norms. In this model all real-axis derivatives have Gaussian-polynomial or auxiliary rapid-decay envelopes, so the real functions are Schwartz; absolute convergence of the Fourier series also verifies the periodization identity. Analyticity alone would not suffice.

For the teacher the grid is the full-weight two-dimensional lattice on the stated rectangle. In its polystrip each logit avoids sigmoid poles, and the sum of absolute likelihoods over all labels factorizes. This bounds the complete probability vector without multiplying a per-label bound by 256.

Specifically, $\sum_y|A_y(x+it)|\le\prod_j\sec(B_j/2)$, and $|\varphi(x_i+it_i)|=e^{t_i^2/2}\varphi(x_i)$. Fourier shifts in the two coordinates give the product of their geometric alias sums; deleting the zero frequency gives the factor in the next formula.

$$
\begin{aligned}
\theta_{\rm T}&=(b_*,W_*),\quad h_t=\tfrac18,\quad L_t=10,\quad d_t=1,\\
q_y^f&=h_t^2\sum_{|k_1|,|k_2|\le80}\varphi(h_tk_1)\varphi(h_tk_2)A_y(h_tk;\theta_{\rm T}),\\
B_j&=d_t(|W_{*,j1}|+|W_{*,j2}|)<\pi,\quad
M_t=e^{d_t^2}\prod_j\sec(B_j/2),\quad \rho_t=e^{-2\pi d_t/h_t},\\
\|q_*-q^f\|_1&\le\delta_t:=M_t\frac{4\rho_t}{(1-\rho_t)^2}
+2(1+h_t\varphi(0))\frac{2\varphi(L_t)}{L_t}<4.894\,10^{-21}.
\end{aligned}
\tag{I.28}
$$

The second term is an omitted-lattice Gaussian mass bound: the infinite one-axis mass is at most one plus the central grid weight, and each omitted right tail is bounded by its adjacent continuous tail. All retained edges and corners have full weights. Finite-sum arithmetic radii are retained separately in the probability balls.

The candidate inner and outer grids, including full endpoint weights, are fixed as follows. Set the total loading sum, sector angle and secant constants below. Each sigmoid factor has argument between zero and its logit imaginary part, so the likelihood product has a continuous argument bounded by the sector angle. Hence its real part is positive in the inner strip, uniformly for every real auxiliary coordinate.

$$
\begin{aligned}
h_z&=\tfrac1{40},\ L_z=11,\ d_z=\tfrac18,\qquad
h_v=\tfrac18,\ v_-=-32,\ v_+=64,\ d_v=\tfrac12,\ W_v=96.125,\\
C_a&=\sum_i|a_{0,i}|,\quad\beta=C_ad_z<\pi/2,\quad
c_i=\sec(|a_{0,i}|d_z/2),\ C_R^2=\sum_i c_i^2,\ c_*^2=\max_i c_i^2,\\
\operatorname{Re}A_y(x+it)&\ge|A_y(x+it)|\cos\beta>0,\quad
|e^{-uA_y}|\le1,\quad |uA_ye^{-uA_y}|\le(e\cos\beta)^{-1},\\
F_z&=\frac{e^{d_z^2/2}}{e\cos\beta},\quad J_z=2+d_z^2,\quad
\rho_z=e^{-2\pi d_z/h_z},\qquad
\delta_X=\frac{2M_X\rho_z}{1-\rho_z}+T_X.
\end{aligned}
\tag{I.29}
$$

There are 881 inner nodes and 769 outer nodes. On the strip, residual norms and conditional-Hessian norms are bounded by the secants; Gaussian boundary integration gives the following envelopes. The middle column is a scalar, vector or operator boundary-integral bound as appropriate; the last column is an omitted-inner-grid bound.

The finite-sum implementation can enclose $e^{-x}$ and $xe^{-x}$ by $[0,e^{-128}]$ and $[0,128e^{-128}]$ when the interval input proves $x=uA\ge128$. These are conservative saturation intervals, not substituted zero values; their radii remain in the final finite-sum boxes.

|Moment|$M_X$|$T_X$|
|---|---|---|
|$\Psi$|$e^{d_z^2/2}$|$p_L$|
|$T_0$|$F_z$|$p_L/e$|
|$\mathcal R$|$F_z\sqrt{C_R^2J_z}$|$2\sqrt{2m}\varphi(L_z)/e$|
|$\mathcal Q$|$F_z C_R^2J_z$|$m j_L/e$|
|$\mathcal B$|$F_z c_*^2J_z/4$|$j_L/(4e)$|
|$\mathcal Q_p$|$F_z C_R^2$|$m p_L/e$|
|$\mathcal B_p$|$F_z c_*^2/4$|$p_L/(4e)$|

The tail symbols are $p_L=2\varphi(L_z)/L_z$ and $j_L=2\varphi(L_z)(L_z+2/L_z)$, with $m=8$. They follow from monotone right-rectangle comparison beyond the retained endpoint, including the first and second Gaussian moments. Ordinary transposes, not conjugate transposes, occur in the analytic density; complex rank-one matrices still satisfy the operator bound by the squared vector norm.

Uniform inner errors propagate through the nested powers and matrix products. For this estimate only, let lower-case letters denote upper bounds on the true moment norms, and use the following product-error operator.

$$
\begin{aligned}
P&=1+\delta_\Psi,\quad
\mathcal E_n(b,\delta b)=P^n\delta b+nP^{n-1}\delta_\Psi b,\\
C&=T_0\mathcal Q-\mathcal R\mathcal R^\top,\quad
\|C\|\le tq+r^2,\quad
\delta_C\le\delta_{T_0}q+(t+\delta_{T_0})\delta_{\mathcal Q}+2r\delta_{\mathcal R}+\delta_{\mathcal R}^2,\\
C_p&=T_0\mathcal Q_p,\quad\|C_p\|\le tq_p,\quad
\delta_{C_p}\le\delta_{T_0}q_p+(t+\delta_{T_0})\delta_{\mathcal Q_p},\\
E_{{\rm in},ba}&=W_v\{K\mathcal E_{K-1}(b,\delta_{\mathcal B})
+K(K-1)\mathcal E_{K-2}(tq+r^2,\delta_C)\},\\
E_{{\rm in},p}&=W_v\{K\mathcal E_{K-1}(b_p,\delta_{\mathcal B_p})
+K(K-1)\mathcal E_{K-2}(tq_p,\delta_{C_p})\}.
\end{aligned}
\tag{I.30}
$$

The factor $W_v=769h_v=96.125$ is the total weight of the finite outer sum, including full endpoint weights. The terms $E_{{\rm in},ba}$ and $E_{{\rm in},p}$ compare exact and approximated inner moments within this finite sum; outer alias and omitted-grid tails are bounded separately.

The valid real-axis bounds are $t=1/e$, $r=\sqrt{2m}/e$, $q=2m/e$, $b=1/(2e)$, $q_p=m/e$, and $b_p=1/(4e)$. Telescoping powers and the matrix triangle inequality prove (I.30); no factor close to one is silently removed. The four-gradient computation uses the same argument for $D=E[tG_\xi]$, with $\|D\|\le E_G/e$ and the following vector envelope.

$$
\begin{aligned}
E_G&=\sqrt m\sqrt{2/\pi},\quad
M_D=F_z\sqrt{C_R^2(1+d_z^2)},\quad
\delta_D=\frac{2M_D\rho_z}{1-\rho_z}+2\sqrt m\varphi(L_z)/e,\\
E_{{\rm in},g}&=KW_v\{P^{K-1}\delta_D+(K-1)P^{K-2}\delta_\Psi E_G/e\}.
\end{aligned}
\tag{I.31}
$$

For the outer strip, real-bank likelihood weights remain real; only the auxiliary scalar becomes complex. Integrating the absolute bank densities after the change of variable to the bank-scaled auxiliary radius gives

$$
\begin{aligned}
M_{ba}&=(1+K)\{(4\cos d_v)^{-1}+m(\cos d_v)^{-2}\},\\
M_p&=(4\cos d_v)^{-1}+m(\cos d_v)^{-2},\quad M_g=KE_G/\cos d_v,\\
E_{{\rm alias},X}&=\frac{2M_X\rho_v}{1-\rho_v},\qquad
\rho_v=e^{-2\pi d_v/h_v},\quad X\in\{ba,p,g\}.
\end{aligned}
\tag{I.32}
$$

At the lower end, the densities and their derivatives decay exponentially. At the upper end split Gaussian banks at a latent threshold proportional to the auxiliary coordinate: on the bounded-latent part the likelihood sum gives double-exponential decay; on its complement Gaussian tails give rapid decay. Differentiating produces polynomials in the bank-scaled auxiliary radius times its damped exponential, so the same split applies at every fixed derivative order. This establishes the contour-edge and Schwartz conditions for the outer use of (I.27).

Omitted outer points require discrete-tail bounds. For the two bank kernels, the total variations below control a right-grid tail by its continuous integral plus a mesh-times-variation term. If the boundary is already past the peak, monotone right rectangles give the tighter continuous bound.

$$
\begin{aligned}
f_1(v)&=se^{-s},\quad f_2(v)=s^2e^{-s},\quad
\int_{\mathbb R}|f_1'(v)|\,dv=2/e,\quad\int_{\mathbb R}|f_2'(v)|\,dv=8/e^2,\\
l_1&=\frac{h_ve^{v_-}}{e^{h_v}-1},\quad l_2=\frac{h_ve^{2v_-}}{e^{2h_v}-1},\quad
E_{{\rm low},p}\le Kl_1/4+mK^2l_2,\quad E_{{\rm low},ba}\le(1+K)E_{{\rm low},p},\\
T&=4,\quad U=e^{v_+},\quad\eta=2^{-m}e^{-C_aT},\quad U\eta>2,\\
p_T&=2\varphi(T)/T,\quad t_{2T}=2\varphi(T)(T+1/T),\quad
C_{\rm bad}=\tfrac14(1+2h_v/e)+m(1+8h_v/e^2),\\
G_p&=\{\tfrac14+m(1+U\eta)\}e^{-U\eta},\quad
E_{{\rm up},p}\le G_p+C_{\rm bad}p_T^K,\\
E_{{\rm up},ba}&\le(1+K)G_p+C_{\rm bad}(p_T^K+Kt_{2T}p_T^{K-1}).
\end{aligned}
\tag{I.33}
$$

Here $Z_k$ is the scalar first latent coordinate used by the collinear likelihood and its one-dimensional inner integration. Define $G=\{\min_{1\le k\le8}|Z_k|\le T\}$ and $G^c=\bigcap_{k=1}^8\{|Z_k|>T\}$. On $G$, the likelihood sum satisfies $S=\sum_k A_y(Z_k)\ge\eta$. On $G^c$, independence and double-sided Gaussian tails give $\Pr(G^c)\le p_T^K$ and $E[(1+M^2)\mathbf1_{G^c}]\le p_T^K+Kt_{2T}p_T^{K-1}$, where $M=\max_k|Z_k|$. Thus the mesh correction is retained on the bad set rather than applying a continuous Laplace tail to every omitted lattice point. For the gradient the corresponding bounds are

$$
\begin{aligned}
E_{{\rm low},g}&\le KE_Gl_1,\\
E_{{\rm up},g}&\le\sqrt m\{\sqrt K e^{-U\eta}+(1+2h_v/e)2K\varphi(T)p_T^{K-1}\}.
\end{aligned}
\tag{I.34}
$$

Sum the inner replacement, outer alias and the two discrete tails to obtain a uniform error for each label. The finite teacher mass is at most one plus its L1 error. The labelwise true gradient bound and conservative Hessian bounds below transfer teacher error only once per vector or matrix norm.

$$
\begin{aligned}
E_X^{\rm unif}&=E_{{\rm in},X}+E_{{\rm alias},X}+E_{{\rm low},X}+E_{{\rm up},X},\\
\|g_{\xi,y}\|&\le\sqrt m E\max_k|Z_k|\le\sqrt{mK}=8,\\
\|H_{ba,y}\|&\le(2m+\tfrac14)(1+K)=146.25,\quad\|H_{p,y}\|\le m+\tfrac14=8.25,\\
E_g&\le(1+\delta_t)E_g^{\rm unif}+8\delta_t,\quad
E_{ba}\le(1+\delta_t)E_{ba}^{\rm unif}+146.25\delta_t,\\
E_p&\le(1+\delta_t)E_p^{\rm unif}+8.25\delta_t.
\end{aligned}
\tag{I.35}
$$

For the Hessian, use each label’s block-diagonal bias/radial principal blocks and its transverse block; folding preserves exactly these objects. The exact teacher mixed block is zero. Finite teacher probabilities need not satisfy a bitwise symmetry test. The interval finite matrices retain their weight and arithmetic uncertainty; Frobenius upper bounds dominate operator norms when comparing them with saved dyadic matrices.

The final outward-rounded bounds are as follows. The saved matrices are exact dyadic interpretations of the previous symmetric binary64 matrices; their positivity is proved by interval LDL. A rational basis of the fixed transverse slice is tested against its Gram matrix, rather than assumed orthonormal.

The rational basis has columns $V_j=e_j-(a_{0,j}/a_{0,8})e_8$, $1\le j\le7$. The test is $V^\top(S_p-.014I)V\succ0$, which certifies the physical curvature bound relative to $V^\top V$.

$$
\begin{aligned}
\|\nabla F_8^{q_*}(\theta_0)\|_2&<7.572\,10^{-10},\\
\|H_{ba}(\theta_0)-S_{ba}\|_{\rm op}&<1.138\,10^{-8}<.001,\\
\|H_p(\theta_0)-S_p\|_{\rm op}&<2.531\,10^{-9}<.001,\\
S_{ba}&\succ.050I_{16},\qquad
w^\top S_pw>.014\|w\|_2^2\quad(0\ne w\perp a_0).
\end{aligned}
\tag{I.36}
$$

|Contribution|Gradient L2|Radial Hessian operator|Transverse Hessian operator|
|---|---:|---:|---:|
|Inner replacement|2.49106e−10|9.04057e−9|2.27090e−9|
|Outer alias|5.00387e−10|2.33628e−9|2.59587e−10|
|Outer lower tail|2.14647e−13|2.14005e−13|2.37783e−14|
|Total mathematical error, including teacher|<7.497076e−10|<1.137708e−8|<2.530514e−9|
|Finite-box norm or distance to saved matrix|<7.490176e−12|<1.456e−13|<2.663e−14|

Table decimals summarize bounds; exact dyadic endpoints and rational comparisons determine (I.36). The omitted upper-tail contribution is included in the total. No 41-class, 256-label or matrix-entry multiplier is inserted into a bound already expressed in the corresponding vector or operator norm.

### I.8.4. Root inclusion and 23-direction local minimality

A uniform full-parameter Hessian variation bound turns the centre certificate into a neighbourhood certificate. Let the bank envelope below include both latent coordinates. Conditional log-kernel derivatives of orders one, two and three are bounded by the stated powers. Differentiating log-sum-exp gives a mean third derivative, three covariance terms and a centred third moment; their bounds give the displayed constant. Gaussian moments justify differentiation under expectation.

$$
\begin{gathered}
q_Z=\sqrt{1+\max_{k\le K}\|Z_k\|^2},\\
\|D\ell_k\|\le\sqrt m q_Z,\quad\|D^2\ell_k\|\le q_Z^2/4,\quad\|D^3\ell_k\|\le q_Z^3/4,\\
L_F\le(\tfrac14+\tfrac32\sqrt m+8m^{3/2})
[1+2dK+dK(dK+2)]^{3/4}<14069,\quad m=K=8,\ d=2,\\
\|H_F(\theta)-H_F(\theta')\|_{\rm op}\le14069\|\theta-\theta'\|_2.
\end{gathered}
\tag{I.37}
$$

For example each covariance term is at most twice the product of the first- and second-derivative bounds, and the centred third moment is at most eight times the cube of the first bound. The bank’s squared envelope is dominated by one plus a chi-square variable with sixteen degrees of freedom. The integer cap is checked by rational upper bounds, not fitted from numerical derivatives. With the preassigned physical radius,

$$
\begin{aligned}
r&=5\,10^{-7},\qquad
m_r=m_{16}=.050-.001-14069r=.0419655,\\
m_T&=.014-.001-14069r=.0059655,\qquad
\epsilon_g=7.572\,10^{-10}<m_rr=2.098275\,10^{-8}.
\end{aligned}
\tag{I.38}
$$

The four-coordinate closed ball has a strongly convex restricted objective. On its sphere, integrating the Hessian along the centre-to-boundary line gives the outward inequality below. A closed-ball minimum cannot lie on the boundary, since an inward move would decrease the objective. It is therefore an interior restricted root, unique within this ball by strong convexity. Exact symmetry in (I.22) lifts it to a full stationary point. Strong monotonicity further gives its location.

$$
\begin{aligned}
\delta^\top\nabla f(\xi_0+\delta)&\ge-\epsilon_g r+m_rr^2>0\quad(\|\delta\|=r),\\
\nabla F_8^{q_*}(\theta_*)&=0,\qquad\theta_*=(0,a_*,0),\qquad
\|\theta_*-\theta_0\|_2<\epsilon_g/m_r<1.805\,10^{-8}.
\end{aligned}
\tag{I.39}
$$

The fixed slice is $a_0^\top v=0$. The derivative of this constraint along a common latent rotation is $a_0^\top u$, positive throughout the small neighbourhood. The implicit function theorem gives a unique small rotation taking any nearby point into the slice. At the true root, reflection makes the radial/transverse mixed blocks zero; (I.38) makes its 16 radial and seven slice-transverse directions positive. Taylor’s theorem gives a strict slice minimum, and rotation invariance lifts it to a strict local minimum modulo common rotation. The 24-dimensional Hessian is singular along the common-rotation direction, as rotation invariance requires; the strict minimum is modulo that rotation, with the 16 bias/radial and seven transverse directions certified positive.

### I.8.5. Risk direction, fixed small step and scope

The same centre and a fixed paired dyadic vector define the mathematical unit direction $d_{\mathrm{risk}}\in\mathbb R^{24}$. Candidate probabilities and their directional derivatives are enclosed by 177 one-dimensional full-weight nodes with spacing one eighth on the interval from minus eleven to eleven. Strip and Gaussian-tail errors are added; teacher probability errors transfer through the posterior ratios. Jensen’s inequality supplies a uniform positive probability bound on the physical ball, and the posterior negative-Hessian identity bounds risk curvature.

The direction-defining vector $h$ in (I.40) differs from the grid spacings $h_t,h_z,h_v$ and the generic scalar mesh in (I.27). Its entries come from pair symmetrization of a numerical radial risk gradient; certification uses the fixed exact dyadics without assuming equality to the true gradient. Its first four components are the following exact dyadics; set $h_{9-j}=h_j$ and use the mathematical norm $\|h\|_2=\sqrt{\sum_{j=1}^8h_j^2}$.

|Component|Exact dyadic|
|---|---|
|$h_1$|$-8589911184852041/1152921504606846976$|
|$h_2$|$-2443582330405755/288230376151711744$|
|$h_3$|$-5282230754430109/576460752303423488$|
|$h_4$|$-2739850785036709/288230376151711744$|

$$
\begin{aligned}
d_{\mathrm{risk}}&=(0,-h/\|h\|_2,0),\\
B_{h,y}&=E[A_y(Z)Z\sum_i h_i\{y_i-\sigma(a_{0,i}Z)\}],\\
DR(\theta_0)[d_{\mathrm{risk}}]&=\sum_yq_*(y)\frac{B_{h,y}}{Q_y(\theta_0)\|h\|_2}\\
&\in[-.024564580310120128,-.024564580310120124],\\
Q_y(\theta)&\ge q_{\min}:=2^{-8}
\exp\{-3r/2-[r^2+(\|a_0\|_2+r)^2]/8\},\\
\|H_R(\theta)\|&\le M_R:=24/q_{\min}<8784.371,\\
DR(\theta)[d_{\mathrm{risk}}]&<-\nu,\quad\nu=.0201723,\qquad\|\theta-\theta_0\|_2\le r.
\end{aligned}
\tag{I.40}
$$

The numerator $B_{h,y}$ is the derivative of $Q_y$ along $+(0,h,0)$, so $DQ_y(\theta_0)[d_{\mathrm{risk}}]=-B_{h,y}/\|h\|_2$. The negative log probability in the risk contributes a second minus sign, giving the positive sign in (I.40).

For completeness, the candidate probability and numerator boundary norms in a width-one strip are at most the Gaussian shift factor times the product of loading secants, and this quantity times the secant-weighted absolute direction sum times square root of two, respectively. Formula (I.27) bounds their lattice errors; omitted tails are at most twice the Gaussian endpoint density divided by the endpoint, and twice that density times the absolute direction sum. Positive probability lower bounds permit interval division. The teacher L1 allowance multiplies the largest absolute posterior ratio once.

To verify the curvature bound, a label’s negative-log-probability Hessian is the posterior mean conditional negative Hessian minus the posterior score covariance. Both terms are positive semidefinite, so their difference has norm at most the larger individual norm. The likelihood is at most one; dividing prior score moments by the uniform probability lower bound gives the stated bound. The direction is not selected after evaluating the root.

The preassigned step remains inside the certified ball even when started at the unknown exact root. It preserves zero biases, hence each observed Bernoulli marginal is exactly fair by Gaussian central symmetry, while row logit variances change with the row lengths. It is not the fixed-row-norm angular path of I.7. Radial curvature and the uniform risk derivative give

$$
\begin{aligned}
t&=10^{-7},\quad \|\theta_*-\theta_0\|_2+t<r,\\
F_8^{q_*}(\theta_*+td_{\mathrm{risk}})-F_8^{q_*}(\theta_*)&\ge\tfrac12m_{16}t^2=2.098275\,10^{-16}>0,\\
R(\theta_*)-R(\theta_*+td_{\mathrm{risk}})&>\nu t=2.01723\,10^{-9},\qquad
R(\theta_*)>2.01723\,10^{-9}.
\end{aligned}
\tag{I.41}
$$

This is an analytic consequence of the certified neighbourhood, not a subtraction of nearly equal float64 losses. It certifies a strict conflict at one population local minimum, not the size of a typical training improvement. The positivity of KL proves the final inequality.

**Risk magnitude from stored probability intervals.** The preceding small-step bound remains the core existence certificate. The teacher and candidate probability intervals also allow a deterministic KL enclosure at the centre and its transfer to the exact root. The observed score information is bounded by the complete-data information, so the following derivative bound uses only probability ratios.

$$
\begin{aligned}
\chi^2(q_*\Vert Q_0)&=\sum_y\frac{(q_*(y)-Q_y(\theta_0))^2}{Q_y(\theta_0)},\quad
I_{\rm obs}(\theta_0)\preceq I_{\rm complete}(\theta_0)\preceq\tfrac14I_{24},\\
\|\nabla R(\theta_0)\|_2&\le\tfrac12\sqrt{\chi^2(q_*\Vert Q_0)},\\
|R(\theta_*)-R(\theta_0)|&\le\tfrac12\sqrt{\chi^2(q_*\Vert Q_0)}\,\eta_*+\tfrac12M_R\eta_*^2,\quad
\eta_*=\epsilon_g/m_r.
\end{aligned}
\tag{I.42}
$$

Indeed the conditional complete-data score has cross-row information zero and within-row information at most one quarter times the expected outer product of the augmented Gaussian input, which is the identity. Conditional expectation gives the observed score, decreasing information by Jensen’s inequality. Its model-probability-weighted mean is zero; Cauchy–Schwarz applied to the target-minus-model weights then yields (I.42). No target probability is renormalized, and no old floating-point gradient is substituted for the full gradient norm.

Endpoint-only interval composition gives a centre risk about 0.01043357299286302, chi-square divergence about 0.021791388777069481, a full gradient bound below 0.073809534, and a root-transfer allowance below 1.334e−9. Outward rational rounding yields the rigorous enclosure

$$
R(\theta_*)\in[.0104335716,\ .0104335744].
\tag{I.43}
$$

This additional bound is a deterministic composition of saved probability intervals and the certified root distance, with no new integral, fit or random sample. It explains why the conservative positive-risk lower bound in (I.41) should not be read as the actual risk magnitude.

The evidence consists of complete interval finite sums plus the analytic errors above. Independent alternative implementations recomputed eight selected teacher labels over their full grid and two of the 41 gradient/Hessian classes over their full grids; this is not independent recomputation of every class. All endpoint uncertainties enter the final certificates. The earlier empirical fits, weak-loading K2 precision failures, K11/K12 numerical continuation and mixed profile comparisons retain their original scope. The result does not certify a globally optimal solution, an optimizer’s probability of entering the neighbourhood, a universal critical K, or numerical detectability of the tiny guaranteed finite step.

## I.9. Qualitative stability under target perturbations {#local-persistence}

**Corollary I.1 (qualitative teacher-law persistence).** There is a neighbourhood of the exact teacher probability vector in the interior of the probability simplex such that every target in that neighbourhood has a nearby rank-one strict local minimum of its K8 objective modulo common rotation, with negative integrated-risk derivative along the same fixed direction $d_{\mathrm{risk}}\in\mathbb R^{24}$. The perturbed probabilities need not preserve complement or reversal symmetry. The conclusion also holds for the distributions induced by all sufficiently nearby teacher parameters in the original two-factor model. The neighbourhood is not quantified.

**Proof.** Use all 23 coordinates on the fixed slice $\mathcal S=\{a_0^\top v=0\}$, not the four-coordinate symmetric branch. Let $E_{\mathcal S}$ have orthonormal columns spanning its tangent space and use affine coordinates for probabilities summing to one. The finite-particle objective is linear in the target and smooth in the parameters, with Gaussian polynomial domination permitting each derivative. At the certified root define

$$
\begin{aligned}
\mathcal F(s,q)&=E_{\mathcal S}^\top\nabla_\theta F_8^q(\theta_*+E_{\mathcal S}s),\\
\mathcal F(0,q_*)&=0,\qquad
D_s\mathcal F(0,q_*)=E_{\mathcal S}^\top H_F(\theta_*)E_{\mathcal S}\succ0.
\end{aligned}
\tag{I.44}
$$

The implicit function theorem gives a unique local smooth slice-critical branch $\theta(q)$. Take a sufficiently small slice neighbourhood centred at $\theta_*=(0,a_*,0)$ that is invariant under $T(b,u,v)=(b,u,-v)$ and lies inside the theorem’s uniqueness neighbourhood; shrink the target neighbourhood so that the branch remains there. For every target, latent reflection leaves the objective and the fixed slice invariant. The reflection of its slice-critical point is therefore another slice-critical point in the same neighbourhood. Uniqueness makes them equal. Continuity keeps $u(q)$ nonzero, so

$$
\begin{aligned}
F_8^q(T\theta)&=F_8^q(\theta),\qquad T\mathcal S=\mathcal S,\qquad T\theta(q)=\theta(q),\\
v(q)&=0,\qquad \operatorname{rank}W(q)=1.
\end{aligned}
$$

Slice criticality means that the ambient gradient is proportional to the slice normal. Rotation invariance makes that gradient orthogonal to the orbit tangent; the normal has nonzero product with this tangent because the original transverse condition persists. Thus the ambient gradient is zero, without imposing bias or paired-row constraints on the perturbed target. Positive slice curvature persists by continuity, giving strict local minimality modulo rotation.

$$
\begin{aligned}
\theta(q)&=\theta_*+E_{\mathcal S}s(q),\qquad
\nabla F_8^q(\theta(q))=0,\\
R_q(\theta)&=D_{\rm KL}(q\Vert p_\theta),\qquad
D R_q(\theta(q))[d_{\mathrm{risk}}]<0.
\end{aligned}
\tag{I.45}
$$

The final strict inequality follows by continuity of the directional risk derivative, since all probabilities remain positive near the interior teacher vector and the certified root. It also gives positive risk. For the correctly specified teacher subclass, each observed probability is continuous in the teacher parameters: its likelihood is continuous and bounded by one, so dominated convergence applies. The inverse image of the probability neighbourhood is thus a neighbourhood of the original teacher parameters, and the conclusion holds there. This does not assert that every nearby probability vector belongs to the original two-factor model.

This is a qualitative consequence of nondegeneracy, not a new teacher-grid experiment or a quantified applicability region. No additional K range is claimed. The nearby branch remains rank one, while its biases and paired row lengths need not retain their original symmetry. In particular, the fair observation marginals at the original root and along its certified risk step in I.8.5 are not asserted for perturbed teachers or their roots.

## I.10. A fixed-marginal angle conflict at the same exact root {#same-root-angle}

We strengthen the witness in I.8 using its same exact teacher $q_*$, centre $\theta_0=(0,a_0,0)$ and root $\theta_*=(0,a_*,0)$. No root is refitted. The new path changes relative angles while keeping every row length fixed; it diagnoses the local minimum in the full model, rather than imposing a constraint on the original learning problem.

### I.10.1. Direction and full path derivative

Define the integer vector and mathematical unit angle direction by

$$
\begin{aligned}
(u_1,u_2,u_3,u_4)&=(-8440215,-6758298,-4585241,-1673686),\\
u_{9-i}&=-u_i\quad(1\le i\le4),\qquad S=\sum_i u_i^2=281474961915412,\\
d=d_{\mathrm{ang}}^{(8)}&=u/\sqrt S,\qquad \|d\|_2=1,\qquad \mathbf1^\top d=0.
\end{aligned}
\tag{I.46}
$$

Saved numerical risk-angle information at the earlier P point motivated this direction. Its exact construction was fixed before the new integral evaluation and was not changed after evaluation; this is an exploratory existence witness, not a blindly selected direction. All quadratic coefficients use exact ratios such as $(u_i-u_j)^2/S$, without rounding $\sqrt S$. The angle normalization is distinct from unit Cartesian path speed. This direction is also distinct from both $d_{\mathrm{risk}}$ in I.8.5 and $d_{\mathrm{ang}}^{(12)}$ in I.6–I.7.

For a collinear loading $a$, let $Z,X$ be independent standard Gaussians and set $\eta_i(t)=a_i\{Z\cos(td_i)+X\sin(td_i)\}$. Write $p_i(z)=\sigma(a_i z)$, $r_i=y_i-p_i(z)$, $D_i=p_i(1-p_i)$, $A_y(z)=\prod_i p_i^{y_i}(1-p_i)^{1-y_i}$ and $Q_y=E A_y(Z)$. For this paragraph define $U=\sum_i a_i d_i r_i$, $V=\sum_i a_i d_i^2r_i$, $G=\sum_i a_i r_i$ and $T=\sum_i a_i^2d_i^2D_i$. Differentiating the likelihood and integrating out $X$ gives

$$
\begin{aligned}
W_i(t)&=a_i(\cos(td_i),\sin(td_i)),\qquad b_i(t)=0,\\
\dot\eta_i(0)&=a_i d_iX,\qquad \ddot\eta_i(0)=-a_i d_i^2Z,\\
Q_y'(0)&=0,\qquad Q_y''(0)=E B_y(Z),\\
B_y(z)&=A_y(z)\{U^2-T-zV\}.
\end{aligned}
\tag{I.47}
$$

The first derivative vanishes by reflection in $X$. In particular, this is not a negative first-order risk direction. The acceleration term $-zV$ cannot be dropped: a Cartesian transverse Hessian alone differentiates a different path. Bounded real sigmoid derivatives and Gaussian polynomial envelopes justify these differentiations.

Set $P_y(z)=\sum_{i<j}a_ia_j(d_i-d_j)^2r_ir_j$. Expansion gives $P_y=GV-U^2$, $A_y'=A_yG$ and $V'=-T$. Consequently

$$
\begin{aligned}
B_y+A_yP_y&=(A_yV)'-zA_yV
=\{\varphi A_yV\}'/\varphi,\\
Q_y''(0)&=-E[A_y(Z)P_y(Z)],\qquad N_y:=E[A_y(Z)P_y(Z)],\\
\mathcal H_R(a;d)&:=\left.\frac{d^2R(\theta(t))}{dt^2}\right|_{t=0}
=\sum_yq_*(y)\frac{N_y}{Q_y}
=\sum_yq_*(y)E_{\pi_y}P_y,\\
\pi_y(dz)&=A_y(z)\varphi(z)\,dz/Q_y.
\end{aligned}
\tag{I.48}
$$

Here $\varphi A_yV$ vanishes at both real infinities because $A_y\le1$ and $V$ is bounded. Thus the first line is a Gaussian weighted total derivative, and the second line follows only after integration. The integrands $B_y$ and $-A_yP_y$ are not asserted to be equal pointwise. Differentiating the negative log probability supplies the second minus sign; $Q_y'(0)=0$ removes its squared-first-derivative term. Constant $d$ gives $P_y=0$, as required for common rotation.

### I.10.2. Centre enclosure and an alternative path expression

At the exact centre in (I.21), $|a_{0,i}|<1$ and $\sum_{i<j}(d_i-d_j)^2=8$, so $|P_y|\le8$. All 256 labels are evaluated individually. On the closed strip $|\operatorname{Im}z|\le1$, set $c=\sec(1/2)$. The inequality $|1+s e^{iv}|\ge(1+s)\cos(v/2)$ for $s\ge0$, $|v|\le1$, bounds each complex Bernoulli factor by $c$ times its real-axis value. Hence $|A_y|\le c^8$, $|r_i|\le c$ and $|D_i|\le c^2/4$. The two numerator expressions have separate one-sided boundary-integral and omitted-lattice bounds:

$$
\begin{aligned}
h&=1/8,\quad N=88,\quad L=Nh=11,\quad c=\sec(1/2),\\
M_{\rm pair}&=e^{1/2}c^8\,8c^2,\qquad M_Q=e^{1/2}c^8,\\
M_{\rm path}&=e^{1/2}c^8\{8c^2+c^2/4+c\sqrt2\},\\
T_{\rm pair}&=16\varphi(L)/L,\qquad T_Q=2\varphi(L)/L,\\
T_{\rm path}&=(8+1/4)2\varphi(L)/L+2\varphi(L),\\
\epsilon_j&=\frac{2M_j}{e^{2\pi/h}-1}+T_j,\qquad j\in\{{\rm pair},{\rm path},Q\}.
\end{aligned}
\tag{I.49}
$$

For the pair expression the integrand is $\varphi A_yP_y$; for the alternative it is $-\varphi B_y$. On a strip boundary, $|U|\le\sqrt8c$, $|T|\le c^2/4$, $|V|\le c$ and $E|Z\pm i|\le\sqrt2$, which prove $M_{\rm path}$. Contour translation gives Fourier decay $M_j e^{-|\xi|}$; summing the nonzero Poisson frequencies proves the alias term, as in (I.27). The Gaussian envelope makes the contour end boundaries vanish. On the real axis the two numerator factors are bounded by $8$ and $8+1/4+|z|$, respectively. The omitted full-weight nodes $|k|>N$ are bounded by adjacent continuous tails, using the monotonicity of $\varphi(x)$ and $x\varphi(x)$ for $x\ge L\ge1$. The identities $\int_L^\infty x\varphi(x)dx=\varphi(L)$ and $\int_L^\infty\varphi(x)dx\le\varphi(L)/L$ give the stated tails.

Each 177-node sum includes both endpoints with full weight and uses 192-bit Arb arithmetic. Its outward arithmetic interval is retained separately from the analytic error. The pair error is approximately $1.44001098659947\,10^{-20}$ per label, not an error for the final curvature. The alternative implementation forms the complete path derivative in (I.47), retaining acceleration, with separate integer-label probability products and its own bounds in (I.49); it also recomputes $Q_y$.

Let $[F_y]$ be the pair finite-sum interval and $[Q_y]$ the original strictly positive true-probability enclosure used in I.8.5. Let $\widehat q_y$ be the exact midpoint of the two outward endpoints of the saved teacher finite-sum interval. No normalization is applied to these midpoints. The teacher certificate gives $\|q_*-\widehat q\|_1<\epsilon_q:=4.894\,10^{-21}$. Interval composition is

$$
\begin{aligned}
[N_y]&=[F_y]+[-\epsilon_{\rm pair}^{\rm up},\epsilon_{\rm pair}^{\rm up}],\\
[\mathcal H_0]&=\sum_y\widehat q_y[N_y]/[Q_y]+[-8\epsilon_q,8\epsilon_q],\\
|N_y/Q_y|&=|E_{\pi_y}P_y|\le8,\qquad
8\epsilon_q=3.9152\,10^{-20}.
\end{aligned}
\tag{I.50}
$$

Here $\epsilon^{\rm up}$ denotes a certified outward upper endpoint. The teacher allowance occurs once, through the true posterior ratio bound; the label-wise numerator error is propagated through division and weighting. The alternative centre enclosure substitutes its path sum and $\epsilon_{\rm path}$, but uses the same old $[Q_y]$. Its newly computed probabilities are a cross-check, not a different denominator for this composition.

After their respective analytic enlargements, all 256 pairs of numerator intervals overlap; all 256 recomputed probability intervals overlap their old counterparts. These checks compare enclosures of equal integrals, not identical finite sums, and do not replace the analytic error proof. The implementations share Arb, teacher inputs and the root certificate. A separate exact rational-arithmetic check recomposes saved endpoints only; it does not recompute transcendental functions or integrals. They are different-expression implementation checks, not independent end-to-end certifications with different arithmetic backends.

### I.10.3. Scalar transfer to the certified root

We require a variation bound for the one scalar $\mathcal H_R(a;d)$ at fixed $q_*,b=0,d$, not a bound on the full risk Hessian. Suppose $\max_i|a_i|\le A$ along the segment in question, with $m$ rows and $\|d\|_2=1$. Integrating $(zA_y\varphi)'$ over the real line gives

$$
\begin{aligned}
E_{\pi_y}Z^2&=1+E_{\pi_y}[Z\,\partial_z\log A_y],\qquad
|\partial_z\log A_y|\le mA,\\
\mu_y^2&\le E_{\pi_y}Z^2\le1+mA\mu_y,\qquad \mu_y:=E_{\pi_y}|Z|,\\
\mu_y&\le M_1:=\frac{mA+\sqrt{m^2A^2+4}}2.
\end{aligned}
\tag{I.51}
$$

The boundary vanishes by Gaussian decay, and Cauchy–Schwarz gives the second line. No numerical lower bound on $Q_y$ is needed for this posterior-moment estimate. Write $c_{ij}=(d_i-d_j)^2$, so $\sum_{i<j}c_{ij}\le m$ and $|P_y|\le mA^2$. For a unit loading direction $h_0$, direct differentiation gives

$$
\begin{aligned}
D_aP_y[h_0]&=\sum_{i<j}c_{ij}\big[(h_{0,i}a_j+a_ih_{0,j})r_ir_j\\
&\hspace{20mm}-za_ia_j(D_ih_{0,i}r_j+r_iD_jh_{0,j})\big],\\
|D_aP_y[h_0]|&\le2mA+(mA^2/2)|z|,\qquad
|D_a\log A_y[h_0]|\le\sqrt m|z|,\\
D_a E_{\pi_y}P_y[h_0]&=E_{\pi_y}D_aP_y[h_0]
+\operatorname{Cov}_{\pi_y}(P_y,D_a\log A_y[h_0]),\\
|\operatorname{Cov}_{\pi_y}(P_y,D_a\log A_y[h_0])|
&\le2m^{3/2}A^2 E_{\pi_y}|Z|.
\end{aligned}
\tag{I.52}
$$

The covariance retains differentiation of the posterior denominator; its two terms are each bounded using $|P_y|\le mA^2$. Weighting by the fixed probability vector and integrating the derivative bound along the loading segment proves

$$
\begin{aligned}
|\mathcal H_R(a;d)-\mathcal H_R(a';d)|&\le L_{\rm ang}\|a-a'\|_2,\\
L_{\rm ang}&=2mA+(mA^2/2+2m^{3/2}A^2)M_1,\\
L_{\rm ang}\big|_{m=8,A=1}
&=16+(4+16\sqrt8)(8+\sqrt{68})/2<417.
\end{aligned}
\tag{I.53}
$$

For an exact rational verification of the last cap, $\sqrt8<2829/1000$ and $\sqrt{68}<8247/1000$ give an upper bound $52024513/125000<417$. All coordinates of $a_0\pm(1.805\,10^{-8})\mathbf1$ lie in $(0,1)$. By (I.39), the segment to the same true root therefore satisfies $A=1$, with no bias or teacher variation. Thus

$$
|\mathcal H_R(a_*;d)-\mathcal H_R(a_0;d)|
<417(1.805\,10^{-8})=7.52685\,10^{-6}.
\tag{I.54}
$$

This uses neither the risk-Hessian norm cap in (I.40) nor the finite-objective variation constant in (I.37) as a risk-curvature variation bound. Combining (I.50) and (I.54) yields the following conservative decimal enclosures, each containing both corresponding saved dyadic certificates:

| Object | Lower endpoint | Upper endpoint |
|---|---:|---:|
| $\mathcal H_R(a_0;d)$ | $-0.004511298531248709$ | $-0.004511298531248699$ |
| $\mathcal H_R(a_*;d)$ | $-0.004518825382$ | $-0.004503771681$ |

### I.10.4. Fixed-marginal local conflict

Finally take $a=a_*$ in (I.47). The preceding enclosure and reflection give $R'(0)=0$ and $R''(0)<-0.00450377$. Twice differentiability implies risk descent for every sufficiently small nonzero step on either side. Since $d$ is nonconstant and all $a_{*,i}>0$, some two-row minor is $a_{*,i}a_{*,j}\sin(t(d_j-d_i))\ne0$ for all sufficiently small nonzero $t$. Such rank-two points lie outside the common-rotation orbit of the rank-one root. Its full-space strict local minimality modulo rotation, established in I.8.4, then proves that for some $\varepsilon>0$,

$$
\begin{gathered}
R'(0)=0,\qquad R''(0)<-0.00450377,\\
0<|t|<\varepsilon\quad\Longrightarrow\quad
R(\theta(t))<R(\theta_*),\\
F_8^{q_*}(\theta(t))>F_8^{q_*}(\theta_*),\quad\operatorname{rank}W(t)=2.
\end{gathered}
\tag{I.55}
$$

The entire path preserves every candidate logit law $N(0,a_{*,i}^2)$ and every observation marginal $\operatorname{Bernoulli}(1/2)$. The latter correctly match the teacher; the candidate logit variances need not match the teacher's $(4/5)^2$. Relative covariances change as $a_{*,i}a_{*,j}\cos(t(d_i-d_j))$, so this is a change of dependence, not a common rotation. It supplies the fixed-logit-marginal part of Theorem 3 at the original exact root, while preserving the distinct radial finite-step guarantee in (I.41).

No numerical $\varepsilon$, specified finite-step gain, full angle-spectrum certificate or K11/K12 transition is asserted. In particular the curvature bound is not a certified loss change at $t=.02$. I.9 remains a qualitative result for the radial direction; I.10 alone does not extend all its angle-path properties to perturbed teachers. I.11 gives that extension for a specified zero-bias family. Exact inputs, finite intervals and their evidence mapping are retained with the AngularWitness-v1 certificate; the proof here depends on the displayed scalar identity, integral envelopes and transfer argument rather than on execution logs.

## I.11. Quantitative persistence in a changing teacher family {#quantitative-teacher-family}

::: {#teacher-family-full-statement}

**Corollary (quantitative local persistence in an explicit teacher family).** Keep the teacher biases zero and all teacher row lengths $4/5$, and replace only its first angle by $-.7+\lambda$. Let $q_\lambda$ be its observation law and $\mathcal R_\lambda(\theta)=D(q_\lambda\Vert Q_\theta)$. There is a smooth branch $\theta_\lambda=(0,a(\lambda),0)$ on the entire interval $|\lambda|\le10^{-3}$, connected to Theorem 3's root, such that every $\theta_\lambda$ is a strict local minimum of its own $F_8^{q_\lambda}$ in the full parameter space modulo common latent rotation. With the same fixed $d=d_{\mathrm{ang}}^{(8)}$ as in (14), set $\theta_\lambda(t)=(0,W_\lambda(t))$, $f_\lambda(t)=F_8^{q_\lambda}(\theta_\lambda(t))$ and $r_\lambda(t)=\mathcal R_\lambda(\theta_\lambda(t))$. Then

$$
\begin{aligned}
W_{\lambda,i}(t)&=a_i(\lambda)(\cos(td_i),\sin(td_i)),\\
f_\lambda'(0)&=r_\lambda'(0)=0,\\
f_\lambda''(0)&>.0011,\qquad
r_\lambda''(0)\le-.004351070278<0.
\end{aligned}
$$

For each $\lambda$, some $\varepsilon_\lambda>0$ makes every $0<|t|<\varepsilon_\lambda$ raise $F_8^{q_\lambda}$, lower its own true risk $\mathcal R_\lambda$ and give $\operatorname{rank}W_\lambda(t)=2$. The path preserves the candidate logit marginals $N(0,a_i(\lambda)^2)$ and the correct fair observation marginals.

:::

### I.11.1. Teacher expansion with a joint remainder

Keep the teacher of (13), except for its first angle, and write $\rho=4/5$, $\phi=(-.7,-.5,-.3,-.1,.1,.3,.5,.7)$. For $z\in\mathbb R^2$, let $A_y^*(z;\lambda)$ denote its Bernoulli product. Differentiating only the first factor gives

$$
\begin{aligned}
t_i^*&=\rho\{\cos(\phi_i+\lambda\mathbf1_{i=1})z_1
+\sin(\phi_i+\lambda\mathbf1_{i=1})z_2\},\\
\partial_\lambda A_y^*&=A_y^*(y_1-\sigma(t_1^*))\partial_\lambda t_1^*,
\qquad q_\lambda(y)=E A_y^*(Z;\lambda),\\
q_\lambda&=q_0+\lambda q'_0+e_\lambda,\qquad
\sum_y e_{\lambda,y}=0,\\
\|q''_\lambda\|_1&\le\frac{\rho^2}{3\sqrt3}
+\frac\rho2\sqrt{2/\pi}<.442323,\qquad
\|e_\lambda\|_1\le .2211615\lambda^2.
\end{aligned}
\tag{I.56}
$$

The second-derivative bound holds for all real $\lambda$: sum the two first-bit derivatives before the other seven factors, use $\sup|\sigma''|=1/(6\sqrt3)$, $\sup\sigma'=1/4$, $E(\partial_\lambda t_1^*)^2=\rho^2$ and $E|\partial_\lambda^2t_1^*|=\rho\sqrt{2/\pi}$. Exact probability mass gives the zero sum. Gaussian reflection preserves complement symmetry and fair marginals, without preserving reversal symmetry.

All 256 components of $q_0,q'_0$ are enclosed by the two-dimensional full-weight trapezoidal rule with spacing $\delta=1/8$, cutoff $L=10$ and 192-bit arithmetic. For complex coordinate shifts of modulus at most $\eta=1$, put $\beta_i=\rho\eta(|\cos\phi_i|+|\sin\phi_i|)/2$. The Bernoulli modulus sum is at most $\sec\beta_i$; the derivative factor has bound $\rho\sec^2\beta_1/2$ times a linear Gaussian envelope. With $s=e^{-2\pi\eta/\delta}$ and $\kappa=|\sin\phi_1|+|\cos\phi_1|$, explicit boundary and omitted-node bounds give

$$
\begin{aligned}
M_q&=e^{\eta^2}\prod_i\sec\beta_i,\\
M_{q'}&=e^{\eta^2}\frac\rho2\sec^2\beta_1
\left(\prod_{i>1}\sec\beta_i\right)\kappa(\sqrt{2/\pi}+\eta),\\
S_0&=1+\delta\varphi(0),\quad S_1=\sqrt{2/\pi}+4\delta\varphi(1),\\
T_0&=2\varphi(L)/L,\quad T_1=2\varphi(L),\\
T_q&=2S_0T_0,\qquad T_{q'}=(\rho/2)\kappa(T_1S_0+T_0S_1),\\
\epsilon_j&\le M_j\frac{4s}{(1-s)^2}+T_j,\qquad j\in\{q,q'\},\\
\|q_0-q_0^f\|_1&<4.894\times10^{-21},\qquad
\|q'_0-(q'_0)^f\|_1<5.927\times10^{-21}.
\end{aligned}
\tag{I.57}
$$

The alias coefficient sums both single-axis and double-axis frequencies. Discrete tails use monotonic Gaussian tails outside the cutoff and the total variation of $|z|\varphi(z)$ for its whole-axis moment. These are joint label-$L^1$ bounds, paid once. Arithmetic intervals remain separate; the resulting component enclosures are neither averaged nor normalized.

### I.11.2. Eight free candidate row lengths

Reversal pairing cannot reduce the new candidate. The eight row-length parameters $a_i$ can vary separately. We apply the Laplace representations in I.3 and I.8.3 with all eight row lengths free. At $u=e^v$, let $p_i=\sigma(a_i z)$, $\widetilde p_i=p_i-1/2$, $\widetilde y_i=y_i-1/2$, $t=uA_y e^{-uA_y}$ and $\Psi=E e^{-uA_y}$. The complete centered moments are

$$
\begin{aligned}
T_l&=E[tz^l],\quad \widetilde P_l=E[tz^l\widetilde p],\quad
\widetilde{PP}_l=E[tz^l\widetilde p\widetilde p^\top],\\
R_1&=T_1\widetilde y-\widetilde P_1,\\
\Gamma_l&=T_l\widetilde y\widetilde y^\top
-\widetilde y\widetilde P_l^\top-\widetilde P_l\widetilde y^\top
+\widetilde{PP}_l,\\
D_l&=\operatorname{diag}(T_l/4-\operatorname{diag}\widetilde{PP}_l),\qquad
c_K=K\Psi^{K-1},\quad d_K=K(K-1)\Psi^{K-2}.
\end{aligned}
\tag{I.58}
$$

Use $l=0,1,2$ for $T,\widetilde P$ and $l=0,2$ for $\widetilde{PP}$, retaining every matrix entry by symmetry. Conditional gradient and Hessian blocks are the integrals over all real $v$ of

$$
\begin{aligned}
g_y(v)&=-c_KR_1,\\
H_{bb,y}(v)&=c_KD_0-d_K(T_0\widetilde{PP}_0-\widetilde P_0\widetilde P_0^\top),\\
H_{aa,y}(v)&=c_KD_2-d_K(T_0\Gamma_2-R_1R_1^\top),\\
H_{\perp,y}(v)&=c_KD_0-d_KT_0\Gamma_0,\qquad K=8.
\end{aligned}
\tag{I.59}
$$

Centering is an exact algebraic change of moments. The transverse block lacks a mean-score square because the second Gaussian coordinate has zero mean. Reflection $A_{1-y}(z)=A_y(-z)$ makes these four quantities complement-invariant, allowing 128 representatives weighted by $q_y+q_{1-y}$. No 41-class paired reduction is used. The I.8 contour and discrete-tail bounds apply with $\sum_i\bar a_i$ and $\sum_i\sec^2(\bar a_i d_z/2)$ over all eight individual upper endpoints. Their uniform gradient and operator errors are denoted $\epsilon_g,\epsilon_H,\epsilon_P$; interval dependence and arithmetic radii remain in the finite matrices.

### I.11.3. One root box for the entire interval

Let $a_0$ be the exact center (I.21), $h=.001$, $X=a_0+\prod_i[-r_i,r_i]$, $r=(.00075,.00015,\ldots,.00015)$ and $t_h=.2211615h^2$. Write $g_y^f,H_y^f$ for the finite conditional enclosures. Preserve the signed teacher derivative by first forming

$$
\begin{aligned}
G_1^f(X)&=\sum_y[q'_{0,y}]g_y^f(X),\\
H_{j,\mathrm{fin}}(X)&=\sum_y[q_{0,y}]H_{j,y}^f(X)
+[-h,h]\sum_y[q'_{0,y}]H_{j,y}^f(X),\\
E_H&=\epsilon_H(1+hQ_1)+146.25t_h,\qquad
E_P=\epsilon_P(1+hQ_1)+8.25t_h,\quad Q_1\ge\|q'_0\|_1.
\end{aligned}
\tag{I.60}
$$

Here $j=bb,aa,\perp$; $E_H$ applies to both principal blocks of the full bias/radial Hessian. Teacher component intervals already contain (I.57). The global conditional caps are those of I.8: 8, 146.25 and 8.25. Multiplying each linearized weight by an unrelated interval for $\lambda$ before summation would discard useful sign cancellation.

Use the certificate's fixed rational symmetric preconditioner $A$, whose positive definiteness is checked by strict interval LDL, and $T_\lambda(a)=a-AG(a,\lambda)$. For row $A_i$, let $b_i$ be the midpoint of the joint range of $(Ag_y^f(X))_i$. Zero remainder mass permits subtraction of this same constant from every label. Thus

$$
\begin{aligned}
\widehat C_i&=\min\left\{8\|A_i\|_2,
\max_y\sup|(Ag_y^f(X))_i-b_i|+\|A_i\|_2\epsilon_g\right\},\\
Y_i&=7.572\times10^{-10}\|A_i\|_2
+h\sup|(AG_1^f(X))_i|
+h\|A_i\|_2\epsilon_gQ_1+t_h\widehat C_i.
\end{aligned}
\tag{I.61}
$$

The old center residual is retained. Although $G_1^f$ covers $X$, $Y_i$ is needed only for the displacement at $a_0$. For the derivative of $T_\lambda$, an operator-error allowance gives

$$
\begin{aligned}
Z_i&=\sum_j\sup|(I-AH_{aa,\mathrm{fin}}(X))_{ij}|r_j
+\|A_i\|_2E_H\|r\|_2,\\
\max_i Z_i/r_i&<.264019,\qquad
\max_i(Y_i+Z_i)/r_i<.753428,\\
\min_i(r_i-Y_i-Z_i)&>8.37210\times10^{-5}.
\end{aligned}
\tag{I.62}
$$

Banach's theorem supplies a unique radial root in $X$ for every $\lambda$, uniformly in the radius-weighted maximum norm. Invertibility of $A$ equates fixed points with roots. Uniform contraction gives continuity. The old true root is inside $X$ by (I.39), so uniqueness identifies $a(0)=a_*$. Smoothness follows from the implicit function theorem and the uniformly invertible radial Hessian below; local branches agree by uniqueness.

### I.11.4. Full stability from original-matrix uncertainty

For each original symmetric $8\times8$ finite interval matrix with exact endpoints $l_{ij},u_{ij}$, form the midpoint before projection:

$$
\begin{aligned}
(M_0)_{ij}&=(l_{ij}+u_{ij})/2,\qquad B_{ij}=(u_{ij}-l_{ij})/2,\\
\|H_{j,\mathrm{fin}}-M_0\|_{\rm op}&\le\|B\|_F
\le\bar\rho_F,\qquad \sum_{i,j}B_{ij}^2\le\bar\rho_F^2.
\end{aligned}
\tag{I.63}
$$

The final squared comparison is checked with exact rational arithmetic, including all 64 entries. Let $U_0$ be the last seven columns of the Householder matrix for $u_0=a_0/\|a_0\|$, $v_0=u_0+e_1$. Its exact orthogonality and $U_0^\top a_0=0$ give unit operator norm. Strict interval LDL proves, with each block's own $M_0,\bar\rho_F$,

$$
\begin{aligned}
M_{0,bb}-(.1+E_H+\bar\rho_{F,bb})I_8&\succ0,\\
M_{0,aa}-(.02+E_H+\bar\rho_{F,aa})I_8&\succ0,\\
U_0^\top M_{0,\perp}U_0-(.005+E_P+\bar\rho_{F,\perp})I_7&\succ0.
\end{aligned}
\tag{I.64}
$$

Bounding original-matrix uncertainty avoids the extra dependency loss from projecting an interval matrix first. The thresholds remain unchanged. Complement and latent-coordinate reflections, as in I.2, make the other gradient components and mixed blocks zero. At the root, rotation invariance supplies the Ward identity:

$$
\begin{gathered}
\nabla F_8^{q_\lambda}(0,a(\lambda),0)=0,\qquad
H_\perp a(\lambda)=0,\qquad a_0^\top a(\lambda)>0,\\
H_{bb}\succ .1I_8,\qquad H_{aa}\succ .02I_8,\qquad
U_0^\top H_\perp U_0\succ .005I_7.
\end{gathered}
\tag{I.65}
$$

The fixed slice is transverse to the rotation direction throughout $X$; its positivity therefore gives full-space strict local minimality modulo rotation. The number .005 is a bound on $a_0^\perp$, and is not asserted unchanged on the moving $a(\lambda)^\perp$.

### I.11.5. Each teacher's own angular risk

Use the fixed angular direction $d=d_{\mathrm{ang}}^{(8)}$ defined in (I.46), with the full path from I.10, including its acceleration. For every $a\in X$, independently enclose all 256 quantities

$$
\begin{aligned}
Q_y(a)&=E A_y(Z;a)>0,\\
N_y(a)&=E[A_y(Z;a)P_y(Z;a,d)],\qquad
\rho_y(a)=N_y(a)/Q_y(a),\\
\sum_y A_y(z;a)P_y(z;a,d)&=0,\qquad \sum_y N_y(a)=0.
\end{aligned}
\tag{I.66}
$$

The node identity follows by summing independent Bernoulli residual cross-products. The 177-node rule and separate probability/numerator bounds in (I.49) apply directly to every member of $X$ because $0<a_i<1$. This encloses candidate movement without a 417-times-displacement transfer. Positive denominators, analytic errors and arithmetic intervals are all included. Set $B_*$ to the midpoint of the joint $\rho_y(X)$ range. Then

$$
\begin{aligned}
R_0(a)&=\sum_y q_{0,y}\rho_y(a)
=\sum_y(q_{0,y}-Q_y(a))\rho_y(a),\\
[r_\lambda''(0)]&\subset [R_0(X)]
+[-h,h]\sum_y[q'_{0,y}]\rho_y(X)\\
&\hspace{8mm}+[-t_h,t_h]\max_y\sup|\rho_y(X)-B_*|\\
&\subset[-.004671548834,-.004351070278].
\end{aligned}
\tag{I.67}
$$

Intersecting the two valid enclosures of $R_0$ tightens its bound; their overlap alone would not certify arbitrary inputs. Both centerings use exact zero-mass identities, without changing target probabilities. Reflection gives $r_\lambda'(0)=0$. For finite-objective curvature put $v=\operatorname{diag}(a)d$, $c=(a_0^\top v)/(a_0^\top a)$ and $w=v-ca$. From (I.65),

$$
\begin{aligned}
w&\in a_0^\perp,\qquad
f_\lambda''(0)=v^\top H_\perp v=w^\top H_\perp w\\
&>.005\|w\|^2\ge .005a_{\min}^2\|d-c\mathbf1\|^2
\ge .005a_{\min}^2>.0011,\\
a_{\min}&=\min_i(a_{0,i}-r_i),\qquad
\|d-c\mathbf1\|^2=1+8c^2.
\end{aligned}
\tag{I.68}
$$

Stationarity gives $f_\lambda'(0)=0$. Twice differentiability now proves the teacher-family corollary and its consequences for each teacher's own sufficiently small two-sided steps. Positive lengths and nonconstant $d$ give rank two, while fixed lengths preserve candidate Gaussian logits and fair observation marginals. No common numerical step radius or family-wide KL interval is inferred.

### I.11.6. Nontriviality and certification boundary

Taylor's inequality and the enclosed $\|q'_0\|_1$ show a genuine law change:

$$
\begin{aligned}
.159347929859&\le\frac{\|q_\lambda-q_0\|_1}{|\lambda|}
\le .159790252860\quad(0<|\lambda|\le .001),\\
\|q_{\pm.001}-q_0\|_1&\in[.000159347929,.000159790253].
\end{aligned}
\tag{I.69}
$$

The seven first-row spin-pair derivatives remain strictly positive after the allowance $.442323h$; the joint law of the other seven observations is unchanged. Hence the formerly equal $(1,2)$ and $(8,7)$ moments separate for every nonzero $\lambda$, breaking reversal symmetry. To enclose the implicit derivative $a'=-H_{aa}^{-1}G_\lambda$, choose $g_C$ coordinatewise as the midpoint of the joint conditional-gradient range, and set $C_g=\min\{8,\max_y\sup\|g_y^f(X)-g_C\|_2+\epsilon_g\}$. Since $q'_\lambda-q'_0$ has zero mass and $L^1$ norm at most $.442323h$, the error in approximating $G_\lambda$ by $G_1^f$ is at most $\epsilon_gQ_1+C_g\,.442323h$. Enlarge every entry of the finite $H_{aa}$ enclosure by $E_H$ and perform interval linear solution. This gives

$$
\begin{aligned}
a'_1(\lambda)-a'_8(\lambda)&\in[.429945195966,.565673464386],\\
\frac{a_1(\lambda)-a_8(\lambda)}{\lambda}
&\in[.429945195966,.565673464386]\quad(\lambda\ne0),\\
|a_1(\pm.001)-a_8(\pm.001)|&\in[.000429945195,.000565673465].
\end{aligned}
\tag{I.70}
$$

Here $a_1(0)=a_8(0)$ comes from the old root's exact symmetry. The branch therefore leaves the paired-length subspace. The three prescribed box halfwidths retain the following outcomes; ``unresolved'' means insufficient enclosure, not a counterexample.

| Halfwidth $h$ | Original root inclusion | Original $bb/aa/\perp$ LDL | Own-risk sign | After midpoint spectral bound |
|---:|---|---|---|---|
| $.01$ | Unresolved | Pass / unresolved / unresolved | Negative | Unresolved |
| $.001$ | Pass | Pass / pass / unresolved | Negative | Full certificate |
| $.0001$ | Pass | Pass / pass / pass | Negative | Full certificate |

The midpoint argument was a subsequent deterministic refinement of saved matrices; it preserves the original failures and all thresholds. Seven floating-point development points reaching $\pm.1$ do not certify that interval. The endpoint law change in (I.69) locates the proved result at a small quantitative local scale. Exact intervals and their expression checks reside in R03TeacherFamily-v1; no development trajectory is needed for this proof.
