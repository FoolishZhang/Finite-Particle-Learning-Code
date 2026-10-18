<a id="model-proof"></a>

# A. Model, identification, and localization

## A.1. A known-channel baseline and a learnable-channel extension

There are three latent groups. The candidate correlations are the three physical off-diagonal entries of

$$
C(\theta)=\begin{pmatrix}1&\theta_{12}&\theta_{13}\\
\theta_{12}&1&\theta_{23}\\\theta_{13}&\theta_{23}&1\end{pmatrix},\qquad
X\sim N(0,C(\theta)),\qquad
\Theta_\delta=\{\theta:C(\theta)\succeq\delta I_3\},\quad 0<\delta<1.
\tag{A.1}
$$

The domain is compact and convex, and 0 is an interior point in the three correlation coordinates. Conditional on X, the observations $O_i=(Y_i,B_i)\in\{-1,1\}\times\{0,\ldots,M\}$ are independent, with

$$
\begin{gathered}
\ell_{i,\alpha}(y,n\mid x)
=\sigma(yx){M\choose n}H_{i,\alpha}(x)^n[1-H_{i,\alpha}(x)]^{M-n},\\
H_{i,\alpha}(x)=E_G\sigma(\alpha_i x+\tau_iG),\qquad
\sigma(t)=(1+e^{-t})^{-1},\qquad G\sim N(0,1).
\end{gathered}
\tag{A.2}
$$

M is a fixed positive integer; the numerical model uses $M=8$ and $\tau_i=3$. The label slope, latent marginal variances, intercepts, and private-noise scales are fixed throughout. In the known-channel baseline, $(\alpha_i,\tau_i)$ is fixed and may be heterogeneous, including a zero count slope. In the extension, all count slopes equal one candidate parameter $\alpha\in\mathcal I_\alpha=[\alpha_-,\alpha_+]$, whose true value $\alpha_*$ is interior, and the private-noise scales equal the fixed $\tau=3$. Write $H_\alpha$ for that common function. The numerical interval is $[.35,.65]$ and $\alpha_*=1.4/\sqrt8$. Alpha is the count-logit coefficient itself; it is not divided again by $\sqrt M$.

The theory uses the normalized continuous channel in (A.2) and exact active-variable integrals. One may instead fix a positive, normalized, reflection-symmetric finite rule for G; that defines another continuous-X channel, to which the same arguments apply. Replacing the remaining active integrals or posterior normalization by finite quadrature is a numerical approximation, not an equality in this probability model.

For either specified channel, define the absolute local log normalizer, its zero-tilt probability, and the independent posterior:

$$
\begin{gathered}
\mathcal A_{i,\alpha}(h,s)
=\log\int\phi(x)\ell_{i,\alpha}(s\mid x)e^{hx}\,dx,\qquad
Z_{i,\alpha}(s)=e^{\mathcal A_{i,\alpha}(0,s)},\\
q_{0i,\alpha}(x\mid s)=\phi(x)\ell_{i,\alpha}(s\mid x)/Z_{i,\alpha}(s),\qquad
\kappa_{i,\alpha}(h,s)=\mathcal A_{i,\alpha}(h,s)-\mathcal A_{i,\alpha}(0,s),\\
\mu_{i,\alpha}(s)=E_{q_{0i,\alpha}}X_i,\qquad
q_{i,\alpha,h}(x\mid s)=q_{0i,\alpha}(x\mid s)e^{hx-\kappa_{i,\alpha}(h,s)}.
\end{gathered}
\tag{A.3}
$$

Here the subscript alpha on heterogeneous known-channel quantities denotes the fixed vector of slopes; no slope is optimized in that baseline. Write $p_{\theta,\alpha}(x,o)=\phi_{C(\theta)}(x)\prod_i\ell_{i,\alpha}(o_i\mid x_i)$ and $\pi_{\theta,\alpha}(x\mid o)$ for its exact joint posterior. Each local state has positive probability and a positive continuous posterior density on the real line.

The proposal classes differ in their inputs, while their outputs remain products:

$$
\begin{aligned}
q_{\alpha,u}(x\mid o)&=\prod_iq_{i,\alpha,h_i(o)}(x_i\mid o_i),\\
F:\quad h_i(o)&=0,\\
L:\quad h_i(o)&=b_i(o_i),\\
A:\quad h_i(o)&=b_i(o_i)+\sum_{j\ne i}a_{i\leftarrow j}\mu_{j,\alpha}(o_j).
\end{aligned}
\tag{A.4}
$$

F is the untilted, no-free-tilt baseline: it is fixed when the channel is known, but its base posterior moves with candidate alpha in the extension. L is local-input adaptation; A is cross-group-input factorized adaptation. All local coefficients lie in a fixed box $[-b_0,b_0]^{3S_{\rm loc}}$ and all six directed coefficients in $[-a_0,a_0]^6$, where $S_{\rm loc}=2(M+1)$ and $a_0,b_0>0$. The numerical half-widths are .25 and .15. Denote the full free vector by u and its box by $B_Q$; $B_F$ is a singleton. At $M=8$, L and A have 54 and 60 free proposal coefficients. The exact joint posterior is a mathematical oracle, not a fourth implemented adaptation class.

The teacher is $P_*=P_{\eta,\alpha_*}$ in the extension, and $P_\eta$ with the same fixed channel in the baseline. Conditional particles are iid from the candidate proposal. For each fixed finite integer $K\ge1$, the conditional loss and its population mean are

$$
\begin{gathered}
W_{\theta,\alpha,u}(x,o)=\frac{p_{\theta,\alpha}(x,o)}{q_{\alpha,u}(x\mid o)},\\
\mathcal L_{K,Q}(\theta,\alpha,u;o)
=E_{q_{\alpha,u}^{\otimes K}}\!\left[-\log\left\{K^{-1}\sum_{k=1}^KW_{\theta,\alpha,u}(X_k,o)\right\}\right],\\
F_{K,Q}(\theta,\alpha,u;\eta)=\sum_op_{\eta,\alpha_*}(o)\mathcal L_{K,Q}(\theta,\alpha,u;o),\\
\log W=\sum_i\log Z_{i,\alpha}(o_i)
+\log\frac{\phi_{C(\theta)}(x)}{\phi_I(x)}
-\sum_i h_i(o)x_i+\sum_i\kappa_{i,\alpha}(h_i(o),o_i).
\end{gathered}
\tag{A.5}
$$

For a known channel, alpha is held fixed and omitted from the argument list. The teacher probability in this formula is fixed when differentiating candidate parameters. In particular, the candidate-dependent $\sum_i\log Z_{i,\alpha}$ cannot be discarded as a constant. Using the exact posterior as the proposal makes the weights constant for every observation, so the oracle loss is the integrated negative log likelihood. Subtracting a teacher-only constant centers the population objective without changing any candidate derivative or minimizer.

## A.2. Smoothness and identification

**Lemma A.1 (regularity).** On the specified compact domains, each conditional loss is finite and continuous. Near an interior parameter point it has every fixed finite order of continuous derivatives needed below, with integration and parameter differentiation interchangeable.

To prove the claim, first note that the finite set of positive $Z_{i,\alpha}(s)$ has a uniform positive lower bound on a compact alpha interval. Thus $q_{0i,\alpha}\le c\phi$, and bounded tilt coefficients give an envelope $c\phi(x)e^{c|x|}$ for the tilted densities. Moving mean features and their fixed-order derivatives are bounded on the same set. For every integer $j\ge1$, logistic derivatives satisfy both $|\sigma^{(j)}(t)|\le c_j\sigma(t)$ and $|\sigma^{(j)}(t)|\le c_j\sigma(-t)$. Therefore

$$
|\partial_\alpha^jH_\alpha(x)|\le c_j|x|^jH_\alpha(x),\qquad
|\partial_\alpha^j[1-H_\alpha(x)]|\le c_j|x|^j[1-H_\alpha(x)],\qquad j\ge1.
\tag{A.6}
$$

Log-likelihood derivatives have polynomial envelopes. Formula (A.5), rather than a nonexistent zero-order version of (A.6), bounds log weights by $c(1+\|x\|^2)$ under the spectral lower bound. The negative log mean weight lies between the minimum and maximum negative log weights in the bank. Its fixed-order derivatives are finite sums of their polynomially bounded derivatives with bounded softmax coefficients. Differentiating the sampling density introduces only further polynomial factors. The Gaussian envelopes prove the claim for fixed K. This argument does not require finite second moments of importance weights over the whole candidate domain.

**Lemma A.2 (generating identification).** The complete observation distribution identifies all three correlations. In the common-channel extension it also identifies alpha, with the other parameters in (A.2) fixed.

For $f(x)=\tanh(x/2)$, the Gaussian density identity $\partial_\rho\phi_\rho=\partial_x\partial_y\phi_\rho$ and two integrations by parts give the Gaussian differentiation formula [@price1958]

$$
\begin{gathered}
\frac{d}{d\rho}E_\rho[Y_iY_j]=E_\rho[f'(X_i)f'(X_j)]>0,\\
\frac{d}{d\alpha}E_\alpha[Y_iB_i]
=M E_{X,G}[X\tanh(X/2)\sigma'(\alpha X+\tau G)]>0.
\end{gathered}
\tag{A.7}
$$

The first moment depends only on the corresponding correlation and is independent of count slopes; summing the count probabilities gives one. The second depends only on the unit Gaussian marginal and alpha, not on any correlation. Its derivative is dominated by $M|X|/4$ and is strictly positive when $X\ne0$. Consequently equality of complete observation PMFs implies equality of the generating parameters. This is a global identification argument, stronger than positive Fisher information at one point. It does not cover simultaneous estimation of the label slope, latent variance, or private-noise scale.

At a matching independent model, omit the fixed channel index and set $p_i(s)=Z_i(s)$, $\mu_i=\mu_i(O_i)$, $d_i=\operatorname{Var}(X_i\mid O_i)$, $v_i=E\mu_i^2$. Conditional expectation gives $E\mu_i=0$ and $Ed_i=1-v_i$. Strictly positive posterior variance gives $v_i<1$; $E[X_iY_i]=E[X_i\tanh(X_i/2)]>0$ forces $v_i>0$. A zero count slope therefore does not make a group uninformative: its label still carries information.

**Lemma A.3 (proposal identification at fixed alpha).** The map from the coefficients in (A.4) to the full collection of proposal densities is injective. A continuous exponential tilt is identifiable because equality of two densities would make $(h-h')x$ constant on the real line. For L this identifies each state coefficient. For A, a zero difference in every tilt gives $\Delta b_i(o_i)+\sum_{j\ne i}\Delta a_{i\leftarrow j}\mu_j(o_j)=0$. Averaging the other independent groups gives $\Delta b_i=0$; multiplying by $\mu_j$ and averaging gives $\Delta a_{i\leftarrow j}v_j=0$. Thus every coefficient difference vanishes. Distinct count states remain distinct positive-probability inputs even when they share a posterior.

## A.3. A unique global anchor and its continuation

**Proposition A.4 (anchor).** For every fixed finite K, each class has a unique global population minimizer at the matching independent teacher. On the known-channel domain $D_Q=\Theta_\delta\times B_Q$ it is $(0,0)$; on the extension domain $D_Q=\Theta_\delta\times\mathcal I_\alpha\times B_Q$ it is $(0,\alpha_*,0)$.

Indeed, Jensen and generating identification yield

$$
F_{K,Q}(\theta,\alpha,u;0)
\ge \mathsf H(P_{0,\alpha_*})+
D_{\rm KL}(P_{0,\alpha_*}\Vert P_{\theta,\alpha})
\ge\mathsf H(P_{0,\alpha_*}).
\tag{A.8}
$$

The anchor attains equality. Any other equality point must first have the same generating parameters. Since every observation has positive probability, equality in the first inequality holds separately at each observation. Strict convexity of minus log makes the bank mean weight constant almost surely. For finite iid particles this forces a single weight to be constant: condition on the first weight and average the remaining independent terms. Only its finite first moment is needed. Constant importance weights imply that the proposal is the exact posterior, and Lemma A.3 then gives $u=0$.

**Proposition A.5 (unquantified localization).** For each fixed domain above and fixed finite K, there is $\varepsilon_Q>0$ such that every teacher with $\|\eta\|<\varepsilon_Q$ has a unique interior global minimizer in that domain. It is the smooth branch issuing from the anchor. A corresponding neighborhood of $(\theta,\eta)=(0,0)$, or $(\theta,\alpha,\eta)=(0,\alpha_*,0)$ in the extension, has a unique global proposal profile over the entire specified box $B_Q$.

The complete anchor Hessians are positive definite by (B.4) and (C.4). Lemma A.1 gives a parameter ball V and teacher neighborhood E on which they remain uniformly positive definite. The implicit function theorem supplies a unique interior stationary branch. Choose a smaller open ball U with closure inside V and the domain, and shrink E until the branch stays in U. Proposition A.4 and compactness imply

$$
\begin{gathered}
\gamma_U=\min_{z\in D_Q\setminus U}\{F_{K,Q}(z;0)-F_{K,Q}(z_*;0)\}>0,\\
\sup_{z\in D_Q}|F_{K,Q}(z;\eta)-F_{K,Q}(z;0)|
\le\sum_o|p_{\eta,\alpha_*}(o)-p_{0,\alpha_*}(o)|
\sup_{z\in D_Q}|\mathcal L_{K,Q}(z;o)|\longrightarrow0.
\end{gathered}
\tag{A.9}
$$

If the complement is empty, localization already holds. Otherwise restrict the last bound below $\gamma_U/3$; comparison with the feasible anchor excludes every minimizer outside U. Compactness supplies existence and strict convexity within V supplies uniqueness. For a proposal profile, repeat the same argument on $B_Q$, using the positive proposal Hessian and allowing the nearby candidate and teacher to vary. F has no profiling coordinates.

These propositions establish that sufficiently small nonzero teachers can have biased **global** finite-objective minimizers; the phenomenon is not confined to an inferior local stationary point. The neighborhood depends on K, the domain, and the proposal class. No numerical radius, arbitrary-start convergence guarantee, or global certification of the finite-amplitude numerical points follows.
