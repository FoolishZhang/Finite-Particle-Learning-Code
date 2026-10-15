# A. Proposal structure and dependence selection {#proposal-proof}

This appendix proves the conditional proposal limit and the same-observation-model ordering reversal used in the main text. All particle expectations are part of the population criterion; they are not conditional training losses on a fixed dataset. The finite source model is defined in [A.1](#proposal-model), the critical limit in [A.2](#proposal-critical), its global integrability argument in [A.3](#proposal-ui), and the selection and reversal results in [A.4](#proposal-selection)–[A.5](#proposal-reversal).

## A.1. A fixed observation model and conditional proposals {#proposal-model}

Let $\sigma$ be the logistic function and $\phi$ the standard Gaussian density. Fix three groups, a finite integer $K\ge2$, $\tau>0$, $a\ge0$, and strictly positive definite teacher and candidate correlation matrices $C_*$ and C. Given the teacher latent vector, all observations below are conditionally independent. The same model with C in place of $C_*$ defines the candidate observation law. Counts retain the information about C in the labeled private observations.

$$
\begin{gathered}
\sigma(x)=(1+e^{-x})^{-1},\qquad
h_\tau(s)=\mathbb E_{G\sim N(0,1)}\sigma(\tau G+s),\\
X_*\sim N(0,C_*),\qquad
\mathbb P(Y_i=y\mid X_*)=\sigma(yX_{*i}),\\
\mathbb P(U_{ij}=u\mid X_*)=h_\tau(u\epsilon_M X_{*i}),
\quad y,u\in\{-1,1\},\quad i=1,2,3,\quad j=1,\ldots,M,\\
\epsilon_M=a/\sqrt M,\qquad
Z_i^M=M^{-1/2}\sum_{j=1}^M U_{ij}.
\end{gathered}
\tag{A.1}
$$

For a private variable use the conditional proposal proportional to $\phi(g)\sigma(u(\tau g+\lambda\epsilon x))^\beta$, with $\beta,\lambda\in[0,1]$. After the sign change $T=ug$, put $s=\epsilon ux$ and define the following densities and normalizers. Different private positions and particles are independent conditional on the data and active particles. Active particles are iid $N(0,C)$, independent of the teacher data.

$$
\begin{gathered}
H_\beta(z)=\mathbb E_\phi\sigma(\tau T+z)^\beta,\qquad
q_{\beta,z}(t)=\frac{\phi(t)\sigma(\tau t+z)^\beta}{H_\beta(z)},\\
p_s(t)=q_{1,s}(t),\qquad
T_{kij}\mid X,U\sim q_{\beta,\lambda s_{kij}},\qquad
s_{kij}=\epsilon_M U_{ij}X_{ki},\\
\bar w_{Mk}(C)=
\prod_i\sigma(Y_iX_{ki})
\prod_{i,j}
\frac{2H_\beta(\lambda s_{kij})
      \sigma(\tau T_{kij}+s_{kij})}
     {\sigma(\tau T_{kij}+\lambda s_{kij})^\beta},\\
F_M=-\log\!\left(K^{-1}\sum_k\bar w_{Mk}\right),
\qquad L_M(C)=\mathbb E F_M.
\end{gathered}
\tag{A.2}
$$

The weights have been divided by the candidate-independent factor $2^{-3M}$. Binomial multiplicities, if using counts, are also candidate-independent and do not affect candidate differences or correlation derivatives. The exact identity below makes the model–criterion distinction explicit. At $\beta=\lambda=1$, the posterior density ratio $p_s/q_{\beta,\lambda s}$ in (A.3) equals one for every finite $M$. The full private importance factor remains $2h_\tau(s)$: private sampling disappears without removing the observation likelihood. Finite K integration over the active prior remains. For conditional proposals, common random numbers must be transformed under the correct candidate-dependent density; the same saved T values cannot simply be reused under different q densities.

$$
\frac{2H_\beta(\lambda s)\sigma(\tau t+s)}
     {\sigma(\tau t+\lambda s)^\beta}
=2h_\tau(s)\frac{p_s(t)}{q_{\beta,\lambda s}(t)}.
\tag{A.3}
$$

Integrating the right side under its proposal gives $2h_\tau(s)$. Thus the exact observation model is independent of $\beta$ and $\lambda$, whereas the finite K expectation of the negative log average need not be.

## A.2. The critical conditional proposal limit {#proposal-critical}

Fix $\gamma\ge0$ and $\lambda\in[0,1]$ and set $\beta_M=1-\gamma/\sqrt{3M}$, considering only M for which $\beta_M\in[0,1]$. In this appendix $\nu$ denotes active-coupled computational noise variance, not a teacher correlation. The endpoint moments are

$$
\begin{gathered}
d_\tau=\mathbb E_\phi\sigma'(\tau T),\qquad
q_{1,0}(t)=2\phi(t)\sigma(\tau t),\\
\ell(t)=\log\sigma(\tau t),\qquad f(t)=\sigma(-\tau t),\qquad
v_{\mathrm{priv}}=\operatorname{Var}_{q_{1,0}}\ell(T),\\
\mathbb E_{q_{1,0}}f=2d_\tau,\qquad
\mathbb E_{q_{1,0}}f^2=d_\tau,\qquad
v_f=d_\tau-4d_\tau^2,\\
b=2ad_\tau,\qquad t=\gamma^2v_{\mathrm{priv}},\qquad
\nu_\lambda=(1-\lambda)^2a^2v_f.
\end{gathered}
\tag{A.4}
$$

The endpoint identities follow by pairing T and $-T$ and using $\sigma(x)+\sigma(-x)=1$. Keep the original pair $(X_*,Y)$, and let E, $W_k$, $V_k$ be independent standard Gaussians of dimensions 3, 3, 1, respectively, independent across k and of the data and iid active particles. The limiting experiment and weights are

$$
\begin{gathered}
Z=bX_*+E,\qquad X_k\stackrel{\mathrm{iid}}{\sim}N(0,C),\\
\widetilde w_k(C)=
\prod_i\sigma(Y_iX_{ki})
\exp\!\left(bZ^\top X_k-\frac{b^2}{2}\|X_k\|^2\right)
\exp\!\left(\sqrt{\nu_\lambda}W_k^\top X_k
                    -\frac{\nu_\lambda}{2}\|X_k\|^2\right)
\exp\!\left(\sqrt t V_k-\frac t2\right),\\
\mathcal L_{K,t,\nu_\lambda}(C)
=-\mathbb E\log\!\left(K^{-1}\sum_k\widetilde w_k(C)\right).
\end{gathered}
\tag{A.5}
$$

**Proposition A.1 (pointwise conditional limit).** Under these fixed parameters, each normalized population objective and each physical off-diagonal correlation derivative converge as follows. This also gives differences for any fixed finite set of candidates; a candidate neighborhood used for derivatives must stay a positive distance from singularity.

$$
L_M(C)\longrightarrow\mathcal L_{K,t,\nu_\lambda}(C),
\qquad
\partial_{c_{ij}}L_M(C)
\longrightarrow
\partial_{c_{ij}}\mathcal L_{K,t,\nu_\lambda}(C),
\quad i<j.
\tag{A.6}
$$

**Local weak-limit calculation.** Conditional on $X_*$, a bounded triangular-array central limit theorem gives $Z^M\Rightarrow bX_*+E$, jointly with $(X_*,Y)$. Indeed, the conditional mean of each coordinate is $bX_{*i}+O(M^{-1}|X_{*i}|^3)$, its variance tends to one, and groups are conditionally independent. Conditional Hoeffding bounds and $\sqrt M|2h_\tau(aX_{*i}/\sqrt M)-1|\le a|X_{*i}|/2$ give uniform moments of every fixed order for $Z^M$.

For the private posterior ratio fix an active x and put $\delta=1-\beta$, $r=1-\lambda$. Its exact normalized log form is

$$
\begin{gathered}
h_{\delta,s}(T)=\log\sigma(\tau T+s)
 -(1-\delta)\log\sigma(\tau T+\lambda s),\quad
q=q_{1-\delta,\lambda s},\\
R_{\delta,s}
=h_{\delta,s}-\log\mathbb E_q e^{h_{\delta,s}},
\qquad
h_{\delta,s}=\delta\ell+rsf+O(\delta|s|+s^2).
\end{gathered}
\tag{A.7}
$$

The first three shift derivatives of log $\sigma$ are bounded. Along the exponential interpolation between q and the true posterior, uniform bounds for fixed moments follow from the density-slope bounds proved in [A.3](#proposal-ui). Taylor expansion of the log normalizer therefore gives, locally in $(\delta,s)$,

$$
\begin{aligned}
R_{\delta,s}
={}&\delta(\ell-\mathbb E_q\ell)+rs(f-\mathbb E_qf)
+\zeta_{\delta,s}\\
&-\frac12\left[
\delta^2v_{\mathrm{priv}}+r^2s^2v_f
+2\delta rs\,c_{\ell f}\right]
+O((\delta+|s|)^3),\\
c_{\ell f}={}&\operatorname{Cov}_{q_{1,0}}(\ell,f),\qquad
\mathbb E_q\zeta_{\delta,s}=0,\qquad
\|\zeta_{\delta,s}\|_{L^p(q)}
=O(\delta|s|+s^2).
\end{aligned}
\tag{A.8}
$$

Constants here are local, for each fixed p; this expansion alone is not a tail argument. With $\delta=\gamma/\sqrt{3M}$, $s=\epsilon_M U_{ij}x_i$, the deterministic remainders summed over 3M positions are $O(M^{-1/2}(1+\|x\|^3))$; the centered $\zeta$ sum has $L^2$ size $O(M^{-1/2}(1+\|x\|^2))$. The displayed mixed drift is $-\delta ra c_{\ell f}\sum_i x_i Z_i^M$ and tends to zero. The conditional centered sums satisfy Lindeberg's condition, since $\ell$ grows at most linearly in T and f is bounded. Their variances tend to t and $r^2a^2v_f\|x\|^2$. Cross-covariances are proportional to $\delta\epsilon_M\sum_jU_{ij}$, up to vanishing local moment changes, hence vanish. Conditional characteristic functions give independent limiting V and W, independent of the data and active x; zero x coordinates cause no difficulty.

Finally $h_\tau(0)=1/2$, $h_\tau'(0)=d_\tau$ and $h_\tau''(0)=0$ imply

$$
\log(2h_\tau(s))
=2d_\tau s-\frac{(2d_\tau)^2}{2}s^2+O(|s|^3).
\tag{A.9}
$$

Summing gives the correct observation factor in (A.5). Together with (A.8), this proves joint weak convergence of logweights, data and active particles for each candidate. The negative half-variance drifts are required by the exact normalization; they are not optional centering conventions.

## A.3. Global log moments and correlation derivatives {#proposal-ui}

We now supply the global argument needed to integrate over unbounded active Gaussians. For arbitrary real s, $\beta,\lambda\in[0,1]$, let $h_s$ be the first line of (A.7) and introduce an interpolation parameter $u\in[0,1]$.

$$
q_u(T)\propto\phi(T)\sigma(\tau T+s)^u
                     \sigma(\tau T+\lambda s)^{\beta(1-u)},
\qquad
-T\le\partial_T\log q_u(T)\le-T+\tau .
\tag{A.10}
$$

The density ratio to $N(0,1)$ is increasing and the ratio to $N(\tau,1)$ decreasing. Thus $q_u$ is stochastically bracketed by these two Gaussians. A quantile coupling gives, for every fixed $p\ge1$, the uniform bound below. The logistic 1/4-Lipschitz bound also controls $h_s'$.

$$
\begin{gathered}
\mathbb E_{q_u}|T|^p\le
M_{\tau,p}:=\mathbb E|G|^p+\mathbb E|G+\tau|^p,\qquad
|h_s'(T)|\le L_s:=\tau(\delta+r|s|/4),\\
\|h_s-\mathbb E_qh_s\|_p
\le2L_sM_{\tau,p}^{1/p},\\
\kappa_s:=\log\mathbb E_qe^{h_s}-\mathbb E_qh_s
=\int_0^1(1-u)\operatorname{Var}_{q_u}(h_s)\,du,
\qquad
0\le\kappa_s\le\tfrac12M_{\tau,2}L_s^2.
\end{gathered}
\tag{A.11}
$$

The identity uses the second derivative of the interpolating log normalizer; the variance bound follows by comparing $h_s(T)$ to $h_s(0)$. These inequalities hold globally in s. Conditional on x,U, the private variables are independent. Expanding any even moment of their centered sum, terms with an index occurring once vanish; the remaining terms are bounded by a constant times the corresponding power of the sum of squared $L_s$. Hence, for fixed even $p\ge2$,

$$
\begin{gathered}
\sum_{i,j}L_{s_{ij}}^2
\le2\tau^2\left(\gamma^2+\frac{r^2a^2}{16}\|x\|^2\right),\\
\mathbb E\!\left[
\left|\sum_{i,j}R_{\delta,s_{ij}}\right|^p
\,\middle|\,x,U\right]
\le C_{p,\tau,a,\gamma}(1+\|x\|^{2p}).
\end{gathered}
\tag{A.12}
$$

For the collapsed factor, $\psi(s)=\log h_\tau(s)$ satisfies $\psi'(0)=2d_\tau$ and $|\psi''(s)|\le1/4$: under the true private posterior, $\psi''(s)=\operatorname{Var}_{p_s}[\sigma(-\tau T-s)]-\mathbb E_{p_s}\sigma'(\tau T+s)$, with both terms in [0,1/4]. Therefore

$$
|\log(2h_\tau(s))-2d_\tau s|\le s^2/8,
\quad
\left|\sum_{i,j}\log(2h_\tau(s_{ij}))\right|
\le2ad_\tau\sum_i|x_iZ_i^M|+\frac{a^2}{8}\|x\|^2 .
\tag{A.13}
$$

Equations (A.12)–(A.13), the uniform moments of $Z^M$ and $|\log\sigma(x)|\le\log2+|x|$ give uniform logweight moments of every fixed order, also on any positive definite compact candidate neighborhood. Since $|F_M|\le\max_k|\log\bar w_{Mk}|$, weak convergence plus uniform integrability proves the objective limit.

For the derivative, first average the private variables conditional on active x and data. That conditional kernel has no further explicit dependence on C. Differentiating the active Gaussian density, with the symmetric off-diagonal coordinate counted once, gives

$$
\partial_{c_{ij}}L_M(C)
=\mathbb E\!\left[
F_M\sum_{k=1}^K
\left\{(C^{-1}X_k)_i(C^{-1}X_k)_j-(C^{-1})_{ij}\right\}
\right].
\tag{A.14}
$$

The score is a quadratic polynomial with bounded coefficients on the neighborhood. Hölder's inequality and the preceding high-order moments give uniform integrability of loss times score and justify density differentiation. Joint weak convergence then proves the derivative limit in (A.6). This avoids omitting derivatives of the candidate-dependent private sampling law. It does not assert convergence of the old fixed-private-sample Price estimator. The case $a=0$ or $\gamma=0$ is covered by deleting the corresponding zero variance term. This completes the proof of Proposition A.1.

## A.4. Signed selection with a fixed observation model {#proposal-selection}

Now regard (A.5) as a Gaussian experiment in its own right, with arbitrary fixed $b>0$, $\nu\ge0$, $K\ge2$ and $t\ge0$. This standalone proof permits any strictly positive definite correlation matrix. To apply its equicorrelation comparison on the main text's fixed domain $\Theta_\delta$, choose $0<c\le1-\delta$; the entire path from $I_3$ to $C_c$ then has minimum eigenvalue at least $\delta$. The broader positive-definite calculation below retains $0<c<1$. Replace $\nu_\lambda$ by $\nu$ and write $D=b^2+\nu$. Integrating W,V conditional on X,Y,Z gives one for both noise factors, so the complete observation density is

$$
p_C(y,z)=\phi_3(z)\,
\mathbb E_{X\sim N(0,C)}
\left[\prod_i\sigma(y_iX_i)
 \exp\!\left(bz^\top X-\frac{b^2}{2}\|X\|^2\right)\right].
\tag{A.15}
$$

This density is independent of t and $\nu$, although its finite K criterion is not. Define the logweight without the independent baseline and its coordinate score:

$$
\begin{gathered}
A_k=\sum_i\left[
\log\sigma(Y_iX_{ki})+bZ_iX_{ki}
+\sqrt\nu W_{ki}X_{ki}-\frac D2 X_{ki}^2\right],\\
r_{ki}=Y_i/2-(\sigma(X_{ki})-1/2)+bZ_i+\sqrt\nu W_{ki}-DX_{ki},\\
\pi_k=\frac{e^{A_k+\sqrt tV_k}}{\sum_l e^{A_l+\sqrt tV_l}},
\qquad
\partial_{c_{ij}}\mathcal L_{K,t,\nu}(C)
=-\mathbb E\sum_k\pi_k(1-\pi_k)r_{ki}r_{kj}.
\end{gathered}
\tag{A.16}
$$

To justify the last equality, Gaussian covariance differentiation sums the mixed derivative over particles. The two symmetric covariance entries cancel the usual factor 1/2. Within a particle $A_k$ has zero cross-coordinate second derivative, so the mixed derivative of the negative logsumexp is precisely the displayed integrand. For finite t, the log loss and required derivatives are controlled by Gaussian polynomials; differentiation and integration by parts are therefore valid on positive definite neighborhoods.

The needed high-noise estimate is exact after conditioning. For fixed finite shifts A and $s>0$, condition on $V_l$, $l\ne k$, and write $T_s=\log\sum_{l\ne k}\exp(sV_l+A_l)$. A change of variable gives

$$
s\,\mathbb E_{V_k}[\pi_k(1-\pi_k)]
=\int_{\mathbb R}\sigma'(u)
  \phi\!\left(\frac{u+T_s-A_k}{s}\right)\,du
\le\frac1{\sqrt{2\pi}}.
\tag{A.17}
$$

As $s\to\infty$, $T_s/s\to\max_{l\ne k}V_l$. Dominated convergence first in u and then in the other V coordinates proves

$$
s\,\mathbb E_V[\pi_k(1-\pi_k)]\longrightarrow J_K/K,
\qquad
J_K=K(K-1)\int_{\mathbb R}\phi(v)^2\Phi(v)^{K-2}\,dv
=\mathbb E\max_{k\le K}V_k>0.
\tag{A.18}
$$

The last identity follows by integration by parts in the maximum density. If an integrable mark R and the shifts A are independent of V and their joint law remains fixed as s grows, (A.17) permits another dominated convergence step: the limit with an extra R is $(J_K/K)\mathbb E R$. Shifts and R need not be independent of each other.

Set $d_1=\mathbb E\sigma'(G)$ and define $\kappa(c)$ below. Independence of candidate particles from teacher data, symmetry, and Gaussian integration by parts give the cross-score moment $H_{ij}$. In particular $\mathbb E(Y_iZ_j)=2bd_1c_{*ij}$, $\mathbb E(Z_iZ_j)=b^2c_{*ij}$ for $i\ne j$, and $\mathbb E[(\sigma(X_i)-1/2)X_j]=d_1c_{ij}$.

$$
\begin{gathered}
\kappa(c)=\mathbb E[\tanh(G_1/2)\tanh(G_2/2)],
\qquad
(G_1,G_2)\sim N\!\left(0,\begin{bmatrix}1&c\\c&1\end{bmatrix}\right),\\
H_{ij}:=\mathbb E(r_ir_j)
=\frac{\kappa(c_{*ij})+\kappa(c_{ij})}{4}
 +(2b^2d_1+b^4)c_{*ij}+(2Dd_1+D^2)c_{ij},\\
\sqrt t\,\partial_{c_{ij}}\mathcal L_{K,t,\nu}(C)
\longrightarrow-J_K H_{ij}.
\end{gathered}
\tag{A.19}
$$

The marked result applies because $(A_k,r_{ki}r_{kj})$ is independent of V and has finite moments, with a law fixed as t grows. To integrate along a compact strictly positive definite $C^1$ path, observe that $|r_i|\le1+b|Z_i|+\sqrt\nu|W_i|+D|X_i|$. All candidate coordinate marginals are $N(0,1)$, giving a uniform second-moment bound $R_2$. Equation (A.17) bounds each scaled expected gradient by $KR_2/\sqrt{2\pi}$; multiplied by the path's coordinate total variation, this is integrable. Dominated convergence therefore permits integrating (A.19).

For the independent teacher and equicorrelation path this yields the signed selection result.

$$
\begin{gathered}
C_*=I_3,\qquad C_c=(1-c)I_3+c\mathbf1\mathbf1^\top,\quad 0<c<1,\\
\Delta_K(t)=\mathcal L_{K,t,\nu}(C_c)
                  -\mathcal L_{K,t,\nu}(I_3),\\
\sqrt t\,\Delta_K(t)\longrightarrow-\delta_c,\qquad
\delta_c=3J_K\left[
\frac14\int_0^c\kappa(u)\,du+
\frac{2Dd_1+D^2}{2}c^2\right]>0 .
\end{gathered}
\tag{A.20}
$$

Here $\kappa(0)=0$ and $\kappa'(u)=4\mathbb E[\sigma'(G_1)\sigma'(G_2)]>0$ for $-1<u<1$. Thus for every fixed $K\ge2$,$b>0$,$\nu\ge0$,$c\in(0,1)$, all sufficiently large finite t satisfy $\Delta_K(t)<0$. The selected candidate has strictly positive complete joint risk: its Z marginal covariance is $I+b^2C_c$, different from the teacher's $(1+b^2)I$. Marginalization gives

$$
\operatorname{KL}(P_{I_3}(Y,Z)\|P_{C_c}(Y,Z))
\ge
\operatorname{KL}\!\left(
N(0,(1+b^2)I_3)\|N(0,I_3+b^2C_c)\right)>0 .
\tag{A.21}
$$

This proves a specified candidate beats the teacher, not where a global minimum lies or whether an optimizer reaches it.

## A.5. Strict reversal and transfer to the finite model {#proposal-reversal}

It remains to establish a positive comparison at low noise for the *same fixed K*. Fix b,$\nu$,c as above and first set $t=0$. Conditional on (Y,Z), the weights for each $C\in\{I_3,C_c\}$ are iid and positive, with mean $m_C=p_C(Y,Z)/\phi_3(Z)$. Write $S_{K,C}=K^{-1}\sum_kw_{k,C}$. For any $0<\alpha<\min\{1,1/(1+b^2)\}$, conditional Jensen and completing the square give

$$
\mathbb E[S_{K,C}^{\alpha}\mid Y,Z]
\le m_C^\alpha
\le e^{\alpha\|Z\|^2/2},
\qquad Z\sim N(0,(1+b^2)I_3)\quad(C_*=I_3).
\tag{A.22}
$$

The last bound is integrable and uniform in K. For a small fixed $\delta>0$, convexity gives $S_{K,C}^{-\delta}\le K^{-1}\sum_kw_{k,C}^{-\delta}$. Remove Y by $\sigma(yx)^{-\delta}\le2^\delta e^{\delta|x|}$; then integrate the independent Gaussian Z,W. This yields

$$
\begin{gathered}
\mathbb E w_C^{-\delta}
\le2^{3\delta}\mathbb E_{X\sim N(0,C)}
\exp\!\left[
\delta\sum_i|X_i|+
\frac{\delta D+\delta^2\{b^2(1+b^2)+\nu\}}{2}\|X\|^2
\right]<\infty,\\
\delta D+\delta^2\{b^2(1+b^2)+\nu\}<\frac1{1+2c}.
\end{gathered}
\tag{A.23}
$$

The strict inequality is possible for small $\delta$ and leaves Gaussian integrability margin for the linear term, since both candidate covariance eigenvalues are at most 1+2c. The elementary bound $|\log x|^p\le C(x^\alpha+x^{-\delta})$, for any fixed $p>1$, therefore gives K-uniform log moments. Conditional strong laws and uniform integrability now imply

$$
\Delta_K(0)
\longrightarrow
\mathbb E\log\frac{m_{I_3}(Y,Z)}{m_{C_c}(Y,Z)}
=\operatorname{KL}(P_{I_3}(Y,Z)\|P_{C_c}(Y,Z))>0
\qquad(K\to\infty).
\tag{A.24}
$$

Consequently there is a finite $K_0$ such that every $K\ge K_0$ has $\Delta_K(0)>0$. Choose one such K and hold it fixed. Equation (A.20) then supplies a finite $t_*>0$ with $\Delta_K(t_*)<0$. The two strict signs concern the same K, teacher, candidate, b,$\nu$ and full observation model. This is the Gaussian ordering reversal.

For a reversal within the original finite model, fix $a>0$ and $\tau>0$, use $\lambda=0$ and its source coefficients $b=2ad_\tau$, $\nu=a^2(d_\tau-4d_\tau^2)>0$. Also $v_{\mathrm{priv}}>0$, because $\tau>0$ and $\log\sigma(\tau T)$ is nonconstant. Choose K as just described, then a finite $t_*$ with negative sign, and only then fix $\gamma_*=\sqrt{t_*/v_{\mathrm{priv}}}$. Proposition A.1 applied separately with $\gamma=0$ and $\gamma=\gamma_*$ gives

$$
\begin{aligned}
L_M^{\beta=1,\lambda=0}(C_c)
-L_M^{\beta=1,\lambda=0}(I_3)
&\longrightarrow \Delta_K(0)>0,\\
L_M^{\beta=1-\gamma_*/\sqrt{3M},\lambda=0}(C_c)
-L_M^{\beta=1-\gamma_*/\sqrt{3M},\lambda=0}(I_3)
&\longrightarrow \Delta_K(t_*)<0 .
\end{aligned}
\tag{A.25}
$$

Take the larger of the two finite M thresholds, also ensuring $\beta\in[0,1]$. Every M beyond it has opposite strict orderings under the two proposals, within the same finite observation model (A.1). The candidate has positive complete joint risk for every such M: its Y marginal has $\mathbb E(Y_iY_j)=\kappa(c)>0$, whereas the teacher's is zero. No KL-convergence assertion is needed.

The quantifier order matters: first a fixed sufficiently large finite K, then a fixed finite $t_*$ and $\gamma_*$, then $M\to\infty$. This proof gives no computable $K_0$ or $M_0$, and it does not certify the numerical K8/M32/M128 examples by these unknown thresholds. Their evidence is the separately reported finite-model experiment. It proves neither a fixed-$\beta<1$ high-dimensional theorem nor uniform candidate convergence, argmin consistency, monotone risk improvement with $\lambda$, or optimizer attainment. The $\lambda=0$ source with $a,\tau>0$ has $\nu>0$; the Gaussian $\nu=0$ case is not a $\lambda=0$ source parameter point. All new arguments here are analytical; no new random samples or numerical integrations are used.

