<a id="proposal-proof"></a>

# A. Proposal structure and dependence selection

This appendix proves the conditional proposal limit and the same-observation-model ordering reversal used in the main text. All particle expectations are part of the population criterion; they are not conditional training losses on a fixed dataset. The finite source model is defined in [A.1](#proposal-model), the critical limit in [A.2](#proposal-critical), its global integrability argument in [A.3](#proposal-ui), and the selection and reversal results in [A.4](#proposal-selection)–[A.5](#proposal-reversal).

<a id="proposal-model"></a>

## A.1. A fixed observation model and conditional proposals

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

<a id="proposal-critical"></a>

## A.2. The critical conditional proposal limit

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

<a id="proposal-ui"></a>

## A.3. Global log moments and correlation derivatives

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

<a id="proposal-selection"></a>

## A.4. Signed selection with a fixed observation model

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

<a id="proposal-reversal"></a>

## A.5. Strict reversal and transfer to the finite model

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

<a id="group-proof"></a>

# B. Group-posterior selection and local dependence

## B.1. Model, regularity, and the global anchor

This appendix proves the group-posterior results for three active variables with fixed standard normal marginals and the known local observation channel defined in the main text. Write the candidate as $\theta$ and the teacher as $\eta$. They share the compact spectral domain below; the origin is an interior point. Each local alphabet is finite, every state has positive conditional probability, and the observations are conditionally independent across groups given the active variables.

$$
\begin{gathered}
\theta=(c_{12},c_{13},c_{23}),\qquad
\Theta_\delta=\{\theta:C(\theta)\succeq\delta I_3,\ \operatorname{diag}C(\theta)=\mathbf1\},\quad 0<\delta<1,\\
p_\theta(o)=\int\phi_{C(\theta)}(x)\prod_{i=1}^3\ell_i(o_i\mid x_i)\,dx,\\
Z_i(o_i)=\int\phi(x_i)\ell_i(o_i\mid x_i)\,dx_i,\qquad
q_o(x)=\prod_{i=1}^3\frac{\phi(x_i)\ell_i(o_i\mid x_i)}{Z_i(o_i)},\qquad
Z(o)=\prod_i Z_i(o_i),\\
\rho_\theta(x)=\frac{\phi_{C(\theta)}(x)}{\phi_{I_3}(x)},\qquad
\mathcal L_K(\theta,o)=E_{q_o^{\otimes K}}\!\left[-\log\left\{\frac{Z(o)}K\sum_{k=1}^K\rho_\theta(X_k)\right\}\right],\\
F_K(\theta;\eta)=\sum_o p_\eta(o)\mathcal L_K(\theta,o),\qquad
R(\theta,\eta)=\mathrm{KL}(P_\eta\Vert P_\theta).
\end{gathered}
\tag{B.1}
$$

The proposal does not depend on either correlation parameter: its inputs are the observed groups and the known marginal channel. The importance weight retains the Gaussian determinant and inverse-covariance terms, and its conditional expectation is the full observation probability. Throughout, K is fixed and finite; no bounds are asserted uniformly in K or the spectral boundary.

Because $q_i\leq\phi/Z_i$, all conditional polynomial moments exist. On a compact positive-definite neighborhood, $\log\rho$ and its fixed-order parameter derivatives have Gaussian polynomial envelopes. Derivatives of log-sum-exp are bounded by finite products of these envelopes. The finite alphabet therefore permits differentiation under both integrations and observation sums, to every fixed order used below. The same bounds make each $\mathcal L_K$ continuous and uniformly bounded on the entire candidate domain. All observation probabilities are positive and have a common positive lower bound on that compact domain.

At independence the weights are constant. Conditional Jensen gives

$$
F_K(\theta;0)\geq H(P_0)+R(\theta,0),\qquad
F_K(0;0)=H(P_0).
\tag{B.2}
$$

For $\theta\ne0$, $\rho$ is nonconstant on the proposal's full support, so the average of finitely many independent weights is nondegenerate and Jensen is strict. Thus 0 is the unique global minimizer. Observation identifiability also follows directly from the binary Y component of the specified channel. For each pair, its moment is the Gaussian expectation of the product of two tanh functions; differentiation in the corresponding correlation and Gaussian integration by parts give

$$
\frac{d}{dc}E_c\!\left[\tanh(X_i/2)\tanh(X_j/2)\right]
=\frac14E_c\!\left[\operatorname{sech}^2(X_i/2)\operatorname{sech}^2(X_j/2)\right]>0.
\tag{B.3}
$$

The three pair moments identify the three physical correlations. Consequently, the integrated population negative log likelihood has the teacher as its unique global minimizer. Constant weights at $\theta=0$ do not imply zero gradient for an individual observation or particle bank.

## B.2. Information, shrinkage, and global localization

Let $s$ and $t$ denote the first and second candidate derivatives of $\log\rho$ at 0. For edge coordinates $a=(ij)$, their values, posterior moments, and information matrices are

$$
\begin{gathered}
s=(x_1x_2,x_1x_3,x_2x_3),\qquad
t_{ij,ij}=1-x_i^2-x_j^2,\qquad t_{12,13}=-x_2x_3,\\
\mu_i(o_i)=E_{q_i}X_i,\quad \nu_i(o_i)=E_{q_i}X_i^2,\quad
v_i=E_0\mu_i(O_i)^2,\\
m(o)=E_{q_o}s=(\mu_1\mu_2,\mu_1\mu_3,\mu_2\mu_3),\qquad
V(o)=\operatorname{Cov}_{q_o}(s),\\
J=E_0[mm^\top]=\operatorname{diag}(v_1v_2,v_1v_3,v_2v_3),\qquad
E_{\phi_{I_3}}[ss^\top]=I_3.
\end{gathered}
\tag{B.4}
$$

Cyclic permutations supply the other mixed entries. These are derivatives in physical off-diagonal coordinates, without an extra factor of one half. The posterior identities give $E_0\mu_i=0$, $E_0\nu_i=1$ and $v_i=1-E_0\operatorname{Var}(X_i\mid O_i)<1$. Moreover, the observed Y component has $E[X_iY_i]=2E\sigma'(X_i)>0$, which forces $v_i>0$. Thus J is positive definite.

For one bank, differentiating the negative log of the mean density ratio gives the negative weighted mean of $t+ss^\top$ plus the outer product of the weighted mean score. At 0 the weights are uniform. Averaging the independent particles yields

$$
\begin{gathered}
\nabla\mathcal L_K(0,o)=-m(o),\\
H_K(o):=\nabla^2\mathcal L_K(0,o)
=-E_{q_o}t-(1-K^{-1})V(o)
=H_\infty(o)+K^{-1}V(o),\\
H_\infty(o)=\nabla^2[-\log p_\theta(o)]_{\theta=0},\qquad
E_0V=I_3-J,\\
H_K:=\nabla_\theta^2 F_K(0;0)=J+(I_3-J)/K,\qquad
\nabla_\eta\nabla_\theta F_K(0;0)=-J.
\end{gathered}
\tag{B.5}
$$

Here the mean Hessian follows from the complete-data identity $E_{\phi_I}(t+ss^\top)=0$. The teacher derivative acts on the observation distribution, whose score is $m$; it does not act on $q$. This gives the minus sign in the last identity. Positive definiteness and the implicit function theorem now provide a unique smooth stationary branch near 0, with positive Hessian and

$$
\begin{gathered}
\widetilde\theta_K(\eta)=A_K\eta+O(\|\eta\|^2),\qquad
A_K=H_K^{-1}J,\\
(A_K)_{aa}=r_a=\frac{KJ_{aa}}{1+(K-1)J_{aa}}\in(0,1),\\
v_1=v_2=v_3=v:\qquad j=v^2,\quad J=jI_3,\quad
r=\frac{Kj}{1+(K-1)j}.
\end{gathered}
\tag{B.6}
$$

The branch is also the unique global minimizer for all sufficiently small teachers. To prove this stronger statement, choose a closed ball of radius $\rho$ inside the candidate interior, where the Hessian is uniformly positive for nearby teachers. Define the compact set and gap

$$
A_\rho=\{\theta\in\Theta_\delta:\|\theta\|\geq\rho/2\},\qquad
\gamma=\min_{\theta\in A_\rho}\{F_K(\theta;0)-F_K(0;0)\}>0.
\tag{B.7}
$$

If this set is empty, localization is already established. Otherwise continuity and the unique global anchor give the positive gap. The finite alphabet and bounded losses give uniform convergence of $F_K(\cdot;\eta)$ to $F_K(\cdot;0)$ as $\eta\to0$. Restrict the teacher neighborhood until that uniform difference is below $\gamma/3$. Every point in $A_\rho$ then has objective larger than the origin by more than $\gamma/3$, so every global minimizer lies in the open ball of radius $\rho/2$. Existence follows from compactness; strict convexity on the larger ball gives uniqueness. The resulting interior minimizer is the implicit-function branch. No numerical radius or arbitrary-start optimizer guarantee follows. The same argument applies to integrated likelihood and the matched ridge below, using their unique anchor at 0 and positive mean Hessians.

## B.3. The three-edge term and the ridge distinction

For identical channels, group reflection and permutation symmetries constrain the branch. Reflection of group i changes the signs of exactly the two correlations incident to i. At degree two, the only allowed term in coordinate 12 is therefore $\eta_{13}\eta_{23}$. To compute its coefficient, name the edges $a=12$, $b=13$, $c=23$ and write $R_{ab}$ for the corresponding derivative of $\rho$ at 0. Direct differentiation of the Gaussian density gives

$$
R_a=s_a,\qquad R_{ab}=(x_1^2-1)x_2x_3,\qquad
R_{abc}=\prod_{i=1}^3(x_i^2-1).
\tag{B.8}
$$

Bars denote averages over K particles. The third derivative of the negative logarithm is $-\bar R_{abc}$ plus the three terms $\bar R_{ab}\bar s_c$, minus twice $\bar s_a\bar s_b\bar s_c$. Under the independent teacher, the first four terms have zero expectation: posterior averaging recovers standard normal marginal moments, including $E_0(\nu_i-1)=0$. In the last term, the three particle indices are all equal, exactly two equal, or all distinct. Their conditional-then-observation expectations are respectively 1, $v^2$ and $v^3$. Counting these ordered choices gives

$$
T_K=\frac{1+3(K-1)v^2+(K-1)(K-2)v^3}{K^2},\qquad
F_{\theta_a\theta_b\theta_c}(0;0)=-2T_K.
\tag{B.9}
$$

The mixed teacher derivative is equally explicit. Formula (B.5) gives $H_{K,ab}(o)=\mu_2\mu_3[1-(1-K^{-1})(\nu_1-\mu_1^2)]$. Multiplication by the teacher score $m_c$ and averaging independent groups yields the following $B_K$. The pure second teacher derivative of the origin score in coordinate $a$ is zero: $E_\eta\mu_1\mu_2$ depends only on the teacher's 12 marginal, not on $\eta_{13}$ or $\eta_{23}$.

$$
B_K:=F_{\theta_a\theta_b\eta_c}(0;0)
=\frac{v^2[1+(K-1)v]}K,\qquad
F_{\theta_a\eta_b\eta_c}(0;0)=0,\qquad
H=\frac{1+(K-1)v^2}K.
\tag{B.10}
$$

Insert the branch expansion into its score equation. The coefficient of $\eta_b\eta_c$ is $H\chi_K-2T_Kr^2+2B_Kr$; hence

$$
\begin{gathered}
\widetilde\theta_{12}=r\eta_{12}+\chi_K\eta_{13}\eta_{23}+O(\|\eta\|^3),\\
\chi_K=\frac{2r(T_Kr-B_K)}H
=-\frac{2K(K-1)v^5(1-v)^2}{[1+(K-1)v^2]^3},\\
\eta=(t,t,0):\qquad
\widetilde\theta_{12}=\widetilde\theta_{13}=rt+O(t^3),\qquad
\widetilde\theta_{23}=\chi_Kt^2+O(t^4).
\end{gathered}
\tag{B.11}
$$

The last remainder uses reflection of group 1: the first two branch coordinates are odd in t and the third is even. The induced third coordinate is a marginal correlation, not a conditional-independence graph edge. For K>1 it is negative to this order; for K=1 the coefficient vanishes.

Compare mean integrated NLL plus a deterministic quadratic penalty:

$$
\begin{gathered}
\mathcal L_{\rm ridge}(\theta,o)=-\log p_\theta(o)+\tfrac12\theta^\top\Lambda_K\theta,
\qquad \Lambda_K=(I_3-J)/K,\\
\nabla\mathcal L_{\rm ridge}(0,o)=-m(o),\qquad
\nabla^2\mathcal L_{\rm ridge}(0,o)=H_\infty(o)+\Lambda_K,\\
\chi_{\rm ridge}=\frac{2v^3r(r-1)}H
=-\frac{2K^2v^5(1-v^2)}{[1+(K-1)v^2]^3}.
\end{gathered}
\tag{B.12}
$$

Its mean Hessian and first-order branch match, but its per-observation Hessian generally does not. For the coefficient, integrated likelihood has the limits $T_\infty=B_\infty=v^3$, while the quadratic penalty contributes no third derivative; substitution in the preceding score equation gives (B.12). Thus the objectives differ beyond their shared first-order response, including at K=1. A sum-NLL convention requires multiplying the penalty by N. The norm is in the three physical coordinates; the squared Frobenius norm of $C-I$ is twice as large. [Appendix C](#risk-proof) derives the shared first-order sampling risk without identifying the two objectives.

<a id="risk-proof"></a>

# C. Sampling risk and the scope of numerical outputs

## C.1. Exact empirical minimization under local alternatives

Use the fixed model, compact domain, and notation of [Appendix B](#group-proof). For all sufficiently large N, observations are independent from the teacher $\eta_N=h/\sqrt N$, with fixed h. The estimator in this section is any measurable global minimizer of the empirical objective after particle randomness has been averaged exactly. It is distinct from the fixed-program output evaluated in the experiments.

$$
\eta_N=h/\sqrt N,\qquad
\widehat F_{K,N}(\theta)=\frac1N\sum_{n=1}^N\mathcal L_K(\theta,O_{Nn}),\qquad
\widehat\theta_{K,N}\in\operatorname*{argmin}_{\theta\in\Theta_\delta}\widehat F_{K,N}(\theta).
\tag{C.1}
$$

Writing the objective as a finite sum against empirical observation frequencies shows uniform convergence to $F_K(\cdot;0)$. The unique global anchor therefore forces every global empirical minimizer to converge in probability to 0. Uniform convergence of local Hessians to $H_K$ and the strict-gap argument of Appendix B show that, with probability tending to one, this minimizer is unique, interior, and in a fixed strongly convex neighborhood.

The observation score at 0 is m. Finite-alphabet smoothness gives its mean under the moving teacher and a triangular-array central limit theorem:

$$
\begin{gathered}
E_{\eta_N}m=J\eta_N+O(\|\eta_N\|^2),\qquad
\operatorname{Cov}_{\eta_N}(m)\longrightarrow J,\\
\bar m_N=\frac1N\sum_{n=1}^N m(O_{Nn}),\qquad
\sqrt N\bar m_N=Jh+\xi_N+o(1),\qquad \xi_N\Rightarrow\xi\sim N(0,J),\\
\sqrt N\widehat\theta_{K,N}=H_K^{-1}\sqrt N\bar m_N+o_p(1)
\Rightarrow H_K^{-1}(Jh+\xi).
\end{gathered}
\tag{C.2}
$$

The last line follows by integrating the empirical Hessian along the segment from 0 to the stationary point; its origin score is exactly $-\bar m_N$. The same proof applies to matched ridge, with $H_K$, and to integrated maximum likelihood, with J. On shared data these expansions use the same random vector:

$$
\begin{gathered}
\sqrt N\widehat\theta_{{\rm ridge},N}
=H_K^{-1}\sqrt N\bar m_N+o_p(1),\qquad
\sqrt N\widehat\theta_{{\rm ML},N}
=J^{-1}\sqrt N\bar m_N+o_p(1),\\
\sqrt N(\widehat\theta_{K,N}-\widehat\theta_{{\rm ridge},N})\longrightarrow_p0.
\end{gathered}
\tag{C.3}
$$

A local output known to converge to 0 has the same distributional expansion if its interior score residual is $o_p(N^{-1/2})$. A fixed numerical tolerance does not imply that condition. Even such a distributional expansion alone does not imply the expectation result below.

## C.2. Expected joint risk and the three regions

Positive finite-alphabet probabilities give the local KL expansion

$$
R(\theta,\eta)=\tfrac12(\theta-\eta)^\top J(\theta-\eta)
+O((\|\theta\|+\|\eta\|)^3).
\tag{C.4}
$$

To justify taking expectations, choose a fixed neighborhood of the independent observation-frequency vector small enough to preserve the global gap and a local Hessian lower bound $aI$ with $a>0$. Under the local alternatives, empirical frequencies lie outside that neighborhood with probability at most $c_1\exp(-c_2N)$, for all sufficiently large N. On the good event, localization and the score equation imply

$$
\|\widehat\theta_{K,N}\|\leq a^{-1}\|\bar m_N\|,\qquad
\sup_N E\|\sqrt N\bar m_N\|^q<\infty\quad(1\leq q<\infty).
\tag{C.5}
$$

The moment bound holds because m has finite support and its moving mean is $O(N^{-1/2})$. On the bad event the parameter domain and joint KL are bounded; any fixed power of N times its exponentially small probability tends to zero. In particular, scaled parameter moments of order greater than two are uniformly bounded. This gives uniform integrability of the limiting quadratic form, while the expected cubic remainder in (C.4), multiplied by N, is $O(N^{-1/2})$ on the good event and negligible on its complement. Consequently,

$$
\begin{gathered}
W=(A_K-I_3)h+H_K^{-1}\xi,\qquad
NR(\widehat\theta_{K,N},\eta_N)\Rightarrow\tfrac12 W^\top JW,\\
\lim_{N\to\infty}NER(\widehat\theta_{K,N},\eta_N)
=\tfrac12\!\left[
h^\top(A_K-I_3)^\top J(A_K-I_3)h
+\operatorname{tr}(JH_K^{-1}JH_K^{-1})\right].
\end{gathered}
\tag{C.6}
$$

Matched ridge has the same limit by the same argument. In the identical-channel case, define S as below. The integrated estimator and the constant independence estimator supply the two reference risks:

$$
\begin{gathered}
S=j\|h\|^2,\qquad
\mathcal R_K(S)=\tfrac12[(1-r)^2S+3r^2],\\
\mathcal R_{\rm ML}=\tfrac32,\qquad \mathcal R_I(S)=\tfrac S2,\\
\mathcal R_K<\mathcal R_I\ \Longleftrightarrow\ S>\frac{3r}{2-r},\qquad
\mathcal R_K<\mathcal R_{\rm ML}\ \Longleftrightarrow\ S<\frac{3(1+r)}{1-r}.
\end{gathered}
\tag{C.7}
$$

These are limits of expected $N$ times joint KL along changing teachers. They neither locate finite-N crossings nor describe a fixed nonzero teacher as N grows without bound. The numerical study tests selected predicted directions using frozen programs; it does not establish a continuous finite-N risk curve. Matching the first-order ridge constant is not equivalence at finite N. For a general algorithm, an additional uniform-integrability argument is required to transfer a distributional approximation into expected risk.

## C.3. A fixed teacher and the remaining bias

There exists $\varepsilon>0$ such that, for every fixed teacher $\eta_0$ with $0<\|\eta_0\|<\varepsilon$, exact empirical global minimization is consistent for the unique group-posterior population branch, while integrated maximum likelihood is consistent for the teacher. More precisely, for iid observations from this single fixed teacher,

$$
\begin{gathered}
\widehat\theta_{K,N}\longrightarrow_{\rm a.s.}\widetilde\theta_K(\eta_0),\qquad
ER(\widehat\theta_{K,N},\eta_0)\longrightarrow
R(\widetilde\theta_K(\eta_0),\eta_0)>0,\\
\widehat\theta_{{\rm ML},N}\longrightarrow_{\rm a.s.}\eta_0,\qquad
ER(\widehat\theta_{{\rm ML},N},\eta_0)\longrightarrow0.
\end{gathered}
\tag{C.8}
$$

Indeed, the finite-alphabet strong law gives uniform empirical-objective convergence on the compact domain. A unique population minimizer then identifies every measurable empirical argmin by the usual compact separation argument: for every neighborhood of that minimizer, the continuous population objective has a strictly positive gap on its compact complement. For sufficiently large N the empirical gap remains positive.

Shrink $\varepsilon$ to stay in the unique-global-branch neighborhood. Since each eigenvalue of $A_K$ is strictly between zero and one, $A_K-I$ is invertible. The expansion $\widetilde\theta_K(\eta)-\eta=(A_K-I)\eta+O(\|\eta\|^2)$ is therefore nonzero for every sufficiently small nonzero teacher. Identifiability gives the strict KL inequality in (C.8). Continuity and boundedness of risk on the compact candidate domain give the expectation limits by dominated convergence. Integrated likelihood uses identifiability directly to obtain its unique population optimum at $\eta_0$.

It follows that, for each such fixed teacher, some finite $N_0$ makes integrated exact estimation strictly better in expected risk than fixed-K group exact empirical minimization for every $N\geq N_0$. This is a pointwise existence statement: neither $\varepsilon$ nor $N_0$ is numerically supplied, risk need not be monotone, and no unique crossing is asserted. It does not establish the limit of the current numerical program or the risk floor of a different high-dimensional prior-proposal model.

## C.4. What the quadratic control preserves

The implementation uses a control that preserves the origin derivatives of the exact single-observation objective. This identity explains a local connection to the theory and also identifies its boundary. For an arbitrary fixed K-particle bank at one observation, let $f_{\rm raw}$ be the negative log mean importance weight in (B.1), and let bars denote bank averages. Using the exact moments of its actual proposal, define

$$
\begin{gathered}
f_u(\theta,o)=f_{\rm raw}(\theta,o)
+\overline{\log\rho_\theta}-E_{q_o}\log\rho_\theta,\\
D(o)=\bar s\bar s^\top-\overline{ss^\top}+(1-K^{-1})V(o),\qquad
f_c(\theta,o)=f_u(\theta,o)-\tfrac12\theta^\top D(o)\theta,\\
\nabla f_c(0,o)=-m(o),\qquad
\nabla^2f_c(0,o)=-E_{q_o}t-(1-K^{-1})V(o)=H_K(o).
\end{gathered}
\tag{C.9}
$$

To verify the identity, uniform weights at 0 give the raw score $-\bar s$ and raw Hessian $-\bar t-\overline{ss^\top}+\bar s\bar s^\top$. The first control replaces $-\bar s$ by $-m$ and $-\bar t$ by $-E_qt$; subtracting D then gives (C.9), for every observation and every bank. Under independent proposal sampling, $E[\bar s\bar s^\top]=mm^\top+V/K$ and $E[\overline{ss^\top}]=mm^\top+V$, so $E D=0$. Both controls have zero conditional expectation and preserve $\mathcal L_K$. They do not make the group and ridge Hessians equal for each observation.

For a fixed finite collection of banks for each observation-alphabet state, sampled independently of the data and held fixed as N varies, the controlled scalar is smooth near 0, with finite local derivative bounds conditional on the banks. Its per-observation origin score and Hessian are exactly those in (C.9). Hence its consistently selected local stationary branch has the same first-order expansion (C.2); the number of banks need not increase for this conditional local statement. This does not preserve the global Jensen gap of the fixed-bank scalar or show that arbitrary starts and finite solver budgets select that branch. Passing from conditional distributional limits to risk averaged over random banks requires integrable control over the banks or an appropriate bound on the output moments.

Finally, the identities assume exact moments of the actual proposal. Fixed quadrature order, finite-bit sampling, and fixed solver tolerance do not disappear merely because N increases. Sufficient local conditions for an asymptotically exact numerical extension include per-observation score error $o(N^{-1/2})$, Hessian error $o(1)$, controlled higher derivatives, localization, and solver score residual $o_p(N^{-1/2})$; expectation claims additionally need uniform integrability. Evaluating the same limiting $N\times\mathrm{KL}$ requires risk error $o(1/N)$. The finite study instead reports its achieved precision diagnostics and all frozen-program risks. Agreement of its risk directions with (C.7) tests the combined data variation, nonlinear risk, constraints, higher-order terms and numerical outputs; because the origin derivatives are preserved by construction, it is not an independent rediscovery of those derivatives.
