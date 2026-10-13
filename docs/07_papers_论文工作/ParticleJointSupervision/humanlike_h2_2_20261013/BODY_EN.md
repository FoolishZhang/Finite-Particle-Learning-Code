---
title: "Finite Prior-Particle Likelihood Objectives Can Reward Risk-Increasing Dependence"
lang: en-US
---

# Abstract

Latent-variable models are often fitted by estimating the marginal likelihood with an average of $K$ conditional likelihoods at independent prior draws and minimizing the expected negative logarithm of that estimate. For fixed $K$ this is a criterion of its own, not a noisier copy of the log likelihood: estimating it more precisely does not turn it into one. We ask which dependence structures it favours, and whether they agree with the KL risk of the integrated model. In a Gaussian--logistic model we append $n$ observed fair coins. After integration they leave every candidate's risk unchanged; inside the particle average, their per-particle likelihoods change which particles dominate. As $n$ grows, with the active Gaussian marginals fixed and $K\ge2$ at most polynomial in $n$, we identify the first term that distinguishes correlations, uniformly and for the actual non-Gaussian weights. It is a strictly concave potential that rewards larger variance of the active log-likelihood across prior draws. Population near-minimizers that resolve this shrinking term approach the spectral boundary of the correlation class and retain positive risk. With no appended block and all parameters free, a computer-assisted proof gives an eight-particle population local minimum whose single-variable marginals are all correct but whose joint KL is about $0.0104$; from that point, every sufficiently small nonzero step along a marginal-preserving path lowers the risk and raises the objective. Fitting the finite objective accurately, or fitting the marginals correctly, therefore does not settle which dependence is learned.

# Introduction

Many latent-variable models have an intractable marginal likelihood $p_\theta(y)=E_{Z}\,p_\theta(y\mid Z)$. A simple replacement draws $K$ independent prior particles $Z_1,\ldots,Z_K$ and minimizes the expected negative logarithm of their average likelihood. It appears in simulated maximum likelihood [@lee1995], in prior-sampled learning of binary latent models [@tipping1998], and as the importance-weighted bound with its proposal fixed to the prior [@burda2016]. Because the logarithm is concave, the objective is biased, and the bias depends on the parameters [@maddison2017; @sung2007]. At fixed $K$ it is therefore a different criterion from the likelihood. More outer repetitions estimate the same finite-$K$ objective more precisely; they do not turn it into integrated log likelihood.

This paper asks a question that the general bias statement leaves open: *which dependence structure does a finite prior-particle objective prefer, and when does that preference disagree with the KL risk $D(P\Vert Q_\theta)$ of the integrated model $Q_\theta$?*

![The fair-coin control. Three binary variables with independent latent logits ($C_*=I$), $K=100$ particles per logarithm. (a) Population gap $g_n=F_n(I)-F_n(C_b)$ between the truth and the equicorrelated candidate $C_b$; positive values mean the finite objective prefers $C_b$. (b) Curvature $h_n$ of the objective at the truth, where $\nabla^2F_n(I)=h_nI_3$ in the three free correlations; negative values make the truth a local maximum. At $n=256$, the appended fair coins' per-particle likelihoods stay inside the particle average; active observations, active particles and candidates are shared with $n=0$. Bars are 99% simultaneous intervals for each panel under ideal iid sampling and stated numerical allowances (Appendix C.2, C.4); positivity of $h_0$ is also proved analytically (C.3). The footer reports one matched 143-point grid selection (C.1).](figures/finite_instance.png){width=100%}

**A controlled comparison.** Take three binary observations whose latent Gaussian logits are independent, and use $K=100$ particles. Now append $n$ further observed binary variables that are fair coins independent of everything else. Each coin has its own latent logit $G_j\sim N(0,1)$ and emission probability $\sigma(u_jG_j)$ for outcome $u_j\in\{-1,1\}$. Averaged over the prior, every coin has probability $1/2$ under every candidate. Each candidate's integrated model is therefore its old model times $2^{-n}$, and every risk is unchanged. Inside the particle average the coins act differently. Particle $k$ carries the factor $\prod_j\sigma(u_jG_{kj})$, which varies from particle to particle even though its prior mean is $2^{-n}$. These factors decide which particles dominate the average, and hence how the active likelihoods compete. Integrating the coins out *before* averaging would instead contribute the common constant $2^{-n}$ and change nothing.

Figure 1 compares $n=0$ with $n=256$ on the same active observations, active particles and candidates. Without coins, the finite objective prefers the true independent model to an equicorrelated candidate $C_b$ with all correlations $.9$, and the truth is a strict local minimum. With 256 coins the preference reverses: $C_b$ scores better, the truth becomes a local maximum, and a matched grid search returns $C_b$, whose risk is $.031$ nats.

We regard this construction as a control, not as a recommended practice. The appended block is known to be independent, and its likelihood factorizes exactly. Its value is that it changes the estimator while preserving the active likelihood ratios and every candidate's KL risk. Whatever changes in Figure 1 is therefore the criterion's own preference.

**Why a fine scale is needed.** As $n$ grows, nuisance log-weights spread on a scale $\sqrt n$, and the average is dominated by the particle with the largest nuisance weight. With fixed active marginals, the expected loss of one prior particle is the same for every candidate correlation matrix, so the leading term of the objective does not see dependence at all. Heuristically, the leading selection term in this regime comes from events where the two leading particles are nearly tied, and such events have frequency of order $w_{n,K}\asymp\sqrt{\log K/n}$. Selection therefore happens at a scale that shrinks with $n$, and this has two consequences. An approximation that gets the objective right to $o(1)$ can be silent about which correlations are chosen; it must be accurate relative to $w_{n,K}$. And identifying the preference requires controlling near-ties for the actual, non-Gaussian log-likelihood weights, uniformly over candidates. A Gaussian model of the log-weights formally predicts the large-$K$ coefficient but does not supply this control (Section 6).

Theorem 1 supplies it. The resulting potential is strictly concave and has a direct reading: it rewards correlation matrices under which the active log-likelihood of an observation varies more across prior draws. For two variables with true correlation zero, the exact likelihood pulls a fitted correlation back toward zero, while the limiting particle objective pushes it away. Theorem 2 combines concavity with the classical geometry of correlation matrices. Near-minimizers that resolve the scale $w_{n,K}$ are driven to the spectral boundary of the class and keep their risk bounded away from zero.

**Without an appended block.** The coins make the mechanism visible, but they leave a natural question. Suppose there is no designated independent block, every bias and loading is learnable, and a local minimum of the finite objective already reproduces every observation marginal. Does the objective then at least not oppose dependence changes that would lower the risk? Theorem 3 answers no for a specific eight-item, two-factor model with $K=8$. A computer-assisted proof certifies a rank-one strict local minimum of the exact population objective in all 24 parameters, modulo common rotations. All marginals are correct and the joint KL is about $.0104$. Along a path that rotates loading rows without changing their lengths, every sufficiently small nonzero step lowers the risk and raises the objective. The two parts thus answer successive questions about one criterion: an asymptotic statement about *selection* in a controlled family, then a finite, *local* statement about geometry without that control. Neither is derived from the other.

**Contributions.**

- A uniform selection-scale theorem for the actual non-Gaussian weights, covering every fixed $K\ge2$ and $K$ growing polynomially in $n$, together with a conditional-variance reading of the limiting potential (Theorem 1).
- Its geometric consequence: a spectral-boundary preference and positive limiting risk for population near-minimizers with regret $o(w_{n,K})$ (Theorem 2).
- A finite-scale analysis: a signed fixed-$K$ correction isolates the same-particle coupling that scalar overlap misses, and explicit bounds certify the leading sign of one coordinate at $n=4096$ (Section 4).
- A computer-assisted certificate of a marginal-correct, risk-positive local minimum with a same-root, marginal-preserving conflict, persisting over a small explicit teacher family (Theorem 3 and its corollary).

The main theory concerns population objectives with particles drawn from the prior; Section 4.3 adds guarantees for one specified finite program. None describes where a stochastic optimizer converges.

# Prior-Particle Objectives and a Controlled Augmentation

For latent $Z\sim p(z)$ and conditional likelihood $p_\theta(y\mid z)$, the $K$-particle objective is $E_{Y\sim P}E_{Z_{1:K}}[-\log K^{-1}\sum_kp_\theta(Y\mid Z_k)]$, with iid prior particles independent of $Y$. At $K=\infty$ it becomes the integrated cross entropy. Our first model makes the active correlations the only unknowns. Fix $m\ge2$, a finite mean $\mu$ and $0<\delta<1$:

$$
\begin{aligned}
\mathcal E_\delta^{(m)}
&=\{C=C^\top:\operatorname{diag}C=\mathbf1,\ C\succeq\delta I\},\\
X_A&=\mu+L_CZ,\qquad L_CL_C^\top=C,\\
X_{m+j}&=\tau_jG_j,\\
P(Y_i=y_i\mid X_i)&=\sigma(y_iX_i),\qquad y_i\in\{-1,1\}.
\end{aligned}
\tag{1}
$$

All Gaussian inputs are independent standard normals before applying $L_C$, and observations are conditionally independent given their logits. Active variances and the mean are known, and active and nuisance logits are uncorrelated. We write $U$ for the nuisance observations and assume the teacher $C_*$ lies in the relative interior, $C_*\succ\delta I$. Because each nuisance coin is fair after integration,

$$
Q_{C,\tau}^{(n)}(y,u)=2^{-n}Q_C^A(y),\qquad
\mathcal R(C)=D(Q_{C_*}^A\Vert Q_C^A).
\tag{2}
$$

This is also the KL risk of the complete observation. Changing nuisance scales at fixed $n$ leaves the whole observation law unchanged; changing $n$ leaves active likelihood ratios and the KL unchanged. With $K$ iid prior particles $(Z_k,G_k)$ independent of the observation,

$$
\begin{aligned}
\widehat Q_{C,\tau,K}^{(n)}(y,u)
&=K^{-1}\sum_{k=1}^Kp_{C,\tau}(y,u\mid Z_k,G_k),\\
F_{n,K}(C;\tau)&=E_{Y,U,Z,G}[-\log\widehat Q_{C,\tau,K}^{(n)}(Y,U)].
\end{aligned}
\tag{3}
$$

Each realized mixture $\widehat Q$ is a normalized distribution, so $F-H(P)=E_{Z,G}D(P\Vert\widehat Q)$. The finite objective scores a random mixture, while the risk scores its average. Suppressing fixed arguments, the gap between the two is the classical Jensen term:

$$
\begin{aligned}
F_P(C)&=H(P)+\mathcal R(C)+\mathcal J_P(C),\\
\mathcal J_P(C)&=E_P[\log Q_C\\
&\qquad -E_{Z,G}\log\widehat Q_C]\ge0.
\end{aligned}
\tag{4}
$$

Here $P=Q_{C_*}^{(n)}$. Nonnegativity of $\mathcal J_P$ is standard; what matters for learning is how $\mathcal J_P$ varies with $C$, which nonnegativity alone does not determine. This is consistent with properness of the logarithmic score, because $F$ is the expected score of a random predictive rule rather than of $Q_C$. Two further facts fix what the augmentation does and does not do. Transporting the prior, emissions and particles by an invertible latent map leaves every realization of $\widehat Q$ unchanged (Appendix A.1), whereas appending observed coordinates changes the estimator. And at $K=1$ the objective depends only on the fixed marginals, so it is constant in $C$: any dependence selection requires $K\ge2$.

# Which Correlations the Finite Objective Prefers

## The Selection Scale

Write $b_i=EY_i$ and $\kappa_{ij}(c)$ for the covariance of $Y_i,Y_j$ when their latent correlation is $c$. Fixed marginals make $b_i$ independent of $C$, and Price's identity [@price1958] gives

$$
\begin{aligned}
\kappa_{ij}(0)&=0,\\
\kappa_{ij}'(c)&=\tfrac14 E_C[\operatorname{sech}^2(X_i/2)\\
&\qquad\quad{}\times\operatorname{sech}^2(X_j/2)]>0.
\end{aligned}
\tag{5}
$$

So observed covariances identify latent correlations. Differentiating the objective weights a per-particle *dependence mark* by how much that particle shares the average. With $\pi_k$ the normalized **complete** particle likelihoods, which include the nuisance factors,

$$
\begin{aligned}
\partial_{c_{ij}}F_{n,K}
&=-E\sum_k\pi_k(1-\pi_k)H_{k,ij},\\
H_{k,ij}&=Y_iY_j\sigma(-Y_iX_{ki})\sigma(-Y_jX_{kj}),\\
EH_{k,ij}&=\{\kappa_{ij}(C_*)+\kappa_{ij}(C)\}/4.
\end{aligned}
\tag{6}
$$

The mean of the mark follows because labels and candidate particles are independent. At finite $n$, however, a particle's weight and its mark are coupled, which will matter in Section 4.

Set nuisance scales to one, write $F_{n,K}^*(C)=F_{n,K}(C;\mathbf1)$, and define

$$
\begin{gathered}
v=\operatorname{Var}\log\sigma(G),\qquad J_K=E\max_{k\le K}G_k,\\
w_{n,K}=J_K/\sqrt{nv}.
\end{gathered}
\tag{7}
$$

Here $J_K$ is the expected maximum of $K$ standard Gaussians, with $J_K\sim\sqrt{2\log K}$ and $J_K^2\asymp1+\log K$ for integers $K\ge2$ (Appendix A.2). Hence $w_{n,K}\asymp\sqrt{\log K/n}$ in the polynomial range below, and $w_{n,K}\asymp n^{-1/2}$ at fixed $K$. A heuristic explains the scale. If the $K$ nuisance log-weights were Gaussian with standard deviation $\sqrt{nv}$, the density at zero of the gap between the largest two would be $K(K-1)\int\phi^2\Phi^{K-2}/\sqrt{nv}=w_{n,K}$. An active log-likelihood difference of order one changes the dominant particle only on such near-ties.

## The Limiting Potential

**Theorem 1 (uniform direction and potential).** Fix $m,\mu,\delta$, $C_*\in\operatorname{relint}\mathcal E_\delta^{(m)}$ and $C_0,p>0$. As $n\to\infty$, uniformly over $C\in\mathcal E_\delta^{(m)}$ and integers $2\le K\le C_0n^p$,

$$
\begin{aligned}
\frac{\partial_{c_{ij}}F_{n,K}^*(C)}{w_{n,K}}
&\longrightarrow-\frac{\kappa_{ij}(C_*)+\kappa_{ij}(C)}4,\\
\frac{F_{n,K}^*(C)-F_{n,K}^*(I)}{w_{n,K}}
&\longrightarrow V(C),\\
V(C)&=-\frac14\sum_{i<j}\Bigl[\kappa_{ij}(C_*)c_{ij}\\
&\qquad\qquad+\int_0^{c_{ij}}\kappa_{ij}(u)\,du\Bigr].
\end{aligned}
\tag{8}
$$

The proof (Appendix A.1--A.2) works with the actual log-sigmoid nuisance weights. It controls nearly tied pairs of nuisance log-weights through uniform moderate-tail and density estimates valid for polynomial $K$, together with a common envelope for active scores over the fixed-dimensional correlation class. Compactness alone does not give control at a shrinking scale. Integrating the gradient along the feasible segment from $I$ to $C$ gives $V$.

Three readings make (8) concrete.

*The Jensen gap absorbs the risk.* Differences of $F^*_{n,K}$ vanish at rate $w_{n,K}$, while risk differences $\mathcal R(C)-\mathcal R(I)$ do not depend on $n$. By (4), the $C$-dependence of the Jensen term therefore cancels the risk differences to leading order. What survives at scale $w_{n,K}$ is $V$, which can rank correlations differently from observation risk.

*What $V$ rewards.* Let $A_C(y,Z)=\sum_i\log\sigma(y_iX_i)$ be the active log-likelihood of label $y$ at one prior draw, and $T(C)=E_Y\operatorname{Var}_ZA_C(Y,Z)$ its variance across prior draws, averaged over true labels. Price's identity gives $\partial_{c_{ij}}T=\{\kappa_{ij}(C_*)+\kappa_{ij}(C)\}/2$, so

$$
V(C)=-\tfrac12\{T(C)-T(I)\}.
$$

The limiting objective prefers correlation matrices under which an observation's active log-likelihood is more variable across prior draws. This is the conditional variance of the active log-likelihood, not the variance of the likelihood itself, of the nuisance weights, or of a stochastic gradient (Appendix B.1).

*Two variables.* With centred variables, true correlation $\theta$ and spin moment $r(\rho)$, the exact cross entropy has derivative $r'(\rho)\{r(\rho)-r(\theta)\}/\{1-r(\rho)^2\}$. Its descent moves $\rho$ toward $\theta$. At $\theta=0$, the limiting particle gradient $-\kappa(\rho)/4$ has the opposite sign, so descent moves a nonzero correlation away from zero. More generally, by (5) the Hessian of $V$ in correlation coordinates is diagonal with strictly negative entries: $V$ is strictly concave.

## Spectral Boundary and Risk

A strictly concave function on a compact convex set attains its minimum only at extreme points. For correlation matrices, those points are constrained by a classical rank bound [@litam1994].

**Theorem 2 (spectral preference and risk).** Under Theorem 1, let $\mathcal M=\arg\min_{\mathcal E_\delta^{(m)}}V$ and $r_{\max}=\lfloor(\sqrt{8m+1}-1)/2\rfloor$. Every $C\in\mathcal M$ satisfies

$$
\operatorname{rank}\left(\frac{C-\delta I}{1-\delta}\right)\le r_{\max}.
\tag{9}
$$

At least $m-r_{\max}$ eigenvalues therefore equal $\delta$, while $C$ remains positive definite. For integer sequences $2\le K_n\le C_0n^p$ eventually, if $\widehat C_n\in\mathcal E_\delta^{(m)}$ has population regret $F_{n,K_n}^*(\widehat C_n)-\inf_C F_{n,K_n}^*(C)=o(w_{n,K_n})$, then

$$
\begin{gathered}
\operatorname{dist}(\widehat C_n,\mathcal M)\to0,\\
\liminf_n\mathcal R(\widehat C_n)
\ge\varepsilon_*,\\
\varepsilon_*:=\min_{C\in\mathcal M}\mathcal R(C)>0.
\end{gathered}
\tag{10}
$$

The constant depends on $(m,\mu,\delta,C_*)$. The affine map in (9) sends $\mathcal E_\delta^{(m)}$ to a set of correlation matrices, where an extreme point of rank $r$ satisfies $r(r+1)/2\le m$. The monotonicity in (5) identifies latent correlations from observed moments, and compactness separates $\mathcal M$ from the interior teacher in observation KL (Appendix A.3). For example, in the four-variable fixture (A.21), with $\mu=(-.6,-.2,.2,.6)$, $\delta=.1$ and teacher eigenvalues $(.5,1,1.2,1.3)$, at least two limiting eigenvalues equal $.1$ and $\liminf_n\mathcal R(\widehat C_n)\ge9/12800$.

"Spectral boundary" here means eigenvalues at the floor $\delta$. The selected matrix is positive definite, and individual correlations need not all grow. For random outputs with $o_p(w_{n,K_n})$ regret, the distance to $\mathcal M$ tends to zero in probability and $P\{\mathcal R(\widehat C_n)<\varepsilon_*-\eta\}\to0$ for every $\eta>0$. The theorem concerns population regret, not every finite dataset: an empirical label distribution changes the linear term of $V$, and a one-label dataset can share its exact and particle minimizers (Appendix A.6).

**Other parameter regimes.** Two variants show that the regime matters. If active means and scales are also free, with means in a compact box containing the logits of the true marginal probabilities, scales in $[0,S]^m$ and the coins still at unit scale, then $o(1)$ population regret already drives the active scales to zero. The observation law converges to the product of the true marginals, and the risk tends to the teacher's total correlation (Appendix A.8). If instead the common nuisance scale is learnable and may reach zero, joint minimization switches the coins off: the extra nuisance penalty disappears, leaving the active finite-$K$ objective with its own bias (Appendix A.9). Theorems 1--2 hold nuisance scales at one, which is exactly what makes the coins a control.

# How the Asymptotic Picture Meets Finite Scales

Theorem 1 gives no remainder bound at any finite $n$. We therefore examine one instance closely: $m=3$, $\mu=0$, $C_*=I$, $\delta=.1$ and $K=100$. Let $C(\rho)$ have all off-diagonals $\rho$ and $C_b=C(.9)$. At $n=64,256,1024,4096$ we compare the gap $g_n=F_n(I)-F_n(C_b)$ and the free $c_{12}$ derivative $D_n$ at $C(.45)$ with their leading predictions, using $F_n=F_{n,100}^*$ and $w_n=w_{n,100}$. The ratios $A_n=g_n/\{w_n[V(I)-V(C_b)]\}$ and $B_n=-D_n/(m_H^{\mathrm{ref}}w_n)$ equal one when the leading term is exact; $m_H^{\mathrm{ref}}$ is a fixed numerical reference for $\kappa(.45)/4$.

## Weight Overlap Is Not the Learning Signal

A natural shortcut is to track scalar weight overlap $\omega_n=E\sum_k\pi_k(1-\pi_k)$, which measures how much of the average is shared between particles. If marks were independent of weights, (6) would give $D_n=-m_H\omega_n$. At $n=64$ overlap matches its leading scale closely, $\omega_n/w_n\simeq.992$. Yet $A_n\simeq-.143$ and $B_n\simeq-.149$: both the gap and the derivative have the *opposite* sign to their leading predictions, and the conditional intervals exclude zero. The identity $D_n=-m_H^{\mathrm{ref}}\omega_n-\mathcal C_n$, with $\mathcal C_n=E\sum_k\pi_k(1-\pi_k)(H_{k,12}-m_H^{\mathrm{ref}})$, separates the overlap error $O_n$ from weight--mark coupling $J_n$ (Figure 2b). Coupling dominates in magnitude at 64, 256 and 1024. By 4096 the point estimates of $A_n$ and $B_n$ are about $.83$, but the intervals do not certify 10% relative accuracy. Complete weights also contain active scores, so the split is an identity, not a causal decomposition (Appendix C.6).

![Finite scales with 100 particles; $n$ counts appended fair coins. (a) Gap, free-coordinate gradient and scalar overlap relative to their leading predictions; negative ratios reverse the predicted sign. (b) Gradient residuals at $C(.45)$, separated into overlap error $O_n$ and weight--mark coupling $J_n$; the label $m_H$ denotes the fixed numerical reference $m_H^{\mathrm{ref}}$. Bars are conditional 99% simultaneous intervals for 26 targets under ideal iid sampling and stated numerical allowances; the 131,072 banks are shared across $n$, and the last paired increase in $B_n$ is unresolved (Appendix C.6).](figures/bridge_summary_en.png){width=100%}

## A Signed Correction and a Certified Direction

What scalar overlap misses can be identified at fixed $K$. Write $L=\log\sigma(G)$, $\kappa_3=E(L-EL)^3$, $s_n=\sqrt{nv}$ and $B_K=EM_K^2-1$, where $M_K=\max_{k\le K}G_k$. For one candidate particle, let $A_Y=\sum_i\log\sigma(Y_iX_i)$, $H_Y=H_{1,12}$ from (6), $m_H=EH_Y$ and $\bar c=E_Y\operatorname{Cov}_X(H_Y,A_Y)$.

::: {#fixed-k-proposition}

**Proposition (fixed-K correction).** In the centred model (1) with unit nuisance scales, fix $m$, a positive definite candidate $C$, a teacher label law and an integer $K\ge2$. Then, as $n\to\infty$,

$$
\begin{aligned}
D_n:=\partial_{c_{12}}F_{n,K}^*(C)
&=-\frac{m_HJ_K}{s_n}\\
&\quad+\frac{B_K}{s_n^2}
 \left(\frac{m_H\kappa_3}{3v}-\bar c\right)\\
&\quad+O_{K,m,C}(n^{-3/2}).
\end{aligned}
\tag{11}
$$

:::

The new ingredient is $\bar c$: a covariance *within one particle*, between its dependence mark and its active log-likelihood, computed given the label and then averaged over labels. It is not a covariance between particles or between training gradients. Particles with high active likelihood receive more weight; when their marks differ systematically, the weighted mark is shifted. For the instance above at $C=C(.45)$, the $n^{-1}$ coefficient is $b_2\simeq1.026$. It splits into a nuisance-skewness part $\simeq-.077$ and a mark--score part $\simeq1.103$ (Appendix C.13). The correction thus opposes the leading term and comes almost entirely from same-particle coupling. These coefficients are model integrals rather than fits to the gradient panel, although the panel was available when the correction was derived. The proposition holds at fixed $K$ and does not replace the uniform statement of Theorem 1.

::: {#finite-direction-corollary}

**Finite-scale corollary.** At $m=3$, $\mu=0$, $C_*=I$, $K=100$, $C=C(.45)$ and the single free $c_{12}$ coordinate, explicit error bounds give

$$
\begin{gathered}
-.002045393<D_{4096}<-.0003519373,\\
|D_{4096}-D_{4096}^{(0)}|<.001097138798\\
<|D_{4096}^{(0)}|,\qquad
D_n^{(0)}=-m_HJ_{100}/s_n.
\end{gathered}
\tag{12}
$$

:::

At $n=4096$ the leading term has the correct sign for this coordinate. The error bound is about 75.7% of $|D_{4096}^{(0)}|$, so the certificate is about sign, not 10% accuracy. At 64, 256 and 1024 the same recipe leaves the sign unresolved. The enclosures are deterministic and use the actual Gumbel-smoothed nuisance noise and certified model moments ([Appendix C.7 onward](#fixed-k-correction)).

The certified sign has a risk consequence. By (5) and the three-variable law, $\partial_{c_{12}}\mathcal R=\kappa'(.45)\kappa/\{(1-\kappa)(1+3\kappa)\}>0$ with $\kappa=\kappa(.45)>0$. Every sufficiently small feasible increase of $c_{12}$ therefore lowers $F^*_{4096,100}$ while raising the risk. This is a finite, coordinate-level objective--risk conflict, established by its own explicit remainder bounds rather than by Theorem 2.

## Finite-Program Consequences at $n=256$

The instance of Figure 1 also yields finite-program statements at $n=256$, $K=100$ (Appendix C.15, D, E). On a 99% audit event under ideal iid sampling and declared numerical conditions, every correlation matrix in the class with population regret below $.00134$, on or off the grid, has risk above $10^{-6}$; this excludes a continuous low-risk region under a near-optimality condition that no algorithm is shown to attain. On the same event, a fresh run of the fixed 143-point grid program selects risk at least $0.09/(32e)$ with probability at least $.7035$, a bound on a probability for that program rather than an observed frequency.

# Without an Appended Block: A Certified Local Conflict

## A Two-Factor Model with All Parameters Free

We now remove the control. There is no designated independent block; all biases and loadings are learned:

$$
\begin{aligned}
Z&\sim N(0,I_2),\qquad \theta=(b,W),\\
Y_j\mid Z&\sim\operatorname{Bernoulli}
 \{\sigma(b_j+w_j^\top Z)\},\\
F_K^q(\theta)&=E_{Y\sim q,Z_{1:K}}\Bigl[
-\log\Bigl\{K^{-1}\sum_{k=1}^K
p_\theta(Y\mid Z_k)\Bigr\}\Bigr].
\end{aligned}
\tag{13}
$$

Here $b$ collects logit biases, $Y_j\in\{0,1\}$, and the prior particles $Z_{1:K}$ are iid and independent of $Y$. Items are conditionally independent given $Z$. The target $q$ is empirical or the teacher law, and $F_\infty^q$ is the integrated cross entropy. The teacher has eight items, zero biases and rows of length $.8$ at angles $(-.7,-.5,-.3,-.1,.1,.3,.5,.7)$, so all pairwise loading inner products are positive. This class differs from Section 3: the logit covariance $WW^\top$ has rank at most two and no spectral floor, and loading rank is not the rank of $\operatorname{Cov}(Y)$.

## Correct Marginals, Positive Risk, and an Opposed Improvement

**Theorem 3 (computer-assisted local dependence conflict).** For the exact teacher in (13), let $q_*$ be its observation law and $\mathcal R(\theta)=D(q_*\Vert Q_\theta)$. There is a rank-one interior parameter $\theta_*=(0,a_*,0)$, with $a_{*,i}>0$ and $\nabla F_8^{q_*}(\theta_*)=0$, that is a strict local minimum of $F_8^{q_*}$ in the full 24-dimensional parameter space modulo common latent rotations. Every observation marginal is correct, $Q_{\theta_*}(Y_j=1)=q_*(Y_j=1)=1/2$, yet $\mathcal R(\theta_*)\simeq.01043>0$.

There is a fixed $d=d_{\mathrm{ang}}^{(8)}\in\mathbf1^\perp\subset\mathbb R^8$, $\|d\|_2=1$, such that the path $\theta_{\mathrm{ang}}(t)=(0,W(t))$ satisfies

$$
\begin{aligned}
W_i(t)&=a_{*,i}\bigl(\cos(td_i),\sin(td_i)\bigr),\\
r(t)&=\mathcal R(\theta_{\mathrm{ang}}(t)),\\
r'(0)&=0,\qquad r''(0)<-.00450377.
\end{aligned}
\tag{14}
$$

There exists $\varepsilon>0$ such that, for every $0<|t|<\varepsilon$,

$$
\begin{gathered}
F_8^{q_*}(\theta_{\mathrm{ang}}(t))>F_8^{q_*}(\theta_*),\\
\mathcal R(\theta_{\mathrm{ang}}(t))<\mathcal R(\theta_*),\\
\operatorname{rank}W(t)=2.
\end{gathered}
\tag{15}
$$

The path changes only dependence. Row lengths are fixed, so each candidate logit keeps its Gaussian marginal $N(0,a_{*,i}^2)$ and each observation stays a fair coin, while $\operatorname{Cov}(X_i(t),X_j(t))=a_{*,i}a_{*,j}\cos\{t(d_i-d_j)\}$ changes. These preserved logit variances are the candidate's own, not the teacher's $.8^2$. Since the root is a strict local minimum and a nonconstant angle path leaves the rotation orbit, the objective rises along it. The risk falls because its curvature along the path is negative. The conflict is second order and local: every sufficiently small nonzero step along this path lowers risk and raises the objective (Appendix I.10). Fixed row lengths define this comparison path; they are not a constraint on the learning problem. The full statement in [Appendix I.8](#full-local-statement) also certifies a direction $d_{\mathrm{risk}}$ with a negative first-order risk derivative; a certified step of size $10^{-7}$ along it raises $F_8$, lowers the risk and keeps the observation marginals fair, while changing row lengths.

For scale, in four training streams of $4096$ teacher samples, independent-Bernoulli fits have KL about $.107$ and Adam training with $K=8$ ended near $.011$ (Appendix I.4); a refitted stationary point of the integrated objective within the collinear submodel has KL about $.0070$ (I.6). These are numerical comparisons, not certified optima.

**Certificate outline.** Symmetry reduces root finding to four coordinates, and interval root inclusion proves an exact stationary point near a dyadic centre. Full-space stability needs every direction. All 16 bias/radial and seven non-rotation transverse directions have certified positive curvature; an angular check alone would not suffice ([Appendix I.2](#full-hessian-criterion), [I.8](#full-local-statement)). The risk curvature along (14) includes the path acceleration. It is enclosed over all 256 labels by Gaussian integration by parts, analytic quadrature error bounds [@trefethen2014] and midpoint-radius arithmetic [@johansson2017], then transferred from the centre to the root by a Lipschitz bound in the row lengths ([I.10](#same-root-angle)).

::: {#quantitative-persistence-corollary}

**Corollary (quantitative local persistence).** Keep all eight teacher biases zero and row lengths $4/5$, changing only its first angle to $-.7+\lambda$, $|\lambda|\le.001$. With observation law $q_\lambda$, there is a smooth branch $\theta_\lambda=(0,a(\lambda),0)$ connected to Theorem 3's root, each a full-space strict local minimum of its own $F_8^{q_\lambda}$ modulo rotation. Using the same fixed $d$ and replacing $a_*$ by $a(\lambda)$ in (14) gives, for each teacher and every sufficiently small nonzero $|t|<\varepsilon_\lambda$, a rank-two path that raises its finite objective and lowers its own risk $D(q_\lambda\Vert Q_\theta)$, preserving candidate Gaussian logit marginals and correct fair observation marginals.

:::

This teacher family breaks the symmetry of the original teacher, but it is small: endpoint observation laws differ by about $1.6\times10^{-4}$ in $L^1$ ([Appendix I.11](#teacher-family-full-statement)). For broader, unquantified perturbations of the teacher, [Appendix I.9](#local-persistence) shows that a nearby rank-one strict local minimum survives with a first-order risk-decreasing direction.

## Weak-Loading Intuition for Alignment

A weak-loading expansion at fixed $K>2$ gives intuition, though not a quantitative account (Appendix I.1). With row lengths fixed and loadings of order $\varepsilon$, the finite objective is, up to angle-independent terms, $\varepsilon^4a_K\|c-\{K/(K-2)\}c_*\|^2+O(\varepsilon^6)$, where $a_K>0$ is given in (I.2) and $c$ collects pairwise row inner products after removing the common scale $\varepsilon$. It rewards inner products *amplified* relative to the teacher's. With fixed lengths, rows can amplify their inner products only by aligning. In the independent-teacher case of Section 3, the limiting potential likewise favours dependence that the teacher lacks. The two expansions are derived separately, and neither explains the other. The weak-loading stability boundary of the collinear configuration, $K\simeq4.33$, does not predict behaviour at loading $.8$.

Numerical learning is consistent with this picture without establishing it (Appendix I.4--I.7). In four streams, $K=8$ training produced nearly collinear loadings, whereas $K=64$ and the integrated reference did not. Refitted at each $K$, the collinear point numerically changes local type between $K=11$ and $12$; angle probes there change the risk by only about $.02\%$, an effect size that belongs to those probes, not to the certified $K=8$ conflict.

# Related Work

<!-- RELATED_WORK -->

# Discussion

**What the results change.** At fixed $K$ the prior-particle objective has its own dependence preferences, and estimating it more accurately does not move them toward the integrated likelihood. In the controlled family these preferences are of order $w_{n,K}$, yet they decide which correlations are selected and yield positive limiting risk (Theorems 1--2). In the finite-scale example, scalar overlap can track its leading prediction while the correlation gradient has the opposite sign to its own (Section 4). Correct marginals at a full-space local minimum do not mean that the objective welcomes risk-reducing dependence changes (Theorem 3).

**Scope.** The results carry different kinds of guarantee: Theorems 1--2 are uniform asymptotic statements; the corollary in (12) is a deterministic enclosure at one $n$; Section 4.3 rests on a 99% audit event with declared numerical conditions; Theorem 3 is a computer-assisted local proof. All use prior particles, and apart from the grid-program bound of Section 4.3 they concern population objectives. The first part requires a known independent appended block with unit scales, fixed active dimension and marginals, and regret small relative to $w_{n,K}$. The second part is one model with a small explicit teacher family. A natural next question is whether learned, posterior-adapted proposals [@burda2016] remove, shrink or reverse this preference.

**Reproducibility.** The accompanying anonymous supplement provides saved interval inputs, arithmetic checks for the certificates and finite-program bounds, and data and scripts for Figures 1--2. Its README gives the commands and verification scope. These checks reconstruct saved evidence; they do not rerun production quadrature or training.

<!-- APPENDICES -->
