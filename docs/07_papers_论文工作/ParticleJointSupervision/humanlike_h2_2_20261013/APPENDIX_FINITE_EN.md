# Appendix C. Finite selection, a population risk gap and local geometry

## C.1. A fixed matched-weight program

Figure 1 of the main text displays this comparison. A positive truth-minus-candidate anchor gap favors the correlated candidate; the scalar Hessian determines whether the truth is a local minimum or maximum. Each panel uses 262,144 banks and its own prespecified 99% simultaneous audit family, giving at least 98% joint coverage under the numerical qualifications. Selection uses 16,384 outer banks and 65,536 labels; this single realization is distinct from the fresh-run probability bound.

Throughout this appendix, $m=3$, the known mean is zero, $C_*=I$, $K=100$, $C\in\mathcal E_{.1}^{(3)}$, and $\mathcal R(C)=D(Q_I\Vert Q_C)$. Write $F_n=F_{n,100}$. The score, after removing a candidate-independent nuisance constant, is

$$
\begin{aligned}
\phi_{n,y}(C;Z,a)&=-\log\sum_{k=1}^{100}a_{nk}\prod_{i=1}^3\sigma\bigl(y_i(C^{1/2}Z_k)_i\bigr),\\
\widehat F_n(C)&=B^{-1}\sum_{b=1}^B\sum_yq_y\phi_{n,y}(C;Z_b,a_b),\\
F_n(C)&=\mathbb E\,8^{-1}\sum_y\phi_{n,y}(C;Z,a).
\end{aligned}
\tag{C.1}
$$

The 65,536 labels and frequencies $q_y$ are reused from Appendix H. There are $B=16,384$ new shared selection banks, each containing 100 independent standard three-dimensional Gaussian vectors. For $n=0$, $a_{0k}=1/100$. For $n=256$, $a_{256,k}=\exp(S_k)/\sum_l\exp(S_l)$, where $S_k$ sums 256 independent standard-Gaussian log-sigmoid terms, independently of the active inputs. Only this weight input changes between branches.

The grid contains the 143 triples in $\{-.9,-.6,-.3,0,.3,.6,.9\}^3$ satisfying $C\ge.1I$. Selection returns the lexicographically first point within $10^{-12}$ of the smallest binary64 score, with **no fallback**. Scores and outputs were fixed before evaluating risk. The anchor $C_b$ has all off-diagonals $.9$. Data, active particles, grid, scoring function and selection rule are identical across branches; $K=100$ was fixed before this run. This is a shared-bank finite search, not a general per-observation resampling and optimization procedure. In particular, $K$ counts particles inside each logarithm, not the total learning budget.

|Weight condition|Returned point|Numerical observation risk|
|---|---|---:|
|$n=0$|$I$|0|
|$n=256$|$C_b$|.031103092779|

Full 128-bit ball-arithmetic replay of the saved selection inputs gives an empirical continuous-region margin at least $.00390670332106984$ for $n=256$: every $C$ with $\mathcal R(C)\le10^{-6}$ scores above $C_b$. It treats saved binary64 inputs as exact mathematical inputs and includes root-residual allowances (Appendix H); it is deterministic, not a fresh-data failure probability. The risk value uses mutually checked Gaussian quadratures, not a certified quadrature remainder. The old $K=2$ result, where both conditions fail, remains in Appendix H and is not a controlled comparison across budgets.

## C.2. From a joint risk enclosure to a population gap

Put $\kappa(c)=\mathbb E[\tanh(X_1/2)\tanh(X_2/2)]$ for standard Gaussian marginals of correlation $c$. Central symmetry eliminates odd spin moments. Walsh inversion and orthogonality give exactly

$$
\begin{gathered}
Q_C(y)=\frac{1+u_y}{8},\qquad u_y=\sum_{i<j}\kappa(c_{ij})y_iy_j,\\
\mathbb E_{Q_I}u=0,\qquad\mathbb E_{Q_I}u^2=\sum_{i<j}\kappa(c_{ij})^2,\qquad
\mathcal R(C)=-\mathbb E_{Q_I}\log(1+u).
\end{gathered}
\tag{C.2}
$$

For $r_0=10^{-6}$ and $\mathcal R(C)\le r_0$, Pinsker applied to $\{y,-y\}$ gives $|u_y|\le a=\sqrt{8r_0}<1$. For either sign of $u$, and by Price's identity [@price1958],

$$
\begin{aligned}
-\log(1+u)&=-u+u^2\int_0^1\frac{1-t}{(1+tu)^2}\,dt
\ge-u+\frac{u^2}{2(1+a)^2},\\
\kappa'(c)&=\tfrac14\mathbb E[\operatorname{sech}^2(X_1/2)\operatorname{sech}^2(X_2/2)]\\
&\ge\tfrac14\mathbb E e^{-(X_1^2+X_2^2)/4}\ge\beta:=e^{-1/2}/4.
\end{aligned}
\tag{C.3}
$$

The last inequalities use $\log\cosh v\le v^2/2$ and Jensen. Since $\kappa(0)=0$, averaging (C.3) and using (C.2) proves

$$
\mathcal R(C)\le r_0\ \Longrightarrow\ 
\|C-I\|_F^2\le\frac{4(1+\sqrt{8r_0})^2r_0}{\beta^2}=:d^2,
\qquad d\simeq.013227076469306579.
\tag{C.4}
$$

For the curvature bound, set $S(t)=\sqrt{I+tD}$, $M_b=\max_k\|Z_{bk}\|_2$, and $\ell_k=\sum_i\log\sigma(y_i(S(t)Z_{bk})_i)$. With fully normalized weights $\pi$, $\phi''=-\sum_k\pi_k\ell_k''-\operatorname{Var}_{\pi}(\ell_k')$. Logistic derivative bounds and the square-root derivative bounds proved in Appendix H yield, whenever $\|D\|_F\le d<1$,

$$
\begin{aligned}
|\phi''|&\le\left\{\frac{(13/4)M_b^2}{4\lambda}+\frac{\sqrt3M_b}{4\lambda^{3/2}}\right\}\|D\|_F^2,\quad\lambda=1-d,\\
H(U;\lambda)&=\frac{(13/4)U}{4\lambda}+\frac{\sqrt{3U}}{4\lambda^{3/2}},\quad
b_K=\tfrac83(\log K+3\log2)\ge\mathbb E M_b^2,\\
\inf_{\mathcal R(C)\le r_0}\widehat F_n(C)&\ge\widehat F_n(I)-d\|\nabla\widehat F_n(I)\|_2/\sqrt2-\tfrac12H(U;1-d)d^2,\\
\inf_{\mathcal R(C)\le r_0}F_n(C)&\ge F_n(I)-c_d,\qquad c_d=\tfrac12H(b_K;1-d)d^2.
\end{aligned}
\tag{C.5}
$$

Here $U\ge B^{-1}\sum_bM_b^2$. The Gaussian bound follows from $\exp(tM_b^2)\le\sum_k\exp(t\|Z_{bk}\|^2)$ with $t=3/8$. Jensen controls the first moment of $M_b$. Coordinate-sign symmetry makes the population gradient at $I$ zero; the empirical gradient need not vanish. Taylor's theorem gives the last two lines. The maximum norm avoids a coarse linear-$K$ curvature cost.

An independent audit uses $L=262,144$ paired banks and true label weights $1/8$, which the learner does not use. Its prespecified contrasts are $g_n=F_n(I)-F_n(C_b)$ and $\Delta=g_{256}-g_0$. Set $A=\|\sqrt{C_b}-I\|_{\rm op}$, $T=8$, and $M=3AT$. A bank difference is set to zero if any active Gaussian coordinate exceeds $T$ in magnitude. Single-branch truncated differences lie in $[-M,M]$; paired differences lie in $[-2M,2M]$. Cauchy--Schwarz, the Gaussian moment bound and Mills' inequality give

$$
\begin{gathered}
t_T=\sqrt3A\sqrt{b_K\,2K(3)\varphi(T)/T},\qquad\alpha_j=.01/3,\\
\operatorname{hw}_j=\sqrt{\frac{2s_j^2\log(4/\alpha_j)}L}
+\frac{7R_j\log(4/\alpha_j)}{3(L-1)}+t_j+\epsilon_j,
\end{gathered}
\tag{C.6}
$$

Here $s_j^2$ is unbiased sample variance and $R_j$ is range length. For one branch, $(R_j,t_j,\epsilon_j)=(32.821067231192,\; .000003077999,\;10^{-9})$ to displayed precision; all three double for $\Delta$, whose variance is measured from paired differences. The empirical-Bernstein inequality [@maurer2009, Theorem 4] and a union bound give one qualified 99% family event. No bank was truncated. Thirty-two fixed banks per branch were checked with ball arithmetic; these diagnostics do not prove the numerical allowance at every population node. The coverage interpretation requires the stated iid Gaussian sampling model and numerical-error qualification. The all-node score enclosures in Appendix E concern the separate fresh-run moment audit in Appendix D, not these earlier anchor-gap banks.

|Contrast|Mean|99% family lower|99% family upper|
|---|---:|---:|---:|
|$g_0$|-.030106637632|-.032232312506|-.027980962758|
|$g_{256}$|.005905979917|.002342327475|.009469632359|
|$\Delta$|.036012617550|.030377739156|.041647495943|

Endpoints are rounded outward. The original prespecified enclosure used $d_{\rm old}=\sqrt{12r_0}/\beta$ and population cost $.004361765095074056$. Its region lower bound was $-.002019437619389396$: that gate failed and **six of seven original gates passed**. This record is unchanged.

Equation (C.4) is a **post hoc deterministic refinement**, using the same already-covered contrasts. It gives $c_d\simeq.0014471050001073832$ and, on that original event,

$$
\begin{aligned}
\inf_{\mathcal R(C)\le r_0}\{F_{256}(C)-F_{256}(C_b)\}&\ge\eta,
\qquad\eta=.0008952224755772784>0,\\
F_{256}(C)-\inf_{C'\in\mathcal E_{.1}^{(3)}}F_{256}(C')<\eta
&\ \Longrightarrow\ \mathcal R(C)>10^{-6}.
\end{aligned}
\tag{C.7}
$$

The outward bound subtracts the new deterministic cost from the original $g_{256}$ lower endpoint. No new random comparison or error budget is introduced. The implication follows because the infimum is at most $F_{256}(C_b)$. Continuity on the compact class, with integrable envelope $3\log2+3M_b$, ensures a minimizer exists. Every minimizer has risk above $10^{-6}$; neither risk $.0311$ for every minimizer nor global optimality of $C_b$ follows. No claim certifies the high-dimensional asymptotic remainder or a general optimizer's regret.

The independent moment audit in Appendix D bounds the same population gap with its own joint event. Appendix D.5 reuses the deterministic radius and cost above to obtain a region margin greater than $.00134$ on the same event as the fresh-run risk bound. This later same-event consequence leaves the original audit table, failed gate and earlier margin unchanged.

## C.3. An analytic positive-curvature control

Coordinate signs and permutations preserve either population objective. In the free coordinates $(c_{12},c_{13},c_{23})$, they imply exactly

$$
\nabla F_{n,K}(I)=0,\qquad\nabla^2F_{n,K}(I)=h_{n,K}I_3.
\tag{C.8}
$$

We prove $h_{0,100}>.00410$. Let $X_i$ be independent standard Gaussians, $p_i=\sigma(X_i)$, $W=p_1p_2p_3$, and define

$$
\begin{aligned}
A(s)&=\mathbb E e^{-sW},\quad B(s)=\mathbb E[X_1X_2e^{-sW}],\\
D(s)&=\mathbb E[(X_1^2-1)(X_2^2-1)e^{-sW}],\\
h_{0,K}&=K\int_0^\infty A(s)^{K-2}\{A(s)D(s)+(K-1)B(s)^2\}\,\frac{ds}{s}.
\end{aligned}
\tag{C.9}
$$

The identity follows by twice differentiating the Laplace representation of $-\log\sum_kW_k$; the one-particle Gaussian density scores are $X_1X_2$ and $(X_1^2-1)(X_2^2-1)$. Reflecting Gaussian coordinates absorbs any label and leaves this diagonal second derivative unchanged. For justification, write $S=\sum_kW_k$ and let $J$ be its joint second density score. Then $\mathbb EJ=\mathbb E(JS)=0$, so $|\mathbb EJe^{-sS}|\le s^2\mathbb E(|J|S^2)/2$ near zero. For $T>0$, the absolute tail integral is at most $\mathbb E[|J|\{E_1(T)-\log W_1\}]<\infty$, where $E_1(v)=\int_v^\infty e^{-u}du/u$. This uses $S\ge W_1$ and $-\log\sigma(x)\le\log2+|x|$. Gaussian density derivatives have locally uniform polynomial-Gaussian envelopes, also justifying twice-continuous differentiation.

Set $m_r=\mathbb E\sigma(X)^r$, $d_r=\mathbb E[(X^2-1)\sigma(X)^r]$. Symmetry gives $d_0=d_1=0$. For $r\ge2$, the even function $f_r=(p^r+(1-p)^r)/2$ increases strictly with $|X|$, hence $d_r>0$. On $x\ge0$,

$$
\begin{aligned}
\left[\frac{r+1}{r}f_r-f_{r+1}\right]'&=\frac{r+1}{2}p'p(1-p)\{p^{r-2}-(1-p)^{r-2}\}\ge0,\\
0<d_{r+1}&\le\frac{r+1}{r}d_r,\qquad
D(s)=\sum_{r=2}^\infty\frac{(-s)^r}{r!}d_r^2m_r\ge0\quad(0\le s\le4/3).
\end{aligned}
\tag{C.10}
$$

The covariance of increasing functions proves the first inequality for $d_r$. The absolutely convergent alternating series has consecutive magnitude ratio at most $s(r+1)/r^2\le3s/4$, proving its sign. No global nonnegativity of $D$ is assumed.

Write $a_0=1/\sqrt{24}\le a=\mathbb E\sigma'(X)$, from $\operatorname{sech}^2(X/2)\ge e^{-X^2/4}$. Jensen gives $A(s)\ge e^{-s/8}$, and integration by parts gives

$$
\begin{gathered}
B(s)=s\mathbb E[\sigma'(X_1)\sigma'(X_2)p_3(sW-1)e^{-sW}],\\
|B(s)|\ge\tfrac12a^2s(1-s)e^{-s}\quad(0\le s\le1).
\end{gathered}
\tag{C.11}
$$

Integrate this positive contribution only over $[0,1/8]$. For the remaining negative contribution take $T=4/3$ and use $m_2=1/2-a$, $e^{-u}\le1-u+u^2/2$, and $\|(X_1^2-1)(X_2^2-1)\|_2=2$. They give

$$
\begin{aligned}
P&=\frac{100\cdot99}{2304}(7/8)^2e^{-57/32}/128,\\
A(T)&\le q:=1-T/8+(T^2/2)(1/2-a_0)^3,\\
\int_T^\infty|D(s)|\frac{ds}{s}&\le2\{e^{-T}/T+3(\log2+1)\},\\
N&=200q^{99}\{3(\log2+1)+e^{-T}/T\},\qquad h_{0,100}\ge P-N.
\end{aligned}
\tag{C.12}
$$

The tail bound follows from $E_1(TW)\le E_1(T)-\log W$, Cauchy--Schwarz and Minkowski. Equation (C.10) controls the rest of the $D$ integral.

The strict numerical threshold uses rational arithmetic alone. Put $a_L=204124/10^6$, $r=57/32$, and

$$
\begin{aligned}
E_U&=\sum_{j=0}^{12}\frac{r^j}{j!}+\frac{r^{13}/13!}{1-r/14}>e^r,\qquad
q_U=1-\tfrac16+\tfrac89(\tfrac12-a_L)^3,\\
P_L&=\frac{100\cdot99}{2304}(7/8)^2/(128E_U),\quad
N_U=200q_U^{99}\{3(694/1000+1)+1/5\},\\
24a_L^2&<1,\quad\sum_{j=0}^{6}\frac{(694/1000)^j}{j!}>2,\quad
\sum_{j=0}^{5}\frac{(4/3)^j}{j!}>15/4,\\
h_{0,100}&\ge P_L-N_U>41/10000=.00410.
\end{aligned}
\tag{C.13}
$$

The exponential tail is bounded geometrically. The finite sums prove $\log2<.694$ and $e^{-4/3}/(4/3)<1/5$; the last strict comparison is an exact rational inequality ($P_L-N_U\simeq.004101508638$). Thus $I$ is a strict local minimum for $n=0$, without Monte Carlo qualification. This does not establish its global optimality.

## C.4. A bounded curvature audit and its scope

For $n=256$ we use a new statistic and new banks. At $I$, fix label $(1,1,1)$ and let $\pi_k$ normalize $a_k\prod_i\sigma(Z_{ki})$. Define $q_{ki}=\sigma(-Z_{ki})$, $r_{ki}=q_{ki}(2q_{ki}-1)$, $a_k^{ij}=q_{ki}q_{kj}$, $A_{ij}=\sum_k\pi_ka_k^{ij}$ and $B_{ij}=\sum_k\pi_k^2a_k^{ij}$. Squaring the Price operator $\sum_k\partial_{Z_{ki}}\partial_{Z_{kj}}$ gives

$$
\begin{aligned}
H_{ij}={}&\sum_k\left[-\pi_k(1-\pi_k)r_{ki}r_{kj}
+2\pi_k^2(1-\pi_k)(q_{ki}^2r_{kj}+q_{kj}^2r_{ki})\right.\\
&\left.\hspace{8mm}+\pi_k^2(1-4\pi_k)q_{ki}^2q_{kj}^2\right]
+A_{ij}^2-4A_{ij}B_{ij}+6B_{ij}^2,\\
h_{n,K}&=\mathbb EH_{ij},\qquad -33/8\le H_{ij}\le67/16.
\end{aligned}
\tag{C.14}
$$

The four displayed contributions lie respectively in $[-1,1/8]$, $[-1/8,1]$, $[-3,1/16]$, and $[0,3]$, using $-1/8\le r\le1$, $\sum\pi^2(1-\pi)\le1/4$ and $0\le B_{ij}\le A_{ij}\le1$. Logistic derivatives and normalized weights are bounded; conditioning on nuisance weights justifies Price differentiation. At $K=1$ the polynomial cancels exactly. This estimator and the square-root path Hessian have the same expectation, not necessarily the same bankwise value.

There are 262,144 new independent paired banks. The three pair kernels are averaged **within** each bank; they are not three independent observations. Apply (C.6) with zero tail allowance to $h_{0,100}$, $h_{256,100}$ and their difference, using range lengths $133/16$, $133/16$, $133/8$ and numerical allowances $10^{-9}$, $10^{-9}$, $2\times10^{-9}$. This is a separate $\alpha=.01$ family.

|Curvature contrast|Mean|99% family lower|99% family upper|
|---|---:|---:|---:|
|$h_{0,100}$|.028200713775|.027644213668|.028757213882|
|$h_{256,100}$|-.004265410162|-.004912712428|-.003618107895|
|$h_{256,100}-h_{0,100}$|-.032466123936|-.033641746822|-.031290501050|

Endpoints are rounded outward. Manufactured small-$K$ automatic differentiation checks test the polynomial; 32 fixed scientific banks give 192 ball-arithmetic checks with maximum discrepancy $8.35\times10^{-15}$. A subsequent deterministic computation encloses the exact saved-input n256 kernels at all 262,144 banks, including all three directions, and propagates their errors through the sample mean and standard deviation (Appendix E). It leaves the original table and family unchanged. Neither the original diagnostics nor this saved-input computation establishes a uniform Gaussian/log-sigmoid input-generation error bound, so the population sign retains that qualification.

On this event, (C.8) makes $I$ a strict local maximum for $n=256$. Every sufficiently small nonzero correlation perturbation decreases its finite population objective and increases exact observation risk; $\kappa'(c)\ge\beta$ ensures that only $I$ has zero risk. The gradient at $I$ is exactly zero, so an exact deterministic optimizer initialized there need not move. No optimizer was run in this curvature audit.

The two random families in this appendix have conservative joint coverage at least 98%, under their sampling and numerical conditions. Jointly including the mean-robustness family in Appendix F and the fresh-run moment family in Appendix D gives at least 96%; adding the earlier Appendix H family gives at least 95%. This covers those specified families, not every historical experiment. Independence between families is unnecessary. Deterministic refinements, the analytic positive-curvature proof and the saved-input arithmetic in Appendix E spend no additional probability budget.

## C.5. Leading-term calibration at the retained finite instance

For the same $m=3$, independent teacher, $n=256$, $K=100$ and $C_b$ with all off-diagonals $.9$, deterministic coefficient quadratures in (A.4), (A.6) and (A.16) give

$$
\begin{gathered}
v(1)\simeq.271514501801,\quad J_{100}\simeq2.507593636442,\\
w_{256,100}\simeq.300774255953,\quad V(C_b)\simeq-.052200051030.
\end{gathered}
$$

Here $V(I)=0$. The leading objective difference is $w_{256,100}[V(I)-V(C_b)]$. Separately, formal differentiation of the limiting gradient field gives $h^{\rm formal}_{256,100}=-w_{256,100}\kappa'(0)/4$, where $\kappa'(0)=\tfrac14(E\operatorname{sech}^2(G/2))^2\simeq.170768891292$.

|Quantity|Leading or formal value|Retained finite-audit mean|Original qualified 99% family interval|
|---|---:|---:|---|
|$g_{256}$, objective gap|.015700431509|.005905979917|[.002342327475, .009469632359]|
|$h_{256,100}$, formal second-order comparison|−.012840721555|−.004265410162|[−.004912712428, −.003618107895]|

Gap units are nats per complete observation; curvature units are nats per squared free correlation coordinate. The finite gap mean is 37.62% of its leading term, or the leading term is 2.658 times that mean. The formal curvature magnitude is 3.010 times the finite mean magnitude. Both comparisons lie outside their corresponding original intervals. The intervals belong to separate families; jointly their original conditional coverage is at least 98%. The finite gap and curvature entries are the retained audits in C.2 and C.4. These are comparisons with estimates, not certified asymptotic errors. In particular, the theorem supplies no small relative remainder at this $n$; the finite consequence is independently qualified. Uniform convergence of the normalized gradient cannot itself be differentiated to establish Hessian convergence, so the curvature row is a **formal second-order comparison**, not a prediction proved by Theorem 1. The leading gap concerns $g_{256}$, not the nuisance interaction $g_{256}-g_0$.

Deterministic Hermite40/64/96 with correlation Legendre32/48/64, a separately parameterized normal-weighted Legendre128 check, and full-line one-dimensional quadratures agree within $1.2\times10^{-13}$ for the displayed coefficients. These rule sensitivities are not rigorous quadrature remainder bounds. This comparison adds no Monte Carlo data or training.

## C.6. A fixed four-scale comparison: overlap and the marked learning field

This comparison keeps the independent three-variable teacher, $K=100$, $I/C_b$ and the free-coordinate convention of C.1. It uses 131,072 iid complete banks at $n=64,256,1024,4096$, with common active inputs and nuisance prefixes. The same banks are shared across scales, not independent training repetitions. Let $C(\rho)$ have all off-diagonals $\rho$, and let $w_n$ denote the reference in (7). The frozen reference $m_H^{\mathrm{ref}}\simeq.0192712788530$ approximates $\kappa(.45)/4$. Define

$$
\begin{gathered}
g_n=F_n(I)-F_n(C_b),\quad D_n=\partial_{c_{12}}F_n(C(.45)),\quad
\omega_n=E\sum_k\gamma_{nk},\\
\gamma_{nk}=\pi_{nk}(1-\pi_{nk}),\quad
A_n=\frac{g_n}{w_n\{V(I)-V(C_b)\}},\quad B_n=-\frac{D_n}{m_H^{\mathrm{ref}}w_n}.
\end{gathered}
\tag{C.15}
$$

**Primary target and bounded kernel.** An independent uniform path coordinate is drawn once per complete bank and shared across $n$. Price differentiation, a Gaussian envelope on the compact positive-floor path, and Fubini give

$$
\begin{aligned}
g_n&=-\int_0^{.9}\sum_{i<j}\partial_{c_{ij}}F_n(C(\rho))\,d\rho=EX_n,\\
X_n&=.9\,\frac18\sum_y\sum_k\gamma_{nk}(C(U))\sum_{i<j}H_{k,ij}(C(U)),
\qquad U\sim\operatorname{Uniform}(0,.9),\\
X_n&\in[-.675s,1.35s],\quad D_{n,\mathrm{bank}}\in[-s/2,s/2],\quad
\omega_{n,\mathrm{bank}}\in[0,s],\qquad s=1-1/K=.99.
\end{aligned}
\tag{C.16}
$$

The eight labels are enumerated with weight $1/8$. A free $c_{12}$ changes both symmetric covariance entries, so its Price derivative has no extra factor $1/2$; the path derivative sums all three free coordinates. For the path support, the two equal-sign labels give a three-mark sum in $[0,3]$, and the six mixed labels give $[-1,1]$, yielding the narrower average support in (C.16). This is an unbiased estimator of the objective gap under the mathematical sampling model, not the same random variable as a realized square-root-path derivative.

**Fixed-coordinate decomposition.** At $C(.45)$, with the same fixed reference $m_H^{\mathrm{ref}}$ used in every bank,

$$
\begin{aligned}
\mathcal C_n&=E\sum_k\gamma_{nk}(H_{k,12}-m_H^{\mathrm{ref}}),\qquad D_n=-m_H^{\mathrm{ref}}\omega_n-\mathcal C_n,\\
\frac{D_n}{w_n}+m_H^{\mathrm{ref}}&=\underbrace{m_H^{\mathrm{ref}}\left(1-\frac{\omega_n}{w_n}\right)}_{O_n}
+\underbrace{\left(-\frac{\mathcal C_n}{w_n}\right)}_{J_n},\\
\mathcal C_{n,\mathrm{bank}}&\in[-(1/2+m_H^{\mathrm{ref}})s,(1-m_H^{\mathrm{ref}})s/2].
\end{aligned}
\tag{C.17}
$$

The bank forms average over all eight labels. The identity holds for any fixed $m_H^{\mathrm{ref}}$; its interpretation as an asymptotic reference retains the reference's numerical qualification. The residuals below share one absolute normalized scale. Complete weights contain active as well as nuisance scores, so $J_n$ is a marked-coupling residual, not an identification of every deeper source of non-Gaussian discrepancy.

|$n$|$A_n$: mean [interval]|$B_n$: mean [interval]|$\omega_n/w_n$: mean [interval]|
|---:|---:|---:|---:|
|64|-0.14273 [-0.16133, -0.12413]|-0.14859 [-0.17300, -0.12419]|0.99160 [0.98741, 0.99580]|
|256|0.33789 [0.30066, 0.37511]|0.32796 [0.27900, 0.37692]|1.02633 [1.01697, 1.03569]|
|1024|0.65135 [0.58246, 0.72023]|0.65173 [0.56171, 0.74175]|1.01815 [1.00267, 1.03363]|
|4096|0.82920 [0.70351, 0.95489]|0.83086 [0.66679, 0.99493]|1.00822 [0.98451, 1.03193]|

|$n$|$O_n$: mean [interval]|$J_n$: mean [interval]|
|---:|---:|---:|
|64|0.00016179 [0.00008101, 0.00024257]|0.02197309 [0.02146427, 0.02248191]|
|256|-0.00050746 [-0.00068772, -0.00032720]|0.01345856 [0.01246015, 0.01445698]|
|1024|-0.00034978 [-0.00064800, -0.00005156]|0.00706136 [0.00528073, 0.00884199]|
|4096|-0.00015841 [-0.00061528, 0.00029846]|0.00341803 [0.00022369, 0.00661238]|

At $n=64$, the conditional intervals support $g_n<0$ and $D_n>0$, opposite to their respective leading signs although scalar overlap is close to its scale. The coupling term $J_{64}\simeq.02197309$ also exceeds $m_H^{\mathrm{ref}}$, whereas $O_{64}\simeq.00016179$. At 64, 256 and 1024, the displayed joint intervals establish that $J_n$ exceeds $|O_n|$; they do not establish this ordering at 4096. The conclusion concerns this coordinate and decomposition, not every point of the $I\to C_b$ path or a causal decomposition of training risk.

**All targets and simultaneous intervals.** The family contains five raw targets at each of four scales and six adjacent paired contrasts, for 26 targets. The raw targets are the path gap, fixed-coordinate gradient, overlap, coupling, and a secondary direct anchor-loss difference. With unbiased sample variance $\widehat v_j$ and prespecified range length $R_j$, the two-sided empirical-Bernstein radius is

$$
\begin{gathered}
\delta_j=.01/26,\qquad B=131072,\\
\operatorname{hw}_j=\sqrt{\frac{2\widehat v_j\log(4/\delta_j)}B}
+\frac{7R_j\log(4/\delta_j)}{3(B-1)}+t_j+\epsilon_j.
\end{gathered}
\tag{C.18}
$$

Here $\epsilon_j=10^{-9}$ for each raw target; their affine transforms and contrasts scale this allowance. The first four raw supports are in (C.16)--(C.17), with zero tail allowance. For the direct gap, set the bank value to zero if any active $|Z|>8$, as in C.2. Its support is $[-M,M]$, $M=16.410533615595888$, with tail allowance $3.077999197742064\times10^{-6}$. No nuisance log-weight clipping is used. The direct-gap control does not replace the primary path estimator.

|$n$|Target|Mean|Sample variance|Joint lower|Joint upper|
|---:|---|---:|---:|---:|---:|
|64|$g_n$|-0.00448185566|0.0004559644|-0.00506565|-0.00389807|
|64|$D_n$|0.00172259379|0.0001016962|0.00143977|0.00200542|
|64|$\omega_n$|0.596498369|0.03940728|0.59397700|0.59901974|
|64|$\mathcal C_n$|-0.0132178802|0.0001418427|-0.01352396|-0.01291180|
|64|$g_n^{\mathrm{dir}}$|-0.00452243934|0.01260892|-0.01126389|0.00221901|
|256|$g_n$|0.00530496451|0.0004581111|0.00472058|0.00588935|
|256|$D_n$|-0.00190094564|0.0001032908|-0.00218471|-0.00161718|
|256|$\omega_n$|0.308694352|0.04976738|0.30588104|0.31150766|
|256|$\mathcal C_n$|-0.0040479893|0.0001304913|-0.00434829|-0.00374769|
|256|$g_n^{\mathrm{dir}}$|0.0055723925|0.04136146|-0.00225117|0.01339595|
|1024|$g_n$|0.00511321392|0.0003143611|0.00457247|0.00565396|
|1024|$D_n$|-0.00188881694|6.786048e-05|-0.00214970|-0.00162793|
|1024|$\omega_n$|0.15311671|0.03318471|0.15078953|0.15544389|
|1024|$\mathcal C_n$|-0.00106193787|7.545536e-05|-0.00132973|-0.00079415|
|1024|$g_n^{\mathrm{dir}}$|0.00613213357|0.06823211|-0.00237854|0.01464281|
|4096|$g_n$|0.00325468235|0.0001887566|0.00276135|0.00374801|
|4096|$D_n$|-0.00120397332|3.956504e-05|-0.00144172|-0.00096623|
|4096|$\omega_n$|0.0758116501|0.01858578|0.07402902|0.07759428|
|4096|$\mathcal C_n$|-.000257014127|4.050285e-05|-0.00049721|-0.00001682|
|4096|$g_n^{\mathrm{dir}}$|0.00365793433|0.08548928|-0.00522308|0.01253895|

Means and variances are displayed approximations; interval endpoints are rounded outward. Adjacent differences are formed bankwise before estimating variance. If $\alpha_n$ is the appropriate affine scale, their variance is $\alpha_b^2\widehat v_b+\alpha_a^2\widehat v_a-2\alpha_a\alpha_b\widehat{\operatorname{Cov}}(X_a,X_b)$. Their support is the difference of the two scaled supports, and their numerical allowances add. The covariance is not replaced by zero, nor is variance reduction assumed from pairing.

|Quantity|Next minus previous $n$|Mean|Joint interval|Raw paired covariance|Paired variance|
|---|---|---:|---|---:|---:|
|$A$|256-64|0.4806169|[0.43092, 0.53031]|-3.293773e-06|2.33423|
|$A$|1024-256|0.3134604|[0.21962, 0.40730]|1.566845e-05|6.705305|
|$A$|4096-1024|0.1778487|[0.00305, 0.35265]|9.030132e-06|16.76677|
|$B$|256-64|0.4765523|[0.41116, 0.54194]|5.843457e-07|3.813732|
|$B$|1024-256|0.3237732|[0.20000, 0.44755]|1.30637e-06|10.99817|
|$B$|4096-1024|0.1791244|[-0.05090, 0.40915]|6.744675e-07|26.60021|

The complete 20-target and 26-target covariance matrices are retained in the machine-readable companion readout. The table gives all six paired covariances needed for the displayed comparisons. All three adjacent increases in $A_n$ are resolved, with the last lower endpoint only about .00305. The last increase in $B_n$ remains unresolved. At 4096 the two ratios' intervals do not establish inclusion in $[.9,1.1]$; thus 10% relative accuracy is unconfirmed. The wider direct-gap intervals and all unresolved comparisons are retained, without outcome-based selection.

These are conditional 99% joint intervals under ideal iid complete-bank sampling, the stated numerical allowances and frozen coefficient references. They are not uniform native-arithmetic or exact coefficient certificates, and not an asymptotic remainder theorem. They spend one additional .01 family beyond the five earlier specified families in C.4; their joint conditional coverage is at least 94% by a union bound, without requiring independence between families. The deterministic local certificates in I.8 spend no additional alpha.

The earlier C.5 gap estimate gives 37.62% of its leading term; this panel gives 33.79% at 256. Different banks and estimators are involved, and the new interval lies inside the earlier gap interval. Figure 2 uses only the new common panel; neither estimate is relabeled as the other's result. This comparison adds interpretation of the finite marked field, not a universal convergence rate, Hessian prediction or particle-budget rule.

In the displayed panel labels, $m_H$ denotes the fixed numerical reference $m_H^{\mathrm{ref}}$ used in this section, not the exact expectation used from C.7 onward.

## C.7. Exact label-conditioned kernel for the finite-scale correction {#fixed-k-correction}

Sections C.7--C.14 establish the fixed-$K$ correction and the finite-scale corollary in (11)--(12). They concern the same zero-mean model, but use deterministic model integrals rather than the conditional Monte Carlo intervals in C.6. First fix $K\ge2$, finite $m$, a positive-definite correlation matrix $C$, and a label law $p_y=P(Y=y)$ independent of $C$. With $Z\sim N(0,1)$ and $X\sim N(0,C)$, put

$$
\begin{aligned}
L&=\log\sigma(Z),\quad \ell=EL,\quad v=\operatorname{Var}L,\\
\kappa_3&=E(L-\ell)^3,\quad s=\sqrt{nv},\\
A_y&=\sum_{i=1}^m\log\sigma(y_iX_i),\quad T_y=A_y-m\ell,\\
H_y&=y_1y_2\sigma(-y_1X_1)\sigma(-y_2X_2),\\
m_y&=E_XH_y,\quad c_y=E_X[(H_y-m_y)T_y],\\
m_H&=E_Ym_Y,\qquad \bar c=E_Yc_Y.
\end{aligned}
\tag{C.19}
$$

The label is shared by every particle in a bank and is independent of the candidate particles. In the retained independent-teacher instance its law is uniform on eight labels. Let $\xi$ be an independent centered standard Gumbel, $S_n=\sum_{j=1}^n(L_j-\ell)$, and let $g,G$ be the density and CDF of $(S_n+\xi)/s$. All kernels below retain this Gumbel. For each fixed label define

$$
\begin{aligned}
f_y(z)&=E_Xg(z-T_y/s),\quad F_y(z)=E_XG(z-T_y/s),\\
h_y(z)&=E_X[H_yg(z-T_y/s)],\quad q=K-2,\\
D_n&=-\frac{K(K-1)}s\sum_yp_y\int_{\mathbb R}h_yf_yF_y^q.
\end{aligned}
\tag{C.20}
$$

The last identity follows from the ordered-pair Price formula in A.2 and the Gumbel maximum representation. A common translation by $m\ell/s$ inside each label integral exactly centers the active score. Conditional particle independence is used before the final label average; replacing the products by products of label-mixture densities would change the target. In particular, $\bar c$ is a same-particle, label-conditional mark--score covariance, not a covariance between particles or a training-noise statistic.

Write $\phi,\Phi$ for the standard Gaussian density and CDF. If $M_K=\max_{k\le K}Z_k$ for iid standard Gaussians, integration by parts gives

$$
\begin{aligned}
I_K&=\int\phi^2\Phi^q,\quad T_K^{\rm G}=\int z\phi^2\Phi^q,\\
J_K&=EM_K=K(K-1)I_K,\\
B_K&=EM_K^2-1=K(K-1)T_K^{\rm G},\\
B_K&=\frac{K(K-1)(K-2)}2\int\phi^3\Phi^{K-3}\quad(K\ge3),\\
B_2&=0.
\end{aligned}
\tag{C.21}
$$

Here $T_y$ is the active centered score, whereas $T_K^{\rm G}$ and $B_K$ are Gaussian order-statistic constants. The identities follow successively from $\phi'=-z\phi$, $z^2\phi=\phi-(z\phi)'$, and $(\phi^2)'=-2z\phi^2$; all boundary terms vanish.

## C.8. Proof and interpretation of the signed second term

For each fixed $K,m,C$ as above, there exist finite constants $C_0,n_0$ such that, for $n\ge n_0$, the remainder in

$$
\begin{aligned}
D_n&=-\frac{m_HJ_K}{s}
+\frac{B_K}{s^2}\left(\frac{m_H\kappa_3}{3v}-\bar c\right)
+O_{K,m,C}(n^{-3/2}),\\
D_n^{(2)}&=-\frac{m_HJ_K}{s}+\frac{b_2}{n},\qquad
b_2=\frac{B_K}{v}\left(\frac{m_H\kappa_3}{3v}-\bar c\right).
\end{aligned}
\tag{C.22}
$$

is bounded by $C_0n^{-3/2}$. This is a fixed-$K$ statement, with no uniformity in growing $K$ or over a parameter class. Its existential constants do not themselves certify any of the four finite scales. The subsequent sections supply a separate explicit finite bound.

For completeness, the density argument supplies the norms needed by the nonlinear kernel. Put $\epsilon=n^{-1/2}$, $\alpha=\kappa_3/(6v^{3/2})$, $e=(z^3-3z)\phi$ and $\mathcal P=(1-z^2)\phi$, so $\mathcal P'=e$. For $p=1,\infty$ and sufficiently large $n$,

$$
\begin{aligned}
\|g-\phi-\epsilon\alpha e\|_p&=O(n^{-1}),\\
\|g'-\phi'\|_p&=O(n^{-1/2}),\qquad \|g''\|_p=O(1).
\end{aligned}
\tag{C.23}
$$

To prove these bounds, the density of $L$ on $x<0$ is $p_L(x)=\phi(x-\log(1-e^x))/(1-e^x)$, extended by zero for $x\ge0$. At $-\infty$ all fixed derivatives have Gaussian decay times a polynomial. At $0-$, writing $t=-\log(-x)$, each derivative is bounded by $\exp(-t^2/2+O(t))$ times a polynomial and tends to zero. Thus this zero extension is Schwartz. Consequently $U=(L-\ell)/\sqrt v$ has all moments, its characteristic function $\psi$ and its derivatives decay rapidly, and $\sup_{|u|\ge\delta}|\psi(u)|<1$ for every $\delta>0$. A nondegenerate density cannot have $|\psi(u)|=1$ at a nonzero frequency. If $p_n$ is the density of $S_n/s$, the local log expansion and its first two derivatives yield

$$
\begin{aligned}
n\log\psi(t/\sqrt n)&=-t^2/2+\epsilon\alpha(it)^3+O(t^4/n),\\
R_n(t)&=\psi(t/\sqrt n)^n-e^{-t^2/2}(1+\epsilon\alpha(it)^3),\\
\int_{\mathbb R}\left|\partial_t^j\{(it)^rR_n(t)\}\right|dt
&=O(n^{-1}),\qquad 0\le r,j\le2.
\end{aligned}
\tag{C.24}
$$

Here is the tail control behind the last line. On $|t|\le n^{1/12}$, Taylor remainders and their first two derivatives are bounded by $n^{-1}$ times a fixed polynomial times $e^{-ct^2}$, including the square of the cubic term. On $n^{1/12}<|t|\le\delta\sqrt n$, choose fixed small $\delta$ so $|\psi(u)|\le e^{-cu^2}$ and $\psi(u)\ne0$; the differentiated products have at most polynomial factors in $n,t$, dominated by this Gaussian tail. Beyond $\delta\sqrt n$, retain the at most two differentiated factors and two undifferentiated factors; the remaining at least $n-4$ factors contribute $r_\delta^{n-4}$ for some $r_\delta<1$. The retained Schwartz factors have finite weighted integrals after rescaling, and all powers of $n$ are dominated by that exponential. These three estimates prove (C.24), rather than leaving an uncontrolled Fourier tail.

Fourier inversion and two integrations by parts in frequency now imply $\sup_z(1+z^2)|p_n^{(r)}-\phi^{(r)}-\epsilon\alpha e^{(r)}|=O(n^{-1})$. This gives both $L^1$ and $L^\infty$ bounds. Since $E\xi=0$, $E\xi^2=\pi^2/6$, convolution with $\xi/s$ gives $\|g-p_n\|_p\le E\xi^2\|p_n''\|_p/(2s^2)$, $\|g'-p_n'\|_p\le E|\xi|\|p_n''\|_p/s$, and $\|g''\|_p\le\|p_n''\|_p$, proving (C.23). The Gumbel therefore contributes no $n^{-1}$ coefficient in (C.22), but remains in its remainder.

Gaussian active moments and $|H_y|\le1$ justify integral Taylor remainders. Centering $ET_y=0$ gives, in both indicated density norms,

$$
\begin{aligned}
\|f_y-g\|_{1,\infty}+\|F_y-G\|_\infty&=O(s^{-2}),\\
h_y-m_yf_y&=-c_yg'/s+O_{1,\infty}(s^{-2}).
\end{aligned}
\tag{C.25}
$$

For $\mathcal I(f)=\int f^2F^q$, bounded densities and $L^1+L^\infty$ errors control all products, with $q$ fixed. Its linear variation at $\phi$ in direction $e$ is

$$
\begin{aligned}
D\mathcal I_\phi[e]&=2\int\phi e\Phi^q+q\int\phi^2\mathcal P\Phi^{q-1}
=-2T_K^{\rm G},\\
\mathcal I(f_y)&=I_K-\frac{\kappa_3}{3vs}T_K^{\rm G}+O_K(n^{-1}),\\
\int(h_y-m_yf_y)f_yF_y^q&=\frac{c_y}{s}T_K^{\rm G}+O_K(n^{-1}).
\end{aligned}
\tag{C.26}
$$

The first identity is obtained by integrating $[(z^2-1)\phi^2\Phi^q]'$; the last sign uses $\phi'=-z\phi$. The product remainder is $O_K(n^{-1})$ because each first-order density or CDF perturbation is $O(n^{-1/2})$. When $q=0$, terms multiplied by $q$ are omitted, with no negative CDF power. When $q=1$, the CDF factor is linear and its second and higher variations vanish. Both cases follow from the same finite polynomial expansion; neither uses the divisions by $q-1$ introduced later for $q=98$. Substitution into (C.20), followed by (C.21), proves (C.22).

Finally, $E_XA_y=m\ell$ is independent of a free correlation coordinate. Price differentiation and the coordinate additivity of $A_y$ show

$$
\begin{aligned}
\partial_{c_{12}}E_XT_y^2&=2m_y,\qquad
\partial_{c_{12}}E_XT_y^3=6c_y,\\
V_A&=E_YE_XT_Y^2,\quad T_A=E_YE_XT_Y^3,\\
m_H&=\tfrac12\partial_{c_{12}}V_A,\qquad
\bar c=\tfrac16\partial_{c_{12}}T_A.
\end{aligned}
\tag{C.27}
$$

The factor is one for the free off-diagonal Price operator: both symmetric covariance entries change together. Thus the second term combines nuisance skewness with the change of a label-conditional active third moment. For $K=2$ it vanishes because $B_2=0$. It cannot be recovered from scalar overlap alone.

## C.9. Five explicit finite remainder budgets

For the rest of the certificate fix $m=3$, $C=C(\rho)$ with $\rho=9/20$, the independent teacher, $K=100$ and $q=98$. Permutations and global reflection reduce the eight-label sum exactly to two classes. In (C.19)--(C.20), replace $H_y$ by the mean of its three free-edge marks:

$$
\begin{aligned}
\widetilde H_y&=\tfrac13(H_{12,y}+H_{13,y}+H_{23,y}),\\
p_+&=1/4\ (y=+++),\qquad p_0=3/4\ (y=++-),\\
m_+&=m_H+1/4,\qquad m_0=m_H-1/12,\\
c_y&=E[(\widetilde H_y-m_y)T_y],\quad v_y=ET_y^2.
\end{aligned}
\tag{C.28}
$$

For the mixed labels, a fixed edge is equal-sign twice and opposite-sign four times; after permutation to the representative this is precisely one equal-sign and two opposite-sign edges averaged. Hence the final derivative is still the free $c_{12}$ derivative, not the three-edge radial derivative.

Set $\lambda=\kappa_3/(6vs)$. Use certified upper bounds for the following norms of the actual Gumbel-smoothed noise:

$$
\begin{aligned}
\|g^{(j)}\|_\infty&\le L_j\quad(j=0,1,2),\\
\|G-\Phi\|_\infty&\le\delta_G,\quad \|g-\phi\|_\infty\le\delta_f,\\
\|g'-\phi'\|_\infty&\le\delta_p,\quad \|g'-\phi'\|_1\le V_{\rm diff},\\
\|g-\phi\|_2^2&\le a_2^2,\quad \|g-\phi-\lambda e\|_\infty\le R_0,\\
\|G-\Phi-\lambda\mathcal P\|_\infty&\le R_{\rm CDF}.
\end{aligned}
\tag{C.29}
$$

$$
\begin{aligned}
M_2&=\sum_yp_yE[|\widetilde H_y-m_y|T_y^2],\\
M_{mv}&=\sum_yp_y|m_y|v_y,\quad M_{cv}=\sum_yp_y|c_y|v_y,\\
v_{\max}&=\max_yv_y,\qquad \Delta=\delta_G+L_1v_{\max}/(2s^2).
\end{aligned}
\tag{C.30}
$$

For an integer $r\ge0$ and a real $d\ge0$, define Gaussian comparison weights, retaining the extreme-value CDF factor:

$$
\begin{aligned}
W_1(r,d)&=\int_{\mathbb R}|\phi'|\min(\Phi+d,1)^r,\\
W_{1,+}(r,d)&=\int_0^\infty|\phi'|\min(\Phi+d,1)^r,\\
W_2(r,d)&=\int_{\mathbb R}|\phi''|\min(\Phi+d,1)^r,\\
\mathcal W_r&=W_1(r,\Delta)+V_{\rm diff}.
\end{aligned}
\tag{C.31}
$$

The five nonnegative budgets are

$$
\begin{aligned}
\mathsf E_1&=\frac{KL_2M_2}{2s^3},\\
\mathsf E_2&=\frac{K(K-1)M_{mv}}{2s^3}
\left[\frac{L_2}{K-1}+L_1\mathcal W_q\right],\\
\mathsf E_3&=\frac{K(K-1)M_{cv}}{2s^4}
\left[L_2\mathcal W_q+qL_1L_0\mathcal W_{q-1}\right].
\end{aligned}
\tag{C.32}
$$

$$
\begin{aligned}
\mathsf E_4&=\frac{K(K-1)|\bar c|}{s^2}
\left[\frac{\delta_p}{K-1}+\delta_GW_2(q,\delta_G)\right],\\
\mathsf E_5&=\frac{K(K-1)|m_H|}{s}\left\{
\frac{2R_0+J_KR_{\rm CDF}}{K-1}+a_2^2+2\delta_f\delta_G\right.\\
&\hspace{18mm}\left.+\frac q2\delta_G^2[W_{1,+}(q-1,\delta_G)+\delta_f]\right\},\\
|D_n-D_n^{(2)}|&\le\mathsf E_n:=\sum_{j=1}^5\mathsf E_j.
\end{aligned}
\tag{C.33}
$$

This implication holds for every $n\ge1$ whenever its stated finite inputs hold; the noise construction below certifies those inputs at the four specified scales. The budgets $\mathsf E_j$ are distinct from the CDF correction function $\mathcal P$.

We give the propagation proof to make signs and constants explicit. Integral Taylor expansion gives

$$
\begin{aligned}
\|f_y-g\|_\infty&\le\delta_{f,y}:=L_2v_y/(2s^2),\\
\|F_y-G\|_\infty&\le\delta_{F,y}:=L_1v_y/(2s^2),\\
h_y-m_yf_y&=-c_yg'/s+e_y,\\
\|e_y\|_\infty&\le L_2E[|\widetilde H_y-m_y|T_y^2]/(2s^2).
\end{aligned}
\tag{C.34}
$$

Since $\int f_yF_y^q=1/(K-1)$, the $e_y$ contribution in (C.20) is bounded by $\mathsf E_1$. Let $I_g=\int g^2G^q$, $J_g=\int g'gG^q$ and $J_\phi=\int\phi'\phi\Phi^q=-T_K^{\rm G}$. Two exact integrations by parts are

$$
\begin{aligned}
\mathcal I(f_y)-I_g
&=\int(f_y-g)f_yF_y^q
-\frac1{q+1}\int g'(F_y^{q+1}-G^{q+1}),\\
J_g-J_\phi
&=\int(g'-\phi')gG^q
-\frac1{q+1}\int\phi''(G^{q+1}-\Phi^{q+1}).
\end{aligned}
\tag{C.35}
$$

The smooth bounded densities tend to zero at infinity; the derivative bounds make these boundary terms and integrals legitimate. Use $|u^a-v^a|\le a|u-v|\max(u,v)^{a-1}$ and $\max(F_y,G)\le\min(\Phi+\Delta,1)$. Thus $\int|g'|\max(F_y,G)^r\le\mathcal W_r$, and

$$
\begin{aligned}
|\mathcal I(f_y)-I_g|&\le\frac{\delta_{f,y}}{K-1}+\delta_{F,y}\mathcal W_q,\\
\left|\int g'f_yF_y^q-J_g\right|
&\le\delta_{f,y}\mathcal W_q+q\delta_{F,y}L_0\mathcal W_{q-1},\\
|J_g-J_\phi|&\le\frac{\delta_p}{K-1}+\delta_GW_2(q,\delta_G).
\end{aligned}
\tag{C.36}
$$

Multiplying by the corresponding coefficients in (C.20) gives $\mathsf E_2,\mathsf E_3,\mathsf E_4$. Crucially, the common kernels are averaged over labels first: their coefficients are $m_H$ and $\bar c$, retaining their signs, rather than averages of absolute values.

For $\mathsf E_5$, put $d=g-\phi$, $D=G-\Phi$ and use the positive-density interpolation $f_t=\phi+td$, $F_t=\Phi+tD$ for $0\le t\le1$. The Edgeworth truncation is not assumed positive. The linear residual and the exact second variation satisfy

$$
\begin{aligned}
|D\mathcal I_\phi[d]-\lambda D\mathcal I_\phi[e]|
&\le\frac{2R_0+J_KR_{\rm CDF}}{K-1},\\
\frac{d^2}{dt^2}\mathcal I(f_t)
&=2\int d^2F_t^q+4q\int f_tdDF_t^{q-1}\\
&\quad+q(q-1)\int f_t^2D^2F_t^{q-2}.
\end{aligned}
\tag{C.37}
$$

The first bound uses $\int\phi\Phi^q=1/(K-1)$ and $q\int\phi^2\Phi^{q-1}=J_{K-1}/(K-1)\le J_K/(K-1)$. The first two terms of the second variation have absolute bounds $2a_2^2$ and $4\delta_f\delta_G$. For the last one,

$$
\begin{aligned}
\int f_t^2F_t^{q-2}
&=\int(f_t-\phi)f_tF_t^{q-2}-\frac1{q-1}\int\phi'F_t^{q-1}\\
&\le\frac{\delta_f+W_{1,+}(q-1,\delta_G)}{q-1},\\
|I_g-I_K-D\mathcal I_\phi[d]|
&\le a_2^2+2\delta_f\delta_G\\
&\quad+\frac q2\delta_G^2[\delta_f+W_{1,+}(q-1,\delta_G)].
\end{aligned}
\tag{C.38}
$$

On the negative half-line $\phi'>0$, so its contribution with the minus sign is nonpositive and can be discarded. This justifies the positive-half-line weight; dropping the sign first would not. The last line retains the Taylor factor $\int_0^1(1-t)dt=1/2$. Combining (C.37)--(C.38), then (C.21) and (C.26), gives $\mathsf E_5$ and closes (C.33). The divisions by $q-1$ are used only at the fixed $q=98$.

## C.10. Finite Fourier bounds for the actual noise

We use the convention $\widehat f(t)=\int e^{itz}f(z)\,dz$. Put $U=(L-\ell)/\sqrt v$, $\mu_3=EU^3$, $\mu_4=EU^4$, $\alpha=\mu_3/6$, $P_n(t)=\psi(t/\sqrt n)^n$, and $Q_n(t)=e^{-t^2/2}(1+\alpha(it)^3/\sqrt n)$. The latter is the transform of $\phi+\lambda e$, not a probability density. The production split is $\delta=1/4$, $a=9/2$, with low, middle and high frequencies separated at $\delta\sqrt n,a\sqrt n$.

**Low frequencies.** Define

$$
\begin{aligned}
q_\delta&=\tfrac12+|\mu_3|\delta/6+\mu_4\delta^2/24,
\quad z_\delta=\delta^2q_\delta,\\
C_\delta&=\mu_4/24+\frac{q_\delta^2}{2(1-z_\delta)},\\
D_\delta&=\frac{\mu_4/6+(1+|\mu_3|\delta/2)q_\delta}{1-z_\delta},\\
c&=\tfrac12-C_\delta\delta^2,\quad
c_{\rm mod}=\tfrac12\{1-(2\mu_4+6)\delta^2/24\}.
\end{aligned}
\tag{C.39}
$$

The certified fourth moment is $\mu_4\simeq4.53778313038698$; the saved intervals give $z_\delta<.034973$, $c>.4780439$, $c_{\rm mod}>.4803703$. Taylor expansion through degree three, $|\log(1+w)-w|\le |w|^2/[2(1-|w|)]$, and $\psi'/\psi$ imply $\log\psi(u)=-u^2/2+\alpha(iu)^3+r(u)$, with $|r(u)|\le C_\delta|u|^4$ and $|r'(u)|\le D_\delta|u|^3$. No fifth moment is needed for these finite bounds. For $0\le t\le\delta\sqrt n$,

$$
\begin{aligned}
|P_n-Q_n|&\le A_0(t):=e^{-ct^2}\frac{C_\delta t^4}{n}
+e^{-t^2/2}\frac{\alpha^2t^6}{2n},\\
|P_n'-Q_n'|&\le A_1(t):=tA_0(t)+e^{-t^2/2}\frac{3\alpha^2t^5}{n}\\
&\quad+e^{-ct^2}\left(\frac{3|\alpha|C_\delta t^6}{n^{3/2}}
+\frac{D_\delta t^3}{n}\right),\\
|P_n(t)|&\le e^{-\max(c,c_{\rm mod})t^2}.
\end{aligned}
\tag{C.40}
$$

The cubic phase is purely imaginary, so $|e^{ib}-1-ib|\le b^2/2$ retains it without an $e^{|b|}$ loss. The additional modulus bound follows by applying the fourth-order cosine bound to $U-U'$, whose fourth moment is $2\mu_4+6$. Positive coefficients are rounded upward and decay rates downward.

**Middle and high frequencies.** At the 273 frequencies $u_j=1/4+j/64$, $j=0,\ldots,272$, certified complex finite sums give upper bounds $b_j$ for $|\psi(u_j)|$. Since $|\psi'|\le E|U|\le1$, each of the 272 cells has the valid cap

$$
\begin{aligned}
\rho_j&=\min\{1,\max(b_j,b_{j+1})+1/128\}<1,\\
|\psi(u)|&\le \min(1,B/|u|),\qquad
B=\frac{2\sqrt v(1+e^{1/2})}{\sqrt{2\pi}}<1.102.
\end{aligned}
\tag{C.41}
$$

For the high-frequency bound, $p_L(\log\sigma(x))=\phi(x)(1+e^x)$ and its log derivative in $x$ is $-x+\sigma(x)$, strictly decreasing with opposite endpoint signs. The density is unimodal and vanishes at both ends. Hence $\|p_L'\|_1=2\sup p_L\le2(1+e^{1/2})/\sqrt{2\pi}$; scaling and integration by parts give (C.41). This supplies an explicit tail constant. On a middle cell $|P_n|\le\rho_j^n$ and $|P_n'|\le\sqrt n\rho_j^{n-1}$. For high-frequency moments,

$$
\begin{aligned}
\int_{a\sqrt n}^{\infty}t^p(B\sqrt n/t)^kdt
&=\frac{n^{(p+1)/2}B^ka^{p-k+1}}{k-p-1},\quad k>p+1,\\
I_p(c;t_0)&:=\int_{t_0}^{\infty}t^pe^{-ct^2}dt,\\
I_0(c;t_0)&=\frac{\sqrt\pi}{2\sqrt c}\operatorname{erfc}(\sqrt c\,t_0),
\quad I_1(c;t_0)=\frac{e^{-ct_0^2}}{2c},\\
I_p(c;t_0)&=\frac{t_0^{p-1}e^{-ct_0^2}+(p-1)I_{p-2}(c;t_0)}{2c}\quad(p\ge2).
\end{aligned}
\tag{C.42}
$$

Middle-cell polynomial integrals are exact endpoint power differences. Formula (C.42) handles powers $n,n-1,2n,2n-2$ and all Gaussian polynomial tails. Low-frequency positive envelopes may be extended to infinity conservatively; their squares include every cross term. Gaussian comparison tails start at $t_0=\delta\sqrt n$ and are counted once, not once per middle cell. No Fourier numerical quadrature is required.

**Gumbel and norm conversion.** For $\chi_n(t)=Ee^{it\xi/s}$, centering and $E\xi^2=\pi^2/6$ imply

$$
\begin{aligned}
|\chi_n|&\le1,\quad |\chi_n-1|\le b(t):=\pi^2t^2/(12nv),\\
|\chi_n'|&\le d(t):=\pi^2|t|/(6nv),\\
R_g&:=\chi_nP_n-Q_n,\\
|R_g|&\le A_g:=A_0+b|Q_n|,\\
|R_g'|&\le A_g':=A_1+d\,e^{-\max(c,c_{\rm mod})t^2}+b|Q_n'|
\quad(0\le t\le\delta\sqrt n).
\end{aligned}
\tag{C.43}
$$

Here $A_g'$ denotes a bound for $|R_g'|$, not a derivative of the envelope $A_g$. On the remaining frequencies use $|R_g|\le M+|Q_n|$ and $|R_g'|\le M_1+dM+|Q_n'|$, where $M,M_1$ are the preceding caps for $|P_n|,|P_n'|$. The centered Gumbel is therefore present in every finite error norm.

Let $r_j=g^{(j)}-(\phi+\lambda e)^{(j)}$. Inversion, CDF inversion and Parseval give the following sufficient bounds:

$$
\begin{aligned}
\|r_j\|_\infty&\le\pi^{-1}\int_0^\infty t^jA_g(t)dt\quad(j=0,1,2),\\
\|G-\Phi-\lambda\mathcal P\|_\infty&\le\pi^{-1}\int_0^\infty A_g(t)/t\,dt,\\
\|r_0\|_2^2&\le\pi^{-1}\int_0^\infty A_g(t)^2dt,\\
\|r_1\|_1&\le\left[\int_0^\infty\{t^2A_g^2+(A_g+tA_g')^2\}\,dt\right]^{1/2},\\
\|g^{(j)}\|_\infty&\le L_{j,\rm direct}:=\pi^{-1}\int_0^\infty t^jM(t)dt.
\end{aligned}
\tag{C.44}
$$

For the $L^1$ line, Cauchy--Schwarz with $(1+z^2)^{-1}$ and Parseval gives $\|r_1\|_1^2\le\tfrac12(\|tR_g\|_2^2+\|(tR_g)'\|_2^2)$. Symmetry turns this into the displayed positive-frequency integral, with no additional $\pi$ factor. The CDF integral is finite at zero: the Gumbel error starts at $t^2$ and the remaining error at $t^4$. On middle and high frequencies, the production bounds take the Minkowski sum of the CF and Gaussian vector $L^2$ caps before squaring; this preserves the same normalization. For direct density caps use $|\chi_n|\le1$.

Write $R_1,R_{02},R_{11}$ for the resulting upper bounds on $\|r_1\|_\infty,\|r_0\|_2,\|r_1\|_1$. The inputs in (C.29) are recovered, including the first Edgeworth term, by

$$
\begin{aligned}
\delta_f&=|\lambda|\|e\|_\infty+R_0,\quad
\delta_p=|\lambda|\|e'\|_\infty+R_1,\\
\delta_G&=\min(1,|\lambda|\phi(0)+R_{\rm CDF}),\\
V_{\rm diff}&=|\lambda|\|e'\|_1+R_{11},\quad
 a_2^2=(|\lambda|\|e\|_2+R_{02})^2,\\
L_0&=\min(L_{0,\rm direct},\phi(0)+\delta_f),\\
L_1&=\min(L_{1,\rm direct},\phi(1)+\delta_p),\qquad L_2=L_{2,\rm direct}.
\end{aligned}
\tag{C.45}
$$

All comparisons use upper endpoints. The required Hermite norms are closed scalar expressions. With $a_*=\sqrt{3-\sqrt6}$, $b_*=\sqrt{3+\sqrt6}$ and probabilists' Hermite polynomials $\operatorname{He}_j$,

$$
\begin{aligned}
\|e\|_\infty&=\max_{z\in\{a_*,b_*\}}|\operatorname{He}_3(z)\phi(z)|,\\
\|e'\|_\infty&=\max_{z\in\{0,\sqrt{5-\sqrt{10}},\sqrt{5+\sqrt{10}}\}}
|\operatorname{He}_4(z)\phi(z)|,\\
\|e'\|_1&=4\sqrt6[a_*\phi(a_*)+b_*\phi(b_*)],\qquad
\|e\|_2^2=\frac{15}{16\sqrt\pi}.
\end{aligned}
\tag{C.46}
$$

These follow from the zeros of $e'=-\operatorname{He}_4\phi$, $e''=\operatorname{He}_5\phi$, total variation between extrema, and Gaussian even moments.

**Certification of the finite CF table and fourth moment.** The original Gaussian variable uses a full-weight trapezoidal sum with $h=1/16$, $x_j=jh$, $|j|\le192$, cutoff $R=12$, strip half-width $\eta=1$ and 192-bit ball arithmetic. Set $d_0=E\log\cosh(Z/2)$, so $L-\ell=x/2-\log\cosh(x/2)+d_0$. On $|\operatorname{Im}x|\le\eta$, $|\operatorname{Im}\log\sigma(x)|\le\eta$. The strip integral caps and resulting errors are

$$
\begin{aligned}
M_\psi(u)&=\exp(\eta^2/2+|u|\eta/\sqrt{v_-}),\\
\varepsilon_\psi(u)&=\frac{2M_\psi(u)}{e^{2\pi\eta/h}-1}+\frac{2\phi(R)}R,\\
B_4^*&=d_{0,+}+\eta+\log\sec(\eta/2),\\
M_4&=e^{\eta^2/2}E(|Z|+B_4^*)^4,\\
\varepsilon_4&=\frac{2M_4}{e^{2\pi\eta/h}-1}
+\sum_{k=0}^4{4\choose k}d_{0,+}^{4-k}P_k(R).
\end{aligned}
\tag{C.47}
$$

The fourth-moment strip bound uses $|L-\ell|\le|\operatorname{Re}x|+B_4^*$. Its expectation is expanded with absolute Gaussian moments $1,\sqrt{2/\pi},1,2\sqrt{2/\pi},3$. The infinite trapezoidal alias bound is the strip-boundary integral divided by $e^{2\pi\eta/h}-1$, summed over both boundaries. For the omitted real grid tail, monotonicity of $x^k\phi(x)$ for $x\ge R\ge2$, $k\le4$, gives

$$
\begin{aligned}
P_0(R)&=2\phi(R)/R,\quad P_1(R)=2\phi(R),\\
P_2(R)&=2\phi(R)(R+1/R),\\
P_3(R)&=2\phi(R)(R^2+2),\\
P_4(R)&=2\phi(R)(R^3+3R+3/R).
\end{aligned}
\tag{C.48}
$$

The raw fourth moment receives $\varepsilon_4$ before division by the saved interval for $v^2$. Each complex sum receives the complex-modulus error $\varepsilon_\psi$, and its true modulus cap is $|\text{finite sum}|+\varepsilon_\psi$. All input dyadic intervals and arithmetic radii are retained. The 385 shared Gaussian nodes produce 273 complex CF sums and one real fourth-moment sum: 547 real scalar sums and 105,105 complex exponentials. These are model-noise calculations, not direct evaluations of $D_n$.

## C.11. Active moment caps without new active integrals

All active inputs follow from the previously certified scalars $V_A,V_H,s_0,m_H,\bar c$. Here $V_H=E_Y\operatorname{Var}_XH_{12,Y}$ and $s_0=E\operatorname{sech}^2(Z/2)$. Write $d(x)=\log\cosh(x/2)$, $\mathcal D=\sum_{i=1}^3[d(X_i)-Ed(Z)]$, $S_y=y^TX/2$, and $a_y=Cy/2$. Then $T_y=S_y-\mathcal D$; odd/even symmetry gives $ES_y\mathcal D=0$. Put $\lambda_C=1+2\rho$, $\kappa=4m_H$ and $t_2=1-s_0$.

$$
\begin{aligned}
V_d&=E\mathcal D^2=V_A-3/4,\\
\sigma_+^2&=3\lambda_C/4,\quad \sigma_0^2=(3-2\rho)/4,\\
v_+&=\sigma_+^2+V_d,\qquad v_0=\sigma_0^2+V_d.
\end{aligned}
\tag{C.49}
$$

For the class covariances, Price differentiation along the three-edge $\rho$ path is used only as a moment identity. Twice integrating by parts gives $ES_y^2\mathcal D=(s_0/4)\|a_y\|^2$, hence

$$
\begin{aligned}
ET_y^3&=-\tfrac{3s_0}4\|a_y\|^2-E\mathcal D^3,\\
\frac d{d\rho}ET_y^3&=6\sum_{i<j}E[T_yH_{ij,y}]=18c_y,\\
\|a_+\|^2-\|a_0\|^2&=4\rho+2\rho^2,\\
c_+&=\bar c-\frac{s_0(1+\rho)}8,\qquad
c_0=\bar c+\frac{s_0(1+\rho)}{24}.
\end{aligned}
\tag{C.50}
$$

The factor 18 is the cubic mixed-derivative factor 6 times three edges. Dividing by it and using $\bar c=(c_++3c_0)/4$ proves the last line; no radial derivative is substituted for $D_n$.

For the fourth moment, write $X=C^{1/2}Z_3$. Since $|d'|\le1/2$, $\|\nabla_{Z_3}\mathcal D\|^2\le3\lambda_C/4$. Gaussian Poincare applied to $\mathcal D^2$ yields

$$
E\mathcal D^4-V_d^2\le4E[\mathcal D^2\|\nabla_{Z_3}\mathcal D\|^2]
\le3\lambda_CV_d,\qquad Q_4:=V_d^2+3\lambda_CV_d.
\tag{C.51}
$$

One may obtain the Poincare inequality here from $\operatorname{Var}u=2\int_0^\infty E\|\nabla P_tu\|^2dt$, $\nabla P_tu=e^{-t}P_t\nabla u$ and Jensen. The functions and their gradients are square-integrable by Gaussian moments.

The potentially signed mixed term requires more than an absolute-value bound. Twice integrating by parts gives

$$
\begin{aligned}
ES_y^2\mathcal D^2
&=\sigma_y^2V_d+2E(a_y\cdot\nabla\mathcal D)^2
+2E[\mathcal D(a_y\cdot\nabla)^2\mathcal D],\\
E[\mathcal D(a_y\cdot\nabla)^2\mathcal D]&\le0.
\end{aligned}
\tag{C.52}
$$

To prove the second line, condition on $X_i=x$. Each $E[d(X_j)\mid X_i=x]$ is even and convex in $x$, hence nondecreasing in $|x|$, whereas $d''(x)=\tfrac14\operatorname{sech}^2(x/2)$ is nonincreasing in $|x|$. For independent copies $R,R'$ of $|X_i|$, the covariance identity $2\operatorname{Cov}(a(R),b(R))=E[(a(R)-a(R'))(b(R)-b(R'))]\le0$ proves $\operatorname{Cov}(\sum_jd(X_j),d''(X_i))\le0$. The Hessian of $\mathcal D$ is diagonal, so summing with coefficients $(a_y)_i^2\ge0$ proves the required sign.

$$
\begin{aligned}
Q_y&:=E(a_y\cdot\nabla\mathcal D)^2
=\tfrac14\{(t_2-\kappa)\|a_y\|^2+\kappa(\mathbf1^Ta_y)^2\},\\
ET_y^4&\le3(\sigma_y^2)^2+6\sigma_y^2V_d+12Q_y+Q_4,\\
\overline T_4&:=\frac{27+36\rho^2}{16}
+\left(\frac92+3\lambda_C\right)V_d+V_d^2\\
&\quad+\frac94(1+2\rho^2)t_2+\frac92\rho(2+\rho)\kappa,\\
\sum_yp_yET_y^4&\le\overline T_4.
\end{aligned}
\tag{C.53}
$$

Odd powers of $S_y$ vanish under global reflection. The label averages used in the last two lines are $E_y\sigma_y^2=3/4$, $E_y(\sigma_y^2)^2=(9+12\rho^2)/16$, $E_y\|a_y\|^2=(3+6\rho^2)/4$, and $E_y(\mathbf1^Ta_y)^2=3\lambda_C^2/4$. Jensen for the three-edge average and label-permutation symmetry imply $\sum_yp_y\operatorname{Var}(\widetilde H_y)\le V_H$. Cauchy--Schwarz on $(y,X)$ now gives

$$
M_2\le\sqrt{V_H\overline T_4}<.307468813.
\tag{C.54}
$$

This is an upper bound for a remainder moment, not a measured value of $M_2$. Together with (C.28), (C.49) and (C.50), it supplies every active input in (C.30), without new active Gaussian integration.

## C.12. Gaussian weight enclosures

The four weights required for each scale are $W_1(98,\Delta)$, $W_1(97,\Delta)$, $W_2(98,\delta_G)$ and $W_{1,+}(97,\delta_G)$. Their computation uses only $\phi,\Phi$ on the 3073-node grid $[-12,12]$ with spacing $1/128$, at 192 bits. For a cell $[a,b]$, the weight $w(z)=\min(\Phi(z)+d,1)^r$ is nondecreasing. Use $w_- =\min(\underline\Phi(a)+d,1)^r$ and $w_+=\min(\overline\Phi(b)+d,1)^r$, clipping the CDF enclosures to $[0,1]$. The grid includes $-1,0,1$, so no cell crosses an unaccounted derivative sign change.

$$
\begin{aligned}
\int_a^b|\phi'|&=|\phi(b)-\phi(a)|,\\
\int_a^b|\phi''|&=|-b\phi(b)+a\phi(a)|,\\
M_-w_-&\le\int_a^bw(z)|\phi^{(j)}(z)|dz\le M_+w_+,\\
\int_{|z|>R}|\phi'|&=2\phi(R),\quad \int_R^\infty|\phi'|=\phi(R),\\
\int_{|z|>R}|\phi''|&=2R\phi(R)\qquad(R\ge2).
\end{aligned}
\tag{C.55}
$$

The first two lines apply on cells not crossing $0$ and $\{-1,1\}$, respectively; $[M_-,M_+]$ is the interval for the relevant derivative mass. Summing lower and upper products, and adding the displayed tail only to the upper sum, encloses each weight. The propagation uses upper endpoints and an upper value of $d$. Thus all 16 full-line weight bounds follow from saved Gaussian values and monotonicity, with no quadrature remainder inferred from grid agreement.

## C.13. Model constants and complete four-scale synthesis

For clarity, the strict constants are model integrals, not fitted gradient coefficients. A low-dimensional evaluation of (C.19) uses $t(x)=\tanh(x/2)$, $d(x)=\log\cosh(x/2)$ and $X_i=\sqrt\rho U_0+\sqrt{1-\rho}V_i$, with independent standard Gaussians. Define $\tau(u)=E_Vt(X)$, $\beta(u)=E_Vd(X)$, $\gamma(u)=E_V[t(X)d(X)]$, $\nu(u)=E_Vt(X)^2$ at $X=\sqrt\rho u+\sqrt{1-\rho}V$. Label averaging and Gaussian integration by parts give

$$
\begin{aligned}
d_0&=Ed(Z),\quad s_0=1-Et(Z)^2,\quad v=\tfrac14+\operatorname{Var}d(Z),\\
\kappa_3&=-3s_0/16-E[(d(Z)-d_0)^3],\quad \kappa=E\tau(U_0)^2,\\
m_H&=\kappa/4,\\
\bar c&=-\rho s_0/8-\{2E[\tau\gamma]+E[\tau^2\beta]-3d_0\kappa\}/4,\\
V_H&=\{2Et(Z)^2+E\nu(U_0)^2-\kappa^2\}/16,\\
V_A&=3/4+3(v-1/4)+6\{E\beta(U_0)^2-d_0^2\}.
\end{aligned}
\tag{C.56}
$$

Here each outer expectation is over $U_0$. The third active coordinate contributes $E[\tau^2\beta]-d_0\kappa$ and is not discarded. These expressions evaluate free-coordinate moments at the equicorrelation point; they do not differentiate the three-edge path.

Their certified evaluation used the full-weight Gaussian trapezoid with spacing $1/16$, cutoff $10$, strip half-width $3/4$ and 192 bits. The bounds $|t(x+iy)|\le1$ for $|y|\le\pi/2$ and $|d(x+iy)|\le|x|/2+\log\sec(|y|/2)+|y|/2$ give the same alias formula as (C.47) with explicit polynomial Gaussian strip and tail envelopes. Each inner conditional integral is enlarged by its own alias and tail bounds before products are formed; the outer sums then receive their own errors. The Gaussian order-statistic integrals in (C.21) are treated separately, using $|\Phi(x+i\eta)|\le1+\eta e^{\eta^2/2}/\sqrt{2\pi}$. The full envelopes and exact saved intervals are provided with the companion constant calculation; no ordinary floating-point rule difference is used as a strict error bound.

```{=latex}
\Needspace{12\baselineskip}
```

|Constant|Approximate value|
|---|---:|
|$v$|.271514501800558731839|
|$\kappa_3$|−.16208683226381804873|
|$m_H$|.01927127885302918387252|
|$\bar c$|−.0547240726452943675562|
|$J_{100}$|2.50759363644168|
|$B_{100}$|5.47243065910866|
|$V_H$|.02360133572724289470720|
|$V_A$|.84051923403528451804|

$$
\begin{aligned}
b_2^{\rm nuisance}&=B_{100}m_H\kappa_3/(3v^2)\simeq-.0772914085493864,\\
b_2^{\rm mark}&=-B_{100}\bar c/v\simeq1.102974945903173,\\
1.02568353735378&<b_2<1.02568353735380.
\end{aligned}
\tag{C.57}
$$

The large positive correction is primarily the signed mark--score term. This is a mechanism in the analytic expansion, not a causal attribution of training outcomes. The table decimals are for reading; the strict coefficient interval and the final outward bounds below use the full saved dyadic inputs.

|$n$|$D_n^{(2)}$, approximately|Strict upper bound on $\mathsf E_n$|Outward enclosure of $D_n$|Sign|
|---:|---:|---:|---|---|
|64|+.00443369615|99.704837|[−99.70041, 99.70928]|Unresolved|
|256|−.00178972824|.35027010|[−.35205983, .34848037]|Unresolved|
|1024|−.00189650820|.011695409|[−.013591917, .009798901]|Unresolved|
|4096|−.00119866512|.000846727778|[−.002045393, −.0003519373]|Strictly negative|

All five budget expressions are retained below as approximate readouts; none is an actual-error lower bound.

|$n$|$\mathsf E_1$|$\mathsf E_2$|$\mathsf E_3$|$\mathsf E_4$|$\mathsf E_5$|
|---:|---:|---:|---:|---:|---:|
|64|.135264699|12.7311527|82.8304548|.445542322|3.56242253|
|256|.0114049517|.0790174116|.223704548|.00979174301|.0263514419|
|1024|.00140485780|.00341057177|.00422225078|.00071844314|.00193928492|
|4096|.00017560722|.00026772948|.00014413379|.00006758914|.00019166814|

The valid but loose 64-scale bound is not an explanation of its old positive derivative estimate. Its second-order center also lies outside the old C.6 interval. At 256 and 1024, a negative center does not settle the actual sign. At 1024, even setting $\mathsf E_2=\mathsf E_3=0$ while retaining the other budgets leaves $\mathsf E_1+\mathsf E_4+\mathsf E_5\simeq.00406258585509>|D_{1024}^{(2)}|$; $\mathsf E_5$ alone already exceeds that window. Thus reducing only those two budgets cannot make this fixed recipe pass. This says nothing about a lower bound on the true error or the impossibility of another method.

At the retained 4096-scale point the full finite synthesis proves

$$
\mathsf E_{4096}<.000846727778,\qquad
-.002045393<D_{4096}<-.0003519373<0.
\tag{C.58}
$$

There is also a direct certified connection to the original leading field. Let $D_n^{(0)}=-m_HJ_{100}/\sqrt{nv}$. Then

$$
\begin{aligned}
D_{4096}^{(0)}&\simeq-.00144907613957,\qquad b_2/4096\simeq.000250411019862,\\
|D_{4096}-D_{4096}^{(0)}|
&\le |b_2|/4096+\mathsf E_{4096}\\
&<.001097138798<|D_{4096}^{(0)}|.
\end{aligned}
\tag{C.59}
$$

The first-order relative upper budget is about 75.7%, and the second-order remainder budget is about 70.6% of its center magnitude. This is a sign-level finite applicability witness, not a 10% approximation guarantee. The assertion is at $n=4096$, not a threshold or a statement for all larger $n$; it does not establish a full-matrix finite remainder, a parameter region, or an optimizer outcome.

The old four-scale panel was visible before the correction was derived. Model constants were not regressed on its gradients, but the comparison remains informed retrospective analysis. The new deterministic enclosures are not confidence intervals, spend no alpha, and do not upgrade the old ideal-iid or native-arithmetic qualifications. The former numerical reference for $J_{100}$ is preserved in C.6 with its own provenance.

The strictness of (C.58)--(C.59) comes from analytic envelopes and saved ball arithmetic, followed by exact endpoint composition. Model-constant certification used 1296 scalar sums and 103041 nested nodes; the later noise and weight computations used the counts in C.10 and C.12. Separate ordinary Gaussian-Hermite and Gaussian-Legendre expressions check agreement, including all 273 CF values, but are not second strict certificates. Independent rational reconstruction checks the 16 weight enclosures, 20 budget expressions and four final intervals from saved endpoints; it does not re-prove all analytic envelopes. No direct $D_n$ integral, new random bank or optimization is used. The companion records contain the full finite sums, error envelopes and arithmetic receipts. The alternative quantile-window interface is not part of this full-line certificate: its central window was not computed, and its separate tail budget is not added here.

## C.14. A local observation-risk consequence at the certified finite point

For three centered Gaussian logits, global reflection eliminates all odd spin moments. Conditional independence and expansion of the three Bernoulli factors therefore give the first identity below. The teacher is uniform on eight labels, and $\kappa(c)=E_c[\tanh(X_1/2)\tanh(X_2/2)]$. Each two-variable marginal depends only on its own correlation, so a free $c_{12}$ change leaves $\kappa(c_{13}),\kappa(c_{23})$ unchanged:

$$
\begin{aligned}
Q_C(y)&=\tfrac18\left(1+\sum_{i<j}\kappa(c_{ij})y_iy_j\right),\\
\left.\partial_{c_{12}}\mathcal R(C)\right|_{C=C(\rho)}
&=-\frac{\kappa'(\rho)}8
\left(\frac2{1+3\kappa(\rho)}-\frac2{1-\kappa(\rho)}\right)\\
&=\frac{\kappa'(\rho)\kappa(\rho)}{(1-\kappa(\rho))(1+3\kappa(\rho))}>0,
\qquad \rho=9/20.
\end{aligned}
\tag{C.60}
$$

Among the two all-equal labels, $y_1y_2$ sums to 2 and the sum of the three pair products is 3. Among the other six labels these values are −2 and −1, respectively; differentiating uniform-teacher cross entropy gives (C.60), with KL differing only by a constant. Equation (5) gives $\kappa'(c)>0$; since $\kappa(0)=0$ and $|\tanh|<1$, $0<\kappa(9/20)<1$. This is an analytic positivity argument, not a new numerical integral for $\kappa'$. Changing all three edges along $\rho$ would multiply this derivative by 3.

For $A=e_1e_2^\top+e_2e_1^\top$, set $C_h=C(9/20)+hA$. Here $\|A\|_{\rm op}=1$ and $\lambda_{\min}(C(9/20))=11/20$, so $|h|<9/20$ keeps $C_h$ inside the paper's $\delta=.1$ correlation class. Both Gaussian and fair observation marginals stay fixed. This bound ensures feasibility only. Combining differentiability, (C.58) and (C.60), there exists $\varepsilon>0$ such that every $0<h<\varepsilon$ is feasible and satisfies $F_{4096,100}^*(C_h)<F_{4096,100}^*(C(9/20))$ and $\mathcal R(C_h)>\mathcal R(C(9/20))$. No explicit monotonicity radius or finite risk increase is supplied. The conclusion is one-sided, local and coordinate-specific; it does not assert a stationary point or an optimizer outcome.

## C.15. Complete finite-program statements {#finite-evidence-summary}

This instance tests matched-run selection, low-risk population near-minimizers, curvature at truth and fresh-run risk. Its separate comparisons and audits do not require a finite remainder bound for Theorem 1.

Fix $m=3$, $\mu=0$, $C_*=I$, $\delta=.1$, $K=100$. Two programs share 65,536 labels, active particles, 143 feasible candidates and lexicographic tolerance $10^{-12}$. Replacing uniform weights by weights from 256 independent nuisance factors changes one matched realization's selection from $I$ (KL zero) to equicorrelation $C_b$ with off-diagonals .9, spectrum (.1,.1,2.8) and KL .03110309. Selection does not use risk. Scores average 16,384 shared outer banks over empirical label frequencies; 100 is the inner particle count.

Write $F_n=F_{n,100}$ after removing candidate-independent nuisance constants. For the continuous low-risk set $\mathcal R(C)\le r_0=10^{-6}$, Price--Pinsker and matrix-square-root derivative bounds give $\widehat F(C)\ge\widehat F(I)-L_r$. Hence an output no worse than $C_b$ obeys

$$
\widehat F(I)-\widehat F(C_b)>L_r
\quad\Longrightarrow\quad \mathcal R(\widehat C)>r_0.
$$

Interval replay gives a saved-input empirical margin .003906703321, excluding the continuous low-risk set, not only the grid point $I$.

Appendix C retains the failed original population-region gate. A deterministic refinement gives population cost $c_d<.001447106$; subtracting it from the separate-bank audit bound on $F_{256}(I)-F_{256}(C_b)$ leaves a margin above $.00134$ on the same 99% event (Appendix D.5). With $\eta=.00134$ and the declared sampling/input qualifications,

$$
\begin{gathered}
\inf_{\mathcal R(C)\le10^{-6}}F_{256}(C)
>F_{256}(C_b)+\eta,\\
F_{256}(\widehat C)-\inf_C F_{256}(C)<\eta
\ \Longrightarrow\ \mathcal R(\widehat C)>10^{-6}.
\end{gathered}
$$

This neither proves global optimality of $C_b$ nor gives its point risk .03110309 to every minimizer.

Under the declared ideal iid sampling and input numerical qualifications, a separate $L=262{,}144$-bank moment audit gives, with at least 99% audit confidence, the following bound for a **fresh run** of the mathematical nuisance program with 143 candidates, $K=100$, $N=65{,}536$, $B=16{,}384$ and the same tolerance:

$$
\begin{gathered}
\Pr_{\rm fresh}\{\mathcal R(\widehat C)\ge r_{\rm grid}\}\ge .7035,\\
r_{\rm grid}=\frac{0.09\,\exp(-1)}{32}>.00103466.
\end{gathered}
$$

Here 70.35% bounds a probability, not an observed frequency; 99% is confidence in that bound. Every non-$I$ grid point has a correlation of magnitude at least .3, giving $r_{\rm grid}$ by Price--Pinsker, while a positive empirical anchor gap excludes $I$. Appendix D combines Cantelli with shared-bank moment and variance bounds; the $NB$ pairs are dependent. The population-region and fresh-run statements share an audit event but assert different conclusions. Neither covers continuous optimizers, finer grids or uniform-branch recovery. Native floating-point execution additionally needs a future-run error guarantee beyond the fresh-run bound's sampling/input qualifications.
