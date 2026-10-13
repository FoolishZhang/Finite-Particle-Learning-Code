# Appendix F. Known means and observation-equivalent scales

## F.1. Matched means and smoothness

Fix finite $K$, three unit Gaussian marginal variances and a known mean $\mu$ shared by teacher and candidate. Only $\theta=(c_{12},c_{13},c_{23})$ is fitted, with $C(\theta)$ the unit-diagonal correlation matrix and teacher $C_*=I$. Write $p_i=E\sigma(\mu_i+X)$ for $X\sim N(0,1)$, $q_i(+1)=p_i$, $q_i(-1)=1-p_i$. For normalized nonnegative weights $a_k$ summing to one, independent of active Gaussians and labels, with law independent of $(\theta,\mu)$, define

$$
\begin{gathered}
Q_{I,\mu}(y)=\prod_iq_i(y_i),\\
\Phi_y(X,\mu,a)=-\log\sum_{k=1}^Ka_k\prod_{i=1}^3\sigma\{y_i(\mu_i+X_{ki})\},\\
F_n(\theta,\mu)=\sum_yQ_{I,\mu}(y)E_{a,X}\Phi_y(X,\mu,a),\quad X_k\stackrel{\rm iid}{\sim}N(0,C(\theta)),\\
G_n=\nabla_\theta F_n,\qquad
Q_{C,\mu}(y)=E_{X\sim N(0,C)}\prod_i\sigma\{y_i(\mu_i+X_i)\},\\
\mathcal R(\theta,\mu)=D(Q_{I,\mu}\Vert Q_{C(\theta),\mu}).
\end{gathered}
\tag{F.1}
$$

Here $n=0$ means $a_k=1/K$; $n=256$ uses the independent nuisance weights. Normalization removes only a parameter-independent additive term. No inverse weight moments are required: uniformly over weights,

$$
0\le\Phi_y(X,\mu,a)\le3\log2+\|\mu\|_1+\sum_i\max_k|X_{ki}|.
\tag{F.2}
$$

On compact positive-definite covariance neighborhoods, Gaussian density derivatives have polynomial-times-Gaussian envelopes. Every positive-order derivative of $\log\sigma$ is bounded; derivatives of the log sum are finite combinations of these derivatives and normalized posterior weights. These bounds permit differentiation under expectation to every fixed order, including derivatives of the teacher probabilities. Thus $F_n$ is jointly $C^\infty$. This concerns known matched means, not means fitted jointly or estimated with error.

For $D=\operatorname{diag}(d_1,d_2,d_3)$, $d_i\in\{-1,1\}$, let $S_D=\operatorname{diag}(d_1d_2,d_1d_3,d_2d_3)$. Gaussian sign changes and label relabeling give

$$
F_n(S_D\theta,D\mu)=F_n(\theta,\mu),\qquad
G_n(S_D\theta,D\mu)=S_DG_n(\theta,\mu).
\tag{F.3}
$$

At truth, $G_{n,ij}(0,\mu)$ is odd separately in $\mu_i,\mu_j$ and even in the remaining mean. Two applications of smooth division, followed by coordinate permutation symmetry, yield

$$
G_{n,ij}(0,\mu)=\mu_i\mu_jb_{n,ij}(\mu)
=b_n\mu_i\mu_j+O(\|\mu\|^4).
\tag{F.4}
$$

Each smooth $b_{n,ij}$ is even in every mean coordinate and shares $b_{n,ij}(0)=b_n$. At most one nonzero mean coordinate therefore leaves $I$ exactly stationary for any finite mean magnitude. Its extremum type need only persist locally for small means. General matched means need not leave $I$ stationary; Appendix G supplies a counterexample, without resolving the corresponding coefficient at $K=100$.

## F.2. Nearby stationary branches and exact risk

At $\mu=0$, Appendix C gives $G_n(0,0)=0$ and $D_\theta G_n(0,0)=h_nI_3$: at $K=100$, $h_0>.00410$ analytically, while $h_{256}<0$ on the centered curvature event with its numerical qualification. Whenever $h_n\ne0$, the implicit function theorem gives a unique smooth stationary branch in a neighborhood of truth. Hessian continuity preserves its strict local minimum or maximum type. Sign equivariance and local uniqueness imply

$$
\begin{gathered}
\theta_{n,ij}(\mu)=\mu_i\mu_jt_{n,ij}(\mu)
=-\frac{b_n}{h_n}\mu_i\mu_j+O(\|\mu\|^4),\\
\theta_n(\mu)=O(\|\mu\|^2).
\end{gathered}
\tag{F.5}
$$

The neighborhood can lie within $C\succ.1I$. It is local, need not be uniform in $K$, and does not certify that a specified moderate mean belongs to it.

Set $v_i=E\sigma'(\mu_i+X)>0$. Price's identity at $I$ gives the probability derivative $y_iy_jv_iv_j\prod_{\ell\ne i,j}q_\ell(y_\ell)$. Under the teacher product law, $E[Y_i/q_i(Y_i)]=0$ and $E[\{Y_i/q_i(Y_i)\}^2]=1/[p_i(1-p_i)]$. Consequently the exact observation-risk Hessian is diagonal Fisher information:

$$
\begin{gathered}
\nabla_\theta^2\mathcal R(0,\mu)=\operatorname{diag}_{i<j}r_{ij}(\mu),\qquad
r_{ij}(\mu)=\frac{v_i^2v_j^2}{p_i(1-p_i)p_j(1-p_j)}>0,\\
\mathcal R(\theta,\mu)=\frac12\sum_{i<j}r_{ij}(\mu)\theta_{ij}^2+O(\|\theta\|^3),\\
\mathcal R(\theta_n(\mu),\mu)
=\frac{r_*}{2}\left(\frac{b_n}{h_n}\right)^2\sum_{i<j}\mu_i^2\mu_j^2
+O(\|\mu\|^6)=O(\|\mu\|^4),\\
r_*=16\{E\sigma'(X)\}^4.
\end{gathered}
\tag{F.6}
$$

The coefficient $b_n$ has not been proved nonzero. The risk order is an upper bound, not $\Theta(\|\mu\|^4)$; risk is exactly zero on the mean axes. More generally, every feasible $C\ne I$ has strictly positive risk: the pair probability $E[\sigma(\mu_i+X_i)\sigma(\mu_j+X_j)]$ has strictly positive correlation derivative $E[\sigma'(\mu_i+X_i)\sigma'(\mu_j+X_j)]$, so a nonzero correlation changes an observed pair marginal.

## F.3. Full derivative audit at an asymmetric mean

The fixed point $\mu_*=(-.2,.1,.3)$, $K=100$, uses 262,144 fresh iid banks, with active Gaussians shared between the two weight conditions. No mean, particle count or bank count was selected from these results. Teacher probabilities are approximately $(.458758831114,.520651702363,.561706803459)$, enclosed by 128-bit Acb integration on $[-12,12]$ with Mills tail bounds and checked against GH64/96. Each bank averages **all eight labels with their teacher probabilities**; the single all-positive-label shortcut is unavailable here.

For fixed $y$, put $x_{ki}=\mu_i+Z_{ki}$ and let $\pi_k$ be proportional to $a_k\prod_i\sigma(y_ix_{ki})$. Define

$$
\begin{gathered}
q_{ki}=y_i\sigma(-y_ix_{ki}),\quad
r_{ki}=\sigma(-y_ix_{ki})\{2\sigma(-y_ix_{ki})-1\},\\
A_p=\sum_k\pi_kq_{ki}q_{kj},\quad B_p=\sum_k\pi_k^2q_{ki}q_{kj},\quad p=(i,j).
\end{gathered}
\tag{F.7}
$$

The Price operator $L_p=\sum_k\partial_{x_{ki}}\partial_{x_{kj}}$ has no extra factor $1/2$. It gives the gradient kernel below and, for distinct pairs $p=(a,b)$ and $q=(a,c)$, the off-diagonal Hessian kernel:

$$
\begin{aligned}
\mathsf G_p={}&-\sum_k\pi_k(1-\pi_k)q_{ki}q_{kj},\\
\mathsf H_{pq}={}&\sum_kq_{kb}q_{kc}\left[-\pi_k(1-\pi_k)(1-2\pi_k)r_{ka}
+3\pi_k^2(1-2\pi_k)q_{ka}^2\right]\\
&+A_pA_q-2(B_pA_q+A_pB_q)+6B_pB_q.
\end{aligned}
\tag{F.8}
$$

The diagonal kernel is (C.14), with these signed $q_{ki}$. Formula (F.8) follows by applying the fourth-order derivative partitions of a negative log sum, then combining equal- and distinct-particle terms. Since $|q_{ki}|\le1$, $r_{ki}\in[-1/8,1]$, the gradient lies in $[-1,1]$ and the diagonal kernel in $[-33/8,67/16]$. For each pair, all $q_{ki}q_{kj}$ have the same label sign, so $B_p=sA_p$ with $s\in[0,1]$. The last three terms in (F.8) have magnitude at most $3$, since $|1-2s-2t+6st|\le3$; the local sum has magnitude at most $4$. Thus $\mathsf H_{pq}\in[-7,7]$. Label averaging preserves these bounds.

The two arms' three gradients and six unique Hessian entries form one 18-component family with $\alpha=.01$. For unbiased bank variance $s_j^2$, range length $L_j$ and $N=262144$, its empirical Bernstein intervals use

$$
\widehat m_j\ \pm\left\{\sqrt{\frac{2s_j^2\log7200}{N}}
+\frac{7L_j\log7200}{3(N-1)}+\varepsilon_j\right\},\qquad
L_j\in\{2,133/16,14\}.
\tag{F.9}
$$

Here $\varepsilon_j=10^{-9}+M_j\delta_Q$, where $M_j\in\{1,67/16,7\}$ bounds the absolute kernel and $\delta_Q\le5.56\times10^{-16}$ bounds teacher-probability rounding. Banks, rather than labels, coordinates or arms, are the independent units. All 18 intervals follow, with endpoints rounded outward to 12 decimal places; Hessian indices follow $(12,13,23)$.

|Component|$n=0$ interval|$n=256$ interval|
|---|---|---|
|$g_{12}$|[-0.000184565302, 0.000179529158]|[-0.000253339714, 0.000229858472]|
|$g_{13}$|[-0.000195341459, 0.000168423016]|[-0.000270877759, 0.000210500179]|
|$g_{23}$|[-0.000177779284, 0.000186080091]|[-0.000235312039, 0.000246075882]|
|$H_{12,12}$|[0.027349272457, 0.028693509363]|[-0.004963286860, -0.003481915149]|
|$H_{13,13}$|[0.027054242599, 0.028398354055]|[-0.004906087758, -0.003426349185]|
|$H_{23,23}$|[0.027163138163, 0.028507306983]|[-0.004915420006, -0.003435231666]|
|$H_{12,13}$|[-0.001116578916, 0.001113107665]|[-0.001128737702, 0.001143888725]|
|$H_{12,23}$|[-0.001109989535, 0.001119719329]|[-0.001123088450, 0.001149345972]|
|$H_{13,23}$|[-0.001113738311, 0.001115930316]|[-0.001136064635, 0.001136714384]|

Gershgorin bounds from the full, unrounded simultaneous intervals give

$$
\lambda_{\min}(H_0)\ge .02482173336837,\qquad
\lambda_{\max}(H_{256})\le-.00114574607762.
\tag{F.10}
$$

All six gradient intervals contain zero; this establishes neither zero nor nonzero gradients, and does not identify $I$ as an extremum at $\mu_*$. On 32 prespecified banks, 576 derivative components were recomputed with 128-bit Arb; the largest discrepancy was $2.19\times10^{-16}$. These checks support the explicit allowance, not a global floating-point proof for every Gaussian, log-sigmoid or weight evaluation. The population conclusion retains that numerical qualification.

## F.4. Risk conflict and scale continuity

Negative definiteness at truth implies $\nabla_\theta^2F_{256}\preceq-\lambda I_3$ in some neighborhood, with $\lambda>0$. Writing $g=G_{256}(0,\mu)$ gives

$$
\begin{gathered}
F_{256}(\theta,\mu)-F_{256}(0,\mu)
\le g^\top\theta-\frac\lambda2\|\theta\|^2,\\
F_{256}(\theta,\mu)+F_{256}(-\theta,\mu)-2F_{256}(0,\mu)
\le-\lambda\|\theta\|^2<0\quad(\theta\ne0).
\end{gathered}
\tag{F.11}
$$

Thus every sufficiently short unoriented line through $I$ has at least one side that lowers the finite objective; both nonzero sides increase exact risk. Every direction descends only when $g=0$. This applies at $\mu_*$ on its own event, and to sufficiently small means on the centered event. It makes no optimizer claim.

At $K=100$, keep all 256 nuisance observations and replace their logits by $\tau G_j$, $0\le\tau\le1$. Each remains an independent fair bit, so

$$
\begin{gathered}
Q_{C,\mu,\tau}(y,u)=Q_{C,\mu}(y)2^{-256},\\
a_k(\tau)=\frac{\prod_{j=1}^{256}\sigma(\tau G_{kj})}
{\sum_\ell\prod_{j=1}^{256}\sigma(\tau G_{\ell j})}.
\end{gathered}
\tag{F.12}
$$

At $\tau=0$ the weights are uniform, while the observations remain present. Pointwise continuity and weight-independent envelopes make the correlation Hessian continuous in $\tau$. At $\mu=0$ it is $h(\tau)I_3$, with $h(0)>.00410$ and $h(1)<0$ on the centered event. At least one interior zero-curvature scale follows. For sufficiently small general means, or for $\mu_*$ on its own event, endpoint Hessians are positive and negative definite, so their continuous eigenvalues must cross zero. With nonstationary truth this is a Hessian-signature change, not a stationary bifurcation. No scale scan, location, uniqueness, monotonicity or global stationary-branch continuation is established.

The matched-weight population audit, centered curvature audit and asymmetric-mean audit each have a 99% family event. Their joint coverage is conservatively at least 97%; including the older $K=2$ finite-selection family gives at least 96%, without requiring independence. Deterministic risk, branch and scale consequences use the same relevant events and consume no additional probability budget. These probability statements preserve each audit's numerical qualification.
