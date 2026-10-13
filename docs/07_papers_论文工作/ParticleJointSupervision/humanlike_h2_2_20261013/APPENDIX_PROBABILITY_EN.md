# Appendix D. Fresh-run risk for a fixed finite search

## D.1. The mathematical program and its risk separation

Fix the centered three-variable Gaussian-logistic model, independent teacher $C_*=I$, unit marginal variances, $K=100$ and $n=256$. The candidate set $\mathcal C_{\rm grid}$ consists of the 143 triples in $\{-.9,-.6,-.3,0,.3,.6,.9\}^3$ whose unit-diagonal correlation matrix satisfies $C\succeq.1I$. The truth $I$ and the equicorrelation anchor $C_b$, with off-diagonals $.9$, both belong to this set. All square roots below are positive symmetric roots.

A fresh run draws $N=65,536$ iid active labels from the uniform distribution on $\{-1,1\}^3$, and independently draws $B=16,384$ iid banks. Bank $W_b$ contains $K$ standard three-dimensional active Gaussian vectors and, for each particle, 256 independent standard Gaussian nuisance coordinates. The same banks are shared across every observation and candidate. Define

$$
\begin{aligned}
S_k&=\sum_{j=1}^{256}\log\sigma(G_{kj}),\qquad
a_k(W)=\frac{e^{S_k}}{\sum_{l=1}^{K}e^{S_l}},\\
\phi(C;y,W)&=-\log\sum_{k=1}^{K}a_k(W)
 \prod_{j=1}^{3}\sigma\!\left(y_j(C^{1/2}Z_k)_j\right),\\
\widehat F(C)&=\frac1{NB}\sum_{\ell=1}^{N}\sum_{b=1}^{B}\phi(C;Y_\ell,W_b),\\
d(y,w)&=\phi(I;y,w)-\phi(C_b;y,w),\quad
g=\mathbb E d(Y,W),\quad
\widehat g=\frac1{NB}\sum_{\ell,b}d(Y_\ell,W_b).
\end{aligned}
\tag{D.1}
$$

The mathematical program returns the lexicographically first candidate within $\mathrm{tol}=10^{-12}$ of the minimum mathematical score. It uses no observation-risk comparison or fallback. This definition distinguishes mathematical evaluation from its floating-point implementation.

Every nonidentity grid point has some $|c_{ij}|\ge.3$. For the pair-spin moment $\kappa$, Appendix C proves $\kappa(0)=0$ and $\kappa'(c)\ge\beta=e^{-1/2}/4$. A bounded pair-spin statistic gives total variation at least $|\kappa(c_{ij})|/2$, and Pinsker in the true-to-candidate KL direction gives

$$
\begin{aligned}
\mathcal R(C)&\ge\tfrac12\beta^2(.3)^2
=r_{\rm grid}:=\frac{0.09\,\exp(-1)}{32}>.00103466
 \quad(C\in\mathcal C_{\rm grid}\setminus\{I\}),\\
\widehat g>\mathrm{tol}
&\ \Longrightarrow\ 
\widehat F(I)>\min_{C\in\mathcal C_{\rm grid}}\widehat F(C)+\mathrm{tol}
\ \Longrightarrow\ \mathcal R(\widehat C)\ge r_{\rm grid}.
\end{aligned}
\tag{D.2}
$$

The first implication uses $\min_C\widehat F(C)\le\widehat F(C_b)$. It does not require identifying the winner among the other 142 points. The exact risk constant is used throughout; its upward-rounded approximation $.00103466093$ is not a certified lower threshold.

## D.2. Shared-bank variance and Gaussian tails

Let $m(y)=\mathbb E_Wd(y,W)$ and $q(w)=\mathbb E_Yd(Y,w)$. The crossed-average decomposition is

$$
\begin{aligned}
d(y,w)&=g+a(y)+b(w)+r(y,w),\qquad
a(y)=m(y)-g,\quad b(w)=q(w)-g,\\
\mathbb E[r(Y,W)\mid Y]&=\mathbb E[r(Y,W)\mid W]=0,\\
\operatorname{Var}(\widehat g)
&=\frac{\sigma_Y^2}{N}+\frac{\sigma_W^2}{B}+\frac{\sigma_r^2}{NB},\\
\sigma_Y^2&=\mathbb E a(Y)^2,\qquad
\sigma_W^2=\mathbb E b(W)^2,\qquad
\sigma_r^2=\mathbb E r(Y,W)^2.
\end{aligned}
\tag{D.3}
$$

Conditional centering makes the residual covariance vanish whenever either index differs. The three terms are also mutually orthogonal. This proves (D.3) without treating the $NB$ crossed terms as independent observations.

Coordinate permutations preserve $C_b$ and its root; global sign reflection preserves the Gaussian bank law. Consequently $m(y)$ has only two values: $m_{\rm same}$ for the two all-equal labels, and $m_{\rm split}$ for the remaining six. Hence

$$
g=\tfrac14m_{\rm same}+\tfrac34m_{\rm split},\qquad
\Delta=m_{\rm same}-m_{\rm split},\qquad
\sigma_Y^2=\tfrac3{16}\Delta^2.
\tag{D.4}
$$

This is a symmetry of conditional expectations. The eight realized differences in one bank need not take only two values, and the program does not symmetrize its inputs.

The function $\log\sigma$ is 1-Lipschitz, as is log-sum-exp in the maximum norm. The nuisance weights are identical at the two anchors, so they disappear from the following envelope. The Gaussian exponential-moment argument in Appendix C, using $t=3/8$, gives

$$
\begin{aligned}
A&=\|\sqrt{C_b}-I\|_{\rm op}
=\max(1-\sqrt{.1},\sqrt{2.8}-1),\qquad
M_Z=\max_{k\le K}\|Z_k\|_2,\\
|d(y,W)|&\le\sqrt3\,A M_Z,\qquad
\mathbb E M_Z^2\le b_K:=\tfrac83(\log K+3\log2),\\
\sigma_r^2&\le\mathbb E d(Y,W)^2\le3A^2b_K.
\end{aligned}
\tag{D.5}
$$

The last bound follows from orthogonality in (D.3). It controls the residual term directly, without an additional random audit.

Set $D=3K=300$, $T=6$ and $E_T=\{\max_{k,j}|Z_{kj}|\le T\}$. Define a bank's class contrast $u(w)$ by the mean of its two same-sign differences minus the mean of its six split-sign differences. Then $\mathbb Eu(W)=\Delta$. The three truncated audit variables and their ranges are

$$
\begin{gathered}
X=q(W)\mathbf1_{E_T},\qquad
D_T=u(W)\mathbf1_{E_T},\qquad M=3AT,\\
X\in[-M,M],\qquad X^2\in[0,M^2],\qquad D_T\in[-2M,2M].
\end{gathered}
\tag{D.6}
$$

Write $\varphi$ and $\overline\Phi$ for the standard Gaussian density and upper tail. The coordinate union bound and Mills' inequality control the tail probability. To control the second moment, bound $M_Z^2$ by the sum of all $D$ squared coordinates and the tail-event indicator by the sum of coordinate exceedance indicators. Diagonal terms use $\mathbb E[Z^2\mathbf1_{\{|Z|>T\}}]=2(T\varphi(T)+\overline\Phi(T))$; off-diagonal terms use independence. No disjointness of exceedance events is assumed. Thus

$$
\begin{aligned}
\mathbb P(E_T^c)&\le p_T:=2D\varphi(T)/T,\\
\mathbb E[M_Z^2\mathbf1_{E_T^c}]
&\le2D\{T\varphi(T)+D\overline\Phi(T)\}
\le Q_T:=2D\varphi(T)(T+D/T),\\
\tau_2&:=3A^2Q_T,\qquad \tau_1:=\sqrt{\tau_2p_T},\\
|g-\mathbb EX|&\le\tau_1,\qquad
|\Delta-\mathbb ED_T|\le2\tau_1,\qquad
\sigma_W^2\le\mathbb E q(W)^2\le\mathbb EX^2+\tau_2.
\end{aligned}
\tag{D.7}
$$

Cauchy--Schwarz proves the mean-tail bounds. Numerically, $\tau_1\simeq1.31901920513\times10^{-5}$ and $\tau_2\simeq2.86347137776\times10^{-4}$; the actual calculation encloses these constants outward.

## D.3. A three-component moment audit on separate banks

This is an author computation using a separately generated bank sample, not a third-party audit. The audit uses $L=262,144$ new banks, with all eight labels and both anchors evaluated within each bank. These are separate from the future program's $N$ labels and $B$ banks. Earlier independent banks were used only for development and choosing the truncation design; none enter the new confidence statistics. The formal run fixed $T=6$, the three targets and the error allocations before generating its fresh banks. Reproducible generation used PCG64DXSM with seed sequence $[202610060501,j]$, with separate active and nuisance streams $j=1,2$. The probability interpretation assumes the specified iid Gaussian model; reproducibility does not establish that assumption.

For each saved binary64 scalar sequence $\widetilde V_1,\ldots,\widetilde V_L$, choose a common dyadic denominator and accumulate integers exactly:

$$
\widetilde V_b=2^{-e}z_b,\quad S_1=\sum_bz_b,\quad S_2=\sum_bz_b^2,\qquad
\overline{\widetilde V}=\frac{S_1}{L2^e},\quad
\widetilde s_V^2=\frac{LS_2-S_1^2}{L(L-1)2^{2e}}.
\tag{D.8}
$$

The exponents are 56, 95 and 68 for $X$, $X^2$ and $D_T$, respectively. High-precision ball arithmetic then encloses the standard deviations, confidence endpoints and final probability. The displayed sample moments are descriptive rounded values, not bounds on true variances.

|Audit variable|Sample mean|Unbiased sample variance|
|---|---:|---:|
|$X$|.00562444984697731|.04106054244035259|
|$X^2$|.04109202024289478|.011140404592328061|
|$D_T$|.23838858527527407|.05025621128454873|

No bank exceeded the truncation threshold. Allocate $\alpha_j=.01/3=1/300$ to each component: a one-sided lower mean bound for $X$, a one-sided upper mean bound for $X^2$, and a two-sided mean interval for $D_T$. For range length $R_j$, the empirical-Bernstein inequality [@maurer2009, Theorem 4], with numerical sample errors propagated through both mean and standard deviation, uses

$$
\begin{aligned}
\epsilon_X&=10^{-9}+10^{-12},\qquad
\epsilon_D=2\cdot10^{-9}+10^{-12},\\
\epsilon_2&=(2M+\epsilon_X)\epsilon_X+10^{-12},\\
B_j&=\sqrt{\frac{2\ell_j}{L}}\!
 \left(\widetilde s_j+\sqrt{\frac{L}{L-1}}\epsilon_j\right)
 +\frac{7R_j\ell_j}{3(L-1)},\\
(R_X,R_2,R_D)&=(2M,M^2,4M),\qquad
(\ell_X,\ell_2,\ell_D)=(\log600,\log600,\log1200).
\end{aligned}
\tag{D.9}
$$

Here $|\widetilde V_b-V_b|\le\epsilon_j$ is the relevant numerical condition. The centered Euclidean norm proves $|s_j-\widetilde s_j|\le\sqrt{L/(L-1)}\epsilon_j$. Thus (D.9) uses the exact mathematical ranges, inflates the sample standard deviation, and also allows $\epsilon_j$ at the mean endpoint. Appendix E gives the complete arithmetic method and remaining input qualifications. In particular, all 2,097,152 saved bank-label differences were subsequently enclosed, with maximum arithmetic error below $3.331\times10^{-14}$ for exact saved $Z/S$; the original $10^{-9}$ allowance was retained. That deterministic replay introduces no new confidence family and does not certify input generation or all future runs.

On the simultaneous audit event, which has probability at least $.99$ under those conditions, define outward endpoints by

$$
\begin{aligned}
g_L&=\overline{\widetilde X}-\epsilon_X-B_X-\tau_1,\\
U_2&=\overline{\widetilde{X^2}}+\epsilon_2+B_2+\tau_2,\\
\Delta_-&=\overline{\widetilde D_T}-\epsilon_D-B_D-2\tau_1,\qquad
\Delta_+=\overline{\widetilde D_T}+\epsilon_D+B_D+2\tau_1,\\
U_\Delta&=\max(|\Delta_-|,|\Delta_+|),\\
g&\ge g_L,\quad \sigma_W^2\le U_2,\quad |\Delta|\le U_\Delta,\qquad
\operatorname{Var}(\widehat g)\le V_U:=
\frac{3U_\Delta^2}{16N}+\frac{U_2}{B}+\frac{3A^2b_K}{NB}.
\end{aligned}
\tag{D.10}
$$

The bank contribution uses a population second-moment upper bound, not an unqualified replacement of true variance by sample variance. The following values display the certified endpoints conservatively; inference uses the full outward endpoints.

|Quantity|Conservative display|
|---|---:|
|$g_L$|.00279405023657203|
|$U_2$|.05074113097360336|
|$U_\Delta$|.24317070176659291|
|$V_U$|$3.289456534426161\times10^{-6}$|
|$p_L$|.70355036|

The three contributions to $V_U$ are shown below, rounded upward. They are contributions to this variance **upper bound**, not an empirical variance decomposition or a causal attribution.

|Bound contribution|Upper display|
|---|---:|
|Data: $3U_\Delta^2/(16N)$|$1.691779809884757\times10^{-7}$|
|Banks: $U_2/B$|$3.096992857275596\times10^{-6}$|
|Residual: $3A^2b_K/(NB)$|$2.328569616208936\times10^{-8}$|

## D.4. The probability statement and its limits

Since $g_L>\mathrm{tol}$, one-sided Cantelli applied to the fresh-run crossed average, followed by (D.2), gives on the same audit event

$$
\begin{aligned}
\mathbb P_{\rm fresh}\{\mathcal R(\widehat C)\ge r_{\rm grid}\}
&\ge\mathbb P_{\rm fresh}\{\widehat g>\mathrm{tol}\}\\
&\ge\frac{(g_L-\mathrm{tol})^2}{V_U+(g_L-\mathrm{tol})^2}
\ge p_L>.7035.
\end{aligned}
\tag{D.11}
$$

For completeness, Cantelli bounds $\mathbb P\{\widehat g\le\mathrm{tol}\}$ by $V/(V+(g-\mathrm{tol})^2)$, which increases with $V$ and decreases with $g>\mathrm{tol}$. The recorded lower endpoint is approximately $.703550362875$; the table and main text use downward conservative displays.

**The 99% statement is audit confidence in a lower bound on fresh-run risk probability.** The 70.35% value is a lower bound, not an observed failure frequency. The audit neither repeatedly ran the 143-point search nor drew fresh training labels; it established the moments needed for this sufficient-condition argument. Nonnegativity of KL also gives the deterministic consequence

$$
\mathbb E_{\rm fresh}\mathcal R(\widehat C)
\ge .7035\,r_{\rm grid}>.00072788.
\tag{D.12}
$$

It uses the same audit event and spends no additional error budget.

This result concerns the $n=256$ program alone. It does not establish that at least 70.35% of paired fresh runs change from a correct $n=0$ output to an incorrect $n=256$ output. Nor does it assert that every bad output is $C_b$ or has risk $.0311$. Finer grids may contain additional low-risk points and invalidate the single-anchor shortcut. The continuous-region result of Appendix C and the present coarse-grid probability result provide different guarantees. Neither is a guarantee for Adam or arbitrary continuous optimization.

A floating-point program requires a further condition. Suppose that, on an event of fresh-run probability at least $1-\delta_{\rm alg}$, all candidate scores are finite, the program returns a feasible grid point according to its stated tolerance rule, and $|\widehat g_{\rm fp}-\widehat g|\le\epsilon_{\rm alg}$. With $t_{\rm alg}=\mathrm{tol}+\epsilon_{\rm alg}<g_L$, a union bound gives

$$
\mathbb P_{\rm fresh}\{\mathcal R(\widehat C_{\rm fp})\ge r_{\rm grid}\}
\ge\left[
\frac{(g_L-t_{\rm alg})^2}{V_U+(g_L-t_{\rm alg})^2}
-\delta_{\rm alg}\right]_+,\qquad [x]_+=\max(x,0).
\tag{D.13}
$$

The event on the left counts only valid grid outputs meeting the risk threshold; unsuccessful executions are not counted. For a uniform valid-output and error guarantee ($\delta_{\rm alg}=0$), $\epsilon_{\rm alg}\le.00022$ would retain a lower bound exceeding $.66824>2/3$. This is a conditional sensitivity statement: such a uniform guarantee has not been established. Saved-input arithmetic enclosures do not certify Gaussian generation, accumulated nuisance transforms, the ideal iid sampling law, or this future-program condition.

This audit is a new 99% family. Combining it with the anchor-gap, centered-curvature and known-mean families of Appendices C and F gives conservative joint coverage at least 96%; including the older finite-selection family in Appendix H gives at least 95%. These bounds concern the named families, not all historical experiments in the paper. Independence between families is unnecessary. The full saved-input arithmetic refinements and (D.12) add no statistical comparisons.

## D.5. A continuous-region consequence of the same audit event

The quantity $g$ in (D.1) is exactly $F_{256}(I)-F_{256}(C_b)$ from Appendix C. Both definitions average the same normalized-weight loss over uniform three-bit teacher labels and the same Gaussian bank law, with known zero means, unit marginal variances, $K=100$, $n=256$ and the equicorrelation anchor $.9$. A candidate-independent nuisance normalization cancels from the point difference and from population regret. The cutoff $T=6$ truncates the audit statistic, not the population target: the mean-tail correction in (D.10) already makes $g_L$ a lower bound for the untruncated $g$.

The radius and population cost in (C.4)–(C.5), $c_d=H(b_{100};1-d)d^2/2$, are deterministic. They require no additional random event. Applying that regional inequality on the same 99% joint event as the present moment audit, with its input, numerical and iid-model qualifications, gives

$$
\begin{aligned}
\inf_{\substack{C\in\mathcal E_{.1}^{(3)}\\\mathcal R(C)\le10^{-6}}}
\{F_{256}(C)-F_{256}(C_b)\}
&\ge g_L-c_d>.00134,\\
F_{256}(C)-\inf_{C'\in\mathcal E_{.1}^{(3)}}F_{256}(C')<.00134
&\ \Longrightarrow\ \mathcal R(C)>10^{-6}.
\end{aligned}
\tag{D.14}
$$

Outward scalar arithmetic verifies the strict margin using the existing lower endpoint for $g$ and upper endpoint for $c_d$. The second line follows because the population infimum is no larger than $F_{256}(C_b)$. Thus the audit event supporting the fresh-run probability statement also supports the continuous-region implication, both restated in Section C.15, with conservative regret threshold $.00134$.

This is a deterministic consequence of the same event, with no additional statistical comparison or alpha allocation. It supplies no fresh-run failure probability for a continuous optimizer. The earlier margin near $.0008952$ and the original six-of-seven gate record in Appendix C remain unchanged. The joint coverage statements for the four or five named families remain 96% and 95%, respectively; this corollary does not change their scope.
