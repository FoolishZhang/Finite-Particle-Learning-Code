# Appendix H. Earlier finite selection and its negative control

## H.1. Selection conditions and the role of a region certificate

For a neighborhood $U$ of $\mathcal M$, define $\Delta_U=\inf_{C\notin U}V(C)-\min_CV(C)>0$, and let $e=\sup_C|F(C)-F(I)-wV(C)|$, $\zeta=\operatorname{osc}_C(\widehat F-F)$ and $\eta=\widehat F(\widehat C)-\inf_C\widehat F(C)$. Comparing with any minimizer of $V$ gives the familiar sufficient condition

$$
2e+\zeta+\eta<w\Delta_U\ \Longrightarrow\ \widehat C\in U.
\tag{H.1}
$$

Indeed an output outside $U$ has empirical excess over that minimizer at least $w\Delta_U-2e-\zeta$, contradicting the bound $\eta$. Writing this condition alone gives no finite numerical certificate. We instead bound the empirical criterion on a whole low-risk region directly, and separately audit a finite-population comparison. We do not estimate global $e$, $\zeta$ or optimizer regret from random restarts or from a grid minimum.

## H.2. A simultaneous lower bound on the continuous low-risk set

Here fix $m=3$, $\mu=0$, $C_*=I$, $\delta=.1$ and $K=2$. Let $q_y$ be any empirical active-label frequencies, and let $Z_{bk}\in\mathbb R^3$ and positive nuisance weights $a_{bk}$ be any fixed finite bank inputs. The symmetric root $S(C)=C^{1/2}$ couples candidates. Removing only a candidate-independent additive constant gives

$$
\widehat F(C)=\frac1B\sum_{b=1}^{B}\sum_yq_y\phi_y(S(C)Z_b),\qquad
\phi_y(X)=-\log\sum_{k=1}^K a_{bk}\prod_{i=1}^m\sigma(y_iX_{ki}),\qquad
a_{bk}=\frac{e^{S_{bk}^{\rm nuis}}}{\sum_\ell e^{S_{b\ell}^{\rm nuis}}}.
\tag{H.2}
$$

The nuisance weights remain random likelihood factors; they are not analytically removed as in FACT. For $r_0=10^{-6}$, let $\mathcal G_{r_0}=\{C\in\mathcal E_{.1}^{(3)}:\mathcal R(C)\le r_0\}$. Price, $\log\cosh u\le u^2/2$ and Jensen imply $\kappa'(c)\ge\beta=e^{-1/2}/4$. Pinsker applied to $y_iy_j$ gives $|\kappa(c_{ij})|\le\sqrt{2r_0}$, since the true pair moment is zero. Therefore

$$
C\in\mathcal G_{r_0}\ \Longrightarrow\
\|C-I\|_F\le d:=\frac{\sqrt{12r_0}}{\beta}=.02284535206697714\ldots,
\qquad\lambda:=1-d>0.
\tag{H.3}
$$

For $D=C-I$, every matrix on $I+tD$ has minimum eigenvalue at least $\lambda$. Differentiating the root identity gives $SS'+S'S=D$ and $SS''+S''S=-2(S')^2$. The Sylvester inverse bound yields

$$
\|S'\|_F\le\frac{\|D\|_F}{2\sqrt\lambda},\qquad
\|S''\|_F\le\frac{\|D\|_F^2}{4\lambda^{3/2}},\qquad
\|\nabla_X\phi_y\|_2\le\sqrt m,\quad
\|\nabla_X^2\phi_y\|_{\rm op}\le m+\tfrac14.
\tag{H.4}
$$

The last bound uses the weighted emission Hessians (norm at most $1/4$) and the covariance of the log-sum-exp gradients (norm at most $m$). If $a_b^2=\sum_k\|Z_{bk}\|^2$ and $A_2=B^{-1}\sum_ba_b^2$, the chain rule followed by $B^{-1}\sum_b a_b\le\sqrt{A_2}$ bounds the second directional derivative. Writing $g$ for the three free-coordinate derivatives at $I$, Taylor's formula gives

$$
H(A_2)=\frac{(m+1/4)A_2}{4\lambda}
+\frac{\sqrt{mA_2}}{4\lambda^{3/2}},\qquad
L_r=\frac{\|g\|_2d}{\sqrt2}+\frac{H(A_2)d^2}{2},\qquad
\inf_{C\in\mathcal G_{r_0}}\widehat F(C)\ge\widehat F(I)-L_r.
\tag{H.5}
$$

The factor $1/\sqrt2$ converts free off-diagonal coordinates to Frobenius norm. This bound holds simultaneously for every matrix in the continuous low-risk set, regardless of whether it is on the search grid. Consequently,

$$
\widehat g:=\widehat F(I)-\widehat F(C_b)-L_r>0,\qquad
\widehat F(\widehat C)\le\widehat F(C_b)
\quad\Longrightarrow\quad \mathcal R(\widehat C)>r_0.
\tag{H.6}
$$

## H.3. A separate finite-population comparison

The true uniform label distribution and the Gaussian prior make $F(C)$ invariant under simultaneous sign conjugation of $C$. Thus its free-coordinate gradient at $I$ is zero, for either nuisance condition. Taking expectations in (H.4), with $Ea_b^2=Km=6$ and $Ea_b\le\sqrt6$, gives the same bound with $A_2=6$:

$$
\inf_{C\in\mathcal G_{r_0}}F(C)-F(C_b)
\ge g-\frac{H(6)d^2}{2},\qquad
g=F(I)-F(C_b),\qquad \frac{H(6)d^2}{2}=.00158844598226789635\ldots.
\tag{H.7}
$$

The independent audit estimates only this prespecified point difference. It averages the eight active patterns exactly using true weights $1/8$; these weights never enter the learner. If $D_b=h_b(I)-h_b(C_b)$ and $A=\|C_b^{1/2}-I\|_{\rm op}=\max(1-\sqrt{.1},\sqrt{2.8}-1)$, Lipschitz continuity gives $|D_b|\le\sqrt3 A\max_k\|Z_{bk}\|$. Set $T=8$, $E_b=\{\max_{k,i}|Z_{bki}|\le T\}$, $D_b^t=D_b1_{E_b}$ and $M=3AT$. Gaussian Mills and Cauchy bounds yield

$$
|D_b^t|\le M=16.410533615595888\ldots,\qquad
|ED_b-ED_b^t|\le t_T:=\sqrt3 A\sqrt{6\cdot12\varphi(T)/T}
<2.526\times10^{-7}.
\tag{H.8}
$$

For $L=2097152$ independent banks, the scaled two-sided empirical Bernstein inequality [@maurer2009, Theorem 4] gives the half-width below. Here $s^2$ is the unbiased sample variance of $D_b^t$, $\alpha_j=.005$, and the union over two conditions has total error at most $.01$ under the iid sampling model and the stated numerical allowance:

$$
b_L=\sqrt{\frac{2s^2\log(4/\alpha_j)}{L}}
+\frac{7(2M)\log(4/\alpha_j)}{3(L-1)}+t_T+10^{-9},
\qquad g\in[\overline D^t-b_L,\overline D^t+b_L].
\tag{H.9}
$$

## H.4. Fixed program, inputs and complete outcomes

The single training dataset has 65,536 iid three-bit observations from the independent teacher. Nuisance labels are integrated by Gaussian sign symmetry; the $n$ random nuisance likelihood factors remain in every bank. Each correlation coordinate takes $\{-9,-6,-3,0,3,6,9\}/10$. All principal minors of $10(C-.1I)$ are checked with integers, leaving 143 feasible candidates. The lexicographic first candidate within $10^{-12}$ of the smallest double score is a proposal. An interval comparison either confirms its score is no worse than the fixed reference $(.9,.9,.9)$ or returns that reference. Risk and spectrum are computed only after outputs are locked. Neither the true correlation direction nor a risk oracle selects among candidates.

PCG64DXSM SeedSequence uses `[202610060101,role]`: roles 0/1/2 are training labels, active Gaussian banks and nuisance Gaussian factors; 3/4 are independent audit active/nuisance Gaussians. Banks are generated in blocks of 4096 in [bank, particle, coordinate] order. Both conditions share labels and active priors; $n=0$ sets nuisance weights to $1/2$, while $n=256$ uses the same active inputs with 256 independent nuisance factors. There are 65,536 training banks and 2,097,152 independent audit banks. No seed, dimension, particle count, risk threshold or family changed after execution. Shared inputs count once: 4,325,376 prior vectors, each with three active and 256 nuisance Gaussian coordinates, and 65,536 observed active patterns. No neural training or gradient-based optimizer updates are performed.

The realized bounds, in nats per observation, are:

|n|Empirical point gap|Linear cost|Quadratic cost|Empirical region lower|
|---|---|---|---|---|
|0|.003594355617|.000015488961|.001585398366|.001993468290|
|256|.004741770692|.000019724039|.001585398366|.003136648287|

The independent point audit gives:

|n|Population point mean|99% family interval|Population region lower|
|---|---|---|---|
|0|.002478573642|[.001864904388, .003092242896]|.000276458406|
|256|.003266233361|[.002280561286, .004251905435]|.000692115304|

Displayed values are rounded; all positivity decisions use the unrounded enclosures. Both proposals already equal $C_b$; no fallback or comparison-based change occurs. Both output spectra are $(.1,.1,2.8)$ and their numerical observation KL is $.031103092779$, whereas the true independent reference has zero risk. The analytic certificate proves only $\mathcal R>10^{-6}$, not a sharp lower bound of $.0311$. In particular, the $n=0$ control also excludes the low-risk region. The study resolves finite wrong selection for these specified programs, but does not isolate nuisance addition as necessary, determine the full global minimizing set, or establish failure for typical stochastic optimizers.

## H.5. Numerical verification and retained failure

Manufactured inputs compare double scores and free-coordinate derivatives to 128-bit Arb ball arithmetic; score/gradient discrepancies are below $3\times10^{-17}$ and central finite-difference gradient error below $2.4\times10^{-11}$. Every saved training bank is replayed with ball arithmetic at $I$, $C_b$ and the returned point, including the gradient at $I$. Binary64 bank inputs are treated as exact numbers for this realized-program certificate. For a positive definite rounded symmetric root $S_0$, a ball residual bound gives $\|S_0-C^{1/2}\|_F\le\|S_0^2-C\|_F/\sqrt\delta$; the resulting score allowance is at most $\sqrt{mA_2}$ times that bound. Positivity of $S_0$ is checked by Sylvester principal minors.

The population audit stores every paired bank difference and recomputes its mean and variance. Thirty-two prespecified evenly spaced banks per condition are checked with Arb. These checks qualify the explicit $10^{-9}$ numerical allowance in (H.9), but do not prove a global floating-point error bound or constitute interval quadrature at every population node. Thus the population coverage statement retains that numerical qualification. The deterministic empirical region certificate does not rely on this sampled population check.

After selection, observation probabilities use the exact centered three-spin representation $Q_C(y)=\{1+\sum_{i<j}\kappa(c_{ij})y_iy_j\}/8$. Odd spin moments vanish by symmetry. Pair Gaussian quadrature orders 32/48/64 are checked against independent three-dimensional orders 32/48. Maximum probability disagreement is below $3.21\times10^{-13}$, risk change below $3.23\times10^{-13}$ and mass error below $1.12\times10^{-16}$. These are numerical qualifications, not certified Gaussian integration remainders; the region proof uses Price and Pinsker, not a quadrature error claim.

After completing data generation, grid proposals, population differences and the full interval selection certificate, the run failed on a NumPy Boolean in the variance-summary serialization. A separately frozen continuation corrected scalar types and completed only the unsaved diagnostics on the same banks and locked selections. Original errors, saved inputs/outputs and recovery costs remain; no data, prior particles or learning output were added. Computation used NumPy/SciPy double precision and python-flint 0.8.0 (Arb, 128-bit) for ball replay. Full raw nuisance Gaussian coordinates are regenerable from the fixed RNG; their sufficient log-sums, active banks, labels and per-bank differences are retained locally. Hash records are not a replacement for those local arrays.
