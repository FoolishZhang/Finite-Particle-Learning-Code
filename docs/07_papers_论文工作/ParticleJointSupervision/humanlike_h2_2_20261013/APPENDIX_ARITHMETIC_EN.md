# Appendix E. Arithmetic enclosures for saved audit inputs

## E.1. Three distinct levels of guarantee

The finite-population results use an ideal iid Gaussian sampling model. We distinguish this probability assumption from two numerical questions. First, **saved-input arithmetic** treats each stored binary64 active Gaussian coordinate $Z_{ki}$ and nuisance log-weight sum $S_k$ as an exact real number, then encloses the corresponding mathematical sigmoid, weights, curvature or score. Second, **input transformation error** concerns how these saved numbers approximate intended inputs, including the generation and accumulation of 256 log-sigmoid terms into $S_k$. Complete arithmetic enclosures address the first level; they do not set errors in the second level to zero. The iid Gaussian law remains an explicit statistical model assumption; equivalence of a pseudorandom implementation to that law is not asserted.

Two deterministic passes now cover every saved centered $n=256$ curvature kernel of Appendix C and every saved bank–label anchor difference in the new probability audit of Appendix D. They give numerical support for those specified mathematical quantities. They do not guarantee the arithmetic error of every future native floating-point execution. Their statistical interpretation retains the stated input and sampling qualifications.

## E.2. Uniform curvature sensitivity

For one pair $(i,j)$, put $q_{ki}=\sigma(-Z_{ki})$, $r_{ki}=q_{ki}(2q_{ki}-1)$, and let $\pi_k\propto e^{S_k}\prod_i\sigma(Z_{ki})$ denote exactly normalized weights for the centered all-positive-label curvature. The diagonal Price kernel in (C.14) is

$$
\begin{aligned}
H_{ij}(\pi,q)={}&-\sum_k\pi_k(1-\pi_k)r_{ki}r_{kj}\\
&+2\sum_k\pi_k^2(1-\pi_k)(q_{ki}^2r_{kj}+q_{kj}^2r_{ki})\\
&+\sum_k\pi_k^2(1-4\pi_k)q_{ki}^2q_{kj}^2+P(A,B),\\
A={}&\sum_k\pi_kq_{ki}q_{kj},\quad B=\sum_k\pi_k^2q_{ki}q_{kj},\quad
P(A,B)=A^2-4AB+6B^2.
\end{aligned}
\tag{E.1}
$$

On the probability simplex and $0\le q\le1$, we have $0\le B\le A\le1$, $|r|\le1$ and $|r'|\le3$. The following constants bound the derivative contribution of each term, using the $\ell_1$ norm for weight changes and the maximum norm over all logistic inputs:

|Term in (E.1)|Weight constant|Logistic-input constant|
|---|---:|---:|
|First local sum|1|6|
|Second local sum|4|5|
|Third local sum|10|12|
|$P(A,B)$|14|12|

For the weight column, use $|1-2\pi|\le1$, $|2\pi-3\pi^2|\le1$ and $|2\pi-12\pi^2|\le10$. For the input column, the two-coordinate derivative sums of $r_ir_j$, $q_i^2r_j+q_j^2r_i$ and $q_i^2q_j^2$ are at most $6,10,4$. Their weight sums are bounded by $1,1/2,3$, respectively: in particular, $2\sum\pi_k^2(1-\pi_k)\le1/2$ and $\sum\pi_k^2|1-4\pi_k|\le3$.

The global polynomial includes the dependence of $B$ on squared weights. Writing $a_k=q_{ki}q_{kj}$, its derivatives and the resulting bound are

$$
\begin{gathered}
\partial_{\pi_k}P=a_k\{(2-8\pi_k)A+(-4+24\pi_k)B\},\\
\partial_{q_{ki}}P=\pi_kq_{kj}\{(2-4\pi_k)A+(-4+12\pi_k)B\},\\
|H_{ij}(\widetilde\pi,\widetilde q)-H_{ij}(\pi,q)|
\le29\|\widetilde\pi-\pi\|_1+35\|\widetilde q-q\|_\infty.
\end{gathered}
\tag{E.2}
$$

For $A>0$, divide each bracket by $A$ and put $t=B/A\in[0,1]$. The first bilinear expression has corner values $2,-6,-2,14$, so its magnitude is at most 14. The second has corner values $2,-2,-2,6$; summing $\pi_kq_{kj}$ over particles gives at most 6 for each of two coordinates. The case $A=0$ follows directly. Integrating these derivative bounds along the simplex and cube proves (E.2). Averaging the three pair kernels within one bank preserves the constants 29 and 35; it neither multiplies them by three nor creates independent observations.

For log weights $\ell$ and perturbation $e$, the softmax Jacobian yields an expectation of $|e-Ee|$. If $m=\min e_k$ and $M=\max e_k$, then $\operatorname{Var}(e)\le(M-Ee)(Ee-m)\le(M-m)^2/4$. Integrating along $\ell+te$ gives

$$
\begin{gathered}
\|\operatorname{softmax}(\ell+e)-\operatorname{softmax}(\ell)\|_1
\le\int_0^1E_{\operatorname{softmax}(\ell+te)}|e-Ee|\,dt\\
\le\tfrac12\operatorname{osc}(e)
=\inf_c\|e-c\mathbf1\|_\infty.
\end{gathered}
\tag{E.3}
$$

A common log-weight shift therefore cancels. These bounds require weights in the exact simplex; rounded stored weights need not satisfy that premise. The interval implementation encloses exact normalization itself rather than assuming that the rounded weights are exactly normalized.

## E.3. Propagation through sample statistics

Let $v_b$ be an exact target bank kernel, $t_b$ its stored floating reference, and $|v_b-t_b|\le\epsilon_b$ for every one of $L$ banks. With the centering projection $J=I_L-\mathbf1\mathbf1^T/L$, the sample standard deviation based on unbiased sample variance, $s(t)=\|Jt\|_2/\sqrt{L-1}$, satisfies

$$
\begin{gathered}
|\bar v-\bar t|\le\bar\epsilon,\qquad
|s(v)-s(t)|\le\frac{\|J(v-t)\|_2}{\sqrt{L-1}}
\le d_s:=\sqrt{\frac{\sum_b\epsilon_b^2}{L-1}},\\
|s(v)^2-s(t)^2|\le2s(t)d_s+d_s^2.
\end{gathered}
\tag{E.4}
$$

Thus numerical error affects both the mean and the standard deviation entering the empirical Bernstein bound. Let $\bar t_U,s_U,\bar\epsilon_U,d_{s,U}$ be reliable upper enclosures for these quantities and $R$ the exact target kernel's range length. On the corresponding empirical Bernstein event,

$$
h\le\bar t_U+\bar\epsilon_U
+\sqrt{\frac{2\log(4/\alpha_j)}L}(s_U+d_{s,U})
+\frac{7R\log(4/\alpha_j)}{3(L-1)}.
\tag{E.5}
$$

The lower endpoint is obtained symmetrically. For centered curvature, $L=262144$, $R=133/16$ and $\alpha_j=.01/3$, giving $\log1200$. The range remains that of the exact target because its sample statistics have been enclosed; this is not an unchanged variance calculation with an allowance added only to the mean. If every bank error is at most $\epsilon$, the extra mean-and-standard-deviation contribution is at most $\{1+\sqrt{2\log1200/(L-1)}\}\epsilon$, approximately $1.007355\epsilon$. Reliable aggregation separately covers its own rounding.

For the score audit, with its fixed truncation indicators, a per-label arithmetic error at most $\eta$ gives error at most $\eta$ in the eight-label mean and $2\eta$ in the difference of the two label-class means. If the exact truncated mean satisfies $|X|\le M$, its squared summary has error at most $2M\eta+\eta^2$. These bounds enter (E.4) for the scalar summaries; the original numerical allowances in Appendix D were retained after the full score pass.

## E.4. Outward arithmetic for the full panels

The implementation assumes correctly rounded, separately executed IEEE binary64 addition, subtraction, multiplication and division, gradual underflow, and reliable nextafter and binary exponent scaling. Each endpoint operation is expanded outward by one representable step. Reductions use an explicit pairwise tree. Manufactured probes exercise ordinary values, subnormals and rejected inputs; they do not prove these platform assumptions. Invalid intervals, nonfinite inputs or lost positive denominators cause failure rather than sample deletion.

For exponentials, an Arb-verified binary64 enclosure for $\log2$ is fixed once. An integer $k$ is used only if the entire reduced interval $r=x-k\log2$ lies in $[-1,1]$. Interval Horner evaluation gives

$$
\begin{gathered}
e^x=2^ke^r,\qquad
e^r\in\left[\sum_{j=0}^{20}\frac{r^j}{j!}-\frac3{21!},\;
\sum_{j=0}^{20}\frac{r^j}{j!}+\frac3{21!}\right],\quad |r|\le1,\\
x\le-800\quad\Longrightarrow\quad 0<e^x<2^{-1074}.
\end{gathered}
\tag{E.6}
$$

The Taylor remainder uses $e<3$. The last bound follows from $\log2<.7$ and $800>1074(.7)$, allowing a rigorous subnormal enclosure. Power-of-two scaling receives an outward rounding step. From $u=e^{-|z|}$, the two sigmoid signs are evaluated as $u/(1+u)$ and $1/(1+u)$, avoiding subtraction from one. The curvature implementation directly encloses all terms of (E.1).

For positive logarithms, exact binary normalization writes $x=2^em$ with $1\le m<2$. Set $u=(m-1)/(m+1)\in[0,1/3)$ and use

$$
\begin{gathered}
\log x=e\log2+2\sum_{j=0}^{19}\frac{u^{2j+1}}{2j+1}+R_{20},\\
0\le R_{20}\le\frac{2u^{41}}{41(1-u^2)}
\le\frac9{164\,3^{41}}.
\end{gathered}
\tag{E.7}
$$

The remainder bounds the positive tail geometrically. Monotonicity permits endpoint evaluation for positive interval inputs. No ordinary library exp/log output is used as the exact target at these nodes. A small number of constants and scalar aggregation log/square-root operations are enclosed with Arb.

For the score audit, the equicorrelation anchor uses its exact closed-form square root, not a rounded eigendecomposition. With $T_C=\sqrt C$ and $S_{\max}=\max_kS_k$,

$$
\begin{gathered}
C_b=\tfrac1{10}I+\tfrac9{10}\mathbf1\mathbf1^T,\qquad
T_{C_b}=aI+b\mathbf1\mathbf1^T,\\
a=\sqrt{1/10},\qquad b=\{\sqrt{14/5}-a\}/3,\\
M_C(y;Z,S)=\sum_ke^{S_k-S_{\max}}\prod_{i=1}^3\sigma\{y_i(T_CZ_k)_i\},\\
d_y(Z,S)=\log M_{C_b}(y;Z,S)-\log M_I(y;Z,S).
\end{gathered}
\tag{E.8}
$$

The constants $a,b$ are enclosed using 192-bit Arb. The common nuisance normalization and any common weight shift cancel exactly in the score difference. All eight labels are evaluated, with the stored score difference used only for error comparison.

## E.5. Coverage, outcomes and interpretation

The arithmetic methods, manufactured checks, source versions and saved-input identities were fixed before each full pass. These were post hoc deterministic additions to already completed audits; original statistical contracts and results were retained. The underlying curvature banks used seed 202610060301 and the new probability-audit banks used seed 202610060501, with PCG64DXSM SeedSequence roles 1/2 for active/nuisance normals and blocks of 256 banks. The arithmetic passes drew no new randomness. Indexed lower/upper bounds, original references, error envelopes and content hashes permit complete local replay and link each pass to its inputs.

|Arithmetic target|Complete saved coverage|Largest error envelope, rounded upward|
|---|---:|---:|
|Centered $n=256$ curvature|262144 banks; 786432 pair kernels|$4.382\times10^{-13}$ per bank average|
|New probability-audit anchor differences|262144 banks; 2097152 bank–label differences|$3.331\times10^{-14}$ per difference|

The unrounded recorded maxima are approximately $4.3818074169088566\times10^{-13}$ and $3.33066907387547\times10^{-14}$. For curvature, the average error bound is at most $2.268\times10^{-14}$ and the standard-deviation perturbation bound at most $3.630\times10^{-14}$. Propagating them through (E.5), with all aggregation errors enclosed, gives the outward conditional interval

$$
h_{256,100}\in[-.004912711427809,\;-.003618108895746].
\tag{E.9}
$$

This remains conditional on the declared input and sampling qualifications. The score pass satisfies the original $10^{-9}$ per-label arithmetic allowance at every saved point; it changes neither the original exact scalar statistics nor the probability lower bound in Appendix D. The two full passes took 126.438 and 339.672 seconds, respectively.

Independent Arb calculations at 32 fixed banks checked 96 $n=256$ curvature entries at 192 bits and 256 label differences at 256 bits; all reference balls lay inside the full-pass intervals. These are auxiliary implementation checks. Complete coverage comes from applying the outward algorithm to every saved input, not from extrapolating the sampled checks. The earlier matched-weight point-gap audit retains its own 32-bank diagnostics; the 18-component asymmetric-mean derivative audit in Appendix F is also not covered by these two passes.

Both passes are deterministic computations on existing samples and spend no additional alpha. They neither prove a deterministic expectation sign nor establish a uniform error bound for the native algorithm on future data and banks.

For clarity, the input-error interface for the two-anchor score is

$$
|d_y(\widetilde Z,\widetilde S)-d_y(Z,S)|
\le\operatorname{osc}(\widetilde S-S)
+3(1+\sqrt{2.8})\|\widetilde Z-Z\|_\infty.
\tag{E.10}
$$

Each candidate's log mixture changes by a value between the smallest and largest nuisance log-weight errors, giving the oscillation term. The matrix $T_{C_b}$ has nonnegative entries and row sum $\sqrt{2.8}$, while $\log\sigma$ and log-sum-exp have the required unit Lipschitz bounds. This proves (E.10) for all labels and banks. It is a conditional bridge, not a numerical bound on unknown input errors. Future native-program conditions on $\epsilon_{\rm alg}$ are stated separately in Appendix D; no such condition follows merely from complete saved-input coverage.
