# Appendix A. Proofs and representation boundaries

Theorem 1 is proved in A.1--A.2 and Theorem 2 in A.3. Here the star on F denotes the fixed teacher: the appendix notation F with star and nuisance scales is the same population score as in the main text. The constant \(\varepsilon_*\) in Theorem 2 is the minimum risk on the potential minimizing set; \(R_*\) in (A.19) is an explicit lower bound for it. A.4--A.9 give the additional representation and scale boundaries.

All logarithms are natural. Risk is the KL divergence for a complete observation, without division by dimension. Throughout A.1–A.6, the active dimension \(m\ge2\), mean vector \(\mu\in\mathbb R^m\), and spectral floor \(0<\delta<1\) are fixed. Spin labels are \(Y_i\in\{-1,1\}\); native API bits are \(Y_i^{(01)}=(Y_i+1)/2\). Particle samples are independent of the observation being scored. The population statements below do not assume that a finite optimization run attains their required regret.

## A.1. Model, exact task and the finite-particle derivative

Learn the upper-triangular entries \(c_{ij}\), \(i<j\), of

$$
\begin{gathered}
\mathcal E_\delta^{(m)}
=\{C=C^\top:\operatorname{diag}C=\mathbf1,\ C\succeq\delta I\},\\
|c_{ij}|\le1-\delta,\qquad
\delta\le\lambda_{\min}(C)\le\lambda_{\max}(C)\le m-(m-1)\delta .
\end{gathered}
\tag{A.1}
$$

This is a compact convex set with nonempty relative interior in the affine space of unit-diagonal symmetric matrices. Let \(X\sim N(\mu,C)\), with conditionally independent emissions \(P(Y_i=y_i\mid X_i)=\sigma(y_iX_i)\). Independent nuisance logits are \(\tau_jG_j\), \(G_j\sim N(0,1)\), with the same logistic emission. Symmetry gives

$$
\begin{aligned}
Q_C^A(y)&=E_C\prod_{i=1}^m\sigma(y_iX_i),&
Q_{C,\tau}^{(n)}(y,u)&=2^{-n}Q_C^A(y),\\
\widehat Q_{C,\tau,K}^{(n)}(y,u)
&=\frac1K\sum_{k=1}^K
\prod_{i=1}^m\sigma(y_iX_{ki})\prod_{j=1}^n\sigma(u_j\tau_jG_{kj}),\\
F_{n,K}^{*,\tau}(C)&=E_{(Y,U)\sim Q_{C_*}^{(n)},\,Z}
[-\log\widehat Q_{C,\tau,K}^{(n)}(Y,U)],&
\mathcal R_*(C)&=D(Q_{C_*}^A\Vert Q_C^A).
\end{aligned}
\tag{A.2}
$$

The true and candidate means and marginal variances agree. At a fixed \(n\), changing nuisance scales preserves the whole observation law. Across different \(n\), sample spaces differ; active likelihood ratios and complete joint KL remain unchanged. Each realized particle predictor is normalized, and its expectation is \(Q_C\); its expected log score nevertheless differs from the log score of the integrated model.

This is distinct from a pure latent-coordinate change. For an invertible measurable \(T_C\), push the prior forward, transport the emission by \(p'_C(y\mid z')=p_C(y\mid T_C^{-1}z')\), and pair the particles. For every realization and \(C\),

$$
\frac1K\sum_kp'_C(y\mid T_C(Z_k))
=\frac1K\sum_kp_C(y\mid Z_k).
\tag{A.3}
$$

Differentiable versions preserve path derivatives with respect to the same physical \(C\) when all transport derivatives are included. Changing optimization coordinates still requires the chain rule and need not preserve Adam updates. Without paired samples, the estimator laws agree rather than their realizations. Changing Gaussian nuisance scales while retaining the uncompensated logistic emission is an observation-equivalent representation change, not (A.3).

Write \(b_i=E Y_i=E\tanh((\mu_i+G)/2)\) and \(\kappa_{ij}(c)=E_CY_iY_j-b_ib_j\). Fixed marginal means and variances make this a function only of the pair correlation \(c\). Gaussian integration by parts gives

$$
\kappa_{ij}(0)=0,\qquad
\kappa_{ij}'(c)=\frac14E_c[
\operatorname{sech}^2(X_i/2)\operatorname{sech}^2(X_j/2)]>0 .
\tag{A.4}
$$

\Needspace{10\baselineskip}

For one complete bank, let \(\pi_k\) be normalized complete particle likelihoods and \(\gamma_k=\pi_k(1-\pi_k)\). Its mixed logit Hessian and the off-diagonal Gaussian Price identity imply

$$
\begin{aligned}
\partial_{c_{ij}}F_{n,K}^{*,\tau}(C)
&=-E\sum_k\gamma_kH_{k,ij},\\
H_{k,ij}
&=Y_iY_j\sigma(-Y_iX_{ki})\sigma(-Y_jX_{kj})\\
&=(Y_i^{(01)}-\sigma(X_{ki}))(Y_j^{(01)}-\sigma(X_{kj})),\\
EH_{k,ij}&=\frac{\kappa_{*,ij}+\kappa_{ij}(c_{ij})}{4},
\qquad \kappa_{*,ij}=\kappa_{ij}(c_{*,ij}).
\end{aligned}
\tag{A.5}
$$

Indeed the mixed derivative for particle \(k\) is \(-\gamma_kH_{k,ij}\); varying both symmetric covariance entries cancels the factor \(1/2\) in Price's formula. Gaussian linear envelopes make the loss integrable and justify differentiation; the mixed derivatives are bounded. Expanding \(H\) as
\([Y_iY_j-Y_iT_j-Y_jT_i+T_iT_j]/4\), \(T_i=\tanh(X_{ki}/2)\), and using observation–particle independence proves its mean. A Price sample is not generally equal to a realized Cholesky path derivative.

## A.2. Uniform near-tie analysis and the matrix potential

We prove the direction result slightly more generally for prescribed nuisance arrays \(a_{nj}\in[a_-,a_+]\), where \(0<a_-\le a_+<\infty\) are fixed. The theorem's unit scales are included. Define

$$
\begin{gathered}
L_a=\log\sigma(aG),\quad v(a)=\operatorname{Var}L_a,\quad
s_n^2=\sum_{j=1}^nv(a_{nj}),\\
I_K=\int_{\mathbb R}\varphi(z)^2\Phi(z)^{K-2}\,dz,\qquad
J_K=K(K-1)I_K=E\max_{k\le K}G_k,\qquad w_{n,K}=J_K/s_n .
\end{gathered}
\tag{A.6}
$$

Since \(L_a=aG/2-\log2-\log\cosh(aG/2)\), odd/even orthogonality gives \(v(a)\ge a^2/4\), while a Gaussian envelope gives \(s_n^2\asymp n\) uniformly. Integration by parts proves the identity for \(J_K\). The real-exponent argument below supplies \(J_K\sim\sqrt{2\log K}\) and \(J_K^2\asymp1+\log K\) for integer \(K\ge2\).

The following steps keep the constants uniform. In the density estimates they may depend on fixed \(a_-,a_+\) and the displayed truncation levels, but not on \(n\), the nuisance array or its ordering. In the pair estimate they may additionally depend on fixed \(C_0,p,C_1\). Only the active-score envelope will introduce dependence on fixed \(m,\mu,\delta\). Each \(o(1)\) has this uniform meaning. No explicit convergence modulus for (A.13) is asserted.

### Step 1. Uniform tilted densities: low and high frequencies

The density of \(L_a\), extended by zero for \(x\ge0\), is

$$
f_a(x)=\frac{\varphi(z_a(x))}{a(1-e^x)},\qquad
z_a(x)=\frac{x-\log(1-e^x)}a,\qquad x<0 .
\tag{A.7}
$$

These densities and each fixed-order derivative have uniform rapidly decreasing envelopes for \(a\in[a_-,a_+]\). The left tail is Gaussian up to polynomial factors. Near zero set \(t=-\log(-x)\); each fixed number of derivatives contributes at most \(e^{ct}\) times a polynomial, dominated by \(\exp[-t^2/(2a_+^2)+O(t)]\). The zero extension is therefore smooth. The same bounds hold after an exponential tilt in a sufficiently small fixed interval.

Let \(k_a(\lambda)=\log E e^{\lambda(L_a-EL_a)}\). Its fixed-order derivatives are uniformly bounded for \(|\lambda|\le\lambda_0<1/4\); the tilted variances are bounded away from zero by continuity and nondegeneracy on this compact interval. For the centered tilted characteristic functions \(\psi_{a,\lambda}\), a nonzero frequency has modulus strictly below one because the tilted law has a density. Continuity makes this strict bound uniform on compact frequency annuli. Uniform integration by parts gives decay at large frequencies, joining the two bounds and yielding

$$
0<c\le k_a''(\lambda)\le C,\qquad
\sup_{a,\lambda,|u|\ge\varepsilon}|\psi_{a,\lambda}(u)|\le q_\varepsilon<1,
\qquad
\sup_{a,\lambda}\int|\psi_{a,\lambda}(u)|^2\,du<\infty .
\tag{A.8}
$$

Let \(S_n=\sum_j(L_{a_{nj},j}-EL_{a_{nj}})\) and add an independent centered standard Gumbel \(E\). Its log MGF \(g_0\) is analytic for \(\lambda<1\). Put \(\mathcal K_n=\sum_jk_{a_{nj}}+g_0\), and let \(f_{n,\lambda}\) denote the density of the exponentially tilted \(S_n+E\). Write \(B_{n,\lambda}^2=\mathcal K_n''(\lambda)\asymp n\), and let \(\Psi_{n,\lambda}\) be the characteristic function after centering at \(\mathcal K_n'(\lambda)\) and dividing by \(B_{n,\lambda}\). Uniform third cumulant bounds give

$$
\begin{gathered}
\log\Psi_{n,\lambda}(u)=-u^2/2+R_{n,\lambda}(u),\\
|R_{n,\lambda}(u)|\le C|u|^3/\sqrt n,\\
\int_{|u|\le\varepsilon\sqrt n}
|\Psi_{n,\lambda}(u)-e^{-u^2/2}|\,du\le Cn^{-1/2}.
\end{gathered}
$$

For sufficiently small fixed \(\varepsilon\), the real part of the exponent, and of its interpolation with \(-u^2/2\), is at most \(-cu^2\). Thus the integrand difference is bounded by \(C|u|^3e^{-cu^2}/\sqrt n\), proving the integral estimate even though its range grows. In the complementary range, \(|u|/B_{n,\lambda}\) is bounded away from zero. Retain two distinct nuisance factors, use (A.8) on the other \(n-2\), and apply Cauchy--Schwarz:

$$
\int_{|u|>\varepsilon\sqrt n}|\Psi_{n,\lambda}(u)|\,du
\le B_{n,\lambda}q^{n-2}
\int_{\mathbb R}|\psi_{a_{n1},\lambda}(t)\psi_{a_{n2},\lambda}(t)|\,dt
\le C\sqrt n\,q^{n-2}.
$$

The tilted Gumbel factor has modulus at most one, so it does not affect this upper bound. The normal integral outside the low-frequency range is exponentially small. Fourier inversion now gives

$$
f_{n,\lambda}(\mathcal K_n'(\lambda))
=\frac{1+O(n^{-1/2})}{\sqrt{2\pi\mathcal K_n''(\lambda)}},
\qquad
\sup_x s_nf_{n,\lambda}(x)\le C .
\tag{A.9}
$$

Indeed the standardized density differs from \(\varphi\) by \(O(n^{-1/2})\) uniformly in its argument; at the centered argument this is a relative error because \(\varphi(0)>0\). The same Fourier integral bounds the density everywhere. This establishes both parts of (A.9) uniformly over the prescribed arrays and small tilts.

### Step 2. Relative density and upper-tail estimates

For \(x=s_nz\), \(|z|\le M\sqrt{\log n}\), strict convexity gives a unique small solution of \(\mathcal K_n'(\lambda)=x\). Taylor expansion at zero, using \(\mathcal K_n'(0)=0\), \(\mathcal K_n''(0)=s_n^2+\operatorname{Var}E\) and uniformly \(O(n)\) third derivatives, gives
\(\lambda=z/s_n+O_M((1+z^2)/n)\),
\(\lambda x-\mathcal K_n(\lambda)=z^2/2+O_M((1+|z|^3)/\sqrt n)\), and
\(s_n/\sqrt{\mathcal K_n''(\lambda)}=1+O_M((1+|z|)/\sqrt n)\).
Let \(q_n,H_n\) be the density and CDF of \((S_n+E)/s_n\). Exact untilting, namely \(q_n(z)=s_ne^{\mathcal K_n(\lambda)-\lambda s_nz}f_{n,\lambda}(s_nz)\), and (A.9) prove the first line of

$$
\begin{aligned}
\sup_{|z|\le M\sqrt{\log n}}\left|\frac{q_n(z)}{\varphi(z)}-1\right|
&=O_M\!\left(\frac{1+(\log n)^{3/2}}{\sqrt n}\right),\\
\sup_{-1\le z\le M\sqrt{\log n}}
\left|\frac{1-H_n(z)}{1-\Phi(z)}-1\right|&=o(1).
\end{aligned}
\tag{A.10}
$$

To prove the second line, an absolute CLT is insufficient. Tilt the global density bound in (A.9) at \(h=M\sqrt{\log n}/s_n\). The same cumulant expansion gives, for \(z\ge M\sqrt{\log n}\),

$$
q_n(z)\le C_Mn^{-M^2/2}
\exp\{-M\sqrt{\log n}(z-M\sqrt{\log n})+o(1)\}.
\tag{A.11}
$$

Fix \(M'>M\), put \(T=M\sqrt{\log n}\), \(T'=M'\sqrt{\log n}\), and let \(\rho_n(M')\) denote the relative density error on \([-T',T']\). Integrating that estimate up to \(T'\), then using (A.11) with \(M'\) beyond \(T'\), gives

$$
\left|\frac{1-H_n(z)}{1-\Phi(z)}-1\right|
\le \rho_n(M')+
\frac{\int_{T'}^\infty[q_n(t)+\varphi(t)]\,dt}{1-\Phi(T)}
=o(1),\qquad -1\le z\le T.
$$

Here Mills' lower bound gives \(1-\Phi(T)\ge c_M n^{-M^2/2}/\sqrt{\log n}\), whereas the numerator from the omitted tails is at most \(C_{M'} n^{-M'^2/2}/\sqrt{\log n}\). The strict inequality \(M'>M\) makes their ratio vanish. This proves the relative upper-tail statement, uniformly down to \(z=-1\). The zero-tilt part of (A.9) also gives \(\sup_zq_n(z)\le C\). Moreover the standardized variable is centered and has second moment \(1+\operatorname{Var}E/s_n^2\), uniformly bounded. Integrating the relative density estimate on fixed compact intervals and using this moment bound gives compact-uniform CDF convergence to \(\Phi\).

### Step 3. Conditional pair identity and small-shift CDF powers

For fixed labels absorb nuisance signs into the Gaussian expectation. Conditional on active log scores \(A_1,\ldots,A_K\), the weights are the softmax of \(S_{nk}+A_k\). The Gumbel maximum representation, differentiated in a second score, gives

$$
E_S[\pi_k\pi_\ell\mid A]
=\frac1{s_n}\int q_n(z-d_k)q_n(z-d_\ell)
\prod_{j\ne k,\ell}H_n(z-d_j)\,dz,\qquad d_j=A_j/s_n .
\tag{A.12}
$$

The derivative of a softmax choice probability is \(-\pi_k\pi_\ell\); differentiating the corresponding maximum integral differentiates one CDF. Bounded softmax derivatives and an integrable density product justify both operations. We will prove, for every fixed \(C_1,C_0,p>0\),

$$
\sup_{\substack{2\le K\le C_0n^p\\
\max_j|A_j|\le C_1\sqrt{\log n}\\k\ne\ell}}
\left|\frac{s_nE_S[\pi_k\pi_\ell\mid A]}{I_K}-1\right|\longrightarrow0 .
\tag{A.13}
$$

Set \(D_n=C_1\sqrt{\log n}/s_n\), so \(|d_j|\le D_n\), and choose fixed \(M\) with \(M^2>4p+4\). On \(0\le z\le T=M\sqrt{\log n}\), the identity \(\log[\varphi(z-d)/\varphi(z)]=zd-d^2/2\) is uniformly \(O(\log n/\sqrt n)\). Mills bounds give \(\varphi(t)/(1-\Phi(t))\le C(1+t_+)\) for \(t\ge-1\), so integrating this log-tail derivative controls the normal upper-tail shift by the same order. Applying (A.10) on a slightly larger fixed window yields

$$
\sup_{\substack{0\le z\le T\\ |d|\le D_n}}
\left|\frac{q_n(z-d)}{\varphi(z)}-1\right|=o(1),\qquad
\sup_{\substack{0\le z\le T\\ |d|\le D_n}}
\left|\frac{1-H_n(z-d)}{1-\Phi(z)}-1\right|=o(1).
$$

The passage from upper tails to powers is explicit. Put \(u=1-\Phi(z)\) and \(u'=1-H_n(z-d)\). Here \(0<u\le1/2\) and \(u'/u=1+o(1)\) uniformly. On, for example, \(0<u\le1/2\) and \(|u'/u-1|\le1/4\), the derivative of \(\log(1-u)\), together with \(u\le-\log(1-u)\), bounds \(|\log(1-u')/\log(1-u)-1|\) by a constant times \(|u'/u-1|\). Consequently there is a deterministic \(\varepsilon_n\to0\), uniform over all arrays and shifts, such that

$$
\Phi(z)^{1+\varepsilon_n}\le H_n(z-d)\le\Phi(z)^{1-\varepsilon_n},\qquad
\Phi(z)^{(K-2)(1+\varepsilon_n)}
\le\prod_{j\ne k,\ell}H_n(z-d_j)
\le\Phi(z)^{(K-2)(1-\varepsilon_n)}.
$$

Both inequalities hold for all \(0\le z\le T\) and \(|d|\le D_n\). The reversal on multiplying by the negative \(\log\Phi(z)\) explains the order of the powers. Multiplication changes the exponent; it does not require a pointwise error small compared with \(1/K\).

### Step 4. Real-exponent normal pair integrals and uniform ratios

For real \(x>1\), extend (A.6) by

$$
I_x=\int_{\mathbb R}\varphi(z)^2\Phi(z)^{x-2}\,dz,\qquad
J_x=x\int_{\mathbb R}z\varphi(z)\Phi(z)^{x-1}\,dz=x(x-1)I_x.
$$

The CDF \(\Phi^x\) defines a variable \(M_x\) whose mean is \(J_x\). Integration by parts has zero boundary terms since \(\varphi(z)\Phi(z)^{x-1}\to0\) at both ends. On every compact subinterval of \(x>1\), Mills bounds supply an integrable envelope for \(\varphi^2\Phi^{x-2}\); hence \(I_x\) is continuous and strictly positive there.

For completeness, put \(b_x=\sqrt{2\log x}\), take fixed \(0<\eta<1\), and write \(M_x^+=\max(M_x,0)\), \(M_x^-=\max(-M_x,0)\). Mills bounds and \(1-\Phi(t)^x\le x[1-\Phi(t)]\) give

$$
\begin{aligned}
E M_x^-&\le 2^{-(x-1)}\int_0^\infty\Phi(-t)\,dt,\\
E M_x^+&\le b_x+x\int_{b_x}^\infty[1-\Phi(t)]\,dt=b_x+O(b_x^{-2}),\\
P(M_x\le(1-\eta)b_x)
&\le\exp\{-x[1-\Phi((1-\eta)b_x)]\}\longrightarrow0.
\end{aligned}
$$

The second line uses \(x\varphi(b_x)=(2\pi)^{-1/2}\). The last line follows because the Mills lower bound makes \(x[1-\Phi((1-\eta)b_x)]\to\infty\). Thus \(J_x\le b_x+O(b_x^{-2})\) and \(J_x\ge(1-\eta)b_x(1-o(1))-o(1)\), proving \(J_x\sim\sqrt{2\log x}\) as real \(x\to\infty\).

These facts give the ratio needed in Step 3 uniformly in the allowed integer \(K\). If uniformity failed, choose a violating sequence \(n\to\infty\), \(K_n\) and \(\theta_n\), with an error bounded below by a fixed positive number. A bounded subsequence of \(K_n\) has a further subsequence equal to one integer \(K\); then \(x_n\to K\), and continuity gives \(I_{x_n}/I_K\to1\). Otherwise take a subsequence with \(K_n\to\infty\). Since \(x_n/K_n\to1\), the real-exponent asymptotic gives

$$
\frac{I_{x_n}}{I_{K_n}}
=\frac{J_{x_n}}{J_{K_n}}
\frac{K_n(K_n-1)}{x_n(x_n-1)}\longrightarrow1,\qquad
x_n=2+(K_n-2)(1+\theta_n),\quad |\theta_n|\le\varepsilon_n.
$$

Both alternatives contradict the fixed positive error. In particular the conclusion holds for \(\theta_n=\pm\varepsilon_n\). No rate in the growth of \(K_n\) is required for this argument.

### Step 5. Relative pair tails and the bounded-particle case

The remote positive tail is uniformly negligible relative to \(I_K\). Exact untilting at \(M\sqrt{\log n}/s_n\), as in (A.11), also bounds the shifted density for \(z\ge T\) with the factor \(e^{TD_n}\). Integrating the product of the two density bounds gives

$$
\int_T^\infty q_n(z-d_k)q_n(z-d_\ell)
\prod_{j\ne k,\ell}H_n(z-d_j)\,dz
\le \frac{C n^{-M^2}e^{2TD_n+o(1)}}{T}=o(I_K),\qquad
I_K\ge c\frac{\sqrt{1+\log K}}{K^2}.
$$

The lower bound follows from \(J_K^2\asymp1+\log K\) and (A.6), with the finitely many small integers absorbed into the constant. Because \(K\le C_0n^p\) and \(TD_n=o(1)\), the displayed positive-tail ratio is at most \(C n^{2p-M^2}/\sqrt{\log n}\to0\). The normal pair integrals with exponents \((K-2)(1\pm\varepsilon_n)\) have the same upper-tail bound, since their CDF factors are at most one.

For the negative half-line, compact CDF convergence and \(D_n\to0\) imply that \(H_n(z-d_j)\le q_*<1\) for every \(z\le0\), every allowed shift and all sufficiently large \(n\). Using \(\sup q_n\le C\) and integrating the other density gives

$$
\begin{gathered}
\int_{-\infty}^0q_n(z-d_k)q_n(z-d_\ell)
\prod_{j\ne k,\ell}H_n(z-d_j)\,dz\le Cq_*^{K-2},\\
\frac{Cq_*^{K-2}}{I_K}
\le C\frac{K^2q_*^{K-2}}{\sqrt{1+\log K}}\longrightarrow0
\quad(K\longrightarrow\infty).
\end{gathered}
$$

This limit depends only on \(K\), so it also covers arbitrarily slow diverging sequences \(K_n\). The negative normal pair integral is similarly bounded by \(C2^{-(x-2)}\); it is \(o(I_K)\) when \(x=2+(K-2)(1\pm\varepsilon_n)\) and \(K\to\infty\). For such sequences, the positive middle interval can therefore be replaced by the whole normal pair integral. The density estimate in Step 3 and the CDF-power bounds sandwich it between \((1+o(1))I_{2+(K-2)(1+\varepsilon_n)}\) and \((1+o(1))I_{2+(K-2)(1-\varepsilon_n)}\). Step 4 and the two tails prove (A.13) on every diverging sequence.

For bounded \(K\), the negative half-line cannot be discarded: it contributes to \(I_K\), especially at \(K=2\). Pass to a fixed-integer subsequence. On each \([-B,B]\), densities and CDFs with shifts converge uniformly to their normal counterparts. For \(B>D_n\), the remaining tails obey

$$
\begin{gathered}
\int_{|z|>B}q_n(z-d_k)q_n(z-d_\ell)
\prod_{j\ne k,\ell}H_n(z-d_j)\,dz
\le C P(|W_n+d_k|>B)\le\frac{C}{(B-D_n)^2},\\
W_n=(S_n+E)/s_n.
\end{gathered}
$$

The second inequality is Chebyshev's inequality and the uniform second-moment bound from Step 2. The corresponding normal tails also vanish as \(B\to\infty\). First let \(n\to\infty\) for fixed \(B\), then \(B\to\infty\), to obtain convergence of the complete integral to the fixed positive \(I_K\). Finally, a violating sequence for (A.13) has either a fixed-integer subsequence or a diverging one; both have just been excluded. All estimates are uniform in the arrays and the score bound, establishing the stated supremum rather than only a pointwise limit.

### Step 6. Active-score truncation, removal and matrix uniformity

For \(X_k=\mu+C^{1/2}Z_k\), put \(\Lambda=m-(m-1)\delta\). Uniformly in \(C\), labels and particles,

$$
|A_k|\le m\log2+\sqrt m\,\|\mu\|+\sqrt{m\Lambda}\,\|Z_k\|,
\qquad E e^{\|Z_k\|^2/4}=2^{m/2}.
\tag{A.14}
$$

Indeed \(-\log\sigma(t)\le\log2+|t|\), the Euclidean-to-coordinate bound contributes \(\sqrt m\), and \(\|C^{1/2}\|_{\mathrm{op}}\le\sqrt\Lambda\). Markov's inequality and a union bound show that the event \(\mathcal B_n=\{\max_k\|Z_k\|\le2\sqrt{(p+3)\log n}\}\) has complement probability at most \(C_02^{m/2}n^{-3}\). On this event (A.14) implies \(\max_k|A_k|\le C_1\sqrt{\log n}\) for one fixed \(C_1\) depending only on \(m,\mu,\delta,p\), for all sufficiently large \(n\). This is the source of uniformity in \(C\); compactness of the matrix class alone would not upgrade a pointwise pair estimate.

Condition on all active particles and labels. They determine both the scores and the marks, whereas the nuisance sums remain independent. Rewrite (A.5) as \(-E\sum_{k<\ell}\pi_k\pi_\ell(H_{k,ij}+H_{\ell,ij})\). On \(\mathcal B_n\), (A.13) says that each conditional pair mass is \((I_K/s_n)(1+r_{k\ell}(A))\). Since \(|H_{k,ij}|\le1\), the total error is bounded by

$$
\left|\frac{I_K}{s_n}E\!\left[
\mathbf1_{\mathcal B_n}\sum_{k<\ell}
 r_{k\ell}(A)(H_{k,ij}+H_{\ell,ij})\right]\right|
\le \eta_n\frac{K(K-1)I_K}{s_n}=\eta_n w_{n,K},\qquad
\sup_{A,k\ne\ell}|r_{k\ell}(A)|\le\eta_n\longrightarrow0.
$$

Without the event, the main term is exactly \(-K(K-1)I_KEH_{k,ij}/s_n=-w_{n,K}EH_{k,ij}\), by particle exchangeability. Removing the indicator from this main term costs at most \(w_{n,K}P(\mathcal B_n^c)\). For the actual derivative off the event, \(\sum_k\pi_k(1-\pi_k)\le1\) gives an absolute error at most \(P(\mathcal B_n^c)=O(n^{-3})\). Uniformly in \(K\ge2\), \(J_K\ge J_2>0\) and \(s_n\le C\sqrt n\), so \(w_{n,K}\ge c/\sqrt n\); hence this last error is \(o(w_{n,K})\). Substitution of the mark mean in (A.5) proves

$$
\sup_{\substack{\mathbf a_n\in[a_-,a_+]^n,\ 2\le K\le C_0n^p\\
C\in\mathcal E_\delta^{(m)}}}
\left|\frac{\partial_{c_{ij}}F_{n,K}^{*,\mathbf a_n}(C)}{w_{n,K}}
+\frac{\kappa_{*,ij}+\kappa_{ij}(c_{ij})}{4}\right|\to0 .
\tag{A.15}
$$

The \(K\) supremum is over integers. This is an absolute error after scaling, including at gradient zeros. Specialize to unit nuisance scales, so \(s_n=\sqrt{nv(1)}\). Integrating along \(I+t(C-I)\), which stays in the convex feasible set, proves uniformly in \(C\) and allowed \(K\)

$$
\begin{aligned}
\frac{F_{n,K}^*(C)-F_{n,K}^*(I)}{w_{n,K}}&\longrightarrow V(C),\\
V(C)&=-\frac14\sum_{i<j}
\left[\kappa_{*,ij}c_{ij}+\int_0^{c_{ij}}\kappa_{ij}(u)\,du\right].
\end{aligned}
\tag{A.16}
$$

The finite number of coordinates and their uniform bounds make the integrated uniform error vanish. This proves the gradient and value conclusions only: it does not differentiate the remainder to infer Hessian convergence. No assertion here allows \(m\) or \(\mu\) to grow with \(n\).

## A.3. Extreme-point spectrum and observation risk

The Hessian of \(V\) in the free coordinates is diagonal with entries \(-\kappa_{ij}'(c_{ij})/4<0\). Thus \(V\) is strictly concave. A minimizer of a strictly concave function on a convex set must be an extreme point: a nontrivial convex combination would have value strictly above the average of its endpoint values. The affine bijection \(R=(C-\delta I)/(1-\delta)\) maps the feasible set to the real correlation matrices.

For completeness, the required classical rank bound has a short proof. If \(R=UU^\top\) has rank \(r\), take \(U\in\mathbb R^{m\times r}\) of full column rank, with unit-length rows \(u_i\). If \(r(r+1)/2>m\), the linear equations \(u_i^\top S u_i=0\) admit a nonzero symmetric \(S\). For sufficiently small \(\varepsilon>0\), the matrices \(U(I\pm\varepsilon S)U^\top\) are distinct correlation matrices averaging to \(R\); distinctness follows from injectivity of \(S\mapsto USU^\top\). Hence an extreme \(R\) must satisfy

$$
\begin{gathered}
\mathcal M=\operatorname*{argmin}_{\mathcal E_\delta^{(m)}}V,\qquad
r_{\max}=\left\lfloor\frac{\sqrt{8m+1}-1}{2}\right\rfloor,\\
C\in\mathcal M\quad\Longrightarrow\quad
\operatorname{rank}\frac{C-\delta I}{1-\delta}\le r_{\max},\qquad
\lambda_1(C)=\cdots=\lambda_{m-r_{\max}}(C)=\delta .
\end{gathered}
\tag{A.17}
$$

Eigenvalues are ordered increasingly. The rank restriction is the known extreme-correlation geometry of [@litam1994]; its application here is to the particle-induced potential. \(C\) remains positive definite and full rank.

All active pattern probabilities are positive and continuous in \(C\), hence uniformly bounded away from zero on the compact class. Equation (A.4) means that the pair observation moments identify every \(c_{ij}\); therefore \(Q_C^A=Q_{C_*}^A\) only at \(C=C_*\). Every relative-interior \(C_*\) is excluded from \(\mathcal M\), including \(C_*=I\), for which the limiting gradient is zero but strict concavity still excludes a minimum.

\Needspace{9\baselineskip}

An explicit general lower bound is available. Using \(\log\cosh x\le x^2/2\) and Jensen,

$$
\kappa_{ij}'(c)\ge\frac14
E e^{-(X_i^2+X_j^2)/4}
\ge\frac14e^{-(1+\|\mu\|_\infty^2)/2}
=:\beta_\mu>0 .
\tag{A.18}
$$

Put \(q=m-r_{\max}\ge1\) and \(\Delta_*=\lambda_q(C_*)-\delta>0\). For \(C\in\mathcal M\), Weyl's inequality and the zero-diagonal row-sum bound give
\(\Delta_*\le\|C-C_*\|_{\mathrm{op}}\le(m-1)\max_{i<j}|c_{ij}-c_{*,ij}|\).
One spin-product mean thus differs by at least \(\beta_\mu\Delta_* /(m-1)\). Map the full observation to that \(\{-1,1\}\)-valued statistic and apply Bernoulli Pinsker:

$$
\inf_{C\in\mathcal M}\mathcal R_*(C)
\ge R_*:=\frac12\left[\frac{\beta_\mu\Delta_*}{m-1}\right]^2>0 .
\tag{A.19}
$$

No overlapping pair risks are added, and no latent-Gaussian KL lower bound is transferred to observations.

For any prescribed allowed \(K_n\), compactness and the uniform limit (A.16) give

$$
\begin{gathered}
F_{n,K_n}^*(\widehat C_n)-\inf_C F_{n,K_n}^*(C)=o(w_n)
\quad\Longrightarrow\quad
\operatorname{dist}(\widehat C_n,\mathcal M)\to0,\quad
\liminf_n\mathcal R_*(\widehat C_n)\ge R_* .
\end{gathered}
\tag{A.20}
$$

Indeed the potential regret is bounded by the scaled objective regret plus twice the uniform approximation error. Outside every neighborhood of \(\mathcal M\), the continuous potential has a positive gap. For random \(o_p(w_n)\) regret, distance converges in probability; for each \(\varepsilon>0\), \(P(\mathcal R_*(\widehat C_n)<R_*-\varepsilon)\to0\), and \(\liminf E\mathcal R_*(\widehat C_n)\ge R_*\). Multiple minimizers need not share one KL value. Ordinary \(o(1)\) objective regret is insufficient because \(w_n\to0\).

### Four-active numerical fixture

For the controlled experiments,

$$
\begin{gathered}
\mu=(-.6,-.2,.2,.6),\qquad
C_*=\begin{pmatrix}
1&.25&-.15&.10\\.25&1&.10&-.15\\
-.15&.10&1&.25\\.10&-.15&.25&1
\end{pmatrix},\\
\delta=.1,\qquad
\operatorname{spec}(C_*)=(.5,1,1.2,1.3).
\end{gathered}
\tag{A.21}
$$

Here \(r_{\max}=2\), so every \(C\in\mathcal M\) has two eigenvalues equal to .1. Its second eigenvalue differs from truth by .9, forcing some correlation error to be at least .3. Equation (A.18) gives \(\beta_\mu=\tfrac14e^{-.68}>1/8\); the final inequality is analytic, since \(\log2>2(1/3+1/81)=56/81>.68\). Therefore

$$
\mathcal R_*(C)\ge\frac12(.3/8)^2=\frac9{12800}
\quad(C\in\mathcal M),
\qquad
\liminf_n\mathcal R_*(\widehat C_n)\ge\frac9{12800}
\tag{A.22}
$$

under (A.20), with the corresponding random-regret statement. This does not assert two exact floor eigenvalues for finite-\(n\) population minima, or a spectral/risk guarantee for the reported 512-step Adam programs.

Initialization also matters to interpretation. At truth the scaled particle gradient tends to \(-\kappa_{*,ij}/2\), whereas exact likelihood has zero gradient. At independence, with \(h_i=E\sigma'(\mu_i+G)>0\),

$$
\left.\partial_{c_{ij}}E_*[-\log Q_C^A(Y)]\right|_{C=I}
=-\frac{4h_ih_j\kappa_{*,ij}}{(1-b_i^2)(1-b_j^2)} .
\tag{A.23}
$$

Factorization at \(I\) gives the corresponding score
\(4h_ih_j(Y_i-b_i)(Y_j-b_j)/[(1-b_i^2)(1-b_j^2)]\), proving (A.23). Its sign agrees with the leading raw direction at \(I\). Initial movement toward the true signs is not itself a wrong direction; the potential's global preference is constrained by matrix feasibility, not by independent coordinatewise endpoints.

## A.4. Two-active warm-up and matched-variance support

For \(m=2,\mu=0,\rho\in[-R,R]\), \(0<R<1\), let \(\theta\in[-R,R]\) be the fixed true correlation and define \(r(\rho)=E_\rho[\tanh(X_1/2)\tanh(X_2/2)]\). Then

$$
\begin{gathered}
Q_\rho^A(y)=\frac{1+y_1y_2r(\rho)}4,\qquad
\mathcal R_\theta(\rho)=\operatorname{kl}\!\left(
\frac{1+r(\theta)}2\middle\Vert\frac{1+r(\rho)}2\right),\\
\ell_s'(\rho)=\frac{r'(\rho)[r(\rho)-s]}{1-r(\rho)^2},\qquad
\frac{(F_{n,K}^\theta)'(\rho)}{w_{n,K}}\to-\frac{r(\theta)+r(\rho)}4 .
\end{gathered}
\tag{A.24}
$$

Here \(s\) is an empirical agreement moment, not the radial parameter of the older four-active experiments. The function \(r\) is odd and strictly increasing by (A.4). Consequently the limiting raw potential is \(-[r(\theta)\rho+\int_0^\rho r(v)\,dv]/4\), with minimum \(\operatorname{sign}(\theta)R\) for \(\theta\ne0\), or the endpoint set for \(\theta=0\). Interior true parameters have positive near-optimal risk; true boundary points are not asserted to fail.

Equal total log-weight variance does not force the same limit. Let \(v(a)=\operatorname{Var}\log\sigma(aG)\). The identity
\(v(a)=a^2/4+\operatorname{Var}\log\cosh(aG/2)\) and Gaussian Poincaré give \(a^2/4\le v(a)\le a^2/2\). Continuity and
\(v'(a)=a/2+2\operatorname{Cov}(\log\cosh(a|G|/2),|G|\tanh(a|G|/2)/2)\ge a/2\)
give a unique positive \(A_n\) satisfying

$$
\begin{gathered}
v(A_n)+(n-1)v(1/n)=nv(1),\qquad A_n=\Theta(\sqrt n),\\
W_n=2\sigma(A_nG_n)\prod_{j<n}2\sigma(G_j/n),\qquad
EW_n=1,\quad EW_n^2\le2e^{1/(4n)},\quad
\|\log W_n\|_2\le A_n+1 .
\end{gathered}
\tag{A.25}
$$

The second moment follows from
\(E[(2\sigma(aG))^2]=1+E\tanh^2(aG/2)\le1+\min(1,a^2/4)\).
The log bound follows from \(|\log(2\sigma(x))|\le|x|\) and the \(L^2\) triangle inequality. The observation law and total log variance match the diffuse unit-scale representation exactly.

The negative log tail needs explicit control. If iid \(T_k>0\) satisfy \(ET=1\), \(ET^2\le M\), \(\|\log T\|_2\le L\), then

$$
\begin{aligned}
P(\bar T_K<1/2)&\le e^{-K/(8M)},\\
0\le E[-\log\bar T_K]&\le
\frac{2(M-1)}K+(\log K+L)e^{-K/(16M)}=:\Delta_K,\\
E|\log\bar T_K|&\le
2\sqrt{\frac{M-1}K}+(\log K+L)e^{-K/(16M)}=:\varepsilon_K .
\end{aligned}
\tag{A.26}
$$

Use \(e^{-x}\le1-x+x^2/2\) and a negative Laplace bound at \(1/(2M)\) for the first line. The log is integrable because \(\bar T_K\ge T_1/K\) and \((\log\bar T_K)^+\le\bar T_K\). On \(\bar T_K\ge1/2\), use \(-\log x+x-1\le2(x-1)^2\) and \(|\log x|\le2|x-1|\); on its complement use \(-\log\bar T_K\le\log K-\log T_1\) and Cauchy–Schwarz. Finally \(E[-\log\bar T_K]=E[-\log\bar T_K+\bar T_K-1]\). Positivity and a second moment alone would not control this log tail.

For any fixed \(m,\mu,\delta\), write \(f_C(y\mid Z)=\prod_i\sigma(y_i[\mu+C^{1/2}Z]_i)\) and apply (A.26) to
\(T=W_nf_C(y\mid Z)/Q_C^A(y)\). Compactness gives \(\min_{C,y}Q_C^A(y)>0\), \(f^2\le f\), and the Gaussian log envelope gives uniform constants \(M=O(1)\), \(L=O(\sqrt n)\). Thus

$$
\begin{gathered}
0\le F_{n,K}^{*,\mathrm{conc}}(C)-n\log2-E_*[-\log Q_C^A(Y)]
\le\Delta_{n,K},\\
\Delta_{n,n+2}=O(n^{-1})+O((\sqrt n+\log n)e^{-cn}).
\end{gathered}
\tag{A.27}
$$

For a population \(\eta\)-near-minimizer, comparison with \(C_*\) gives \(\mathcal R_*(\widehat C)\le\Delta_{n,K}+\eta\). This proves concentrated population recovery, including on the full matrix class, while (A.20) proves diffuse population failure. The concentrated scales leave every fixed positive compact window.

For clarity, the previously established actual shared-bank separation remains one-dimensional. Here \(\widehat F\) averages the loss over \(N\) complete rows and \(B\) complete prior banks shared across all rows and candidates. For any such dataset and grid \(\mathcal G_n\) of size \(M_n\), write \(\ell_{\mathcal D}^{A}=N^{-1}\sum_j[-\log Q_\rho^A(Y_j)]\). Equation (A.26), triangle inequality and maximum bounded by sum give
\(E[\max_{\rho\in\mathcal G_n}|\widehat F^{\mathrm{conc}}(\rho)-n\log2-\ell_{\mathcal D}^{A}(\rho)|\mid\mathcal D]\le M_n\varepsilon_{n,K}\), for any prescribed \(B\ge1\); no row or candidate independence is needed. At \(K=n+2\), \(M_n=O(n^{1/4})\), mesh tending to zero, iid data \(N_n\to\infty\) independent of the banks, and grid regret \(o_p(1)\), concentrated learning is consistent. Indeed the four active pattern probabilities are uniformly positive, so empirical frequencies give a uniform exact-likelihood law of large numbers; grid approximation and the displayed maximum-error bound complete the argument. Risk is bounded, so convergence also holds in expectation.

The diffuse side requires finer accuracy. Conditional only on nuisance weights \(a_k\), the two-active path gradient \(h\) obeys

\[
h^2\le(\sum a_kf_k(X'_{k2})^2)(\sum a_k/f_k).
\]

Splitting equal and unequal particle indices, using \(f\le1\),
\(E(X_2')^2=(1-\rho^2)^{-1}\), and
\(E(1/f)\le C_R=1+2e^{1/2}+e^{1+R}\), gives \(Eh^2\le C_R/(1-R^2)\).
The bound \(|h|\le\max_k\|(Z_{k1},Z_{k2})\|/\sqrt{1-R^2}\) justifies differentiation. Jensen over arbitrary data rows and integral Cauchy–Schwarz therefore give

$$
E[(\operatorname{osc}_{[-R,R]}e_B)^2\mid\mathcal D]
\le\frac{4R^2C_R}{B(1-R^2)},\qquad
e_B=\widehat F-E(\widehat F\mid\mathcal D).
\tag{A.28}
$$

Sign symmetry makes the conditional mean affine in the empirical agreement \(s\); apply (A.15) at two distinct fixed generating correlations to identify its two coefficients uniformly for \(s\in[-1,1]\). With \(B=n,K=n+2\), (A.28) gives \(\operatorname{osc}e_B=o_p(w_n)\) because \(nw_n^2\asymp\log n\). On a prescribed grid containing the endpoints, diffuse grid regret \(o_p(w_n)\) consequently gives the endpoint limit in (A.24), while concentrated grid regret \(o_p(1)\) gives zero risk under the same data and bank counts. This is the limited shared-bank learning comparison; (A.28) is not a multidimensional supremum bound or an Adam theorem.

## A.5. Observation-preserving scale paths and mechanism identities

Condition on active particles and an observation; let \(\alpha_k>0\) be normalized active likelihoods. Independent fair nuisance coordinates have mean-one weights \(W_k(\tau)=\prod_j2\sigma(u_j\tau_jG_{kj})\). Two applications of Jensen give

$$
\begin{gathered}
0\le E[-\log\sum_k\alpha_kW_k(\tau)]
\le\sum_jE\log\cosh(\tau_jG/2)\le\frac18\|\tau\|_2^2,\\
\widehat F^{\mathrm{fact}}_{n,K}=n\log2+\widehat F^{\mathrm{raw}}_{0,K}.
\end{gathered}
\tag{A.29}
$$

The final identity is per realization with paired active particles; factorization retains active finite-\(K\) bias. To prove strict increase in one positive scale, condition on all other inputs and magnitudes \(r_k=|G_k|\). Its sum has the form \(\sum_kc_k(1+\epsilon_ku_k)\), \(u_k=\tanh(\tau r_k/2)\), with independent fair signs. Holding other signs fixed, \(A=c_k+\sum_{\ell\ne k}c_\ell(1+\epsilon_\ell u_\ell)>c_ku_k\), and
\[
E_{\epsilon_k}[-\log(A+\epsilon_kc_ku_k)]
=-\tfrac12\log(A^2-c_k^2u_k^2),\qquad
\partial_{u_k}=\frac{c_k^2u_k}{A^2-c_k^2u_k^2}>0 .
\]
For \(\tau>0\), \(du_k/d\tau>0\) almost surely. Sum partial derivatives and average; a Gaussian maximum bounds the path derivative and justifies exchange with expectation. Each positive nuisance scale is therefore a population descent opportunity toward zero without changing the observation model.

\Needspace{11\baselineskip}

This boundary is not confined to fair marginals. For any fixed \(p\in(0,1)\), strict monotonicity in \(\mu\) gives a unique smooth even \(\mu_p(\tau)\) with \(E\sigma(\mu_p(\tau)+\tau G)=p\). Put \(X_\tau=\mu_p(\tau)+\tau G\). For \(\tau\ge0\) and differentiable convex \(\psi\) satisfying the integrability needed for differentiation, the implicit function theorem and Gaussian integration by parts yield

$$
\begin{gathered}
\mu_p(0)=\log\frac p{1-p},\qquad
\mu_p'(\tau)=-\tau\frac{E\sigma''(X_\tau)}{E\sigma'(X_\tau)},\qquad
|\mu_p'(\tau)|\le\tau,\\
\frac d{d\tau}E\psi(\sigma(X_\tau))
=a_\tau\operatorname{Cov}_{\nu_\tau}(\psi'(\sigma(X_\tau)),G)\ge0,\\
d\nu_\tau=\frac{\sigma'(X_\tau)}{a_\tau}\,d\Phi,\quad a_\tau=E\sigma'(X_\tau).
\end{gathered}
\tag{A.30}
$$

The mean constraint centers \(G\) under the tilted law. For convex \(\psi\), the covariance is nonnegative because both factors are increasing functions of \(G\), and is positive for strictly convex \(\psi\) when \(\tau>0\). Apply this separately to each particle in
\(-\log\sum_k\alpha_kW_{y,k}\), where \(W_1=\sigma(X)/p\) and \(W_0=(1-\sigma(X))/(1-p)\), then sum partial derivatives along the common-scale path. The conditional functions are strictly convex; their differentiated integrands are bounded by \(\tau+\max_k|G_k|\). Thus the expected score increases strictly with positive scale while preserving Bernoulli \(p\). This is a compensated total derivative, not a fixed-mean partial derivative or an optimizer guarantee. Nonfair nuisance labels do not acquire fair-label canonicalization.

Locally, fix \(p,K\) and all inputs other than the changing nuisance coordinate; let \(\omega_k\) be their normalized likelihood weights, \(q=\sum_k\omega_k^2\), \(c_1=1-p,c_0=-p\), and \(h_y=-\log\sum_k\omega_kW_{y,k}\). Mean preservation and \(W_y'(0)=c_yG\) give \(Eh_y(\tau)=q c_y^2\tau^2/2+O(\tau^4)\), \(Eh_y'(\tau)=q c_y^2\tau+O(\tau^3)\), and \(\operatorname{Var}h_y'(0)=q c_y^2\). Evenness and Gaussian moment envelopes justify the remainders. Under \(\tau=\operatorname{softplus}(u)+\varepsilon\), both mean and SD acquire the exact factor \(1-e^{-(\tau-\varepsilon)}\). With zero floor or \(\varepsilon\ll\tau\ll1\), the raw-gradient mean is of order \(\tau^2\) and SD of order \(\tau\); a small raw gradient does not establish recovery. Fixed positive floor and growing nuisance dimension require separate control.

Finally, for any fixed reference center \(h_{ij}^{\mathrm{ref}}\), the direction experiment uses the exact identity

$$
\begin{aligned}
G_{ij}&=-h_{ij}^{\mathrm{ref}}A-\sum_k\gamma_k(H_{k,ij}-h_{ij}^{\mathrm{ref}}),\\
A&=\sum_k\gamma_k=1-\frac1{\mathrm{ESS}},\qquad EA=1-E(1/\mathrm{ESS}).
\end{aligned}
\tag{A.31}
$$

These are full-likelihood weights, not noise-only weights; \(EA\ne1-1/E(\mathrm{ESS})\) in general. A changed reference center shifts the two terms oppositely and leaves their sum unchanged, so their proportions are not causal contributions. At fixed \(n\) and truth, \(K\to\infty\) makes the expected score gradient tend to zero while \(EA\to1\): the centered coupling cancels the mean-mark term. This follows from the bounded-likelihood law of large numbers and \(\sum_y\partial_{c_{ij}}Q_C(y)=0\). It explains why overlap alone cannot determine a gradient's direction, without turning the decomposition into a new sign criterion.

A separate two-active control remains available for finite \(n\ge0\) and \(K\ge2\): writing the conditional bank mean as \(F_s=U+sV\), the Price marks have the fixed agreement sign and \(\pi_k(1-\pi_k)>0\), giving \(F_+'<0<F_-'\) and \(b=-V'>0\). The deterministic additive correction \(g'=-U'-rV'\) yields \((F_s+g)'=b(r-s)\), hence the exact clipped-likelihood minimizer. This is optimizer agreement, not an unbiased log score or a normalized density. Replacing the whole criterion by an integral of any positive weight times \(r-s\) has the same minimizer; neither construction supplies a universal simulation budget or a matrix-learning algorithm.

## A.6. Conditional label weights and a finite-data boundary

Keep fixed \(m,\mu,\delta\) and unit independent fair nuisance scales. Let \(q\) be any probability vector on the \(2^m\) active spin patterns, including zero components. It remains fixed when comparing candidate matrices and is independent of the new scoring particles. For fixed active pattern \(s\), let \(\phi_{s,n,K}(C)\) be the prior expectation of the raw score with any fixed nuisance label pattern, and put \(F_{q,n,K}=\sum_s q_s\phi_{s,n,K}\). Gaussian reflection removes nuisance label signs in this expectation; it does not replace the inputs of an actual training run. With \(b_{q,i}=E_qS_i\), the correct centered moment is

$$
\begin{aligned}
A_{q,ij}&=E_q[(S_i-b_i)(S_j-b_j)]\\
&=\operatorname{Cov}_q(S_i,S_j)+(b_{q,i}-b_i)(b_{q,j}-b_j),\\
E_{q,C}H_{k,ij}&=\frac{A_{q,ij}+\kappa_{ij}(c_{ij})}{4}.
\end{aligned}
\tag{A.32}
$$

For empirical frequencies, \(\operatorname{Cov}_q\) has denominator \(N\), and empirical marginal offsets cannot in general be omitted. Expansion of the mark in (A.5) and label–particle independence prove (A.32).

The near-tie argument is uniform over labels: nuisance sums have the same law after reflection, the envelope (A.14) is independent of the label, and \(|H_{k,ij}|\le1\). The same truncation has complement probability \(O(n^{-3})\) for every label. On the good event the summed relative pair error is \(o(w_{n,K})\); off it, \(\sum_k\gamma_k\le1\) bounds the derivative error. Prove the bound for each of the finitely many labels and then average with \(q\); no division by a label probability is needed. Hence, uniformly over the probability simplex, \(C\in\mathcal E_\delta^{(m)}\) and integers \(2\le K\le C_0n^p\),

$$
\begin{gathered}
\frac{\partial_{c_{ij}}F_{q,n,K}(C)}{w_{n,K}}
+\frac{A_{q,ij}+\kappa_{ij}(c_{ij})}{4}\longrightarrow0,\\
\frac{F_{q,n,K}(C)-F_{q,n,K}(I)}{w_{n,K}}-V_q(C)\longrightarrow0,\\
V_q(C)=-\frac14\sum_{i<j}
\left[A_{q,ij}c_{ij}+\int_0^{c_{ij}}\kappa_{ij}(u)\,du\right].
\end{gathered}
\tag{A.33}
$$

The value limit follows by line integration as in (A.16). Only the linear term depends on \(q\). For one common \(q\), convex weights \(\alpha_\ell\), a candidate panel \(C_\ell\) and its physical mean \(\bar C\), write \(\rho_\ell\) for the vector of free correlations and \(H_{ij}(c)=\int_0^c\kappa_{ij}(u)\,du\). Then

$$
\begin{aligned}
\nabla^2V_q&=-\tfrac14\operatorname{diag}(\kappa_{ij}')
\preceq-\tfrac{\beta_\mu}{4}I,\\
\mathcal J_{V_q}
&:=\sum_\ell\alpha_\ell V_q(C_\ell)-V_q(\bar C)\\
&=-\tfrac14\sum_{i<j}\left[
\sum_\ell\alpha_\ell H_{ij}(c_{ij}^{(\ell)})-H_{ij}(\bar c_{ij})\right]\\
&\le-\tfrac{\beta_\mu}{8}\sum_\ell\alpha_\ell
\|\rho_\ell-\bar\rho\|^2.
\end{aligned}
\tag{A.34}
$$

The gap for this fixed panel is independent of \(q\), but minimizers and panels produced by training can depend on \(q\). Different label weights for different candidates do not cancel the linear terms. This is a statement about the limiting potential, not global concavity or Hessian convergence of the finite-particle objective.

For each fixed \(q\), strict concavity and (A.17) place the minimizers of \(V_q\) at extreme points with the same floor-eigenvalue bound. An \(o(w)\)-near-minimizer of the conditional prior expectation approaches that fixed minimizer set. Uniform value convergence does not imply uniform distance to a moving minimizer set for arbitrary \(q_n\): nearly tied minima can switch. If \(q_n\to q\), the compactness argument gives distance to the limiting set for \(q\), not necessarily to each moving set.

Extreme-point geometry alone also does not establish an additional simulation error relative to exact empirical likelihood. To see this, concentrate \(q\) on one spin pattern \(s\). For fixed positive normalized weights \(a_k\), independent of \(C\) and active particles, define \(F_s^a(C)=E[-\log\sum_k a_k\prod_i\sigma(s_iX_{ki})]\), with iid \(X_k\sim N(\mu,C)\). When \(K\ge2\), the Price derivative of \(F_s^a\) has strictly the sign \(-s_is_j\), since \(\gamma_k>0\). The exact pattern probability satisfies

$$
\begin{gathered}
\partial_{c_{ij}}Q_C^A(s)=s_is_jE_C\left[
\sigma'(s_iX_i)\sigma'(s_jX_j)
\prod_{\ell\ne i,j}\sigma(s_\ell X_\ell)\right],\\
C_s=\delta I+(1-\delta)ss^\top,\\
\operatorname*{argmin}_{C\in\mathcal E_\delta^{(m)}}[-\log Q_C^A(s)]
=\operatorname*{argmin}_{C\in\mathcal E_\delta^{(m)}}F_s^a(C)=\{C_s\}.
\end{gathered}
\tag{A.35}
$$

Indeed \(C_s\) simultaneously maximizes every signed correlation \(s_is_jc_{ij}\). Along the feasible segment from any other \(C\) to \(C_s\), all coordinate contributions decrease weakly and at least one decreases strictly. Conditioning on positive nuisance weights and integrating preserves this conclusion for the present finite-\(n\) logistic model. Arbitrary random-weight extensions additionally require the compared objectives or their differences to be defined. At \(K=1\) the expected raw score depends only on fixed marginals and is constant in \(C\), so raw uniqueness fails. None of these signs is a claim about each realized path gradient.

Finally, nonnormalized numerical weights require a mass factor. If \(\widetilde q\ge0\) has total mass \(a>0\) and \(\widetilde A_{ij}=\sum_s\widetilde q_s(s_i-b_i)(s_j-b_j)\), linearity gives

$$
\begin{gathered}
\frac{\partial_{c_{ij}}F_{\widetilde q,n,K}(C)}{w_{n,K}}
\longrightarrow-\frac{\widetilde A_{ij}+a\kappa_{ij}(c_{ij})}{4},\\
\nabla^2V_{\widetilde q}=-\frac a4\operatorname{diag}(\kappa_{ij}').
\end{gathered}
\tag{A.36}
$$

The fixed-panel gap also scales by \(a\). These identities do not authorize changing frozen numerical weights, asserting exact machine-level equality across differently normalized panels, or reinterpreting finite-data boundary preferences as universal excess risk over empirical maximum likelihood.

## A.7. Free active means and scales: a finite curvature threshold

We now release the active means and marginal scales. In A.7–A.9, \(Y\in\{0,1\}^m\) denotes **bits**, and the teacher \(P\) may be any law with marginals \(p_i\in(0,1)\). Write

$$
\begin{gathered}
X_i=\mu_i+s_iZ_i,\quad Z\sim N(0,C),\quad s_i\ge0,\quad C\in\mathcal E_\delta^{(m)},\\
\mu^0=\operatorname{logit}p,\quad D=\operatorname{diag}\{p_i(1-p_i)\},\quad
\Gamma=\operatorname{Cov}_P(Y),\quad R_Y=D^{-1/2}\Gamma D^{-1/2},\\
f_\theta(y,Z)=\prod_i\sigma\{(2y_i-1)(\mu_i+s_iZ_i)\},\qquad
\overline F_K(\theta)=E[-\log\sum_k\pi_k f_\theta(Y,Z_k)].
\end{gathered}
\tag{A.37}
$$

Here \(\theta=(\mu,s,C)\), the active particles are iid, and the entire positive nuisance-weight vector \(\pi_k=W_k/\sum_\ell W_\ell\) is independent of \((Y,Z_1,\ldots,Z_K)\). The weights need not be independent across particles for the next identity. Set \(\chi=E\sum_k\pi_k^2\), \(\omega=1-\chi\). These are nuisance-only weights, not the full weights in (A.31). The centered objective avoids requiring an arbitrary nuisance normalizing constant to be integrable.

At \((\mu,s)=(\mu^0,0)\), the model is the product of the true marginals, denoted \(Q_0\), and both the centered score and exact risk have zero mean/scale gradients. For fixed \(C\),

$$
\begin{aligned}
\nabla^2_{\mu\mu}\overline F_K&=D,&\nabla^2_{\mu s}\overline F_K&=0,&
\nabla^2_{ss}\overline F_K&=D-\omega(\Gamma\circ C),\\
\nabla^2_{\mu\mu}\mathcal R&=D,&\nabla^2_{\mu s}\mathcal R&=0,&
\nabla^2_{ss}\mathcal R&=D-\Gamma\circ C.
\end{aligned}
\tag{A.38}
$$

To verify the scale block, put \(r_i=Y_i-p_i\). The log-kernel derivatives are \(r_iZ_{ki}\) and \(-D_{ii}Z_{ki}^2\mathbf1_{i=j}\). The Hessian of negative log-sum-exp subtracts their weighted covariance. Conditional on \(Y,\pi\), independence of active particles gives
\(E_Z[\sum_k\pi_kZ_{ki}Z_{kj}-(\sum_k\pi_kZ_{ki})(\sum_k\pi_kZ_{kj})]=(1-\sum_k\pi_k^2)c_{ij}\).
Averaging proves the first line. Mean scores are identical across particles, and the mixed block vanishes by Gaussian centering. Exact integration gives the second line with coefficient one. Gaussian polynomial envelopes justify differentiation. One may differentiate in signed scales and then restrict to the nonnegative cone. This generalizes the uniform-weight local formula, in which \(\chi=1/K\); the uniform-weight specialization is not claimed as a separate novelty.

There is a sharp feasible threshold over the whole correlation class. Let

$$
\lambda=\lambda_{\max}(R_Y),\qquad B=\delta+(1-\delta)\lambda,
\qquad \sup_{C\in\mathcal E_\delta^{(m)}}\lambda_{\max}(R_Y\circ C)=B.
\tag{A.39}
$$

Indeed write \(C=\delta I+(1-\delta)T\), with \(T\) a correlation matrix. The Schur product theorem applied to \((\lambda I-R_Y)\circ T\) bounds \(R_Y\circ T\preceq\lambda I\). Equality is attained at \(T=qq^\top\), since this conjugates \(R_Y\) by \(\operatorname{diag}q\). If \(v\) is a unit top eigenvector, choose

$$
q_i=\operatorname{sign}(v_i),\quad
C^\dagger=\delta I+(1-\delta)qq^\top,\quad
N=\|D^{-1/2}|v|\|_2,\quad a=D^{-1/2}|v|/N\ge0.
\tag{A.40}
$$

Either sign may be used at zero coordinates. This choice makes the negative-curvature candidate a physically feasible nonnegative scale activation. Along \(\mu=\mu^0,s=ta,C=C^\dagger\),

$$
\overline F_K''(0)=\frac{1-(1-\chi)B}{N^2},\qquad
\mathcal R''(0)=\frac{1-B}{N^2},\qquad
\chi>1-1/B\ \Longleftrightarrow\ (1-\chi)B<1.
\tag{A.41}
$$

If \(\lambda>1\) and the strict inequality holds, the whitened score Hessian is uniformly positive over all \(C\), with gap at least \(1-(1-\chi)B\). Compactness and Gaussian derivative envelopes extend this to a neighborhood: the product-marginal manifold is a local minimum transverse to \(C\), while sufficiently small positive activation (A.40) lowers observation risk and raises the score. The mean block remains positive, so releasing means does not remove the conclusion. A smooth marginal-preserving compensation \(\mu(t)=\mu^0+O(t^2)\) has the same scale curvature because the mean gradient is zero; its existence follows from the implicit function theorem.

If \((1-\chi)B>1\), the constructed activation lowers both quantities, without proving global recovery. Equality is inconclusive. \(\lambda=1\) means pairwise uncorrelated observations, not necessarily joint independence; pure higher-order dependence is invisible to this second-order test. At zero scales \(C\) is unidentifiable, so the minimum is a manifold, not a unique full parameter. A negative unconstrained eigenvector at another fixed \(C\) need not lie in the scale cone. Nor does nuisance collision determine general gradients away from this base point.

## A.8. A leading-order collapse theorem for free active scales

Fix \(m,\delta\), a compact mean box containing \(\mu^0\), and \(s\in[0,S]^m\), with \(|\mu_i|\le U\). The key bound is elementary. For positive iid active kernels independent of \(\pi\), weighted Jensen and \(\log x\le x-1\) give

$$
\begin{gathered}
0\le L(\mu,s)-\overline F_K(\theta)\le\omega M(\theta),\\
L(\mu,s)=\sum_i\{E\operatorname{softplus}(\mu_i+s_iG)-p_i\mu_i\},\\
M(\theta)=E_Y[(E_Zf_\theta(Y,Z))(E_Zf_\theta(Y,Z)^{-1})-1].
\end{gathered}
\tag{A.42}
$$

For clarity, the inverse moment is \(E(1/f)\), not \(1/Ef\). With \(S_f=\sum_k\pi_k f_k\), the pathwise Jensen gap is at most \(S_f\sum_k\pi_k/f_k-1\). In its expectation, equal-index terms give \(\sum_k\pi_k^2\); unequal-index terms give \((1-\sum_k\pi_k^2)Ef\,E(1/f)\). This proves (A.42).

For unit-scale independent Gaussian-logistic nuisance weights, (A.12)–(A.13) with every active log score zero give the overlap asymptotic. Expanding \(f^{-1}\) over coordinate subsets and using the Gaussian MGF supplies the required uniform moment bound:

$$
\begin{gathered}
\sup_{y,\theta}E_Zf_\theta(y,Z)^{-1}\le2^m e^{mU+m^2S^2/2}<\infty,\\
\omega_{n,K}=(1+o(1))w_{n,K},\qquad
\sup_{2\le K\le C_0n^b,\,\theta}|\overline F_{n,K}(\theta)-L(\mu,s)|
=O\!\left(\sqrt{\log n/n}\right)\to0 .
\end{gathered}
\tag{A.43}
$$

The constants \(C_0,b>0\) are fixed and \(K\) is integer. The overlap error is uniform in this range. This argument uses the actual non-Gaussian nuisance weights through the already proved near-tie estimate; it does not substitute Gaussian log weights. The inverse-moment bound is an asymptotic qualification, not a useful numerical error certificate for every finite activation.

Strict Jensen and Bernoulli cross-entropy minimization show
\(L\ge\sum_i[\operatorname{softplus}(\mu_i)-p_i\mu_i]\ge\sum_i h(p_i)\), with equality exactly at \(\mu=\mu^0,s=0\). Thus, for any allowed sequence \(K_n\),

$$
\begin{gathered}
F_{n,K_n}(\theta_n)-\inf_\theta F_{n,K_n}(\theta)=o(1)\\
\Longrightarrow\quad\mu_n\to\mu^0,\ s_n\to0,\ Q_{\theta_n}\to Q_0,\qquad
\mathcal R(\theta_n)\to D(P\Vert Q_0)=\sum_i h(p_i)-H(P).
\end{gathered}
\tag{A.44}
$$

The nuisance-only constant cancels from regret. Compactness, uniform convergence, and the unique mean/scale minimum imply convergence to the product manifold. The finite pattern probabilities are positive and continuous with a uniform lower bound, so KL is continuous. For random \(o_p(1)\) regret the conclusion holds in probability and, since risk is bounded, also in expectation. If zero scales are only in the closure, the same assertion uses the closure infimum and permits arbitrarily small positive scales.

The limiting risk is total correlation and is positive for every dependent \(P\), including purely higher-order dependence. To call this a failure to attain otherwise realizable zero risk, one must additionally assume \(P=Q_{\theta_*}\) for a feasible \(\theta_*\). This leading-order theorem requires \(o(1)\) regret, unlike the \(o(w)\) resolution of the fixed-scale spectral preference. It neither identifies a finite-\(n\) global minimizer nor proves optimizer convergence. In particular, the nuisance scales are fixed at one throughout this theorem.

For active scales \(s\in[\varepsilon,S]^m\) with \(0<\varepsilon\le S\), assume the mean box contains the unique roots \(E\sigma(\mu_{p_i}(\varepsilon)+\varepsilon G)=p_i\). Envelope differentiation gives \(d\min_\mu L_i/ds=sE\sigma'(\mu_{p_i}(s)+sG)>0\). Hence near-optimal outputs approach \(s_i=\varepsilon,\mu_i=\mu_{p_i}(\varepsilon)\), while \(C\) remains free. On that limiting manifold, Price's formula and \(\sigma'\le1/4\) imply

$$
\begin{gathered}
|\operatorname{Cov}_Q(Y_i,Y_j)|\le\varepsilon^2(1-\delta)/16,\\
\liminf_n\mathcal R(\theta_n)\ge\tfrac12
\left(\max\{|\kappa_{*,ij}|-\varepsilon^2(1-\delta)/4,0\}\right)^2,
\quad \kappa_{*,ij}=\operatorname{Cov}_P(2Y_i-1,2Y_j-1).
\end{gathered}
\tag{A.45}
$$

The risk bound applies Pinsker to a single spin-product moment after the marginals converge. One may maximize over pairs, but not add overlapping pair bounds. It is an asymptotic floor-class statement, not a bound for arbitrary finite outputs with unmatched marginals.

## A.9. When nuisance scales are also free

Fix finite \(K\ge2\) in this subsection; the \(K=1\) observation below is a separately noted extension.

Let all \(n\ge1\) independent fair nuisance coordinates share \(\tau\ge0\). Write \(F_{A,K}(\theta)\) for the active finite-particle score. The conditional Jensen bound and strict scale monotonicity in A.5 hold for every active \(\theta\), so

$$
\begin{gathered}
B_{n,K}(\theta,\tau):=F_{n,K}(\theta,\tau)-n\log2-F_{A,K}(\theta),\quad
0\le B_{n,K}\le n\tau^2/8,\quad \partial_\tau F_{n,K}>0\ (\tau>0),\\
\inf_{\theta,\tau\ge0}F_{n,K}=n\log2+\inf_\theta F_{A,K},\qquad
r(\theta,\tau)=r_A(\theta)+B_{n,K}(\theta,\tau).
\end{gathered}
\tag{A.46}
$$

Here \(r,r_A\) are the respective population regrets, both nonnegative. Thus a joint global minimizer has \(\tau=0\) and an active finite-\(K\) minimizer, when the minimum exists. For fixed \(n,K\), compact active parameters and bounded \(\tau\), vanishing joint regret forces \(\tau\to0\): continuity makes \(B\) uniformly positive on each compact subset \(\tau\ge a>0\). There is no rate uniform in growing \(n,K\). If \(\tau\ge\varepsilon>0\), the boundary becomes \(\varepsilon\); a softplus parameter with a positive floor may attain only an infimum at finite raw coordinates.

Under correct specification \(P=Q_{\theta_*}\) for a feasible \(\theta_*\), Jensen gives a useful but standard risk comparison:

$$
e_K(\theta_*)=F_{A,K}(\theta_*)-H(P)\ge0,\qquad
\mathcal R(\theta)\le r+e_K(\theta_*)
\quad\text{or}\quad r+e_K(\theta_*)+n\varepsilon^2/8
\tag{A.47}
$$

for zero or positive nuisance floor, respectively; each \(r\) is defined in its own feasible class. Compare the joint optimum with \((\theta_*,0)\), or \((\theta_*,\varepsilon)\), and use \(F\ge H(P)+\mathcal R+n\log2\). Positive kernels, finite active alphabet and Gaussian log moments allow (A.26) at fixed \(\theta_*\), giving \(e_K\to0\). Hence \(K\to\infty,r\to0\) suffices for ideal population risk recovery at zero nuisance floor; adding \(n\varepsilon^2\to0\) suffices for the positive-floor bound. Fixed \(K\) leaves active simulation bias, and at \(K=1\) its optimum is the product model itself. These are not finite-training guarantees.

At \((\mu,s,\tau)=(\mu^0,0,0)\), the complete Hessian in these coordinates is

$$
\nabla^2_{(\mu,s,\tau)}F_{n,K}
=\operatorname{blockdiag}\{D,\ D-(1-1/K)(\Gamma\circ C),\ n/(4K)\}.
\tag{A.48}
$$

The \(C\) directions and their mixed blocks are zero. For each separate nuisance coordinate, its centered log-kernel derivative is \(G/2\), yielding curvature \(1/(4K)\); independence kills cross terms and a common scale sums the \(n\) entries. Exact observation risk has the scale block \(D-\Gamma\circ C\) and zero nuisance block.

The finite-\(n\) endpoints of nuisance collision also matter. For iid Gaussian \(G_{kj}\),

$$
\begin{gathered}
\chi_{n,K}(\tau)=E\sum_k\left[
\frac{\prod_j\sigma(\tau G_{kj})}{\sum_\ell\prod_j\sigma(\tau G_{\ell j})}\right]^2,
\quad \chi_{n,K}(0)=1/K,\\
\lim_{\tau\to\infty}\chi_{n,K}(\tau)
=\chi_\infty(n,K)=P(M=0)+\sum_{r=1}^K\frac{P(M=r)}r,
\quad M\sim\operatorname{Binomial}(K,2^{-n}).
\end{gathered}
\tag{A.49}
$$

Continuity follows by dominated convergence, and finite positive \(\tau\) gives \(1/K<\chi<1\). If \(M\ge1\), exactly the all-positive rows retain limiting weight, shared equally. If \(M=0\), the sums \(\sum_j\min(G_{kj},0)<0\) have an almost surely unique maximum, so one row dominates. The collision is bounded, proving (A.49). In particular,

$$
\chi_\infty(n,K)\ge1-\binom K2\,4^{-n}.
\tag{A.50}
$$

This uses \(1-\chi_\infty\le P(M\ge2)\le E\binom M2\). The endpoint is generally not one: fixed \(n\), infinite scale differs from increasing \(n\) at fixed positive scale.

At the product base point and arbitrary finite \(\tau\), (A.38) uses \(\chi_{n,K}(\tau)\). Therefore continuity alone yields

$$
\begin{gathered}
(1-1/K)B>1\ \Longrightarrow\ \exists\eta>0:
\overline F_K''(0;\tau)<0\quad(0\le\tau<\eta),\\
\chi_\infty(n,K)>1-1/B\ \Longrightarrow\ \exists T<\infty:
H_s^F(C;\tau)\succ0\quad(C\in\mathcal E_\delta^{(m)},\ \tau>T).
\end{gathered}
\tag{A.51}
$$

The first curvature is along (A.40); exact-risk curvature remains negative. If both conditions hold, the threshold is crossed at least once. Neither monotonicity of \(\chi\), a unique crossing, nor a finite-amplitude guarantee follows. At \(\tau>0\) the joint objective still has a nuisance descent direction, so positive active curvature is not a joint stationary point.

Finally, local topology must be specified. At \(C=I,\tau=0\), the active score Hessian is \(D/K\succ0\), even when (A.51)'s first condition holds elsewhere in the class. A small Euclidean neighborhood in \((\mu,s,C,\tau)\) can therefore have a local minimum manifold. All \(C\) represent the same product model at \(s=0\); a flat path can move to \(C^\dagger\), followed by descent. In covariance-factor coordinates \(\Sigma=AA^\top\), the path \(A=t\operatorname{diag}(a)L_{C^\dagger}\) already lies in every small neighborhood of \(A=0\), excluding that same local-minimum claim when its curvature is negative. The observation distributions also approach \(Q_0\) along descent. This is a coordinate-dependent boundary, not a universal optimizer trap. The experiments use positive physical scales and a specified repair rule; they do not inherit a convergence or trapping theorem from these local identities.
