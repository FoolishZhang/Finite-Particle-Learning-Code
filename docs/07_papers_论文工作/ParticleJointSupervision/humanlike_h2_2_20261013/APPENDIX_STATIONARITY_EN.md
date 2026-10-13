# Appendix G. Matched means need not preserve finite-particle stationarity

## G.1 A two-particle counterexample

Consider three logistic observations, unit Gaussian marginal variances, independent teacher coordinates, and known matched means $\mu_t=(t,t,0)$. Write $\sigma(x)=(1+e^{-x})^{-1}$ and let $P_t$ be the teacher label distribution on $\{-1,1\}^3$. With two independent candidate particles $X_k\sim N(\mu_t,C)$ and no nuisance coordinates, define

$$
\begin{aligned}
F_{0,2}(C;\mu_t)
&=E_{Y\sim P_t}E_X\left[-\log\left\{\frac12\sum_{k=1}^2
\prod_{i=1}^3\sigma(Y_iX_{ki})\right\}\right],\\
g(t)&=\left.\partial_{c_{12}}F_{0,2}(C;\mu_t)\right|_{C=I}.
\end{aligned}
\tag{G.1}
$$

The teacher distribution remains fixed when differentiating the candidate correlation. The other two candidate correlations are zero.

**Proposition G.1.** There is a constant $\Gamma>0$ such that

$$
\lim_{t\to+\infty}e^{2t}g(t)=-e\Gamma<0.
\tag{G.2}
$$

Consequently, there exists a finite $T$ such that $g(t)<0$ for every real $t>T$. Moreover, $g$ is the real restriction of a nonzero entire function, is even, and satisfies $g(0)=0$. Thus there exists $\epsilon_0>0$ such that $g(t)\ne0$ whenever $0<|t|<\epsilon_0$. Neither threshold is quantified. The proposition does not determine the first nonzero Taylor order or the sign near zero.

## G.2 Four label contributions and their limits

Let $\varepsilon=e^{-t}$, $R_{ki}=e^{-Z_{ki}}$ for $i=1,2$, and $B_k=\sigma(Z_{k3})$, where all $Z_{ki}$ are independent standard Gaussians. Put $a=ER_{ki}=e^{1/2}$, $p_t=E\sigma(t+Z)$, and $q_t=1-p_t$. Dominated convergence gives $p_t\to1$ and $q_t/\varepsilon\to a$. For the third label, reflection of $Z_{k3}$ leaves the law of $B_k$ unchanged; its two probabilities sum to one, so there is no additional factor of two.

For $s=(s_1,s_2)\in\{+,-\}^2$ and $\delta_i=1_{\{s_i=-\}}$, define

$$
\begin{aligned}
\pi_k^s(\varepsilon)&=
\frac{B_kR_{k1}^{\delta_1}R_{k2}^{\delta_2}/[(1+\varepsilon R_{k1})(1+\varepsilon R_{k2})]}
{\sum_{\ell=1}^2B_\ell R_{\ell1}^{\delta_1}R_{\ell2}^{\delta_2}/[(1+\varepsilon R_{\ell1})(1+\varepsilon R_{\ell2})]},\\
d_{ki}^+&=\frac{\varepsilon R_{ki}}{1+\varepsilon R_{ki}},\qquad
 d_{ki}^-=-\frac1{1+\varepsilon R_{ki}}.
\end{aligned}
\tag{G.3}
$$

The common factor $\varepsilon^{\delta_1+\delta_2}$ cancels in the weights. Price's identity [@price1958], applied to the mixed derivative of the negative log particle average, gives

$$
\begin{aligned}
G^s&=-\sum_{k=1}^2\pi_k^s(1-\pi_k^s)d_{k1}^{s_1}d_{k2}^{s_2},\\
g(t)&=p_t^2EG^{++}+p_tq_t(EG^{+-}+EG^{-+})+q_t^2EG^{--}.
\end{aligned}
\tag{G.4}
$$

For $0<\varepsilon\le1$, valid dominating bounds are

$$
\begin{aligned}
\varepsilon^{-2}|G^{++}|&\le\sum_kR_{k1}R_{k2},&
\varepsilon^{-1}|G^{+-}|&\le\sum_kR_{k1},\\
\varepsilon^{-1}|G^{-+}|&\le\sum_kR_{k2},&
|G^{--}|&\le1.
\end{aligned}
\tag{G.5}
$$

These lognormal envelopes are integrable. Since $B_k>0$ almost surely, the weights converge pointwise to those proportional to $B_kR_{k1}^{\delta_1}R_{k2}^{\delta_2}$; no inverse moment of $B$ is required. For independent standard Gaussians $U_1,U_2$, also independent of $B_1,B_2$, introduce the collision rate

$$
T_k(v)=B_ke^{\sqrt vU_k},\qquad
c(v)=E\sum_{k=1}^2\left(\frac{T_k(v)}{T_1(v)+T_2(v)}\right)^2,
\qquad v\ge0.
\tag{G.6}
$$

Multiplication by one or two independent $R$ factors gives $v=1$ or $v=2$. After applying (G.5), independence of each unused $R$ coordinate gives

$$
\begin{aligned}
\varepsilon^{-2}EG^{++}&\longrightarrow-a^2[1-c(0)],\\
\varepsilon^{-1}EG^{+-},\ \varepsilon^{-1}EG^{-+}
&\longrightarrow a[1-c(1)],\\
EG^{--}&\longrightarrow-[1-c(2)].
\end{aligned}
\tag{G.7}
$$

In particular, the teacher tail probabilities in (G.4) are essential:

$$
\begin{aligned}
\lim_{t\to\infty}e^{2t}g(t)
&=-a^2[1-c(0)]+2a^2[1-c(1)]-a^2[1-c(2)]\\
&=e\{c(0)-2c(1)+c(2)\}.
\end{aligned}
\tag{G.8}
$$

## G.3 A strictly signed Fourier representation

Set $L=\log B$ and $\phi_L(\omega)=Ee^{i\omega L}$. The two-particle log odds satisfy

$$
\begin{aligned}
D(v)&=L_1-L_2+\sqrt v(U_1-U_2),\\
Ee^{i\omega D(v)}&=|\phi_L(\omega)|^2e^{-v\omega^2},\\
c(v)&=1-\tfrac12E\operatorname{sech}^2(D(v)/2).
\end{aligned}
\tag{G.9}
$$

The nonnegative squared modulus follows from the independent, identically distributed $L_1,L_2$; $L$ itself need not be symmetric. With the convention $\widehat f(\omega)=\int_{\mathbb R}f(x)e^{-i\omega x}\,dx$, substitution $u=e^x$ for $f(x)=\operatorname{sech}^2(x/2)$ yields

$$
\widehat f(\omega)
=4\int_0^\infty\frac{u^{-i\omega}}{(1+u)^2}\,du
=4\Gamma(1-i\omega)\Gamma(1+i\omega)
=\frac{4\pi\omega}{\sinh(\pi\omega)}.
\tag{G.10}
$$

Here the Beta integral has parameters with real part one; the Gamma recurrence and reflection formulas apply [@nistdlmf, Sections 5.12 and 5.5]. At zero the continuous value is four. Since $\widehat f$ is integrable and $|\phi_L|\le1$, Fourier inversion and expectation can be interchanged. Hence

$$
\begin{aligned}
c(v)&=1-\int_{\mathbb R}\frac{\omega}{\sinh(\pi\omega)}
|\phi_L(\omega)|^2e^{-v\omega^2}\,d\omega,\\
c(0)-2c(1)+c(2)&=-\Gamma,\\
\Gamma&=\int_{\mathbb R}\frac{\omega}{\sinh(\pi\omega)}
|\phi_L(\omega)|^2(1-e^{-\omega^2})^2\,d\omega>0.
\end{aligned}
\tag{G.11}
$$

Strict positivity is analytic: $\omega/\sinh(\pi\omega)>0$ (with value $1/\pi$ at zero), and continuity with $\phi_L(0)=1$ makes its squared modulus positive on a neighborhood of zero. The remaining factor is positive away from zero. Equations (G.8) and (G.11) prove (G.2), without numerical integration or a finite-$t$ certificate.

## G.4 Entire continuation with an explicit envelope

For real $x\in\mathbb R^6$ and $y\in\{-1,1\}^3$, keep the bounded Price kernel fixed:

$$
\begin{aligned}
\pi_k^y(x)&=\frac{\prod_{i=1}^3\sigma(y_ix_{ki})}
{\sum_{\ell=1}^2\prod_{i=1}^3\sigma(y_ix_{\ell i})},\\
H_y(x)&=-\sum_{k=1}^2\pi_k^y(x)(1-\pi_k^y(x))
y_1\sigma(-y_1x_{k1})y_2\sigma(-y_2x_{k2}),\\
|H_y(x)|&\le1.
\end{aligned}
\tag{G.12}
$$

Move the mean into the Gaussian density. For $z=a+ib$ with $|z|\le R$, the elementary bound $(x-a)^2\ge x^2/2-a^2$ gives

$$
\left|e^{-(x-z)^2/2}\right|
=e^{-(x-a)^2/2+b^2/2}
\le e^{R^2/2}e^{-x^2/4}.
\tag{G.13}
$$

There are four shifted coordinates and two unshifted ones. Consequently the complexified density and its kernel integral obey

$$
\begin{aligned}
\rho_z(x)&=(2\pi)^{-3}\exp\left[-\frac12\sum_{k=1}^2
\{(x_{k1}-z)^2+(x_{k2}-z)^2+x_{k3}^2\}\right],\\
|H_y(x)\rho_z(x)|&\le(2\pi)^{-3}e^{2R^2}e^{-\|x\|^2/4},\\
h_y(z)&=\int_{\mathbb R^6}H_y(x)\rho_z(x)\,dx.
\end{aligned}
\tag{G.14}
$$

The displayed bound is integrable and uniform on every compact complex disk. Since the integrand is entire in $z$, dominated holomorphic integration makes each $h_y$ entire. The same one-dimensional argument makes $p(z)=\int_{\mathbb R}\sigma(x)\varphi(x-z)\,dx$ entire, where $\varphi$ is the standard Gaussian density continued in its argument. Each teacher weight is $w_y(z)=\tfrac12\prod_{i=1}^2p(z)^{1_{\{y_i=1\}}}[1-p(z)]^{1_{\{y_i=-1\}}}$. The finite sum $\sum_yw_y(z)h_y(z)$ is therefore an entire extension of $g$. This construction evaluates logistic functions only at real $x$; it does not assume that a complex log-sigmoid is entire.

Simultaneously reflecting the first two coordinates preserves $c_{12}$ and changes $t$ to $-t$, giving $g(-t)=g(t)$. Single-coordinate reflection at zero gives $g(0)=0$. Equation (G.2) excludes the identically zero function. Isolated zeros and evenness now imply, for some integer $r\ge1$ and real $A\ne0$,

$$
g(t)=At^{2r}+O(t^{2r+2}),\qquad
0<|t|<\epsilon_0\ \Longrightarrow\ g(t)\ne0.
\tag{G.15}
$$

This refutes universal stationarity under known matched means. It neither proves a nonzero quadratic coefficient $b_2$ nor transfers to $K=100$. In particular, it does not resolve the six gradient intervals containing zero at the asymmetric mean used in Appendix F, establish nonstationarity for every finite $K$, or negate the exact symmetry result when at most one mean coordinate is nonzero.
