# C. Sampling risk and the scope of numerical outputs {#risk-proof}

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
