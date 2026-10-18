<a id="risk-proof"></a>

# D. Sampling risk and the scope of numerical outputs

## D.1. Exact empirical minimization under local alternatives

This section first retains the known-channel F baseline and its matched ridge. Use the fixed model and compact domain of Appendix A. Here $m(O)=(\mu_1\mu_2,\mu_1\mu_3,\mu_2\mu_3)$, $H_K=S_F$ and $A_K=H_K^{-1}J$; these baseline symbols are not the adaptive proposal score A_Q. The full F/L/A and unknown-channel extension is proved in Section D.5. For all sufficiently large N, observations are independent from the teacher $\eta_N=h/\sqrt N$, with fixed h. The estimator in this section is any measurable global minimizer of the empirical objective after particle randomness has been averaged exactly. It is distinct from the fixed-program output evaluated in the experiments.

$$
\eta_N=h/\sqrt N,\qquad
\widehat F_{K,N}(\theta)=\frac1N\sum_{n=1}^N\mathcal L_K(\theta,O_{Nn}),\qquad
\widehat\theta_{K,N}\in\operatorname*{argmin}_{\theta\in\Theta_\delta}\widehat F_{K,N}(\theta).
\tag{D.1}
$$

Writing the objective as a finite sum against empirical observation frequencies shows uniform convergence to $F_K(\cdot;0)$. The unique global anchor therefore forces every global empirical minimizer to converge in probability to 0. Uniform convergence of local Hessians to $H_K$ and the strict-gap argument of Appendix A show that, with probability tending to one, this minimizer is unique, interior, and in a fixed strongly convex neighborhood.

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
\tag{D.2}
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
\tag{D.3}
$$

A local output known to converge to 0 has the same distributional expansion if its interior score residual is $o_p(N^{-1/2})$. A fixed numerical tolerance does not imply that condition. Even such a distributional expansion alone does not imply the expectation result below.

## D.2. Expected joint risk and the three regions

Positive finite-alphabet probabilities give the local KL expansion

$$
R(\theta,\eta)=\tfrac12(\theta-\eta)^\top J(\theta-\eta)
+O((\|\theta\|+\|\eta\|)^3).
\tag{D.4}
$$

To justify taking expectations, choose a fixed neighborhood of the independent observation-frequency vector small enough to preserve the global gap and a local Hessian lower bound $aI$ with $a>0$. Under the local alternatives, empirical frequencies lie outside that neighborhood with probability at most $c_1\exp(-c_2N)$, for all sufficiently large N. On the good event, localization and the score equation imply

$$
\|\widehat\theta_{K,N}\|\leq a^{-1}\|\bar m_N\|,\qquad
\sup_N E\|\sqrt N\bar m_N\|^q<\infty\quad(1\leq q<\infty).
\tag{D.5}
$$

The moment bound holds because m has finite support and its moving mean is $O(N^{-1/2})$. On the bad event the parameter domain and joint KL are bounded; any fixed power of N times its exponentially small probability tends to zero. In particular, scaled parameter moments of order greater than two are uniformly bounded. This gives uniform integrability of the limiting quadratic form, while the expected cubic remainder in (D.4), multiplied by N, is $O(N^{-1/2})$ on the good event and negligible on its complement. Consequently,

$$
\begin{gathered}
W=(A_K-I_3)h+H_K^{-1}\xi,\qquad
NR(\widehat\theta_{K,N},\eta_N)\Rightarrow\tfrac12 W^\top JW,\\
\lim_{N\to\infty}NER(\widehat\theta_{K,N},\eta_N)
=\tfrac12\!\left[
h^\top(A_K-I_3)^\top J(A_K-I_3)h
+\operatorname{tr}(JH_K^{-1}JH_K^{-1})\right].
\end{gathered}
\tag{D.6}
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
\tag{D.7}
$$

These are limits of expected $N$ times joint KL along changing teachers. They neither locate finite-N crossings nor describe a fixed nonzero teacher as N grows without bound. The numerical study tests selected predicted directions using frozen programs; it does not establish a continuous finite-N risk curve. Matching the first-order ridge constant is not equivalence at finite N. For a general algorithm, an additional uniform-integrability argument is required to transfer a distributional approximation into expected risk.

## D.3. A fixed teacher and the remaining bias

There exists $\varepsilon>0$ such that, for every fixed teacher $\eta_0$ with $0<\|\eta_0\|<\varepsilon$, exact empirical global minimization is consistent for the unique group-posterior population branch, while integrated maximum likelihood is consistent for the teacher. More precisely, for iid observations from this single fixed teacher,

$$
\begin{gathered}
\widehat\theta_{K,N}\longrightarrow_{\rm a.s.}\widetilde\theta_K(\eta_0),\qquad
ER(\widehat\theta_{K,N},\eta_0)\longrightarrow
R(\widetilde\theta_K(\eta_0),\eta_0)>0,\\
\widehat\theta_{{\rm ML},N}\longrightarrow_{\rm a.s.}\eta_0,\qquad
ER(\widehat\theta_{{\rm ML},N},\eta_0)\longrightarrow0.
\end{gathered}
\tag{D.8}
$$

Indeed, the finite-alphabet strong law gives uniform empirical-objective convergence on the compact domain. A unique population minimizer then identifies every measurable empirical argmin by the usual compact separation argument: for every neighborhood of that minimizer, the continuous population objective has a strictly positive gap on its compact complement. For sufficiently large N the empirical gap remains positive.

Shrink $\varepsilon$ to stay in the unique-global-branch neighborhood. Since each eigenvalue of $A_K$ is strictly between zero and one, $A_K-I$ is invertible. The expansion $\widetilde\theta_K(\eta)-\eta=(A_K-I)\eta+O(\|\eta\|^2)$ is therefore nonzero for every sufficiently small nonzero teacher. Identifiability gives the strict KL inequality in (D.8). Continuity and boundedness of risk on the compact candidate domain give the expectation limits by dominated convergence. Integrated likelihood uses identifiability directly to obtain its unique population optimum at $\eta_0$.

It follows that, for each such fixed teacher, some finite $N_0$ makes integrated exact estimation strictly better in expected risk than fixed-K group exact empirical minimization for every $N\geq N_0$. This is a pointwise existence statement: neither $\varepsilon$ nor $N_0$ is numerically supplied, risk need not be monotone, and no unique crossing is asserted. It does not establish the limit of the current numerical program or the risk floor of a different high-dimensional prior-proposal model.

## D.4. What the quadratic control preserves

The implementation uses a control that preserves the origin derivatives of the exact single-observation objective. This identity explains a local connection to the theory and also identifies its boundary. For an arbitrary fixed K-particle bank at one observation, let $f_{\rm raw}$ be the negative log mean importance weight in (A.5) with zero tilt, and let bars denote bank averages. Here $s=(X_1X_2,X_1X_3,X_2X_3)$, $m=E_qs$, $V=\operatorname{Cov}_q(s)$, t is the origin Hessian of $\log\rho_\theta$, and $\rho_\theta=\phi_{C(\theta)}/\phi_I$. Using the exact moments of its actual proposal, define

$$
\begin{gathered}
f_u(\theta,o)=f_{\rm raw}(\theta,o)
+\overline{\log\rho_\theta}-E_{q_o}\log\rho_\theta,\\
D(o)=\bar s\bar s^\top-\overline{ss^\top}+(1-K^{-1})V(o),\qquad
f_c(\theta,o)=f_u(\theta,o)-\tfrac12\theta^\top D(o)\theta,\\
\nabla f_c(0,o)=-m(o),\qquad
\nabla^2f_c(0,o)=-E_{q_o}t-(1-K^{-1})V(o)=H_K(o).
\end{gathered}
\tag{D.9}
$$

To verify the identity, uniform weights at 0 give the raw score $-\bar s$ and raw Hessian $-\bar t-\overline{ss^\top}+\bar s\bar s^\top$. The first control replaces $-\bar s$ by $-m$ and $-\bar t$ by $-E_qt$; subtracting D then gives (D.9), for every observation and every bank. Under independent proposal sampling, $E[\bar s\bar s^\top]=mm^\top+V/K$ and $E[\overline{ss^\top}]=mm^\top+V$, so $E D=0$. Both controls have zero conditional expectation and preserve $\mathcal L_K$. They do not make the group and ridge Hessians equal for each observation.

For a fixed finite collection of banks for each observation-alphabet state, sampled independently of the data and held fixed as N varies, the controlled scalar is smooth near 0, with finite local derivative bounds conditional on the banks. Its per-observation origin score and Hessian are exactly those in (D.9). Hence its consistently selected local stationary branch has the same first-order expansion (D.2); the number of banks need not increase for this conditional local statement. This does not preserve the global Jensen gap of the fixed-bank scalar or show that arbitrary starts and finite solver budgets select that branch. Passing from conditional distributional limits to risk averaged over random banks requires integrable control over the banks or an appropriate bound on the output moments.

Finally, the identities assume exact moments of the actual proposal. Fixed quadrature order, finite-bit sampling, and fixed solver tolerance do not disappear merely because N increases. Sufficient local conditions for an asymptotically exact numerical extension include per-observation score error $o(N^{-1/2})$, Hessian error $o(1)$, controlled higher derivatives, localization, and solver score residual $o_p(N^{-1/2})$; expectation claims additionally need uniform integrability. Evaluating the same limiting $N\times\mathrm{KL}$ requires risk error $o(1/N)$. The finite study instead reports its achieved precision diagnostics and all frozen-program risks. Agreement of its risk directions with (D.7) tests the combined data variation, nonlinear risk, constraints, higher-order terms and numerical outputs; because the origin derivatives are preserved by construction, it is not an independent rediscovery of those derivatives.

## D.5. Adapted proposals and an unknown common channel

We now treat the full F/L/A parameter vector of Appendices A--C. Keep K, all dimensions, the spectral domain and the proposal boxes fixed. In the unknown-channel comparison use the homogeneous model and a common alpha in [.35,.65], with $\alpha_*=1.4/\sqrt8$ interior; all other generation parameters remain known. The teacher is $\eta_N=h/\sqrt N$, with fixed three-dimensional h. Each observation is one complete three-group row.

The joint estimator minimizes the exact conditional-bank expectation over the entire fixed domain. The oracle fixes the true channel. The plug-in estimator first minimizes the marginal criterion on the same N rows, then minimizes the same-class correlation/proposal criterion at that channel:

$$
\begin{aligned}
\widehat z_{Q,N}&\in\arg\min_{z\in D_Q}
 N^{-1}\sum_n L_K^Q(z;O_n),\\
\widetilde\alpha_N&\in\arg\min_{\alpha\in\mathcal I_\alpha}
 \left\{-N^{-1}\sum_{n,i}\log Z_{i,\alpha}(O_{ni})\right\},\\
(\widehat\theta^{\rm plug}_{Q,N},\widehat u^{\rm plug}_{Q,N})
&\in\arg\min_{\theta,u}
 N^{-1}\sum_n L_K^Q(\theta,\widetilde\alpha_N,u;O_n).
\end{aligned}
\tag{D.10}
$$

Every selection is defined on every dataset, including ties, missing states and boundary solutions. One rule is successive coordinate minimization over the compact argmin set. The resulting lexicographic selection is measurable because, for every finite N, there are only finitely many possible frequency vectors. This definition does not assert that a practical optimizer computes it.

At the anchor $z_0=(0,\alpha_*,0)$, each observation's Jensen gap is differentiable, nonnegative and zero. Its gradient therefore vanishes. Differentiating the observation probability gives

$$
\begin{gathered}
s_{\theta,ij}=\mu_i\mu_j,\quad
s_\alpha=\sum_i\left.\partial_\alpha\log Z_{i,\alpha}(O_i)\right|_{\alpha_*},\\
\nabla_zL_K^Q(z_0;O)=-(s_\theta,s_\alpha,0_u),\qquad
V_0=\operatorname{diag}(J,I_\alpha,0),\\
H_0=
\begin{pmatrix}
J+C_r/K&0&-M_Q/K\\
0&I_\alpha&0\\
-M_Q^\top/K&0&G_Q/K
\end{pmatrix}\succ0 .
\end{gathered}
\tag{D.11}
$$

The covariance V0 and Hessian H0 are different objects. In the covariance, every correlation/channel cross-term contains an independent zero-mean group posterior mean. The zero proposal coordinates of the loss gradient concern conditional bank expectation; they do not set the proposal-density score $A_Q$ or a random-bank loss gradient to zero. The Hessian is positive definite although V0 is singular. The changing posterior base cancels the channel residual, and moving mean features have coefficient zero at the anchor.

Here is the expectation argument for the complete vector. Let p0 be the independent observation law and p_hat the empirical frequency vector on the fixed alphabet. Appendix A gives a unique global anchor, a positive gap outside a neighborhood, and local uniform positive curvature. Bounded per-state loss and local derivatives imply that a sufficiently small fixed bound on $\|\widehat p-p_0\|_1$ preserves this gap and curvature. The empirical global minimizer is then the unique interior stationary point, a twice continuously differentiable function of the frequencies. Since $p_N-p_0=O(N^{-1/2})$, finite-alphabet concentration bounds the complementary event by $C\exp(-cN)$. Expanding the smooth function on the good event gives

$$
\begin{gathered}
\widehat z_{Q,N}-z_0
=H_0^{-1}\frac1N\sum_n(s_\theta(O_n),s_\alpha(O_n),0_u)+R_N,\\
\|R_N\|\le C\|\widehat p-p_0\|_1^2
\quad\hbox{on the good event},\\
E\|R_N\|^2=O(N^{-2}),\qquad
E\|\widehat z_{Q,N}-z_0\|^3=O(N^{-3/2}).
\end{gathered}
\tag{D.12}
$$

The squared remainder follows from the fourth moment of bounded empirical frequencies; on the bad event, compactness and bounded scores bound the remainder uniformly. The cubic moment and globally bounded generating KL control the local Taylor remainder and all bad-event risk. Thus these conditions establish convergence of the needed moments, rather than only convergence in distribution. The concentration constants are not numerically quantified and give no practical N threshold.

Solving the complete block system in (D.11) yields

$$
\begin{gathered}
\psi_{\theta,Q}=S_Q^{-1}s_\theta,\qquad
\psi_\alpha=I_\alpha^{-1}s_\alpha,\qquad
\psi_{u,Q}=G_Q^{-1}M_Q^\top\psi_{\theta,Q},\\
E_{\eta_N}s_\theta=Jh/\sqrt N+O(N^{-1}),\qquad
E_{\eta_N}s_\alpha=0,\\
\sqrt N\,\widehat\theta_Q\Rightarrow
N(S_Q^{-1}Jh,S_Q^{-1}JS_Q^{-1}),\\
\sqrt N(\widehat\alpha_Q-\alpha_*)\Rightarrow N(0,I_\alpha^{-1}).
\end{gathered}
\tag{D.13}
$$

The two generating blocks are independent in the limit, not necessarily at finite N. The channel score mean is zero for every allowed teacher because its group marginals are unchanged. L's proposal first-order influence is zero; A's directed influences are induced by the correlated-model block. Counting each proposal coefficient as a separate generating-risk dimension would be incorrect.

For known channels, the general adapted risk is the right side of (D.6) with H_K replaced by S_Q and A_K by $S_Q^{-1}J$. For the homogeneous unknown common channel, the generating Fisher matrix is $\operatorname{diag}(jI_3,I_\alpha)$, so (D.12)--(D.13) give

$$
\begin{aligned}
\lim_N NER_Q
&=\frac12j(1-r_Q)^2\|h\|^2+\frac32r_Q^2+\frac12,\\
r_F=r_L&=\frac{Kj}{1+(K-1)j},\qquad
r_A=\frac{Kj}{Kj+(1-v)^2}.
\end{aligned}
\tag{D.14}
$$

All three correlation coordinates are estimated, even for a two-edge teacher direction. Fixing the true channel deletes the final 1/2. Integrated ML uses the four generating coordinates directly and has limit 2; independence fixes theta=0 and estimates only the common channel, giving $j\|h\|^2/2+1/2$. These are not K=infinity assertions about identifiable proposal coordinates. The known-channel matched-ridge result in D.1--D.2 remains the baseline comparison.

For the plug-in estimator, channel identifiability and positive I_alpha give the same compact localization and frequency expansion for its first stage:

$$
\widetilde\alpha_N-\alpha_*
=I_\alpha^{-1}N^{-1}\sum_n s_\alpha(O_n)+r_{\alpha,N},
\qquad E r_{\alpha,N}^2=O(N^{-2}).
\tag{D.15}
$$

The random first-stage channel must also preserve the second-stage global gap. The uniform derivative envelope from Appendix A gives

$$
\sup_{\theta,u}
|\widehat F_N(\theta,\widetilde\alpha_N,u)
 -F_0(\theta,\alpha_*,u)|
\le C\|\widehat p-p_0\|_1+C|\widetilde\alpha_N-\alpha_*|.
\tag{D.16}
$$

On the same type of good event the second-stage minimizer is therefore unique and local. Its derivative with respect to alpha at the anchor is zero, since $H_{(\theta,u),\alpha}=0$. Composing the two smooth frequency maps gives the same first-order influences and an $O(N^{-2})$ mean-square remainder. In particular,

$$
E\|(\widehat\theta^{\rm joint},\widehat\alpha^{\rm joint})
 -(\widehat\theta^{\rm plug},\widetilde\alpha)\|^2
=O(N^{-2}).
\tag{D.17}
$$

Their leading expected risk is consequently identical. Their complete objectives, finite-N outputs and higher-order risks need not agree. Joint minus true-channel oracle empirical correlation differences are only controlled at order $O_{L^2}(N^{-1})$ by this argument; the population t^3 comparison cannot be substituted for a sample-level rate.

Data reuse is substantive. Splitting a total of N rows into a fraction a for the channel and 1-a for the correlation fit changes the leading variance to $3r_Q^2/[2(1-a)]+1/(2a)$, for 0<a<1, while retaining the same local bias term. A different, inefficient first-stage channel estimate need not share the common 1/2 either. Standard M-estimator and two-step arguments underlie this analysis [@newey1994; @westling2019]; the explicit scores and finite-alphabet moment control establish their consequences for these classes. No growing-K/dimension, finite-bank or finite-budget-optimizer extension is asserted.
