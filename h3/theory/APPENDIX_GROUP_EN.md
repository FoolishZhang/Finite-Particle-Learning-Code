# B. Group-posterior selection and local dependence {#group-proof}

## B.1. Model, regularity, and the global anchor

This appendix proves the group-posterior results for three active variables with fixed standard normal marginals and the known local observation channel defined in the main text. Write the candidate as $\theta$ and the teacher as $\eta$. They share the compact spectral domain below; the origin is an interior point. Each local alphabet is finite, every state has positive conditional probability, and the observations are conditionally independent across groups given the active variables.

$$
\begin{gathered}
\theta=(c_{12},c_{13},c_{23}),\qquad
\Theta_\delta=\{\theta:C(\theta)\succeq\delta I_3,\ \operatorname{diag}C(\theta)=\mathbf1\},\quad 0<\delta<1,\\
p_\theta(o)=\int\phi_{C(\theta)}(x)\prod_{i=1}^3\ell_i(o_i\mid x_i)\,dx,\\
Z_i(o_i)=\int\phi(x_i)\ell_i(o_i\mid x_i)\,dx_i,\qquad
q_o(x)=\prod_{i=1}^3\frac{\phi(x_i)\ell_i(o_i\mid x_i)}{Z_i(o_i)},\qquad
Z(o)=\prod_i Z_i(o_i),\\
\rho_\theta(x)=\frac{\phi_{C(\theta)}(x)}{\phi_{I_3}(x)},\qquad
\mathcal L_K(\theta,o)=E_{q_o^{\otimes K}}\!\left[-\log\left\{\frac{Z(o)}K\sum_{k=1}^K\rho_\theta(X_k)\right\}\right],\\
F_K(\theta;\eta)=\sum_o p_\eta(o)\mathcal L_K(\theta,o),\qquad
R(\theta,\eta)=\mathrm{KL}(P_\eta\Vert P_\theta).
\end{gathered}
\tag{B.1}
$$

The proposal does not depend on either correlation parameter: its inputs are the observed groups and the known marginal channel. The importance weight retains the Gaussian determinant and inverse-covariance terms, and its conditional expectation is the full observation probability. Throughout, K is fixed and finite; no bounds are asserted uniformly in K or the spectral boundary.

Because $q_i\leq\phi/Z_i$, all conditional polynomial moments exist. On a compact positive-definite neighborhood, $\log\rho$ and its fixed-order parameter derivatives have Gaussian polynomial envelopes. Derivatives of log-sum-exp are bounded by finite products of these envelopes. The finite alphabet therefore permits differentiation under both integrations and observation sums, to every fixed order used below. The same bounds make each $\mathcal L_K$ continuous and uniformly bounded on the entire candidate domain. All observation probabilities are positive and have a common positive lower bound on that compact domain.

At independence the weights are constant. Conditional Jensen gives

$$
F_K(\theta;0)\geq H(P_0)+R(\theta,0),\qquad
F_K(0;0)=H(P_0).
\tag{B.2}
$$

For $\theta\ne0$, $\rho$ is nonconstant on the proposal's full support, so the average of finitely many independent weights is nondegenerate and Jensen is strict. Thus 0 is the unique global minimizer. Observation identifiability also follows directly from the binary Y component of the specified channel. For each pair, its moment is the Gaussian expectation of the product of two tanh functions; differentiation in the corresponding correlation and Gaussian integration by parts give

$$
\frac{d}{dc}E_c\!\left[\tanh(X_i/2)\tanh(X_j/2)\right]
=\frac14E_c\!\left[\operatorname{sech}^2(X_i/2)\operatorname{sech}^2(X_j/2)\right]>0.
\tag{B.3}
$$

The three pair moments identify the three physical correlations. Consequently, the integrated population negative log likelihood has the teacher as its unique global minimizer. Constant weights at $\theta=0$ do not imply zero gradient for an individual observation or particle bank.

## B.2. Information, shrinkage, and global localization

Let $s$ and $t$ denote the first and second candidate derivatives of $\log\rho$ at 0. For edge coordinates $a=(ij)$, their values, posterior moments, and information matrices are

$$
\begin{gathered}
s=(x_1x_2,x_1x_3,x_2x_3),\qquad
t_{ij,ij}=1-x_i^2-x_j^2,\qquad t_{12,13}=-x_2x_3,\\
\mu_i(o_i)=E_{q_i}X_i,\quad \nu_i(o_i)=E_{q_i}X_i^2,\quad
v_i=E_0\mu_i(O_i)^2,\\
m(o)=E_{q_o}s=(\mu_1\mu_2,\mu_1\mu_3,\mu_2\mu_3),\qquad
V(o)=\operatorname{Cov}_{q_o}(s),\\
J=E_0[mm^\top]=\operatorname{diag}(v_1v_2,v_1v_3,v_2v_3),\qquad
E_{\phi_{I_3}}[ss^\top]=I_3.
\end{gathered}
\tag{B.4}
$$

Cyclic permutations supply the other mixed entries. These are derivatives in physical off-diagonal coordinates, without an extra factor of one half. The posterior identities give $E_0\mu_i=0$, $E_0\nu_i=1$ and $v_i=1-E_0\operatorname{Var}(X_i\mid O_i)<1$. Moreover, the observed Y component has $E[X_iY_i]=2E\sigma'(X_i)>0$, which forces $v_i>0$. Thus J is positive definite.

For one bank, differentiating the negative log of the mean density ratio gives the negative weighted mean of $t+ss^\top$ plus the outer product of the weighted mean score. At 0 the weights are uniform. Averaging the independent particles yields

$$
\begin{gathered}
\nabla\mathcal L_K(0,o)=-m(o),\\
H_K(o):=\nabla^2\mathcal L_K(0,o)
=-E_{q_o}t-(1-K^{-1})V(o)
=H_\infty(o)+K^{-1}V(o),\\
H_\infty(o)=\nabla^2[-\log p_\theta(o)]_{\theta=0},\qquad
E_0V=I_3-J,\\
H_K:=\nabla_\theta^2 F_K(0;0)=J+(I_3-J)/K,\qquad
\nabla_\eta\nabla_\theta F_K(0;0)=-J.
\end{gathered}
\tag{B.5}
$$

Here the mean Hessian follows from the complete-data identity $E_{\phi_I}(t+ss^\top)=0$. The teacher derivative acts on the observation distribution, whose score is $m$; it does not act on $q$. This gives the minus sign in the last identity. Positive definiteness and the implicit function theorem now provide a unique smooth stationary branch near 0, with positive Hessian and

$$
\begin{gathered}
\widetilde\theta_K(\eta)=A_K\eta+O(\|\eta\|^2),\qquad
A_K=H_K^{-1}J,\\
(A_K)_{aa}=r_a=\frac{KJ_{aa}}{1+(K-1)J_{aa}}\in(0,1),\\
v_1=v_2=v_3=v:\qquad j=v^2,\quad J=jI_3,\quad
r=\frac{Kj}{1+(K-1)j}.
\end{gathered}
\tag{B.6}
$$

The branch is also the unique global minimizer for all sufficiently small teachers. To prove this stronger statement, choose a closed ball of radius $\rho$ inside the candidate interior, where the Hessian is uniformly positive for nearby teachers. Define the compact set and gap

$$
A_\rho=\{\theta\in\Theta_\delta:\|\theta\|\geq\rho/2\},\qquad
\gamma=\min_{\theta\in A_\rho}\{F_K(\theta;0)-F_K(0;0)\}>0.
\tag{B.7}
$$

If this set is empty, localization is already established. Otherwise continuity and the unique global anchor give the positive gap. The finite alphabet and bounded losses give uniform convergence of $F_K(\cdot;\eta)$ to $F_K(\cdot;0)$ as $\eta\to0$. Restrict the teacher neighborhood until that uniform difference is below $\gamma/3$. Every point in $A_\rho$ then has objective larger than the origin by more than $\gamma/3$, so every global minimizer lies in the open ball of radius $\rho/2$. Existence follows from compactness; strict convexity on the larger ball gives uniqueness. The resulting interior minimizer is the implicit-function branch. No numerical radius or arbitrary-start optimizer guarantee follows. The same argument applies to integrated likelihood and the matched ridge below, using their unique anchor at 0 and positive mean Hessians.

## B.3. The three-edge term and the ridge distinction

For identical channels, group reflection and permutation symmetries constrain the branch. Reflection of group i changes the signs of exactly the two correlations incident to i. At degree two, the only allowed term in coordinate 12 is therefore $\eta_{13}\eta_{23}$. To compute its coefficient, name the edges $a=12$, $b=13$, $c=23$ and write $R_{ab}$ for the corresponding derivative of $\rho$ at 0. Direct differentiation of the Gaussian density gives

$$
R_a=s_a,\qquad R_{ab}=(x_1^2-1)x_2x_3,\qquad
R_{abc}=\prod_{i=1}^3(x_i^2-1).
\tag{B.8}
$$

Bars denote averages over K particles. The third derivative of the negative logarithm is $-\bar R_{abc}$ plus the three terms $\bar R_{ab}\bar s_c$, minus twice $\bar s_a\bar s_b\bar s_c$. Under the independent teacher, the first four terms have zero expectation: posterior averaging recovers standard normal marginal moments, including $E_0(\nu_i-1)=0$. In the last term, the three particle indices are all equal, exactly two equal, or all distinct. Their conditional-then-observation expectations are respectively 1, $v^2$ and $v^3$. Counting these ordered choices gives

$$
T_K=\frac{1+3(K-1)v^2+(K-1)(K-2)v^3}{K^2},\qquad
F_{\theta_a\theta_b\theta_c}(0;0)=-2T_K.
\tag{B.9}
$$

The mixed teacher derivative is equally explicit. Formula (B.5) gives $H_{K,ab}(o)=\mu_2\mu_3[1-(1-K^{-1})(\nu_1-\mu_1^2)]$. Multiplication by the teacher score $m_c$ and averaging independent groups yields the following $B_K$. The pure second teacher derivative of the origin score in coordinate $a$ is zero: $E_\eta\mu_1\mu_2$ depends only on the teacher's 12 marginal, not on $\eta_{13}$ or $\eta_{23}$.

$$
B_K:=F_{\theta_a\theta_b\eta_c}(0;0)
=\frac{v^2[1+(K-1)v]}K,\qquad
F_{\theta_a\eta_b\eta_c}(0;0)=0,\qquad
H=\frac{1+(K-1)v^2}K.
\tag{B.10}
$$

Insert the branch expansion into its score equation. The coefficient of $\eta_b\eta_c$ is $H\chi_K-2T_Kr^2+2B_Kr$; hence

$$
\begin{gathered}
\widetilde\theta_{12}=r\eta_{12}+\chi_K\eta_{13}\eta_{23}+O(\|\eta\|^3),\\
\chi_K=\frac{2r(T_Kr-B_K)}H
=-\frac{2K(K-1)v^5(1-v)^2}{[1+(K-1)v^2]^3},\\
\eta=(t,t,0):\qquad
\widetilde\theta_{12}=\widetilde\theta_{13}=rt+O(t^3),\qquad
\widetilde\theta_{23}=\chi_Kt^2+O(t^4).
\end{gathered}
\tag{B.11}
$$

The last remainder uses reflection of group 1: the first two branch coordinates are odd in t and the third is even. The induced third coordinate is a marginal correlation, not a conditional-independence graph edge. For K>1 it is negative to this order; for K=1 the coefficient vanishes.

Compare mean integrated NLL plus a deterministic quadratic penalty:

$$
\begin{gathered}
\mathcal L_{\rm ridge}(\theta,o)=-\log p_\theta(o)+\tfrac12\theta^\top\Lambda_K\theta,
\qquad \Lambda_K=(I_3-J)/K,\\
\nabla\mathcal L_{\rm ridge}(0,o)=-m(o),\qquad
\nabla^2\mathcal L_{\rm ridge}(0,o)=H_\infty(o)+\Lambda_K,\\
\chi_{\rm ridge}=\frac{2v^3r(r-1)}H
=-\frac{2K^2v^5(1-v^2)}{[1+(K-1)v^2]^3}.
\end{gathered}
\tag{B.12}
$$

Its mean Hessian and first-order branch match, but its per-observation Hessian generally does not. For the coefficient, integrated likelihood has the limits $T_\infty=B_\infty=v^3$, while the quadratic penalty contributes no third derivative; substitution in the preceding score equation gives (B.12). Thus the objectives differ beyond their shared first-order response, including at K=1. A sum-NLL convention requires multiplying the penalty by N. The norm is in the three physical coordinates; the squared Frobenius norm of $C-I$ is twice as large. [Appendix C](#risk-proof) derives the shared first-order sampling risk without identifying the two objectives.
