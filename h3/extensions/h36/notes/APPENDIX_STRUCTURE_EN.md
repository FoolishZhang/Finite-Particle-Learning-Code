<a id="structure-proof"></a>

# A. Input access, joint outputs, and residual posterior change

## A.1. The independent anchor and its correlation scores

Fix a finite number of groups $d\ge2$ and an edge set $E\subseteq\{ij:1\le i<j\le d\}$, with $m=|E|$. Neither dimension grows in the local limit. The candidate parameter consists of the physical Gaussian correlations on these edges:

$$
\begin{gathered}
C_\theta=I_d+\Theta,\qquad
\Theta_{ii}=0,\qquad
\Theta_{ij}=\Theta_{ji}=\theta_{ij}\ (ij\in E),\qquad
\Theta_{ij}=0\ (ij\notin E),\\
p_\theta(dx,do)=\varphi_{C_\theta}(x)\,dx\prod_{i=1}^dQ_i(do_i\mid x_i).
\end{gathered}
\tag{A.1}
$$

The candidate neighborhood keeps $C_\theta$ positive definite. Each $Q_i$ is a fixed probability kernel, independent of the correlation parameters. The kernels may differ across groups and may be nonlinear, discrete, or continuous. Let $P_\eta$ denote the observation law under teacher correlation $\eta$ in this same model.

At $\theta=0$, the pairs $(X_i,O_i)$ are independent and the posterior factorizes as $q_0(dx\mid o)=\pi_0(dx\mid o)=\prod_iq_{0i}(dx_i\mid o_i)$. This is an exact conditional-distribution identity for almost every observation. Throughout this appendix, unqualified expectations and inner products use the joint anchor law $\Lambda_0(dx,do)=P_0(do)q_0(dx\mid o)$. Define

$$
\begin{gathered}
\mu_i=E[X_i\mid O_i],\qquad e_i=X_i-\mu_i,\qquad
v_i=E\mu_i^2,\qquad w_i=Ee_i^2=1-v_i,\\
s_{ij}=X_iX_j,\qquad u_{ij}=E[s_{ij}\mid O]=\mu_i\mu_j,\\
r_{ij}=s_{ij}-u_{ij}=\mu_je_i+\mu_ie_j+e_ie_j.
\end{gathered}
\tag{A.2}
$$

Here $s$, $u$, and $r$ are respectively the complete-data, observation, and posterior scores at the anchor. The complete score follows by differentiating the Gaussian density in the two symmetric covariance positions. The observation score follows by conditional expectation. Thus the decomposition uses the Gaussian correlation path and the independent, groupwise observation channel; it does not require a Gaussian conditional posterior or constant conditional variance.

Conditional centering gives $E[e_i\mid O_i]=0$, $E\mu_i=Ee_i=0$, and $E[e_i\mu_i]=0$. The three terms of $r_{ij}$ are orthogonal in $L^2(\Lambda_0)$, with squared norms $w_iv_j$, $v_iw_j$, and $w_iw_j$. Terms from different unordered edges are also orthogonal: an unmatched group contributes a single centered mean or residual. Consequently

$$
J=E[uu^\top]=\operatorname{diag}_{ij\in E}(v_iv_j),\qquad
E[ss^\top]=I_m,\qquad C_r=E[rr^\top]=I_m-J.
\tag{A.3}
$$

These are population orthogonality identities. The corresponding terms need not be orthogonal at each fixed observation. Values $v_i=0$ or $1$ are allowed in the projection results. Response formulas involving ordinary inverses below require $J\succ0$ on the parameter submodel being considered.

## A.2. Which posterior changes a block can represent

Let $\mathcal P$ be a partition of the groups. Each block $B\in\mathcal P$ has a specified observation-access set $\mathcal C_B\supseteq B$. The proposal density has the form

$$
\begin{gathered}
q(x\mid o)=\prod_{B\in\mathcal P}q_B(x_B\mid o_{\mathcal C_B}),\\
\mathcal H=\{h\in L^2(\Lambda_0):E[h\mid O]=0\},\\
\mathcal H_B=\{h_B(X_B,O_{\mathcal C_B})\in L^2(\Lambda_0):
E[h_B\mid O_{\mathcal C_B}]=0\},\qquad
\mathcal T_{\mathcal P,\mathcal C}=\bigoplus_{B\in\mathcal P}\mathcal H_B.
\end{gathered}
\tag{A.4}
$$

Every block density shares its declared inputs across its joint output. Thus this model specifies both the output partition and the inputs available to each block. Scores are changes of normalized conditional densities, which explains the conditional centering in $\mathcal H_B$. At the anchor, the conditional law of $X_B$ given all observations depends only on $O_B$; hence centering given $O_{\mathcal C_B}$ also gives centering given $O$. Each $\mathcal H_B$ is closed, being the kernel of conditional expectation on the relevant measurable $L^2$ subspace. Different block spaces are orthogonal conditional on $O$, so their finite direct sum is closed. The space is saturated: it allows every square-integrable score with the declared measurability and centering.

**Proposition A.1 (input and output projection).** Let $B(i)$ be the block containing $i$, and project each coordinate of $r$ onto $\mathcal T_{\mathcal P,\mathcal C}$. A within-block edge has zero residual. A cross-block edge satisfies

$$
\begin{aligned}
\Pi_{\mathcal T_{\mathcal P,\mathcal C}}r_{ij}
&=\mathbf1_{\{j\in\mathcal C_{B(i)}\}}\mu_je_i
 +\mathbf1_{\{i\in\mathcal C_{B(j)}\}}\mu_ie_j,\\
r_{ij}-\Pi_{\mathcal T_{\mathcal P,\mathcal C}}r_{ij}
&=e_ie_j+\mathbf1_{\{j\notin\mathcal C_{B(i)}\}}\mu_je_i
 +\mathbf1_{\{i\notin\mathcal C_{B(j)}\}}\mu_ie_j,
\qquad B(i)\ne B(j).
\end{aligned}
\tag{A.5}
$$

The residual matrix is diagonal, with

$$
\begin{gathered}
R_{\mathcal P,\mathcal C}
=E[(r-\Pi_{\mathcal T_{\mathcal P,\mathcal C}}r)
    (r-\Pi_{\mathcal T_{\mathcal P,\mathcal C}}r)^\top],\\
(R_{\mathcal P,\mathcal C})_{ij,ij}=
\begin{cases}
0,&B(i)=B(j),\\
w_iw_j+\mathbf1_{\{j\notin\mathcal C_{B(i)}\}}w_iv_j
+\mathbf1_{\{i\notin\mathcal C_{B(j)}\}}v_iw_j,
&B(i)\ne B(j).
\end{cases}
\end{gathered}
\tag{A.6}
$$

**Proof.** For $f\in\mathcal H$, its projection onto $\mathcal H_B$ is $E[f\mid X_B,O_{\mathcal C_B}]-E[f\mid O_{\mathcal C_B}]$. The second term is zero because $E[f\mid O]=0$. If $i,j\in B$, then $r_{ij}$ is measurable in $(X_B,O_B)$ and has conditional mean zero given $O_B$, so it belongs to $\mathcal H_B$. For a cross-block edge, condition on the block containing $i$. When $O_j$ is available, integration over $X_j$ leaves $\mu_je_i$; when it is unavailable, integration over the entire independent group $j$ gives zero. The block containing $j$ gives the other term. All other blocks give zero even when they read $O_i,O_j$, because integrating both latent variables leaves $E[r_{ij}\mid O]=0$. The cross-block product $e_ie_j$ is orthogonal to every single-block score by conditional independence and centering. This proves the projection. The component orthogonality established in A.1 gives its covariance and all zero off-diagonal entries. $\square$

For singleton outputs write $\mathcal A_i=\mathcal C_{\{i\}}$, with $i\in\mathcal A_i$. Proposition A.1 gives arbitrary directed access, as well as its two endpoints:

$$
\begin{aligned}
(R_{\mathcal A})_{ij,ij}
&=w_iw_j+\mathbf1_{\{j\notin\mathcal A_i\}}w_iv_j
 +\mathbf1_{\{i\notin\mathcal A_j\}}v_iw_j,\\
R_{\rm local}&=I_m-J,
\qquad \Pi_{\mathcal T_{\rm local}}r_{ij}=0,\\
R_{\rm product}&=\operatorname{diag}_{ij}(w_iw_j),
\qquad \Pi_{\mathcal T_{\rm product}}r_{ij}=\mu_je_i+\mu_ie_j.
\end{aligned}
\tag{A.7}
$$

Adding the input connection $j\to i$ removes $\mu_je_i$ and reduces only the $ij$ residual by $w_iv_j$. It has no second-order benefit when that quantity is zero. All-input block outputs leave zero within-block residual and $w_iw_j$ across blocks. A single all-input joint block leaves zero residual on every edge. Blocks that read only their own observations instead leave $1-v_iv_j$ across blocks. Merging such blocks changes their available information as well as their joint outputs; these two changes must be kept distinct when interpreting a comparison.

## A.3. From projection to a regular local learning response

The preceding identities are statements about score spaces. We next give conditions under which they describe the curvature of an actual finite-particle profile. Fix a finite integer $K\ge1$. Start with a finite-dimensional proposal $q_\beta(x\mid o)$, whose coordinates $\beta$ are independent of the candidate coordinates $\theta$, and whose density at $\beta=0$ is $q_0=\pi_0$. Define the population negative importance-weighted objective

$$
F_K(\theta,\beta;\eta)=
E_{P_\eta}E_{q_\beta(\cdot\mid O)^{\otimes K}}
\left[-\log\left\{\frac1K\sum_{k=1}^K
\frac{p_\theta(X_k,O)}{q_\beta(X_k\mid O)}\right\}\right].
\tag{A.8}
$$

The same conditional mapping is used across observations. Its fitted coefficients cannot be chosen separately for each complete observation while still being called a local-input mapping.

The following regularity requirements apply to the profile result:

1. The anchor is an interior parameter point. Candidate and proposal densities have a common support, and the proposals cover the corresponding posteriors. Conditional particles are iid.
2. The normalized densities, sampling expectations, and observation expectations can be differentiated twice near the anchor, with continuous Hessians and integrable second-order Taylor remainders uniform as the joint candidate, proposal, and teacher increments tend to zero. In particular, differentiating an expectation includes its changing sampling distribution. A sufficient envelope condition bounds the required log-objective and joint sampling-density derivatives through order three, together with products involving inverse importance-weight means, by common integrable functions.
3. The teacher observation law is twice differentiable with anchor score $u$. The quadratic gap term can be evaluated under $P_0$ with a remainder of smaller joint order.
4. After removing redundant regular coordinates, the proposal-score Gram matrix is positive definite. This provides a regular local nuisance minimum. A claim about the complete density infimum additionally requires localization of that infimum to this branch.

**Theorem A.2 (regular local profile).** Under these conditions let $a=\partial_\beta\log q_\beta|_0$, $G=E[aa^\top]\succ0$, $M=E[ra^\top]$, and $\mathcal T_Q=\operatorname{span}\{a_\ell\}$. Then

$$
\begin{aligned}
F_K(\theta,\beta;\eta)-F_K(0,0;\eta)
={}&-\theta^\top J\eta+\tfrac12\theta^\top J\theta\\
&+\frac1{2K}E[(\theta^\top r-\beta^\top a)^2]
+o(\|(\theta,\beta,\eta)\|^2).
\end{aligned}
\tag{A.9}
$$

The local proposal minimum and its profile Hessian satisfy

$$
\begin{gathered}
\beta_*(\theta,\eta)=G^{-1}M^\top\theta+o(\|(\theta,\eta)\|),\\
R_Q=C_r-MG^{-1}M^\top
=E[(r-\Pi_{\mathcal T_Q}r)(r-\Pi_{\mathcal T_Q}r)^\top],\\
H_Q=J+R_Q/K,\qquad
F_{\theta\eta}^{\rm prof}(0;0)=-J.
\end{gathered}
\tag{A.10}
$$

If $J\succ0$, the resulting joint local minimum has generating-parameter response

$$
\widetilde\theta_Q(\eta)=A_Q\eta+o(\|\eta\|),\qquad
A_Q=(J+R_Q/K)^{-1}J.
\tag{A.11}
$$

For a proposal without free coordinates, omit the $G,M,\beta$ blocks.

**Proof.** At a fixed observation factor the importance ratio into $p_\theta(O)U$, where $U=\pi_\theta/q_\beta$ and $E_{q_\beta}[U\mid O]=1$ exactly. Its first variation is $\theta^\top r-\beta^\top a$, whose conditional mean at the anchor is zero. In the expansion of the negative log of the average of $K$ such ratios, exact normalization cancels the expected linear term. Conditional iid sampling divides the variance of the first variation by $K$. The quadratic gap is therefore the last term of (A.9). The observation negative log likelihood contributes $-\theta^\top J\eta+\theta^\top J\theta/2$. Since the gap is already quadratic, changing its observation law from $P_0$ to $P_\eta$ affects only the remainder under the stated envelopes. This proves (A.9), including the sampling-law derivative.

Its joint Hessian has blocks $J+C_r/K$, $-M/K$, and $G/K$. The implicit function theorem applies to the proposal gradient because $G/K\succ0$, and yields $\beta_*$. Substitution, or the Schur complement, gives (A.10). Orthogonal projection onto the proposal scores gives the stated residual identity. If $J\succ0$, then $H_Q\succ0$; equivalently, the full quadratic form is $z^\top Jz+K^{-1}E[(z^\top r-c^\top a)^2]$, positive for nonzero $(z,c)$. A second application of the implicit function theorem yields the joint local minimum and (A.11). $\square$

For a regular realization of the saturated structure in Proposition A.1, $R_Q$ is diagonal and the edge response is

$$
(A_Q)_{ij,ij}=\frac{Kv_iv_j}{Kv_iv_j+(R_Q)_{ij,ij}},\qquad
(A_Q)_{ij,k\ell}=0\quad(ij\ne k\ell).
\tag{A.12}
$$

This yields the local-input response $Kv_iv_j/[1+(K-1)v_iv_j]$ and the all-input product response $Kv_iv_j/(Kv_iv_j+w_iw_j)$. The coefficient $1/K$ is exact at second order for each fixed finite $K$; no large-$K$ limit is taken. The finite-amplitude gap is generally not the residual variance divided by $2K$.

An infinite-dimensional closed score space defines the quadratic projection in (A.10), but that identity alone does not establish a second-order expansion of an infimum over all densities. The theorem applies to a regular finite-dimensional realization, or to a profile whose second-order expansion and localization have been established separately. The special compact-domain global continuation for the three-group Gaussian--logistic tilt classes is proved in Appendices B--C. Its global identification, fixed parameter boxes, and unique anchor are additional assumptions beyond this theorem.

## A.4. Finite-dimensional realization of the saturated coefficients

The saturated residuals need only finitely many tangent features. For each block and edge define $t_{B,ij}=\Pi_{\mathcal H_B}r_{ij}$ and consider

$$
q_{B,\beta}(x_B\mid o_{\mathcal C_B})=
q_{0B}(x_B\mid o_B)
\exp\left\{\sum_{ij\in E}\beta_{B,ij}t_{B,ij}
-\psi_B(\beta_B,o_{\mathcal C_B})\right\}.
\tag{A.13}
$$

Assume the local normalizers and the derivatives required in A.3 are integrable. Since every feature is conditionally centered, its anchor score is exactly $t_{B,ij}$. Delete zero features and regular linear redundancies. The remaining finite score span contains every saturated projection of $r_{ij}$ and is itself contained in $\mathcal T_{\mathcal P,\mathcal C}$, so it has exactly the residual (A.6). Each within-block edge supplies one feature and each cross-block edge at most two; at most $2m$ nonzero features suffice. When $0<v_i<1$ on all groups, the retained features are mutually orthogonal and nonzero, so their Gram is positive definite. Degenerate cases require deleting the zero features explicitly.

There is a particularly direct realization for $O_i=X_i+\epsilon_i$ with independent $\epsilon_i\sim N(0,\tau_i)$ and $\tau_i>0$. Here $v_i=(1+\tau_i)^{-1}$ and the posterior variance is the constant $w_i$. A block-Gaussian path can use

$$
\begin{gathered}
m_i(\theta,O)=\mu_i+w_i
\sum_{j:\,ij\in E,\ j\in\mathcal C_{B(i)}}\theta_{ij}\mu_j,
\qquad \theta_{ji}=\theta_{ij},\\
V_B(\theta)=W_B+W_B\Theta_{BB}W_B,\qquad
W_B=\operatorname{diag}_{i\in B}(w_i).
\end{gathered}
\tag{A.14}
$$

For small enough $\theta$, $V_B\succ0$. The mean score supplies each accessible $\mu_je_i$, and the covariance score supplies each within-block $e_ie_j$. Their sum is precisely the projection in Proposition A.1. The path specifies a first-order tangent, not a finite-amplitude optimal proposal. Using independent feature coordinates realizes a regular nuisance family as required by Theorem A.2.

For these Gaussian proposals, covariance matrices can be kept uniformly positive definite on a compact neighborhood. Reparameterizing particles by parameter-independent standard normal variables makes log weights and their fixed-order derivatives polynomials with locally bounded coefficients. The log mean weight is bounded in absolute value by the largest absolute log weight, and its derivatives are controlled by polynomial combinations of log-weight derivatives with bounded softmax coefficients. Gaussian moments then supply common integrable envelopes for fixed $K$. General groupwise channels require their own normalizer and envelope check for (A.13); square-integrable scores alone do not establish that check.

## A.5. Restricted families: matrix order and coordinate mixing

**Proposition A.3 (nested spaces).** Suppose $\mathcal T_Q\subseteq\mathcal T_{\rm rich}$ are closed subspaces of $\mathcal H$. Their residuals obey $R_Q\succeq R_{\rm rich}$. When both have the regular local-profile interpretation and $J\succ0$, define the Fisher-whitened response $W_Q=J^{1/2}A_QJ^{-1/2}$. Then

$$
\begin{gathered}
z^\top R_Qz=\inf_{h\in\mathcal T_Q}\|z^\top r-h\|_2^2
\ \ge\ \inf_{h\in\mathcal T_{\rm rich}}\|z^\top r-h\|_2^2
=z^\top R_{\rm rich}z,\\
0\prec W_Q=J^{1/2}(J+R_Q/K)^{-1}J^{1/2}
\preceq W_{\rm rich}\preceq I_m.
\end{gathered}
\tag{A.15}
$$

**Proof.** The distance to a smaller closed space is at least the distance to a larger one. This proves the residual order for every $z$. Adding $J$, inverting positive definite matrices, and applying the congruence by $J^{1/2}$ proves the response order. $\square$

In particular, any restricted all-input product family has $R_Q\succeq\operatorname{diag}(w_iw_j)$. Its whitened response is bounded by the diagonal all-input product response in (A.12). The unwhitened $A_Q$ need not be symmetric, so a Loewner order should not be applied directly to it. Every local-input score subspace is orthogonal to all $r_{ij}$ by (A.7), and thus has the same residual $I_m-J$ whenever the regular local-profile interpretation applies.

The matrix order does not imply a coordinatewise bound for every multi-edge teacher. Consider three linear Gaussian channels with $v_i=w_i=1/2$, retaining candidate edges $12$ and $13$. Give factor 1 the sole free feature $a=e_1(\mu_2+\mu_3)$ through a normalized exponential tilt; fix the other factors at their exact anchor posteriors. This is a regular Gaussian mean-tilt family. Direct calculation gives

$$
\begin{gathered}
G=\tfrac12,\qquad M=(\tfrac14,\tfrac14)^\top,\qquad J=\tfrac14I_2,\\
R_Q=\begin{pmatrix}5/8&-1/8\\-1/8&5/8\end{pmatrix},\qquad
R_{\rm product}=\tfrac14I_2,\\
K=1:\qquad
A_Q=\tfrac1{24}\begin{pmatrix}7&1\\1&7\end{pmatrix},\qquad
A_{\rm product}=\tfrac12I_2.
\end{gathered}
\tag{A.16}
$$

The eigenvalues of $R_Q-R_{\rm product}$ are $1/4$ and $1/2$, so the matrix bound holds. But along $\eta=t(1,6)^\top$, the first response coefficient is $13/24>1/2$, exceeding the first coordinate of the saturated product response. Taking $t$ sufficiently small preserves positive definiteness of the generating covariance. This is a counterexample about leading local coefficients, illustrating mixing between edges in a realizable restricted family.

## A.6. Conditions that determine the scope of the result

Regular coordinates matter. For finite score matrices a pseudoinverse computes a linear projection, but it does not repair a singular density parameterization. For example, $q_\phi=\pi_{\phi^3}$ has zero parameter score at zero, while $\phi=\sqrt[3]{\theta}$ can match $\pi_\theta$ exactly. Inferring a complete density profile from that zero score would be incorrect. One must use regular quotient coordinates or establish the profile expansion directly.

If the proposal explicitly depends on the candidate, write $q_{\theta,\beta}$ and $b=\partial_\theta\log q_{\theta,\beta}|_0$. The quadratic gap in (A.9) then contains $\theta^\top(r-b)-\beta^\top a$. A $b$ lying in the same nuisance score span is absorbed by adaptation. More generally, a candidate-dependent path respecting (A.4) still has its total score in the corresponding saturated space, so Proposition A.1 continues to bound its residual.

The Gaussian correlation score is a substantive restriction. Independent, centered, unit-variance latent variables alone do not determine the score of a dependence path. The projection algebra also applies to a non-Gaussian path if its actual complete score is proved to be $X_iX_j$; for example a regular local interaction exponential family can have this score. Its natural interaction parameter need not equal finite-amplitude Pearson correlation. Other scores require rebuilding the decomposition from the actual score statistics.

Finally, a full joint output containing all posterior score directions has $R=0$ and unit local response under Theorem A.2. Equality of the entire profiled objective with observation negative log likelihood follows by Jensen only when the full posterior belongs to the proposal family at every nearby candidate. An approximate anchor, cross-group observation channels, dependent particles, overlapping or autoregressive outputs, and a growing number of groups require separate analysis. When $J$ is singular, the projection identities remain valid, but the inverse and whitening formulas require restriction to an identified submodel that does not mix with unidentified nuisance directions.
