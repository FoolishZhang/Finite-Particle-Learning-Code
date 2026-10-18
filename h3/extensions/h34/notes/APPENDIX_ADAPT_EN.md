<a id="adapt-proof"></a>

# B. What adaptation removes from the posterior change

## B.1. Scores and the complete quadratic problem

Fix a known channel as in Appendix A and a finite integer K. Expectations in this section, unless conditioned otherwise, use the matching independent joint law $P_0(o)q_0(x\mid o)$. Put $\mu_i=E[X_i\mid O_i]$, $d_i=\operatorname{Var}(X_i\mid O_i)$, $v_i=E\mu_i^2$, and $w_i=1-v_i=Ed_i$. The complete Gaussian score, observed score, and posterior residual in an edge coordinate are

$$
\begin{gathered}
s_{ij}=X_iX_j,\qquad m_{ij}=E[s_{ij}\mid O]=\mu_i\mu_j,\\
e_i=X_i-\mu_i,\qquad
r_{ij}=s_{ij}-m_{ij}=\mu_je_i+\mu_ie_j+e_ie_j,\\
J=E[mm^\top]=\operatorname{diag}_{i<j}(v_iv_j),\qquad
C_r=E[rr^\top]=\operatorname{diag}_{i<j}(1-v_iv_j).
\end{gathered}
\tag{B.1}
$$

The diagonal identities use independent groups, conditional centering, and $EX_i^2=1$. They do not assume Gaussian conditional residuals or a constant posterior variance. All coordinates refer to physical correlations in the order $(12,13,23)$.

Let $A_Q$ be the proposal score vector at zero tilt; it has no coordinates for F. For L it contains $A_{i,s}=1_{\{O_i=s\}}e_i$. A additionally contains $D_{i\leftarrow j}=\mu_je_i$, $i\ne j$. Direct conditioning gives

$$
\begin{gathered}
G_Q=E[A_QA_Q^\top],\qquad M_Q=E[rA_Q^\top],\\
G_{(i,s),(i,s)}=p_i(s)d_i(s)>0,\qquad E[r_{ij}A_{k,s}]=0,\\
E[D_{i\leftarrow j}^2]=w_iv_j>0,\qquad
E[r_{k\ell}D_{i\leftarrow j}]
=1_{\{\{k,\ell\}=\{i,j\}\}}w_iv_j.
\end{gathered}
\tag{B.2}
$$

The local Gram is diagonal. All directed scores are mutually orthogonal and orthogonal to the local block: an unmatched group leaves a centered mean or residual. Thus both full proposal Grams are positive definite, including when one count slope is zero. No pseudoinverse or dimension reduction is needed.

For a fixed observation, write the normalized posterior-to-proposal ratio as $1+\Delta=\pi_\theta/q_u$. Its conditional mean under $q_u$ is exactly one. At the anchor, its first variation in increments $(z,c)$ is $z^\top r-c^\top A_Q$. Expanding the log of the mean of K iid ratios, the linear term vanishes and the mean-square term has the factor $1/K$. Consequently the conditional Jensen gap has quadratic term

$$
\mathcal L_{K,Q}(\theta,u;o)+\log p_\theta(o)
=\frac1{2K}E_{q_0}\!\left[(\theta^\top r-u^\top A_Q)^2\mid o\right]
+O((\|\theta\|+\|u\|)^3).
\tag{B.3}
$$

This is a derivative expansion under the integrable envelopes of Lemma A.1, not a uniform-in-x bound on the density ratio. Changing the sampling density is part of the expansion. One can check the generating block directly: if t is the Hessian of the Gaussian log density ratio at zero, then $t_{ij,ij}=1-X_i^2-X_j^2$ and $t_{12,13}=-X_2X_3$, with cyclic versions. A fixed bank has Hessian $-\overline{t+ss^\top}+\bar s\bar s^\top$. Averaging its iid particles gives $-E[t\mid o]-(1-K^{-1})\operatorname{Cov}(s\mid o)$. For integrated likelihood the conditional Hessian is $H_\infty(o)=-E[t\mid o]-\operatorname{Cov}(s\mid o)$, with mean J; its origin gradient is $-m(o)$. This is the observed-information decomposition [@louis1982]. Teacher differentiation acts only on observation probabilities, whose score is m. These facts give the full result:

$$
\begin{gathered}
\nabla_{(\theta,u)}\mathcal L_{K,Q}(0,0;o)=(-m(o),0),\\
H_Q^{\rm full}=\nabla^2_{(\theta,u)}F_{K,Q}(0,0;0)
=\begin{pmatrix}J+C_r/K&-M_Q/K\\-M_Q^\top/K&G_Q/K\end{pmatrix},\qquad
F_{\theta\eta}=-J,\quad F_{u\eta}=0,\\
(z,c)^\top H_Q^{\rm full}(z,c)
=z^\top Jz+K^{-1}E[(z^\top r-c^\top A_Q)^2]>0\quad\text{if }(z,c)\ne0.
\end{gathered}
\tag{B.4}
$$

The last line follows from $J\succ0$ and $G_Q\succ0$ and proves positive definiteness in the full 3, 57, or 63 dimensional space. For F the proposal blocks are absent. This supplies the known-channel Hessian condition used in Proposition A.5.

## B.2. Profiling leaves an interaction residual

By Proposition A.5, the following profile is the minimum over the entire specified proposal box for nearby candidates and teachers. Solving the linear proposal equation and substituting it into the generating equation gives

$$
\begin{gathered}
S_Q=J+K^{-1}(C_r-M_QG_Q^{-1}M_Q^\top),\\
\theta_Q^*(\eta)=S_Q^{-1}J\eta+O(\|\eta\|^2),\qquad
u_Q^*(\eta)=G_Q^{-1}M_Q^\top\theta_Q^*(\eta)+O(\|\eta\|^2).
\end{gathered}
\tag{B.5}
$$

The matrix subtracted from $C_r$ is the covariance of the orthogonal projection of r onto the span of proposal scores. The remaining covariance is nonnegative. Its size describes which posterior changes this class can follow; it is not a count of proposal parameters.

For L, (B.2) gives $M_L=0$. Hence full local-state adaptation cannot remove any of this leading residual, and F and L have the same response:

$$
\begin{gathered}
S_F=S_L=\operatorname{diag}_{i<j}\left[v_iv_j+\frac{1-v_iv_j}{K}\right],\\
r_{F,ij}=r_{L,ij}=\frac{Kv_iv_j}{1+(K-1)v_iv_j},\qquad
\theta_{Q,ij}^*(\eta)=r_{Q,ij}\eta_{ij}+O(\|\eta\|^2),\qquad
b_L^*(\eta)=O(\|\eta\|^2).
\end{gathered}
\tag{B.6}
$$

A represents precisely the two additive terms $\mu_je_i+\mu_ie_j$ in (B.1). The remaining edge residual is $e_ie_j$, with variance $w_iw_j$, and different edges remain orthogonal. Thus

$$
\begin{gathered}
S_A=\operatorname{diag}_{i<j}\left[v_iv_j+\frac{w_iw_j}{K}\right],\qquad
r_{A,ij}=\frac{Kv_iv_j}{Kv_iv_j+w_iw_j},\qquad
0<r_{L,ij}<r_{A,ij}<1,\\
b_A^*(\eta)=O(\|\eta\|^2),\qquad
a_{i\leftarrow j}^*(\eta)=\theta_{A,ij}^*(\eta)+O(\|\eta\|^2).
\end{gathered}
\tag{B.7}
$$

For a fixed nearby candidate, replace the branch value of theta in the final line by that candidate and the remainder by $O((\|\theta\|+\|\eta\|)^2)$. This explains a first-order proposal initialization without identifying it with the finite-amplitude profile optimum. All inequalities hold separately for each edge in a heterogeneous known channel. In particular, averaging the $v_i$ does not recover the heterogeneous response. For a homogeneous channel, $J=jI_3$, $j=v^2$, and $S_Q=s_QI_3$: $s_F=s_L=j+(1-j)/K$, $s_A=j+(1-v)^2/K$, and $r_Q=j/s_Q$.

The local distinction also appears in objective values. For $\theta,\eta=O(t)$, the L profile improves on F only by $O(t^4)$, because its optimal displacement is $O(t^2)$ and its proposal gradient is zero at the anchor. A has the leading profile improvement

$$
F_{K,A}^{\rm prof}(\theta;\eta)-F_{K,F}(\theta;\eta)
=-\frac1{2K}\sum_{i<j}(w_iv_j+w_jv_i)\theta_{ij}^2+O(t^3).
\tag{B.8}
$$

This holds with the same generating candidate. Comparing values at different joint optima answers a different question. The exact posterior oracle instead has zero Jensen gap and response one. Equations (B.6)--(B.7) apply to the specified finite-dimensional classes; they do not order all factorized encoders, all local-input densities, or arbitrary learned features.

**Feature approximation boundary.** Suppose the directed input is a fixed table $m_j(O_j)$ rather than the exact mean. Center it under the true independent observation distribution, let $D_j=\operatorname{Var}(m_j)>0$ and $c_j=E[\mu_j(m_j-Em_j)]$. A local coefficient can absorb its constant component. The tangent projection then gives

$$
R_{ij}^{m}=1-v_iv_j-w_i\frac{c_j^2}{D_j}-w_j\frac{c_i^2}{D_i}
\ge w_iw_j>0,\qquad
S_{A(m),ij}=v_iv_j+R_{ij}^{m}/K.
\tag{B.9}
$$

Cauchy--Schwarz proves the inequality. A nonconstant input suffices for parameter identification, but can have $c_j=0$ and give no projection gain in that direction. For example, a nonconstant reflection-even table is orthogonal to the reflection-odd exact mean. The exact result is recovered when its centered table is a nonzero multiple of the exact posterior mean. The local reparameterization need not preserve the global box. A finite-quadrature mean table cannot be declared exact solely because two quadrature orders agree. Posterior normalization, feature approximation, and optimization accuracy are separate numerical questions.

## B.3. The induced third correlation in a homogeneous channel

Now assume identical reflection-symmetric channels and exact mean features, and put $w=1-v$. The needed finite-order smoothness follows from Lemma A.1. For a candidate direction z, write $\theta=tz$ and, at a fixed observation,

$$
\begin{gathered}
\pi_{tz}/q_0=1+tp+\tfrac12t^2p_2+O(t^3),\qquad
q_t/q_0=1+tL+\tfrac12t^2(L^2-D)+O(t^3),\\
L=\sum_i l_i e_i,\quad D=E[L^2\mid O],\quad r_z=p-L,\\
S_z=\sum_{i<j}z_{ij}X_iX_j,\quad m_z=E[S_z\mid O],\quad
p=S_z-m_z,\quad p_2=R_{zz}-E[R_{zz}\mid O]-2m_zp.
\end{gathered}
\tag{B.10}
$$

Here $R_{zz}$ is the second derivative of the Gaussian prior density ratio in direction z. Conditional means of p, $p_2$, and L vanish. For $\Delta=\pi_{tz}/q_t-1$, expanding both the ratio and the sampling density in $E_{q_t}\Delta^2/(2K)-E_{q_t}\Delta^3/(3K^2)$ yields

$$
\mathcal L_{K,Q}+\log p_{tz}(O)
=\frac{t^2}{2K}E[r_z^2\mid O]
+t^3\left\{\frac{E[r_z(p_2-pL)\mid O]}{2K}
-\frac{E[r_z^3\mid O]}{3K^2}\right\}+O(t^4).
\tag{B.11}
$$

Omitting the changing sampling density gives the wrong cubic coefficient. The second-order correction to the optimal proposal is not needed here: it multiplies the zero proposal derivative of the quadratic criterion at the first-order optimum.

For A, this optimum has $l_i=\sum_{j\ne i}z_{ij}\mu_j$, so $L=\sum_{i<j}z_{ij}(\mu_je_i+\mu_ie_j)$ and $r_z=\sum_{i<j}z_{ij}e_ie_j$. Reflection symmetry permits only the candidate triangle $P_z=z_{12}z_{13}z_{23}$ at cubic order. Its coefficients in the unconditional moments of (B.11) are

$$
\begin{array}{c|rrrr}
\text{moment}&E[r_zR_{zz}]&E[m_zr_zp]&E[r_zpL]&E[r_z^3]\\\hline
[P_z]&0&0&6vw^2&6w^3
\end{array}
\tag{B.12}
$$

For example, the mixed Gaussian derivative $R_{12,13}=(X_1^2-1)X_2X_3$ contributes zero after the first-group average. Conditional orthogonality gives $E[r_zp\mid O]=E[r_z^2\mid O]$, so the second column vanishes. In $r_{23}L_{12}L_{13}$ the only surviving term is $e_2e_3(\mu_1e_2)(\mu_1e_3)$, of mean $vw^2$; the three residual edges and two orders give six terms. The last column consists of the six permutations of $e_1^2e_2^2e_3^2$. Thus the Jensen-gap cubic term is $-(3vw^2/K+2w^3/K^2)P_\theta$.

For completeness, the integrated NLL contributes $-2v^3P_\theta$ and the mixed polynomial $v^3(\theta_{12}\theta_{13}\eta_{23}+\theta_{12}\theta_{23}\eta_{13}+\theta_{13}\theta_{23}\eta_{12})$. The A gap has no cubic teacher-mixed contribution: its conditional quadratic residual is $\sum z_{ij}^2d_id_j$, and multiplication by any teacher-edge score leaves a centered odd factor. Terms linear in theta and quadratic in the teacher also vanish, since the corresponding second Gaussian derivative contains a factor with mean $E(E[X_i^2\mid O_i]-1)=0$.

For F, the Gaussian ratio derivatives are $R_a=X_iX_j$, $R_{12,13}=(X_1^2-1)X_2X_3$, and $R_{12,13,23}=\prod_i(X_i^2-1)$. In the third derivative of minus log bank mean, all terms except the product of three bank scores vanish after observation averaging. The three particle indices are all equal, exactly two equal, or all distinct, giving expectations $1,v^2,v^3$. The mixed teacher derivative follows by multiplying $H_{K,12,13}(o)=\mu_2\mu_3[1-(1-K^{-1})d_1]$ by the remaining observed edge score $\mu_2\mu_3$ and averaging. These calculations give the following common form, up to candidate-independent terms:

$$
\begin{aligned}
F_{K,Q}^{\rm prof}(\theta;\eta)
&=\tfrac12s_Q\|\theta\|^2-v^2\theta^\top\eta\\
&\quad-2T_Q\theta_{12}\theta_{13}\theta_{23}
+U_Q\sum_{\{a,b,c\}=\{12,13,23\},\,a<b}\theta_a\theta_b\eta_c\\
&\quad+O((\|\theta\|+\|\eta\|)^4),\\
T_F=T_L&=\frac{1+3(K-1)v^2+(K-1)(K-2)v^3}{K^2},\qquad
U_F=U_L=\frac{v^2[1+(K-1)v]}K,\\
T_A&=v^3+\frac{3vw^2}{2K}+\frac{w^3}{K^2},\qquad U_A=v^3.
\end{aligned}
\tag{B.13}
$$

The sum has three terms. F and L share the full cubic polynomial because their profile difference starts at degree four, not merely because their first responses agree.

On $\eta=(u,u,0)$, leaf exchange gives $\theta_{12}=\theta_{13}=r_Qu+O(u^3)$. The third score equation at order $u^2$ is $s_Q\chi_Q-2T_Qr_Q^2+2U_Qr_Q=0$. Therefore

$$
\begin{gathered}
\theta_{Q,23}^*(u,u,0)=\chi_Qu^2+O(u^4),\\
\chi_F=\chi_L=-\frac{2K(K-1)v^5(1-v)^2}{[1+(K-1)v^2]^3},\\
\chi_A=\frac{Kv^4(1-v)^2[Kv+2(1-v)]}{[Kv^2+(1-v)^2]^3}>0.
\end{gathered}
\tag{B.14}
$$

For $K>1$, the F/L coefficient is strictly negative; at $K=1$ it is zero while the A coefficient is still positive. Reflection of group 1 sends u to $-u$ but leaves coordinate 23 unchanged. Uniqueness of the branch and closure of the symmetric boxes therefore make that coordinate even; the $O(u^4)$ remainder uses a $C^4$ branch, available from the higher-order regularity above. Without those conditions one should not infer this remainder from a cubic objective alone. For the unit direction $\eta=t(1,1,0)/\sqrt2$, the leading third coordinate is $\chi_Qt^2/2$. It is a marginal correlation, not a conditional-independence graph edge. The formula is homogeneous and class-specific; it is not a sign theorem for heterogeneous channels or arbitrary factorized proposals.

## B.4. The local ridge connection has a boundary

For the F/L first-order response, a deterministic quadratic penalty on integrated NLL can match the population curvature:

$$
\begin{gathered}
\mathcal L_{\rm ridge}(\theta,o)=-\log p_\theta(o)+\tfrac12\theta^\top\Lambda_K\theta,
\qquad \Lambda_K=(I_3-J)/K,\\
\nabla\mathcal L_{\rm ridge}(0,o)=-m(o),\qquad
\nabla^2\mathcal L_{\rm ridge}(0,o)=H_\infty(o)+\Lambda_K,\\
\nabla^2\mathcal L_{K,F}(0,o)=H_\infty(o)+K^{-1}\operatorname{Cov}_{q_0}(s\mid o),\\
\chi_{\rm ridge}=-\frac{2K^2v^5(1-v^2)}{[1+(K-1)v^2]^3}.
\end{gathered}
\tag{B.15}
$$

The mean Hessians and first responses match, whereas the conditional Hessians and cubic terms generally do not. For the last line, integrated likelihood has $T_\infty=U_\infty=v^3$ and the quadratic penalty has no third derivative; substitution in the third score equation gives the expression. A sum-NLL convention multiplies this penalty by N. The norm uses the three physical correlations; $\|C-I\|_F^2=2\|\theta\|^2$.

Odd variation in a three-edge candidate contrast is also not exclusive to finite particles. To separate it, fix w in correlation space, $0<\beta<1$, and let $c_{3,K}$ be the cubic coefficient in $F_{K,F}(\beta tw;tw)-F_{K,F}(tw;tw)$. Put $P_w=w_{12}w_{13}w_{23}$. From (B.13),

$$
\begin{aligned}
c_{3,\infty}&=-P_w(1-\beta)^2(1+2\beta)v^3,\\
c_{3,K}&=P_w\{2(1-\beta^3)T_F-3(1-\beta^2)U_F\},\\
c_{3,K}-c_{3,\infty}
&=P_w\left[\frac3K(1+\beta^2-2\beta^3)v^2(1-v)
+\frac2{K^2}(1-\beta^3)(1-v)^2(1+2v)\right].
\end{aligned}
\tag{B.16}
$$

For $P_w>0$ and $0<v<1$ this increment is positive, but the total cubic coefficient need not be. Quadratic and higher-order terms still enter finite-amplitude rankings. Neither this identity nor ridge matching identifies the complete finite objective with penalized likelihood.
