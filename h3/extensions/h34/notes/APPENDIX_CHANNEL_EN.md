<a id="channel-proof"></a>

# C. Learning one common observation channel

## C.1. Candidate derivatives with a fixed teacher

Use the common-channel extension in Appendix A: one count slope alpha is free, while the label slope, latent marginal variances, $\tau$, and M remain fixed. The teacher is always $P_{\eta,\alpha_*}$. The base posterior, directed mean features, and tilt normalization in every candidate proposal use the **candidate** alpha. All expectations marked $E_*$ below keep the teacher distribution fixed.

First record the derivatives that maintain this distinction. At fixed x and local state s, let $\xi_\alpha=\partial_\alpha\log\ell_\alpha(s\mid x)$ and $\omega_\alpha=\partial_\alpha^2\log\ell_\alpha(s\mid x)$. At fixed tilt h, with q the normalized tilted local distribution, differentiation of a normalized integral gives

$$
\begin{gathered}
\partial_\alpha\mathcal A=E_q\xi_\alpha,\qquad
\partial_\alpha^2\mathcal A=E_q\omega_\alpha+\operatorname{Var}_q(\xi_\alpha),\\
\partial_\alpha\mu=\operatorname{Cov}_q(X,\xi_\alpha),\qquad
\partial_\alpha^2\mu=\operatorname{Cov}_q(X,\omega_\alpha)
+E_q[(X-\mu)(\xi_\alpha-E_q\xi_\alpha)^2],\\
\partial_\alpha m_2=\operatorname{Cov}_q(X^2,\xi_\alpha).
\end{gathered}
\tag{C.1}
$$

These are derivatives of the absolute log normalizer $\mathcal A$ from (A.3), not only of the relative tilt normalizer kappa. A consistent positive finite quadrature sum has the same normalized-sum identities, but is still an approximation to the continuous integral. Differentiating only its unnormalized numerator would not give (C.1).

A's physical feature map also varies with alpha. At a fixed observation,

$$
\begin{gathered}
h_{i,\alpha}=\sum_{j\ne i}a_{i\leftarrow j}\mu_{j,\alpha}',\qquad
h_{i,\alpha\alpha}=\sum_{j\ne i}a_{i\leftarrow j}\mu_{j,\alpha}'',\qquad
h_{i,\alpha a_{i\leftarrow j}}=\mu_{j,\alpha}',\\
\nabla^2_{\rm physical}F
=D^\top\nabla^2_{\rm local}F D+\sum_iF_{h_i}\nabla^2_{\rm physical}h_i.
\end{gathered}
\tag{C.2}
$$

Here D is the Jacobian from physical $(\theta,\alpha,b,a)$ to local $(\theta,\alpha,h)$. The last term and the sampling-density score must be retained in a derivative away from the anchor. Their cancellation at zero coefficients does not justify omitting them generally.

More explicitly, at a fixed particle $\partial_\alpha\log q_i=\xi_{i,\alpha}-E_{q_i}\xi_{i,\alpha}+h_{i,\alpha}(X_i-E_{q_i}X_i)$, whereas $\partial_\alpha\log W=\sum_i\{E_{q_i}\xi_{i,\alpha}-h_{i,\alpha}(X_i-E_{q_i}X_i)\}$. If f is the bank loss, its expected derivative is $E[\partial_\alpha f+f\sum_k\partial_\alpha\log q(X_k\mid O)]$. These identities follow directly from (A.5) and differentiation of the normalized sampling integral; using only the derivative of f would change the objective being differentiated.

## C.2. The complete anchor Hessian preserves the first response

At $\theta=u=0$, the candidate proposal is exactly the candidate independent posterior for every alpha. Its weights are $\prod_iZ_{i,\alpha}(O_i)$, so for every teacher correlation,

$$
\begin{gathered}
F_{K,Q}(0,\alpha,0;\eta)=-\sum_iE_{*,i}\log Z_{i,\alpha}(O_i),\qquad
F_u(0,\alpha,0;\eta)=0,\\
z_i(O_i)=\left.\partial_\alpha\log Z_{i,\alpha}(O_i)\right|_{\alpha_*},\qquad
I_\alpha=\sum_iE_{0,*}z_i^2>0,\\
E_{0,*}[m\textstyle\sum_i z_i]=0,\qquad F_{\alpha\eta\eta}(0,\alpha_*,0;0)=0.
\end{gathered}
\tag{C.3}
$$

The group marginals of every teacher have the same alpha*, which proves the first and last statements. Information and the middle cross moment use $E_{0,*}$, the matching independent teacher. That cross moment vanishes because each edge score contains two independent centered means; every term leaves at least one such mean after averaging. Positivity of $I_\alpha$ follows from the strictly monotone visible moment in (A.7). Along the independent candidate axis, the exact posterior's alpha score and the proposal's alpha score coincide. Thus the posterior-minus-proposal residual in the Jensen gap has no alpha direction. The moving directed feature derivative in (C.2) is multiplied by $a=0$ at the anchor.

In physical order $(\theta,\alpha,u)$, the full result is

$$
\begin{gathered}
\nabla\mathcal L_{K,Q}(0,\alpha_*,0;O)=(-m,-\textstyle\sum_i z_i,0),\\
H_{Q,\alpha}^{\rm full}
=\begin{pmatrix}J+C_r/K&0&-M_Q/K\\0&I_\alpha&0\\-M_Q^\top/K&0&G_Q/K\end{pmatrix},\\
(z,l,c)^\top H_{Q,\alpha}^{\rm full}(z,l,c)
=z^\top Jz+I_\alpha l^2+K^{-1}E[(z^\top r-c^\top A_Q)^2]>0
\quad\text{if }(z,l,c)\ne0.
\end{gathered}
\tag{C.4}
$$

F has no proposal coordinates. The complete dimensions at $M=8$ are 4, 58, and 64. The positivity proof uses the full proposal Gram from (B.2), not a reduced numerical Hessian. It supplies the remaining Hessian condition of Proposition A.5. Eliminating u leaves $\operatorname{diag}(S_Q,I_\alpha)$, so the first-order correlation response remains $S_Q^{-1}J\eta$ and the first-order alpha shift is zero. These zero blocks belong to the matching independent anchor; they do not state that the parameters decouple at nonzero teachers.

## C.3. The fixed-teacher candidate axis

To determine the next effect, allow candidate alpha to vary while averaging against the independent teacher's fixed local probabilities $Z_{i,\alpha_*}$. Define

$$
\begin{gathered}
P_i(\alpha)=E_*m_{2,i,\alpha},\qquad V_i(\alpha)=E_*\mu_{i,\alpha}^2,\qquad
C_i(\alpha)=E_*[\mu_{i,\alpha}\mu_{i,*}],\qquad W_i=P_i-V_i,\\
P_i(\alpha_*)=1,\quad V_i(\alpha_*)=C_i(\alpha_*)=v_i,\quad W_i(\alpha_*)=1-v_i,\\
P_i'=E_*\operatorname{Cov}_{q_{0i,\alpha}}(X_i^2,\xi_{i,\alpha}),\quad
V_i'=2E_*[\mu_{i,\alpha}\operatorname{Cov}_{q_{0i,\alpha}}(X_i,\xi_{i,\alpha})],\\
C_i'=E_*[\mu_{i,*}\operatorname{Cov}_{q_{0i,\alpha}}(X_i,\xi_{i,\alpha})].
\end{gathered}
\tag{C.5}
$$

In particular, $V_i'=2C_i'$ at the matching point, whereas $P_i'$ need not vanish. Setting $P_i(\alpha)\equiv1$ would incorrectly change a fixed-teacher expectation into a moving-model expectation.

Let e denote edge ij. Expanding the complete conditional objective at $(\theta,u,\eta)=(0,0,0)$ for each candidate alpha, and retaining all proposal coordinates, gives the diagonal generating curvature and forcing

$$
\begin{gathered}
B_e(\alpha)=C_iC_j,\qquad
H_e(\alpha)=P_i+P_j-1-(1-K^{-1})(P_iP_j-V_iV_j),\\
S_{F,e}=S_{L,e}=H_e,\qquad
S_{A,e}=H_e-K^{-1}(W_iV_j+W_jV_i),\\
R_e=V_iV_j-(P_i-1)(P_j-1),\qquad
S_{F,e}=R_e+(P_iP_j-V_iV_j)/K,\quad S_{A,e}=R_e+W_iW_j/K.
\end{gathered}
\tag{C.6}
$$

For example, the conditional integrated-likelihood diagonal Hessian is $m_{2,i}+m_{2,j}-1-m_{2,i}m_{2,j}+\mu_i^2\mu_j^2$; averaging independent teacher groups produces $R_e$. The residual variance adds $(P_iP_j-V_iV_j)/K$. Local proposal directions have zero theta mixed block by reflection. The directed direction $\mu_je_i$ has both Gram and theta mixed magnitude $W_iV_j/K$, with opposite sign in the mixed block, giving the stated Schur subtraction. This verifies (C.6) with the candidate base and features updated together.

Up to terms independent of theta, the profiled objective consequently starts as $\frac12\sum_eS_{Q,e}(\alpha)\theta_e^2-\sum_eB_e(\alpha)\theta_e\eta_e$. At alpha*, $B_e=J_{ee}=v_iv_j$ and the curvatures reduce to Appendix B. Also $R_e'=2B_e'$ there; this identity will distinguish generating KL from the finite objective.

## C.4. A second-order channel shift

Let $\eta=td$ for a fixed direction d. All quantities without an explicit alpha argument in the next display are evaluated at alpha*. Differentiating the profiled objective with respect to alpha, using (C.3), yields

$$
\begin{gathered}
r_{Q,e}=B_e/S_{Q,e},\qquad
\theta_Q^{\rm joint}(td)=t\operatorname{diag}(r_Q)d+O(t^2),\\
\alpha_Q^{\rm joint}(td)=\alpha_*+c_Q(d)t^2+o(t^2),\qquad
c_Q(d)=I_\alpha^{-1}\sum_e d_e^2
\left[B_e'r_{Q,e}-\tfrac12S_{Q,e}'r_{Q,e}^2\right].
\end{gathered}
\tag{C.7}
$$

The equation at order $t^2$ is $I_\alpha c_Q+\frac12\sum_eS_{Q,e}'r_{Q,e}^2d_e^2-\sum_eB_e'r_{Q,e}d_e^2=0$. The second-order proposal correction drops out by its first-order stationarity. No teacher-alpha derivative is used. Integrated likelihood has $r=1$ and $S_e'=2B_e'$, so this coefficient is zero in that reference.

In the homogeneous channel take $\|d\|=1$, write $J=jI_3$, $j=v^2$, $w=1-v$, and let $s_Q$ be the common diagonal curvature. Put $P'=\partial_\alpha E_*m_{2,\alpha}$ and $B'=2vE_*[\mu_*\mu_\alpha']$. Formula (C.7) simplifies to

$$
\begin{gathered}
s_F=s_L=j+(1-j)/K,\qquad s_A=j+w^2/K,\qquad r_Q=j/s_Q,\\
c_F=c_L=\frac{j(B'-jP')}{K s_F^2 I_\alpha},\qquad
c_A=\frac{jw(B'-jP')}{K s_A^2 I_\alpha}.
\end{gathered}
\tag{C.8}
$$

For a non-unit direction multiply these coefficients by $\|d\|^2$. The numerator can vanish, and its sign is a property of the fixed-teacher moment derivatives. When $c_F\ne0$, $c_A/c_F=w(s_F/s_A)^2$: the signs agree, but a universal ordering of magnitudes does not follow.

Compare each joint branch with the **same class Q** optimized while alpha is fixed at its true value. The fixed-teacher response function $\widetilde r_Q(\alpha)=B(\alpha)/s_Q(\alpha)$ has derivative $r_Q'=(B'-r_Qs_Q')/s_Q$ at alpha*. The homogeneous unit-direction branches obey

$$
\theta_Q^{\rm joint}(td)-\theta_Q^{\rm known}(td)
=c_Qr_Q'd\,t^3+o(t^3).
\tag{C.9}
$$

This is a difference between branches, not a claim that each branch has no second-order correlation term. In particular, on the forest direction of (B.14), both branches can have a third correlation of order $t^2$. The leading vector in (C.9) includes d, so its third entry is zero when $d_{23}=0$.

## C.5. Finite objective, marginal KL, and joint KL

Define the true generating risk $\mathcal R(\theta,\alpha;\eta)=D_{\rm KL}(P_{\eta,\alpha_*}\Vert P_{\theta,\alpha})$. It does not depend on the proposal coefficients. Relaxing alpha within one profiled class and comparing to that class's known-channel branch gives

$$
\begin{gathered}
F_{K,Q}^{\rm joint}(td)-F_{K,Q}^{\rm known}(td)
=-\tfrac12I_\alpha c_Q^2t^4+o(t^4),\\
\sum_{i=1}^3D_{\rm KL}(Z_{i,\alpha_*}\Vert Z_{i,\alpha_Q^{\rm joint}(td)})
=+\tfrac12I_\alpha c_Q^2t^4+o(t^4).
\end{gathered}
\tag{C.10}
$$

The first identity is obtained by minimizing the local quadratic in the alpha displacement: its linear coefficient at the known branch is $-I_\alpha c_Qt^2+o(t^2)$. The second is the marginal KL expansion with information $I_\alpha$, summed over all three groups. It is not a one-group average. Their opposite signs do not determine the sign of complete joint KL.

For the latter comparison, retain the homogeneous, unit-direction assumptions. At the known-channel branch, the needed derivatives of generating risk are

$$
\begin{gathered}
\mathcal R_\theta=j(r_Q-1)d\,t+O(t^2),\qquad
\mathcal R_\alpha=B'r_Q(r_Q-1)t^2+O(t^3),\\
j r_Q'+r_QB'=2r_QB'-r_Q^2s_Q'=2I_\alpha c_Q.
\end{gathered}
\tag{C.11}
$$

To check the alpha derivative, the risk's candidate-theta curvature is $R_e$ in (C.6), its teacher forcing is $B_e$, and $R_e'=2B_e'$ at the match. Terms involving two teacher derivatives and no candidate correlation vanish because $\mathcal R_\alpha(0,\alpha_*;\eta)=0$ for every eta. Using (C.9), the alpha displacement $c_Qt^2$, and the alpha Hessian $I_\alpha$ in the risk Taylor expansion gives

$$
\begin{aligned}
\mathcal R_Q^{\rm joint}(td)-\mathcal R_Q^{\rm known}(td)
&=\left\{c_Q(r_Q-1)(jr_Q'+r_QB')+\tfrac12I_\alpha c_Q^2\right\}t^4+o(t^4)\\
&=I_\alpha c_Q^2(2r_Q-\tfrac32)t^4+o(t^4).
\end{aligned}
\tag{C.12}
$$

Thus, when $c_Q\ne0$ and $r_Q<3/4$, the leading joint-KL difference is negative despite the positive marginal-KL sum. At $c_Q=0$ or $r_Q=3/4$ the displayed coefficient is zero and does not decide the higher-order sign. Heterogeneous edge responses cannot be replaced by one scalar threshold. These are same-class joint-minus-known differences, not the leading total KL of either branch and not a comparison with a different proposal class.

On the specific forest direction $d=(1,1,0)/\sqrt2$, reflection of group 1 changes t to $-t$ while leaving alpha unchanged. The class and its boxes are closed under that transformation. Branch uniqueness and higher-order smoothness therefore give the following stronger remainders, where $\Delta_Q(t)$ denotes any value difference in (C.10) or (C.12) and $\gamma_Q$ its displayed coefficient:

$$
\alpha_Q^{\rm joint}(td)=\alpha_*+c_Qt^2+O(t^4),\qquad
\Delta_Q(t)=\gamma_Qt^4+O(t^6).
\tag{C.13}
$$

An arbitrary three-edge direction can contain a triangle invariant and need not have this even symmetry. Proposition A.5 gives the local branches a global-minimizer interpretation only in its unquantified teacher neighborhood. Fixed-amplitude stationary points and their quadrature-order checks address separate numerical questions; these expansions supply neither a numerical radius nor a continuous integration certificate. The sampling risk of estimating alpha from finite data is a further effect, treated in [Appendix D](APPENDIX_RISK_EN.md#risk-proof).
