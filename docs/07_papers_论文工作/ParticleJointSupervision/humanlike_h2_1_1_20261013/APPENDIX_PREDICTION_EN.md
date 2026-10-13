# Appendix L. Limits of local geometry for risk-budget decisions

## L.1. Fixed development protocol

This appendix reports a completed development test of whether the empirical suppression/release pattern in Appendix K predicts gains from increasing the particle count. The study is exploratory development, not a confirmatory test. Twelve independent units comprise three generators, sample sizes $N=1024,4096$, and repetitions 0 and 1 in every cell. None of the four discovery training streams is reused.

All generators use (13). D1 and D2 share $b=(-.35,.4,-.15,.6,-.5,.2,.1,-.25)$ and row lengths $a=(.4,.65,.8,.5,.95,.7,.6,.85)$. D1 has zero angles; D2 has angles $(-.06,-.02,0,.01,.03,.045,.07,.09)$. D3 has $b=(-.6,.1,.35,-.45,.65,-.2,.5,-.05)$, $a=(.7,.55,.9,.6,.8,1.05,.45,.75)$, and angles $(-.65,-.43,-.11,.15,.28,.49,.75,.93)$. These describe the prescribed generators, not measured factor detectability.

The anchor fits empirical $F_8$ on a training-selected signed rank-one branch. Signs come from the leading eigenvector of empirical Bernoulli correlation, with sign-resolution requirements of eigenvalue gap $>10^{-8}$ and component magnitudes $>10^{-6}$. Empirical correlation uses variance floor $10^{-8}$; unresolved signs fall back to all-positive signs and D0 (not triggered here). Exact bit complementation recodes negative rows; it neither smooths nor renormalizes the empirical target. Initialization uses recoded Jeffreys-marginal logits and amplitude .5; smoothing applies only to initialization. Positive amplitudes lie in $[.05,2]$ and biases in $[-2,2]$. The two-stage Newton budgets and C/D qualification rules are those in Appendix K.7. No teacher sign or risk selects an anchor.

FULL raises to HIGH only when qualified low-$K$ angular curvature is positive, integrated empirical curvature is negative with a resolved minimum direction, and $F_{64}$ is negative in that same direction. Each sign requires agreement of C/D readings beyond $\tau=\max(5\times10^{-4},10\,\text{rule discrepancy})$; radial stationarity and numerical prerequisites also apply. Let $A=1$ denote this D3 outcome. The simpler reference-only indicator $B$ requires a qualified negative integrated empirical angle mode, without the low/high finite-$K$, radial-stationarity or minimum-direction-gap requirements. The rules satisfy $A\le B$. D1 denotes an already-negative low mode or a positive reference mode; D2 denotes suppression with positive high-direction curvature. D0 is abstention, not evidence that suppression is absent. All non-raise outcomes select LOW.

Each unit receives one independent centered, normalized Gaussian angle kick of magnitude .15, shared by LOW, HIGH and REF-fit and never selected or redrawn for alignment. LOW8 and HIGH64 each run 600 Adam updates with four banks per update, sharing a $600\times4\times64\times2$ prior pool; LOW uses each bank's first eight particles. They process 19,200 and 153,600 prior vectors respectively, not equal particle budgets. Adam uses $\beta=(.9,.999)$ and $\epsilon=10^{-8}$. Learning rates are .01 for 400 steps and .002 thereafter; parameters are clipped to $[-2,2]$, with LAST readout. REF-fit uses the same start and integrated empirical GH32 L-BFGS-B, capped at 120 iterations and 180 actual objective calls; a budget stop retains the last accepted state. It is not a global optimum or equal-cost reference.

All inputs, then all diagnoses, then all LOW/HIGH/REF-fit endpoints are globally locked before teacher-PMF risk evaluation. PCG64DXSM uses SeedSequence $[202610071601,0,\text{teacher index},N,\text{rep},\text{role}]$, with teacher indices 0--2 and roles 0--3 for data latent variables, Bernoulli uniforms, neutral kick and shared training priors respectively. No formal validation unit was generated.

## L.2. Every unit and the risk comparisons

Write $R_L,R_H$ for LOW/HIGH observation KL and $R_A,R_B$ for the two policies, each choosing among those same saved endpoints. $R_B$ uses the same mixture with $B$ in place of $A$. Bars below average all twelve units:

$$
\begin{aligned}
\Delta_i&=R_{L,i}-R_{H,i},&
R_{A,i}&=(1-A_i)R_{L,i}+A_iR_{H,i},\\
R_{A,i}-R_{L,i}&=-A_i\Delta_i,&
R_{A,i}-R_{H,i}&=(1-A_i)\Delta_i,\\
R_{B,i}-R_{A,i}&=(A_i-B_i)\Delta_i,&
I&=\frac1{12}\sum_{i=1}^{12}(A_i-B_i)\Delta_i.
\end{aligned}
\tag{L.1}
$$

The following table retains every unit. Generator names D1--D3 and diagnostic classes D0--D3 are different labels. Risks are per eight-bit observation; displayed values are rounded, not interval bounds.

|Generator|$N$|rep|$A/B$|Class|$R_L$|$R_H$|$R_{\rm REF}$|$\Delta$|
|---|---:|---:|---|---|---:|---:|---:|---:|
|D1|1024|0|1/1|D3|0.00527038|0.00765636|0.01039311|-0.00238598|
|D1|1024|1|0/1|D0|0.00511894|0.00736992|0.01010630|-0.00225098|
|D1|4096|0|0/0|D1|0.00467141|0.00370323|0.00468541|0.00096817|
|D1|4096|1|0/0|D1|0.00379623|0.00292711|0.00374689|0.00086912|
|D2|1024|0|0/0|D0|0.01477750|0.00957973|0.01102989|0.00519777|
|D2|1024|1|1/1|D3|0.00832493|0.00745775|0.01015787|0.00086717|
|D2|4096|0|0/0|D1|0.00458296|0.00193956|0.00259753|0.00264340|
|D2|4096|1|0/0|D1|0.00360152|0.00152366|0.00247655|0.00207786|
|D3|1024|0|0/1|D0|0.00985903|0.00611614|0.00740828|0.00374289|
|D3|1024|1|1/1|D3|0.01190885|0.01196568|0.01371180|-0.00005683|
|D3|4096|0|0/0|D0|0.01177927|0.00501291|0.00406364|0.00676637|
|D3|4096|1|1/1|D3|0.00904123|0.00147159|0.00171710|0.00756964|

All twelve LOW/HIGH risk pairs and all anchor/diagnostic numerical prerequisites passed. FULL raised in four units and reference-only in six. Four D0 outcomes arose solely from the prespecified resolution bands: one low-$K$, two reference, and one high-direction reading; none was a computation/solver failure. Increasing quadrature precision is not the same operation as lowering these semantic thresholds.

HIGH reduced risk in 9/12 units; six gains exceeded the prespecified descriptive scale .001. FULL captured one of those six and reference-only two. The raise/stay mean gains were .00149850/.00250183; $\overline{A\Delta}=.00049950$ and $\overline{(1-A)\Delta}=.00166788$. The observed increment was $I=-.000124326$, so FULL did not improve mean risk over reference-only. For example, D1/1024/0 raised despite a negative gain, whereas D2/4096/0 did not raise despite gain .00264340. These are development observations, not population-probability counterexamples or a causal separation of sampling error, radial adaptation and training randomness.

## L.3. Cost and development scope

Let $C_0$ include the common anchor and neutral-kick computation/generation, $C_D$ the FULL diagnosis, and $C_L,C_H$ the subsequent arm costs. Always-HIGH has cost $C_{\rm AH}=C_0+C_H$:

$$
\begin{aligned}
C_A&=C_0+C_D+(1-A)C_L+AC_H,\\
C_A-C_{\rm AH}&=C_D-(1-A)(C_H-C_L),\\
\overline{C_D}
&\le\overline{(1-A)(C_H-C_L)}
 -.2\,\overline{C_0+C_H}
 \quad\text{for 20\% saving}.
\end{aligned}
\tag{L.2}
$$

The reference-only formulas replace $A,C_D$ by $B,C_{\rm ref}$. Measured mean costs use the same single-threaded float64 implementation and include the common start; they exclude archival I/O and initial process setup. FULL evaluates both reference and finite C/D blocks for every unit, rather than using an optimized short circuit.

|Program or policy|Mean risk|Mean seconds/unit|
|---|---:|---:|
|always-LOW|0.00772769|6.86419|
|always-HIGH|0.00556030|7.47119|
|FULL|0.00722819|21.31930|
|reference-only|0.00710386|7.20431|
|REF-fit|0.00684120|6.54326|

Mean common-start cost was 6.19146 seconds, FULL diagnosis 14.25508, reference-only diagnosis .04356, and LOW/HIGH follow-up .67273/1.27973. The 8-fold prior-processing ratio thus did not imply an 8-fold wall-time ratio. FULL cost 2.85354 times always-HIGH and was worse in both mean risk and time than always-HIGH and reference-only on this panel. Even with free diagnosis and all units choosing LOW, retaining the common-start cost permits only 8.12455% savings; the 20% target is unattainable for these measured costs. This bounds the current prototype and budget, not all possible diagnostics or programs.

Risk uses GH40/60/80 and rotated GH72, with GH60 main values, raw probability masses retained, mass tolerance $2\times10^{-12}$, PMF difference $2\times10^{-8}$, and paired-risk discrepancy $2\times10^{-6}$. Qualification failure would invalidate primary risk totals without dropping units. The fixed 10,000 stratified resamples draw twelve complete units from each two-unit empirical cell; empty conditional groups remain undefined without replacement draws. The resulting 2.5th--97.5th percentile ranges for the raise mean [.000416,.002597] and increment [-.000332,.0000836] are width sensitivities, not confirmatory confidence intervals or power guarantees; two observations per cell can miss tail variation.

Local execution verification covered 99 saved numerical archives: generated-label reconstruction, signed recoding and kicks, saved Newton aggregates, all 14,400 finite-arm update calculations, 72 selected gradient recomputations at zero-based steps 0/400/599, and diagnostic/risk/policy identities. It did not independently retrain the models or recompute every integral and PMF; numerical rule agreement is not a rigorous remainder certificate. Public tables and receipts do not imply that an external reviewer obtained every local archive.

The development result did not justify expanding this prototype to formal 72-unit validation. The following saved-object analyses clarify its limitations without changing the twelve units, alarms or directions.

## L.4. Two distinct extrapolations from saved local geometry

A post-hoc analysis fixes the original empirical minimum direction $d$ at every anchor. It does not select a population minimum mode, change an alarm or use $d$ as the training kick. Let $h_{\rm emp}$ and $h_P$ denote integrated empirical and population curvature along this same $d$, and $g_P$ the population risk derivative along the unit negative empirical $F_{64}$ gradient in biases and row lengths. All twelve objects are retained:

|Generator|$N$|rep|$h_{\rm emp}$|$h_P$|$g_P$|
|---|---:|---:|---:|---:|---:|
|D1|1024|0|-0.0029610851|0.000983312416|-0.0118723777|
|D1|1024|1|-0.00091085198|0.00638551323|-0.0132373414|
|D1|4096|0|0.00137759516|0.00325640224|-0.0170323236|
|D1|4096|1|0.000699872149|0.00286399175|-0.0178011302|
|D2|1024|0|0.00028740652|0.00556901196|-0.0359961876|
|D2|1024|1|-0.00346641642|0.00601140055|-0.0243424188|
|D2|4096|0|0.00157486876|0.00332114038|-0.0244989155|
|D2|4096|1|0.000889443563|0.00275525245|-0.020023896|
|D3|1024|0|-0.00541911655|-0.00101587762|-0.0170359137|
|D3|1024|1|-0.00267833527|-0.00220189265|-0.00557352383|
|D3|4096|0|0.000111312305|0.00103881666|-0.0267891493|
|D3|4096|1|-0.00159105188|-0.000827868167|-0.0227718276|

Three of the six empirically negative directions become positive under the teacher law: D1/1024/0, D1/1024/1 and D2/1024/1. The population signs are 9 positive and 3 negative, agreeing across eight combinations of conditional-integration and teacher-PMF rules and an additional GH80 check. These describe the locked direction, not all population angular modes. D3/1024/1 has negative population curvature but HIGH still ends slightly worse than LOW, with $\Delta=-.0000568324$. This endpoint comparison is not a controlled step along $d$, nor does it say that HIGH is worse than the original anchor. Empirical-to-population meaning and local-direction-to-training-outcome prediction are distinct extrapolations.

The empirical bias/row-length gradient norms are $1.25\times10^{-13}$ to $7.84\times10^{-12}$ for $F_8$, .0204607--.0230120 for $F_{64}$, and .0226746--.0257878 for the integrated target. At all twelve interior anchors $g_P<0$, ranging from -.0359962 to -.00557352. Thus low-$K$ stationarity leaves an available first-order channel at higher $K$, and a sufficiently small feasible step along this measured direction can lower population risk. No such step was trained here. The sign is negative even in the three units where HIGH loses to LOW, so it does not predict the 600-step comparison. Bias and row length change together; sampling, altered angle geometry, the neutral kick and the full stochastic trajectory remain competing explanations.

Risk bookkeeping also distinguishes marginal fit from joint dependence. The dependence score is $E_P\log[Q(Y)/\prod_jQ_j(Y_j)]$, and marginal error is $\sum_jD(P_j\Vert Q_j)$. Across the twelve units, the mean gain .00216738388 splits into .000117968607 marginal-error improvement and .00204941527 joint dependence-score improvement. All three negative total gains improve marginal error but worsen dependence score; five of the nine positive gains worsen marginal error. This is an identity for the saved endpoints, not a coordinate-causal decomposition or a separate dependence KL. All 36 endpoints have lower risk than their same-data independent Bernoulli MLEs (mean baseline risk .0891267). Moreover, D1 is a legal one-factor witness with numerical $D(P_{D2}\Vert P_{D1})\simeq8.11451\times10^{-7}$: D2 is a near-one-factor control, not evidence of an easily detectable second factor. That value is not a rigorously certified upper bound. These checks added no training data, prior draws or optimizer updates.

## L.5. Fixed-direction validation: a small-step observation

For each saved anchor, lock $b,a,d$ and the row-length-preserving candidates $t=.02,.10,.25$. Here $a_j$ is the signed first-column loading in the original label coordinates, so the row length is $|a_j|$. Let $Q_t$ be the exact integrated observation law of (13) on the path below, and $H_{\phi,\infty}^{y}$ the per-label Hessian of $-\log Q(y)$ with respect to the angular parameters, evaluated at the anchor. Define beneficial curvature and finite paired scores by

$$
\begin{aligned}
W_j(t)&=a_j(\cos(td_j),\sin(td_j)),\\
s(y)&=d^\top H_{\phi,\infty}^{y}d,
&X_c(y)&=-s(y),\\
z_t(y)&=\log Q_t(y)-\log Q_0(y),\\
E_Pz_t&=D(P\Vert Q_0)-D(P\Vert Q_t).
\end{aligned}
\tag{L.3}
$$

The last identity targets the risk improvement of this fixed candidate from $Q_0$, not the earlier HIGH/LOW endpoints. There are twelve curvature scores and 36 finite scores; none of the 36 candidates is dropped or selected by its teacher risk. No independent validation label was drawn.

**Observation (reflection and local scaling).** For fixed finite $b,a,d$, every label probability is positive, $Q_t(y)=Q_{-t}(y)$, and the following convergence holds uniformly over the finite alphabet:

$$
\begin{aligned}
z_t(y)&=-\frac{t^2}{2}s(y)+O(t^4),\\
\sup_y\left|\frac{2z_t(y)}{t^2}+s(y)\right|&=O(t^2).
\end{aligned}
\tag{L.4}
$$

To see this, reflect the standard Gaussian latent variable $(Z_1,Z_2)$ to $(Z_1,-Z_2)$. Its law is unchanged and each row inner product at $t$ becomes the one at $-t$. Hence $\ell_y(t)=-\log Q_t(y)$ is even. Fixed-order path derivatives are dominated by Gaussian polynomial moments; positivity and the finite alphabet give a common neighborhood in which $\ell_y$ is four times continuously differentiable. Its odd derivatives vanish at zero, while $\ell_y''(0)=s(y)$, giving (L.4) by Taylor expansion. This is an analytic observation about the fixed path, not a new main theorem or a computed remainder certificate.

Writing $\kappa=E_Ps$ and $\sigma_s^2=\operatorname{Var}_P(s)$ gives, for fixed $n$,

$$
\begin{aligned}
E_Pz_t&=-\frac{t^2}{2}\kappa+O(t^4),\\
\operatorname{Var}_P(z_t)&=\frac{t^4}{4}\sigma_s^2+O(t^6),\\
\frac{E_Pz_t}{\sqrt{\operatorname{Var}_P(z_t)/n}}
&\longrightarrow-\frac{\sqrt n\,\kappa}{\sigma_s}
\quad(\sigma_s>0).
\end{aligned}
\tag{L.5}
$$

The variance remainder follows from the covariance of the $t^2s$ term with the bounded $t^4$ remainder. Signal and standard deviation both shrink as $t^2$; a smaller step does not automatically improve nondegenerate signal-to-noise. If $\sigma_s=0$, the limiting expression in (L.5) is undefined and this conclusion does not apply.

## L.6. Statistical interface and oracle feasibility

Condition on the training data and randomization, then lock the anchors, directions, amplitudes and scoring functions. For a future independent iid validation set from $P$, a fixed bounded score $X$ has one-sided lower bound $E_PX\ge\bar X-r$ with probability at least $1-\delta$. The empirical Bernstein radius below is the affine rescaling of Theorem 4 in [@maurer2009], applied to $-X$; the Hoeffding radius is included for comparison. Here $n\ge2$, and $R$ is the score's maximum minus minimum over **all 256 labels**, including any zero-probability labels, rather than the observed sample range:

$$
\begin{aligned}
\widehat V_n&=\frac1{n-1}\sum_{i=1}^n(X_i-\bar X)^2,\\
r_{\rm EB}&=\sqrt{\frac{2\widehat V_n\log(2/\delta)}n}
 +\frac{7R\log(2/\delta)}{3(n-1)},\\
r_{\rm H}&=R\sqrt{\frac{\log(1/\delta)}{2n}},
&\delta&=\frac{.05}{48\cdot6}.
\end{aligned}
\tag{L.6}
$$

The union bound covers 48 fixed scores at the six prespecified looks $n=1024,4096,16384,65536,262144,1048576$ with total error budget .05. Scores and nested looks may share observations; independence between scores or looks is unnecessary. This budget does not authorize extra adaptive looks or uncounted candidates. Multiple fixed risk gates on one score reuse its same lower-bound event. A constant score with $R=0$ needs no uncertainty radius; zero observed sample variance with $R>0$ still leaves the Bernstein range term.

For $c>0$, both radii are positively scale equivariant: $r(n,c^2V,cR,\delta)=c\,r(n,V,R,\delta)$, with $V$ unused by Hoeffding. Uniform $2z_t/t^2\to X_c$ implies convergence of means, full-label ranges and the variance of any fixed validation sample. Consequently the corresponding tests converge when both score and target are rescaled by $2/t^2$; a strict binary decision need not stabilize exactly on a limiting boundary. Curvature margin $\tau_c$ matches finite margin $t^2\tau_c/2$. A fixed risk gate $10^{-5}$ or $10^{-4}$ does not shrink with step size, so arbitrarily reducing $t$ eventually removes its finite signal. None of these limits predicts the random kick or the unrestricted 600-step trajectory.

The present comparison uses the known teacher mean $\mu$ and population variance $V$ in place of validation-sample quantities. Inserting $V$ into the empirical Bernstein expression is **oracle planning**, not an actual confidence interval, a power guarantee or a necessary sample-size lower bound. Even a Hoeffding curve centered at the known $\mu$ is not an observed-data lower bound. Saved raw PMFs are unchanged; moments explicitly use $\bar P=P/\sum_yP_y$, retaining raw mass and normalization corrections. A displayed planning $n$ is the first integer $2\le n\le2^{30}$ satisfying $r(n,V,R,\delta)<\mu-\tau$, not a guarantee that a random validation set of that size will pass.

A separate full-label comparison can transfer an existing valid curvature lower bound $L_c\le E_PX_c$ to a finite candidate:

$$
\begin{aligned}
e_-(t)&=\max_{y\in\{0,1\}^8}
 \left\{X_c(y)-\frac{2z_t(y)}{t^2}\right\},\\
E_Pz_t&\ge\frac{t^2}{2}\{E_PX_c-e_-(t)\},\\
L_c&>e_-(t)+\frac{2\eta}{t^2}
\quad\Longrightarrow\quad E_Pz_t>\eta.
\end{aligned}
\tag{L.7}
$$

The pointwise inequality follows directly from the maximum defining $e_-$; its sign is retained even if negative. Three fixed candidates can reuse the same curvature event, without a new confidence event for each transfer. The numerical table relationship inherits the qualification of its scores; it is not a rigorous exact-model error enclosure or a Taylor remainder bound.

The next table contains the three objects with negative population curvature. Curvature $n$ targets margin $10^{-4}$, and matched $n$ at $t=.02$ targets finite gain $t^2\cdot10^{-4}/2=2\times10^{-8}$. The transfer column is $t^2(\mu_c-e_-)/2$ with the curvature mean already known, so it contains no statistical error:

|Object|Curvature $n$|Matched $n$, $t=.02$|Gain, $t=.25$|Transfer floor, $t=.25$|
|---|---:|---:|---:|---:|
|D3/1024/0|261,012|261,593|$2.64051\times10^{-5}$|$-4.35048\times10^{-5}$|
|D3/1024/1|68,080|68,157|$6.20650\times10^{-5}$|$-2.62309\times10^{-5}$|
|D3/4096/1|265,403|265,907|$2.22401\times10^{-5}$|$-2.90338\times10^{-5}$|

The matched planning ratios are 1.00113--1.00223, giving no clear data saving for curvature in this comparison. At $t=.02$ the actual numerical gains are only $1.65\times10^{-7}$ to $4.40\times10^{-7}$. At $t=.25$ all three gains are positive but the particular transfer floor stays negative even with the curvature mean known. More validation data therefore cannot make this specific floor establish a positive gain; it does not show that the candidates have no gain or that every tighter transfer fails. Direct paired-score validation avoids this conversion. For direct gain exceeding $10^{-5}$ at $t=.25$, the corresponding oracle Bernstein planning sizes are 750,141, 103,000 and 862,235 in table order, well above the original training sizes.

All 36 fixed finite candidates remain in the public record: 9 have positive numerical gain, 4 exceed $10^{-5}$ and none exceeds $10^{-4}$; the maximum is $6.20650\times10^{-5}$. The other nine objects have positive population curvature along their locked $d$, so this curvature-benefit target lacks an oracle margin there. No independent validation or new training was executed. Numerical qualification covered all 48 scores using 192 GH PMF evaluations and 48 independent GL checks; score moments and critical planning integers were subsequently verified from saved quantities. Cross-rule agreement does not provide a rigorous numerical enclosure: without a strict scoring-error bound, the statistical interface is conditional on exact scores and must not be called an unconditional exact-model certificate.

Together, the development, saved-object interpretation and fixed-direction planning limit the demonstrated utility of this budget rule. They do not negate Appendix K's local geometry or establish a universal failure of curvature diagnostics. The complete public record retains all twelve objects and all 36 finite candidates.
