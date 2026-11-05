<a id="evidence-details"></a>

# G. Numerical evidence and implementation scope

## G.1. Finite-model proposal reversal

The comparison below uses two private-variable proposals at $a=1$, $\tau=3$, independent teacher and equicorrelation $.45$, with active $K=8$ and $\zeta=0$. For each $M$, 16,384 Monte Carlo replicates share the teacher observations and candidate active draws across the two settings, while private latent draws are independent between settings. The primary estimator integrates the correlation gradient along a fixed path with eight-point Gauss--Legendre quadrature and subtracts that setting's zero-mean field at independence. Path-order, quadrature and finite-score checks supplied the numerical qualifications reported with this experiment; these checks are not strict interval integration certificates.

The tempered setting uses $\beta_M=1-\gamma/\sqrt{3M}$ with $\gamma=7.241913488981693$. This level came from a separate exploratory Gaussian comparison at the independent teacher, equicorrelation $.45$ and the natural coupled-noise level, over $t\in\{0,1,4,16,64\}$. After viewing that study, the smallest positive grid value whose control-variate estimate of the candidate difference plus three SEs was negative was selected, giving $t_*=16$. The selection rule was recorded after the exploratory results and before subsequent validation. With the previously computed private-logweight variance $.30507970494299236$, $\gamma=\sqrt{16/v_{\rm priv}}$. This same value was held fixed before the fresh paired finite-model comparison shown here. It is a development-selected design setting, not a numerical threshold supplied by the existence theorem.

|$M$|$\beta$|Candidate minus teacher objective|Approximate simultaneous interval|
|---:|---:|---:|---|
|32|1|+.005779742|[+.005592829, +.005966656]|
|32|.260875300|-.003097069|[-.003376238, -.002817900]|
|128|1|+.005904553|[+.005712986, +.006096121]|
|128|.630437650|-.003556909|[-.003810596, -.003303222]|

The family contains these four two-sided comparisons, with normal-Bonferroni level .05. The Monte Carlo replicates, rather than fitted teachers, are the repetitions. The candidate's complete joint KL is positive by its different $Y$ pair moments. Earlier same-model estimates were $.012582783$ (SE $.001283657$) at $M=32$ and $.010619433$ (SE $.001248098$) at $M=128$; they come from a preceding experiment with 2,048 independent risk observations per $M$, not additional risk samples from this paired comparison. Both arms at one $M$ have exactly the same true risk. This experiment concerns population candidate ordering, with zero optimization updates.

![Changing only the private-variable proposal reverses the finite objective preference. Bars are the four approximate simultaneous 95% intervals in the table, based on 16,384 paired Monte Carlo repetitions per private dimension. The candidate has the same positive true joint KL in the two proposal settings; these bars are objective differences, not risk intervals.](figures/proposal_reversal.png){width=100%}

## G.2. Local-risk design and numerical output

The original risk experiments and figures in G.2--G.5 use known observation channels. Their theoretical curves directly use the known-channel baseline in main-text (12).

The four fixed teachers, all feasible for spectral floor $.15$, are:

|$N$|$S$|Nonzero teacher coordinates $u$|Datasets|
|---:|---:|---:|---:|
|2048|.05|.014819242|24|
|2048|2|.093725115|24|
|2048|12|.229578709|24|
|256|2|.265094659|24|

There are 96 independent datasets, each paired with independent numerical sampling, not 96 randomly sampled teachers. Data are independent across conditions; the two sample sizes are not nested prefixes. Teacher parameters enter data generation and offline evaluation, not fitting, start selection or ridge tuning. The initial signal choices $.2,2,8$ were changed to $.05,2,12$ before data generation, using the Gaussian paired-variance formulas below. Both development and confirmation data were generated after this change.

For shared $\varepsilon\sim N(0,I_3)$, put $a_h=\sqrt jh$, $\|a_h\|^2=S$, $G=\|r\varepsilon-(1-r)a_h\|^2/2$, $M_0=\|\varepsilon\|^2/2$ and $I_0=S/2$. Gaussian odd moments vanish, giving

$$
\begin{aligned}
\operatorname{Var}(G-M_0)
&=\tfrac32(1-r^2)^2+r^2(1-r)^2S,\\
\operatorname{Var}(G-I_0)
&=\tfrac32r^4+r^2(1-r)^2S.
\end{aligned}
\tag{G.1}
$$

These are paired local-limit variances, not exact finite-$N$ power or normality statements. Two starts do not divide the statistical variance by two.

The original program fits three coordinates with a spectral projected-gradient solver, at most 100 iterations and 240 evaluations per start, mapping tolerance $10^{-7}$, initial step 32, Armijo constant $10^{-4}$, at most 16 backtracks and steps bounded by $[.25,256]$. Group fitting uses two independent replicates of 512 particle sets and a quadratic control; independent-reference evaluation uses four replicates of 512 particle sets. The integrated and ridge objectives use Gauss--Hermite orders $(20,128)$. All methods return the last accepted output for each original start. Neither risk nor a reference diagnostic selects the reported output.

The controls in Appendix E use actual group-proposal moments; implementation uses finite quadrature and rejection sampling with recorded moment, envelope and count diagnostics. A fixed finite particle set and a fixed quadrature rule are not exact expectations. Full joint-risk evaluation compares orders $(20,128)$ and $(28,192)$, allowing $(36,256)$ as the final order. The prespecified KL tolerance is $\min(10^{-7},10^{-4}/N)$, with maximum PMF discrepancy $10^{-8}$. All 864 risk records passed; the largest observed $N$-scaled KL order discrepancy was $1.93\times10^{-8}$ and the largest PMF discrepancy $6.18\times10^{-12}$. These are observed order differences, not rigorous upper bounds on integration error.

GROUP's independent-reference diagnostic requires projected mapping infinity norm at most $.0003$ and maximum coordinatewise replicate SE of the reference gradient at most $.0001$. The second criterion is not the SE of the mapping norm. The original group program passed 110/192: 48/48 at weak signal, 44/48 at intermediate signal and 12/48 at stronger signal for $N=2048$, plus 6/48 in the descriptive $N=256$ condition; integrated ML and matched ridge passed their corresponding checks on 384/384 outputs combined. Every output remained in the analysis. All 576 original fits stopped under their own mapping tolerance, which does not resolve the separate reference failures. Thus exact empirical minimizers and numerical outputs remain distinct. The subsequent same-data stronger-signal re-evaluation is reported separately in [G.5](APPENDIX_EVIDENCE_EN.md#budget-reevaluation).

Four development datasets were used to choose numerical completeness and cost, with 24 fits and 36 risk evaluations. Confirmation used 96 datasets, 576 fits and 864 risk records. Development contained 4,608 observations and confirmation 153,600. All teachers were fixed by the design; inference within each condition is over data and numerical randomness at that teacher.

## G.3. Statistical synthesis and limits

For each original confirmatory comparison, first average its two start-specific risks within each dataset and then take the difference between methods. The family of four primary means uses 24 independent dataset differences per condition, sample SE and critical value $t_{23,1-.05/8}=2.7097204201$. Bonferroni does not require the two intermediate-signal comparisons to be independent; the Student-$t$ approximation is not an exact pivot for these risk differences.

|Comparison at $N=2048$|Mean $N\times$ risk difference|Approximate simultaneous interval|
|---|---:|---|
|Weak: group minus $I$|+.169672794|[+.077103286, +.262242302]|
|Intermediate: group minus integrated|-1.173274697|[-1.677668076, -.668881319]|
|Intermediate: group minus $I$|-.293678741|[-.515450570, -.071906912]|
|Stronger: group minus integrated|+1.457811968|[+.967053514, +1.948570422]|

Descriptive mean $N\times$ joint KL, including the separate independent baseline, is:

\begin{center}
\begin{tabular}{lrrrr}
\toprule
Condition & Integrated program & Group program & Matched ridge & $I$ \\
\midrule
$N=2048,S=.05$ & 1.644003 & .194673 & .195492 & .025000 \\
$N=2048,S=2$ & 1.880164 & .706889 & .708474 & 1.000568 \\
$N=2048,S=12$ & 1.424162 & 2.881974 & 2.843391 & 6.020533 \\
$N=256,S=2$ & 1.884548 & .689833 & .686531 & 1.004570 \\
\bottomrule
\end{tabular}
\end{center}

The original two-start untrained baselines are also retained in the saved tables; they are not the $I$ baseline and do not enter the four primary comparisons. Group minus ridge has descriptive scaled means $-.000819$, $-.001585$, $+.038583$ and $+.003302$ in these four conditions. No additional equivalence or superiority test was prespecified for those differences. No teacher-population prevalence, finite-$N$ crossing location, continuous risk-region guarantee or decomposition of the discrepancy from the local constants is inferred.

![Known-channel F program (GROUP) and ML/ridge/IND risks at $N=2048$. Dashed curves are the known-channel F limits in (12); points are means $\pm1$ dataset-level SE across 24 datasets. Shading marks the local-theory advantage shared by F and matched ridge over ML and IND.](figures/local_risk_f_group.png){width=100%}

## G.4. A descriptive condition at a smaller sample size

The $N=256,S=2$ condition uses the same known channel and particle count, but a different teacher from the $N=2048,S=2$ condition because $\eta_N=h/\sqrt N$. Its 24 datasets and numerical samples are independent of the main conditions. It was designated descriptive and does not add a comparison to the four-comparison family.

![Descriptive program risks at $N=256,S=2$ with a known channel. Symbols show means with $\pm1$ dataset SE, averaging the two original-start risks within each of 24 independent datasets. Dashed curves and shading use the known-channel local theory in (12): group posterior and matched ridge share a leading risk below ML and independence between the theoretical endpoints. Only the single $S=2$ condition was evaluated here; the curves do not assert a measured finite-$N$ region. This teacher differs from the $N=2048$ teacher and the data are not nested prefixes. The group stationarity diagnostic passed on 6 of 48 outputs; all 48 remain in the risk summary.](figures/local_risk_n256.png){width=100%}

<a id="budget-reevaluation"></a>

## G.5. Retrospective stronger-signal budget comparison

The original stronger-signal risk direction persists when numerical budgets are increased. This follow-up reuses all 24 datasets at $N=2048,S=12$, the same teacher and both original starts. The condition was selected after viewing the original results; this is a retrospective sensitivity study, not a fresh confirmation of the four-comparison family in G.3. No diagnostic failure or start was removed.

Let $P0$ denote the original outputs and $H$ the higher-budget outputs. A bar denotes the average of the two start-specific joint-KL risks within dataset $i$. Both sets of outputs use a common risk evaluator:

$$
D_i^p=N\{\overline R_{i,\mathrm{group},p}
-\overline R_{i,\mathrm{ML},p}\},\qquad
G_i=D_i^H-D_i^{P0},\qquad p\in\{P0,H\}.
\tag{G.2}
$$

A positive $D_i^p$ means higher group-program risk. The comparison preserves $K=8$, the proposal, the group program's private quadrature order 256 and control-moment order 192, ridge strength, and the complete solver policy in G.2, including the $10^{-7}$ stopping tolerance and last-accepted output rule. In the table below, $R\times B$ means $R$ independent randomizations containing $B$ sets of $K$ particles each.

|Numerical component|Original|Follow-up|
|---|---|---|
|Group fitting, $R\times B$|$2\times512$|$4\times2048$|
|ML/ridge fitting, GH orders|$(20,128)$|$(28,192)$|
|Group independent reference, $R\times B$|$4\times512$|$8\times2048$, fresh randomizations|

All new fitted outputs were frozen before generating the new independent reference. The common risk evaluation compares GH orders $(28,192)$ and $(36,256)$ and uses the latter for both $P0$ and $H$. Thus $G$ does not subtract an old low-order risk estimate from a new high-order one. The mean original difference under this common evaluation is $1.457812$, reported descriptively.

|Quantity, in $N$ times joint KL|Mean|Dataset SE|Approximate simultaneous interval|
|---|---:|---:|---|
|Higher-budget difference $D^H$|1.455840|0.182440|[1.018372, 1.893308]|
|Paired change $G$|-0.001972|0.005289|[-0.014655, 0.010711]|

These two intervals use the 24 dataset differences, $23$ degrees of freedom and critical value $2.3978750647$, with a two-comparison Bonferroni adjustment at level $.05$. That adjustment does not account for selecting the stronger-signal condition after seeing the original study. The intervals retain their retrospective status; they do not replace G.3. The risk direction is stable under this budget change, but the interval for $G$ containing zero is not an equivalence result: no equivalence margin was specified.

The diagnostic counts below distinguish outputs from references. Their unit is a start-specific output, not an independent dataset. The two reference criteria are the mapping and gradient-coordinate SE criteria stated in G.2.

|Group outputs and reference|Passes|Mapping only fails|Gradient SE only fails|Both fail|
|---|---:|---:|---:|---:|
|$P0$, original reference|12/48|2|20|14|
|Same $P0$, new reference|36/48|12|0|0|
|$H$, same new reference|48/48|0|0|0|

With the original outputs held fixed, the new reference removes the SE failures but leaves 12 mapping failures. The higher-budget outputs also pass the mapping criterion under that reference. These observations qualify the original numerical limitation without assigning a unique cause to every error: reference samples and fitted locations change across comparisons, and passing does not certify global optimality. ML and ridge received no new independent gradient-reference check.

All common risk evaluations met the prescribed order-difference gates. The largest KL and PMF differences were $5.46\times10^{-14}$ and $1.12\times10^{-14}$, respectively, against tolerances $4.8828125\times10^{-8}$ and $10^{-8}$. At fixed private order 256, changing control-moment order from 192 to 384 shifted the gradient by at most $4.72\times10^{-16}$. These checks compare the stated numerical rules; they do not bound continuous-integration error or certify the private rule.

<a id="learned-proposals"></a>

## G.6. Learned local proposals and quadratic prediction

This comparison replaces the exact local posterior by a density learned from finite simulator labels, keeping the known channel $M=8$, $\alpha=1.4/\sqrt8$, $\tau=3$ and $K=8$. Each of three independent training pools contains 65,536 local $(X_i,O_i)$ examples and an independent density-validation set of 32,768 examples. The smaller training set is the first 16,384 examples of its pool. Thus the six fitted densities come from three independent pools, not six independent training repetitions.

For each of the 18 local observation states, a two-Gaussian mixture is fitted with defensive mass $.05\phi(x)$, where $\phi$ is the standard normal density. In the raw fit, nondefensive weights, means and standard deviations are constrained to $[.02,.98]$, $[-5,5]$ and $[.15,3]$. A common checkpoint from epochs $0,5,20,80$ minimizes the average validation negative log density of the raw fit and its reflection average. No correlation parameter or candidate contrast enters this choice. The frozen proposal is

$$
q_\varphi(x\mid o)=\tfrac12\{q_{\rm raw}(x\mid o)+q_{\rm raw}(-x\mid\mathcal Jo)\},
\qquad \mathcal J(y,b)=(-y,M-b).
$$

All three groups share this same density table. Reflection averaging retains the $.05$ defensive mass and generally produces four tied Gaussian components. Each factor reads only its own group observation; the construction uses known-channel simulated latent labels, not an encoder learned solely from joint observations.

Write $q_{\varphi,o}(x)=\prod_iq_\varphi(x_i\mid o_i)$ for the product proposal. The target retains the full importance weight:

$$
\begin{aligned}
w_\theta(x,o)&=\frac{\phi_{C(\theta)}(x)\prod_i\ell_i(o_i\mid x_i)}
{\prod_i q_\varphi(x_i\mid o_i)},\\
F_\varphi(\theta;\eta)&=E_{O\sim P_\eta}E_{q_{\varphi,O}^{\otimes K}}
\left[-\log\left\{\frac1K\sum_k w_\theta(X_k,O)\right\}\right].
\end{aligned}
\tag{G.3}
$$

Let $g_\varphi=\nabla_\theta F_\varphi$, $H_\varphi=\nabla^2_{\theta\theta}F_\varphi$ and $B_\varphi=-\partial_\eta\nabla_\theta F_\varphi$, all evaluated at $(0,0)$. With $v=(1,1,0)^\top/\sqrt2$, $t\in\{.1,.2\}$ and the pre-existing exact-proposal value $r_0=.3201223434446316$, define

$$
\begin{aligned}
D(t)&=F_\varphi(r_0tv;tv)-F_\varphi(tv;tv),\\
P(t)&=(r_0-1)t\,g_\varphi^\top v\\
&\quad+t^2\{(r_0^2-1)v^\top H_\varphi v/2-(r_0-1)v^\top B_\varphi v\}.
\end{aligned}
\tag{G.4}
$$

The candidate is fixed before estimating the learned-proposal coefficients. $D<0$ means that the finite objective prefers this shrunken candidate; it is not a risk comparison. Estimates retain the full matrices and nonconstant particle weights.

Independent coefficient estimates use 16 randomizations of 256 particle sets. Their predictions are frozen before a fresh 16-randomization evaluation with 512 sets per randomization. Each set contains eight particles and all $18^3$ observation states are enumerated. The same evaluation banks also provide their own origin derivatives, without fitting coefficients to the finite-$t$ contrasts. For replicate $r$, the two errors are

$$
W_{{\rm ind},r}=\frac{\widehat D_r-\widehat P_{{\rm ind},r}}{t^2},
\qquad
W_{{\rm same},r}=\frac{\widehat D_r-\widehat P_{{\rm same},r}}{t^2}.
\tag{G.5}
$$

Variance reduction centers the uniform log density ratio and a quadratic score-covariance term using analytic moments of the actual frozen Gaussian mixtures. These controls have zero conditional mean under the ideal sampling law. The teacher probabilities and derivatives use the same finite GH36 Cholesky function, with private order 256; consequently the same-bank prediction differentiates the numerical scalar used for $D$.

The comparison family consists of the 12 values of $D$, 12 independent errors and 12 same-bank remainders. Approximate simultaneous 95% intervals use Bonferroni adjustment across these 36 quantities, $15$ degrees of freedom and critical value $3.9112865148$. SE is the sample standard deviation of the 16 replicate differences divided by four: independent streams are paired by index, whereas the same-bank difference retains its shared-input covariance. This uncertainty is conditional on the six frozen proposals, not on a training population. The accuracy criterion requires the whole error interval to lie within $[-.01,.01]$.

The following two tables give every prespecified comparison. Means and interval endpoints for all three quantities are multiplied by $10^3$ and rounded for display; the error tolerance on this scale is $[-10,10]$.

|Pool|$n_{\rm train}$|$t$|$10^3D$: mean [simultaneous interval]|
|---:|---:|---:|---|
|0|16384|0.1|-0.407 [-0.413, -0.401]|
|0|16384|0.2|-1.681 [-1.702, -1.659]|
|0|65536|0.1|-0.408 [-0.412, -0.403]|
|0|65536|0.2|-1.683 [-1.706, -1.660]|
|1|16384|0.1|-0.411 [-0.419, -0.403]|
|1|16384|0.2|-1.694 [-1.725, -1.663]|
|1|65536|0.1|-0.418 [-0.426, -0.409]|
|1|65536|0.2|-1.721 [-1.754, -1.688]|
|2|16384|0.1|-0.408 [-0.413, -0.404]|
|2|16384|0.2|-1.685 [-1.703, -1.667]|
|2|65536|0.1|-0.405 [-0.410, -0.401]|
|2|65536|0.2|-1.671 [-1.687, -1.654]|

|Pool|$n_{\rm train}$|$t$|$10^3W_{\rm ind}$: mean [interval]|$10^3W_{\rm same}$: mean [interval]|
|---:|------:|---:|----------------------------|----------------------------|
|0|16384|0.1|0.150 [-1.281, 1.580]|-0.434 [-0.555, -0.313]|
|0|16384|0.2|-1.158 [-2.573, 0.257]|-1.742 [-2.206, -1.277]|
|0|65536|0.1|-0.651 [-1.118, -0.183]|-0.431 [-0.531, -0.331]|
|0|65536|0.2|-1.960 [-2.505, -1.416]|-1.741 [-2.128, -1.353]|
|1|16384|0.1|-0.673 [-2.259, 0.912]|-0.399 [-0.465, -0.334]|
|1|16384|0.2|-1.888 [-3.464, -0.313]|-1.615 [-1.871, -1.358]|
|1|65536|0.1|-0.896 [-2.603, 0.812]|-0.412 [-0.499, -0.326]|
|1|65536|0.2|-2.149 [-3.892, -0.406]|-1.666 [-2.002, -1.329]|
|2|16384|0.1|-0.362 [-1.033, 0.308]|-0.425 [-0.524, -0.325]|
|2|16384|0.2|-1.653 [-2.367, -0.940]|-1.716 [-2.097, -1.335]|
|2|65536|0.1|-0.221 [-1.227, 0.784]|-0.405 [-0.479, -0.330]|
|2|65536|0.2|-1.448 [-2.504, -0.393]|-1.631 [-1.920, -1.343]|

All 12 candidate-difference intervals are negative. Both kinds of error meet the specified accuracy in every comparison, and all same-bank remainder intervals are also negative. Thus a useful quadratic approximation and a small, resolved finite-amplitude correction coexist. These results do not locate an objective minimizer or extend the exact-posterior risk theorem to learned proposals.

![Independent prediction and finite-amplitude remainder for the same 12 comparisons. Both panels show approximate simultaneous 95% intervals from the 36-quantity family, conditional on six frozen densities from three training pools. The left axis uses the original normalized-error scale; the right repeats the same-bank intervals on an expanded axis whose tick labels carry the displayed $10^{-3}$ multiplier. The shaded band is the prespecified tolerance, not a risk region.](figures/learned_prediction.png){width=100%}

A prespecified ordinary GH28/36 comparison on pool 0's $n_{\rm train}=16384$ proposal, using its complete validation bank, found maximum replicate differences in $D/t^2$ and $W_{\rm same}$ below $10^{-4}$. This check covers one proposal and two numerical rules, not a continuous-integration error bound for all six proposals.

An earlier separate evaluation of these frozen proposals used only the log-density-ratio control and a 24-comparison family; one of its 12 residual intervals met the same tolerance. Those results remain in the research record as an earlier, mostly unresolved numerical evaluation. After control development on the saved banks, fresh coefficient and evaluation streams produced the present 36-comparison family. The earlier results are neither rewritten nor pooled with it.

<a id="adaptation-evidence"></a>

## G.7. Responses when adapted factors can read additional observations

The adaptation comparison uses the same Gaussian model, known observation channel and fixed $K=8$, with teacher $\eta=t(1,1,0)/\sqrt2$. F has no free tilt. L fits all 54 local-state tilt coefficients while each factor reads only its own group's observation. A additionally fits six directed coefficients multiplying other groups' posterior means, while its conditional density remains a product. The exact joint posterior is an analytical oracle, not a fourth fitted program. The response predictions in [Appendix C](APPENDIX_ADAPT_EN.md#adapt-proof) refer to these specified classes and input restrictions.

For homogeneous slopes $\alpha_i=e=1.4/\sqrt8$, the leading response on each edge is $0.320122$ for F/L and $0.432259$ for A. The four saved amplitudes $.025,.05,.1,.2$ give 12 joint stationary points. At $t=.2$, their active-edge responses are $0.318965$, $0.319360$ and $0.430267$, respectively. F and L share a leading coefficient, but their finite-amplitude outputs need not coincide. Eight additional fixed-generator proposal optimizations complete the 20 predeclared targets; these latter points qualify only the proposal coordinates. All targets passed their prespecified ordinary stationarity, curvature and higher-rule checks; the five predeclared endpoints also passed full-state and full-space comparisons. An independent numerical evaluation used eight complete randomized repetitions and a three-comparison family: all intervals for controlled objective minus deterministic value contained zero. These repetitions are numerical calibration units, not training datasets.

Panel (b) below displays the third coordinate of the same 12 $K=8$ points as $2\theta_{23}/t^2$, so its leading coefficient is $\chi_Q$ rather than $\chi_Q/2$. The analytical coefficient (C.14) predicts opposite signs for F/L and A on the pair that is uncorrelated in the teacher. The lines evaluate that coefficient with the independent-point moment; no coefficient is fitted to the markers. Among these 12 points, the three at $t=.2$ are the predeclared joint endpoints with the full-state and full-space comparison. The separate K4 check below was frozen independently.

The heterogeneous comparison fixes $\alpha=(e,0,2e)$. Its second group's count is independent of its latent variable, but its label remains informative. No leaf-exchange restriction is imposed. The three information values are $v=(0.235765,0.170769,0.377527)$, giving the following leading responses:

|Edge|F/L: local information|A: cross-group input|
|---|---:|---:|
|12|0.251274|0.336979|
|13|0.438717|0.599491|
|23|0.355380|0.499799|

At each of $t=.0125,.025,.05$, all three correlations and all proposal coefficients are optimized, giving nine joint targets and 12 predeclared numerical qualification checks. The response figure below displays both nonzero teacher edges; its lines copy the leading coefficients and its symbols copy the saved outputs, with no fitted coefficient or error bar. In particular, a single response based on averaged group information would not represent these two edges. The zero-teacher third coordinate is still optimized. Separately, a homogeneous $K=4$ check at $t=.0125,.025$ gave $2\theta_{23}/t^2$ approximately $.0340213$ and $.0340212$, against the independently derived coefficient $.034021349$; it checks the positive second-order third-edge effect, without fitting an asymptotic coefficient from two points.

These are population numerical branch comparisons, not finite-$N$ risk experiments. Finite GH rules, finite Laplace windows and ordinary order differences do not certify continuous error. The compact-domain theorem has an unknown teacher radius, so it does not certify these particular finite-amplitude stationary points as global minima or prove a continuous branch enclosure.

![Population correlation responses in the three-group Gaussian--logistic model at $K=8$, with teacher $t(1,1,0)/\sqrt2$. Panels (a), (c) and (d) compare leading response predictions with deterministic population stationary branches. Panel (b) shows the opposite signs of the induced third marginal correlation predicted by (C.14), scaled as $2\theta_{23}/t^2$. Homogeneous channels use $\alpha_i=\alpha_0=1.4/\sqrt8$; heterogeneous channels use $(\alpha_0,0,2\alpha_0)$. Numerical design and branch checks are described in this section.](figures/adaptation_response.png){width=100%}

<a id="channel-evidence"></a>

## G.8. Learning a common observation-channel parameter

The channel extension keeps $M=8$, $\tau=3$, the label mechanism and $K=8$ fixed, but estimates a single common slope $\alpha$, with teacher $\alpha_*=1.4/\sqrt8$. Every candidate updates its independent-posterior base; A also updates the posterior-mean features. Complete importance weights retain the candidate marginal normalizers, and the teacher distribution is held fixed. F therefore has no free tilt but is not a proposal frozen along the $\alpha$ direction. Full physical dimensions are 4, 58 and 64. The compact-domain and local expansions in [Appendix D](APPENDIX_CHANNEL_EN.md#channel-proof) concern this same normalized model.

At $t=.1,.2$, each class is compared with its own known-true-channel reference. Those references use oracle channel information. Initial parameters also use their saved values and the predicted second-order channel shift; the actual accepted Newton corrections are recorded separately. These are six pointwise stationary calculations, not a demonstration that an ordinary unknown-channel training program reaches the branch. The table reports saved full-rule values; the two active teacher edges have the same response gain, shown once.

|Class, $t$|$(\alpha-\alpha_*)/t^2$|Active response gain$/t^2$|$\Delta F/t^4$|Sum of three marginal KL$/t^4$|$\Delta$ joint KL$/t^4$|
|---|---:|---:|---:|---:|---:|
|F, .1|0.031507|0.004231|-0.000159537|0.000159418|-0.000274217|
|L, .1|0.031512|0.004231|-0.000159586|0.000159467|-0.000274238|
|A, .1|0.043882|0.008639|-0.000309588|0.000309256|-0.000393349|
|F, .2|0.031403|0.004197|-0.000158895|0.000158424|-0.000272775|
|L, .2|0.031422|0.004198|-0.000159088|0.000158615|-0.000272857|
|A, .2|0.043675|0.008559|-0.000307817|0.000306507|-0.000391327|

Here $\Delta$ means free-channel minus fixed-true-channel within the same class. The marginal column is a sum, not a single-group KL or average. All six main/high full-coordinate stationarity checks and all 11 predeclared high-rule calculations passed. Objective signs use the conservative maximum of the paired 24/32-rule difference and the sum of the two pointwise order differences; KL signs use the paired difference formed separately at orders 48 and 64. Each negative sign exceeded ten times its corresponding sensitivity, with the constituent qualifications passed. Marginal KL increased while complete joint KL decreased in these six comparisons. This is neither a universal risk improvement nor exact equality of finite-amplitude values with their fourth-order leading terms; two amplitudes alone do not establish an order.

The derivative implementation was separately calibrated at every local state, three channel positions and three tilts, using fixed fourth-order stencils. Its maximum primary first- and second-derivative discrepancies were $5.06\times10^{-12}$ and $3.95\times10^{-12}$. A preceding second-order-stencil evaluation failed its frozen tolerance and remains a failed development record. The new qualification does not change that result or supply a continuous integration bound.

The unknown-channel risk result in [Appendix E](APPENDIX_RISK_EN.md#risk-proof) concerns a different object: the exact-bank-expectation empirical M-estimator under $\eta_N=h/\sqrt N$. A deterministic arithmetic check of all 4/58/64 coordinates under the two saved moment rules verifies the observed-score covariance, actual likelihood Hessian, complete Schur elimination and sandwich covariance. At $K=8$, F/L give leading risk $0.012846642\|h\|^2+0.653717472$ and A gives $0.008958357\|h\|^2+0.780271950$ for joint or same-data efficient marginal plug-in. The oracle-channel intercepts are lower by $.5$; the three correlation coordinates contribute $0.153717472$ and $0.280271950$, respectively, and the common channel contributes $.5$. The deterministic $18^3$ summation nodes are not independent datasets. These checks do not add finite-$N$ simulations or extend this theorem to random-bank optimization outputs.

<a id="execution-evidence"></a>

## G.9. Numerical execution and comparison with the population reference

Finite stochastic programs are compared separately with the population stationary points. The known homogeneous channel and teacher $t=.2$ are fixed throughout this evidence. An initial selected-learning-rate panel's nine outputs were decomposed under one full-state GH24 rule: output minus same-class reference equals a fixed-generator proposal-improvement term plus a remaining generator term. F has no proposal term. All nine outputs are retained below, with differences multiplied by $10^6$.

|Class, repetition|Proposal term|Remaining generator term|
|---|---:|---:|
|F, 0|0|127.223506|
|F, 1|0|20.212571|
|F, 2|0|72.268078|
|L, 0|89.968349|120.662749|
|L, 1|85.639404|19.023678|
|L, 2|55.722146|72.855795|
|A, 0|203.958614|174.207850|
|A, 1|193.098093|61.323957|
|A, 2|73.621592|157.129584|

The proposal optimizations qualify all free tilt coordinates with the generator fixed; they are not full joint-stationarity or global-regret certificates. GH16 development mass checks failed, and the predeclared GH24 correction rule was used without relaxing that gate. All six proposal optimizations passed GH24 qualification. The five predeclared repetition-0 raw/profile points also passed full GH32 checks; repetitions 1 and 2 have no pointwise GH32 qualification. This is a selected old-panel diagnostic, not a new independent training sample or a population error-allocation proportion.

A separate fixed-point noise experiment keeps full generator--proposal cross-covariances. It uses three independent numerical repetitions, each with 256 observations and eight independent $K=8$ banks per observation. At the A joint reference, the following entries are means across the three complete repetitions; the uncertainty is sample SD, not SE or a confidence interval. The Newton column is a local Schur-based sensitivity for 32 observations and one bank per observation, not an Adam risk prediction. Here $p$ controls the proposal-score term and $g$ the generator term.

|Controls|Particle-gradient covariance trace|Local Newton sensitivity, mean $\pm$ SD|
|---|---:|---:|
|$p0g0$|0.790786|0.725514 $\pm$ 0.018777|
|$p0g1$|0.446280|0.561223 $\pm$ 0.048102|
|$p1g0$|0.377110|0.958962 $\pm$ 0.028356|
|$p1g1$|0.032603|0.318073 $\pm$ 0.018189|

Proposal-only control reduces the particle-gradient trace while increasing this effective Newton sensitivity. At the independent teacher anchor, exact current posterior moments explain the generator--proposal noise cancellation; the finite nonzero-teacher table is a separate numerical diagnostic. Negative eigenvalues of finite-sample observation-noise component estimates are retained without clipping. Four-control conditional-mean equality does not imply equality of their optimization trajectories.

A fresh program comparison uses ten distinct control variants, four independent complete repetitions, 1,500 Adam updates and a common nominal rate $.001$, with 32 observations and one $K=8$ bank per update. All programs start at zero, without the population reference as initialization. The same repetition shares inputs across controls. Outputs are fixed at 500 steps, 1,500 steps and the physical-parameter average over steps 1001--1500; prefixes and averages are not additional repetitions. All 40 runs, 60,000 updates, 120 outputs and 132 paired evaluations passed their program/numerical/feasibility checks, with no blocked update or boundary hit. This does not certify output stationarity. The comparison is a direct replacement at the same nominal rate and update/sample budget, not separately tuned best performance or equal elapsed time.

The complete matched-control A-minus-L response increments are:

|Output|Controls|Edge 12, mean $\pm$ SD|Edge 13, mean $\pm$ SD|
|---|---|---:|---:|
|500 steps|$p0g0$|0.004781 $\pm$ 0.011284|0.009189 $\pm$ 0.009219|
|500 steps|$p0g1$|0.013637 $\pm$ 0.008737|0.017013 $\pm$ 0.016175|
|500 steps|$p1g0$|0.018112 $\pm$ 0.014471|0.029997 $\pm$ 0.018318|
|500 steps|$p1g1$|0.036184 $\pm$ 0.016586|0.049974 $\pm$ 0.020967|
|1500 steps|$p0g0$|0.039883 $\pm$ 0.012500|0.051019 $\pm$ 0.017201|
|1500 steps|$p0g1$|0.050910 $\pm$ 0.023755|0.071507 $\pm$ 0.038128|
|1500 steps|$p1g0$|0.063486 $\pm$ 0.016722|0.089163 $\pm$ 0.029773|
|1500 steps|$p1g1$|0.084020 $\pm$ 0.029922|0.106189 $\pm$ 0.018148|
|1001--1500 average|$p0g0$|0.033041 $\pm$ 0.002495|0.038504 $\pm$ 0.017196|
|1001--1500 average|$p0g1$|0.043199 $\pm$ 0.015730|0.052532 $\pm$ 0.034944|
|1001--1500 average|$p1g0$|0.055779 $\pm$ 0.012034|0.083058 $\pm$ 0.026503|
|1001--1500 average|$p1g1$|0.081145 $\pm$ 0.020673|0.105180 $\pm$ 0.013075|

SD uses four complete independent repetitions; descriptive SE equals SD/2. The deterministic A-minus-L reference is $.110906$ on both active edges. Longer execution increases the increment even without controls, while controls also affect it. The dual-control average remains short of that reference and no equivalence margin or test was specified. Objective-gap point estimates do not give a uniform ranking of the controls, and signed negative realized gaps were retained. A saved single-case evaluation SE is not the SE of output minus reference; the across-repetition gap SD includes both training and evaluation noise. These program results therefore support a bounded execution comparison, not a finite-$N$ statistical-risk claim or a ranking of optimizers.

<a id="adapted-risk-evidence"></a>

## G.10. Independent finite-sample risk comparison of adapted proposals L and A

The main-text comparison between adapted proposals `L` and `A` uses an independent confirmatory panel of $R=64$ datasets per condition across four prespecified $(N,S)$ conditions ($256$ datasets and $512$ final method outputs total), keeping the known homogeneous channel $M=8$, $\alpha_i=1.4/\sqrt8$, $\tau_i=3$, spectral floor $\delta=.15$, and particle count $K=8$ ($j=.05558506$, $r_L=.32012234$, $r_A=.43225914$). Under the chain teacher $\eta_N=t(1,1,0)/\sqrt2$ with $t=\sqrt{S/(jN)}$, the nonzero teacher correlation coordinate $t/\sqrt2$ equals $.04686256$ (`c0`: $N=2048,S=0.5$), $.13254733$ (`c1`: $N=2048,S=4$), $.02343128$ (`c2`: $N=8192,S=0.5$), and $.06627366$ (`c3`: $N=8192,S=4$). An earlier $R=12$ development panel was used solely to select the confirmatory replication count $R=64$ and is excluded from all confirmatory statistics.

Under the known channel, each dataset aggregates the $N$ independent observations into state counts across the $18^3=5{,}832$ observation states; the conditional $K$-particle log-weight expectation is evaluated by deterministic numerical quadrature (active order $24$, log-$t$ window $[-28,14]$, Legendre order $20$, normalizer order $192$, private order $256$) and weighted by `counts[o]/N` (without random particle banks). On each dataset, `L` ($57$ parameters: $3$ correlations and $54$ local state tilts in $[-.15,.15]$) and `A` ($63$ parameters: $3$ correlations, $54$ local state tilts, and $6$ directed cross-group coefficients in $[-.25,.25]$) are fitted by projected damped Newton under prespecified multi-start rules: `L` runs in fixed order from the origin $(\theta,u)=(0,0)$ and the projected data origin-score moment start (which sets $\theta$ to the empirical origin score divided by the public origin Fisher information diagonal and $u=0$), whereas `A` runs in fixed order from the embedded `L` solution ($a=0$) and the origin while permanently retaining the embedded `L` solution as a feasible candidate. Final method outputs are selected in fixed order solely by the main-rule empirical training objective under a frozen selection tolerance, never using true risk or teacher parameters. Across the $256$ datasets, all $512$ final `L`/`A` method outputs ($256\text{ datasets}\times 2\text{ arms}$) converged to interior points with zero fallbacks, zero boundary hits, and zero failed or unknown offline numerical qualifications (evaluated at order $32$ and expanded probe window $[-32,16]$ with Legendre order $28$).

For each dataset $r\in\{1,\dots,64\}$ in a condition, true risk is evaluated offline on the full $5{,}832$-state observation support under the normalized GH-64 generative PMF $P_N$ (with GH-48 as a sensitivity check), giving the paired scaled joint-KL risk difference $\Delta_{N,r}:=N(\mathrm{KL}_{64,A,r}-\mathrm{KL}_{64,L,r})$. Let $s(O)=\bigl(\mu_0(O_0)\mu_1(O_1),\,\mu_0(O_0)\mu_2(O_2),\,\mu_1(O_1)\mu_2(O_2)\bigr)$ denote the unprojected origin correlation score formed from the public zero-correlation feature table $\mu_i(o_i)$, and let $\bar s_r=N^{-1}\sum_{n=1}^N s(O_{r,n})$ be its empirical average on dataset $r$. With fixed theoretical constants $d=r_A-r_L$, $q=r_A+r_L$, and linear score proxies $\widetilde\theta_{Q,r}=(r_Q/j)\bar s_r$, the prespecified offline control variate is $C_{N,r}=N\bigl\{\tfrac{dq}{2j}\|\bar s_r\|^2-d\,\eta_N^\top\bar s_r\bigr\}=\tfrac12 Nj\bigl(\|\widetilde\theta_{A,r}-\eta_N\|^2-\|\widetilde\theta_{L,r}-\eta_N\|^2\bigr)$, and its exact expectation under the normalized GH-64 generative PMF $P_N$ is $c_{N,64}:=\mathbb E_{P_N}[C_N]=\tfrac{dq}{2j}\bigl(\operatorname{tr}V+N\|m\|^2\bigr)-dN\eta_N^\top m$, where $m=\mathbb E_{P_N}[s(O)]$ and $V=\operatorname{Cov}_{P_N}(s(O))$ ($c_{N,64}=+.091573$ for `c0`, $-.153313$ for `c1`, $+.091577$ for `c2`, and $-.153268$ for `c3`; GH-48/64 center differences are below $10^{-6}$). The control variate $C_{N,r}$ is used solely for offline mean risk-difference estimation with fixed coefficient $\beta=1$ and never enters `L` or `A` fitting: $\Delta_N^{\rm ctrl}:=c_{N,64}+\overline E$, where $E_{N,r}:=\Delta_{N,r}-C_{N,r}$ (reducing Monte Carlo standard errors by $9.8\times$--$38.3\times$ relative to raw $\Delta_N$). The primary analysis uses 4-condition Bonferroni--Student approximate simultaneous 95% intervals ($df=63$, critical value $t_{63,1-.05/8}=2.5714412$); raw paired results use the same $64$ datasets per condition and are reported as unadjusted comparisons.

\begin{center}
\small
\setlength{\tabcolsep}{4.5pt}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lrrrcrcc}
\toprule
Cond. & $N$ & $S$ & $\Delta_\infty(S)$ & Primary $\Delta_N^{\rm ctrl}$ (MCSE) & Primary 95\% CI & Raw $\Delta_N$ (MCSE) & Raw 95\% CI \\
\midrule
\texttt{c0} & 2048 & 0.5 & $+.091578$ & $+.091217$ (.000596) & $[+.089686, +.092749]$ & $+.073199$ (.012269) & $[+.041651, +.104747]$ \\
\texttt{c1} & 2048 & 4.0 & $-.153253$ & $-.152055$ (.001279) & $[-.155344, -.148766]$ & $-.146759$ (.012587) & $[-.179125, -.114393]$ \\
\texttt{c2} & 8192 & 0.5 & $+.091578$ & $+.090977$ (.000620) & $[+.089384, +.092570]$ & $+.107453$ (.013720) & $[+.072173, +.142732]$ \\
\texttt{c3} & 8192 & 4.0 & $-.153253$ & $-.152090$ (.000522) & $[-.153432, -.150749]$ & $-.138330$ (.019991) & $[-.189735, -.086924]$ \\
\bottomrule
\end{tabular}%
}
\end{center}

To examine the bias--variance structure, each arm's scaled quadratic Fisher risk $Q_{Q,r}:=\tfrac12 Nj\|\widehat\theta_{Q,r}-\eta_N\|^2$ satisfies the exact finite-$R$ sample identity $\overline Q_Q = B_Q + V_Q$, where $B_Q:=\tfrac12 Nj\|\overline{\widehat\theta_Q-\eta_N}\|^2$ is the plug-in squared estimation bias ($B_Q^{\rm deb}=B_Q-V_Q/(R-1)$ is the debiased estimate) and $V_Q:=\tfrac{R-1}{R}\tfrac12 Nj\operatorname{tr}(\widehat\Sigma_Q)$ is the trace of the scaled sample parameter covariance ($63/64=.984375$). Consequently, $\Delta B_N+\Delta V_N=\overline Q_A-\overline Q_L$ decomposes the unadjusted mean quadratic risk difference (which matches the unadjusted raw mean $\Delta_N$ up to $\overline E_2$, rather than the control-adjusted primary mean $\Delta_N^{\rm ctrl}$):

\begin{center}
\small
\setlength{\tabcolsep}{4.5pt}
\resizebox{\linewidth}{!}{%
\begin{tabular}{llrrrrrr}
\toprule
Condition & Arm & $N\overline R_Q$ & $\overline Q_Q$ & $B_Q$ ($B_Q^{\rm deb}$) & $V_Q$ & $\Delta B_N$ (theory) & $\Delta V_N$ (theory) \\
\midrule
\texttt{c0} (2048, 0.5) & \texttt{L} & .246457 & .246862 & .116458 (.114388) & .130404 & $-.032587$ ($-.034976$) & $+.106238$ ($+.126554$) \\
\texttt{c0} (2048, 0.5) & \texttt{A} & .319656 & .320513 & .083871 (.080115) & .236641 & & \\
\texttt{c1} (2048, 4.0) & \texttt{L} & 1.152362 & 1.150857 & 1.005658 (1.003354) & .145199 & $-.261188$ ($-.279808$) & $+.114474$ ($+.126554$) \\
\texttt{c1} (2048, 4.0) & \texttt{A} & 1.005603 & 1.004144 & .744471 (.740349) & .259673 & & \\
\texttt{c2} (8192, 0.5) & \texttt{L} & .250494 & .250342 & .075022 (.072240) & .175319 & $-.035650$ ($-.034976$) & $+.142996$ ($+.126554$) \\
\texttt{c2} (8192, 0.5) & \texttt{A} & .357946 & .357687 & .039373 (.034320) & .318315 & & \\
\texttt{c3} (8192, 4.0) & \texttt{L} & 1.092465 & 1.092658 & .923773 (.921092) & .168885 & $-.276754$ ($-.279808$) & $+.138952$ ($+.126554$) \\
\texttt{c3} (8192, 4.0) & \texttt{A} & .954136 & .954856 & .647019 (.642132) & .307838 & & \\
\bottomrule
\end{tabular}%
}
\end{center}

Across all four conditions, the control-variate residual decomposes as $E_{N,r}=\Delta_{N,r}-C_{N,r}=E_{1,r}+E_{2,r}$, where $E_{1,r}:=(Q_{A,r}-Q_{L,r})-C_{N,r}$ is the difference between the fitted quadratic risk difference and the linear score proxy, and $E_{2,r}:=N(\mathrm{KL}_{64,A,r}-\mathrm{KL}_{64,L,r})-(Q_{A,r}-Q_{L,r})$ is the difference between the numerical joint-KL contrast and its Fisher-quadratic counterpart. In `c0`--`c3`, $\overline E$ equals $-.000355$, $+.001258$, $-.000600$, and $+.001178$ (matching $\Delta_N^{\rm ctrl}-c_{N,64}$ identically), which decomposes into $\overline E_1=+.000097$, $+.001304$, $-.000707$, and $+.001706$ (MCSEs $.000778$, $.001766$, $.000787$, and $.000793$) plus $\overline E_2=-.000452$, $-.000046$, $+.000107$, and $-.000528$ (MCSEs $.000223$, $.000527$, $.000180$, and $.000296$). These components quantify the fitted quadratic deviations and nonquadratic KL terms that remain after score control.

<a id="structure-validation"></a>

## G.11. Continuous Gaussian checks of directed access and output blocks

A four-group continuous Gaussian model checks the access and partition predictions beyond the local/all-input endpoints. Observations satisfy $O_i=X_i+\sqrt{D_i}\epsilon_i$, with independent standard Gaussian noise, $v=(.25,.4,.6,.75)$ and $D_i=(1-v_i)/v_i$. All six generating correlations are free. In edge order $12,13,14,23,24,34$, the teacher is $\eta=t(1,-.6,.5,.8,-.4,1.1)/\sqrt{3.62}$, for $t\in\{.08,.04,.02\}$.

Seven structures are fixed. The four product structures have respective input sets $1|2|3|4$ (local), $12|23|34|14$ (directed ring), $124|123|234|134$ (bidirectional cycle), and all four observations for every factor. Two block structures use outputs $12|34$, reading either their own block observations or all observations. Full joint output reads all four observations. Input sets are ordered by output factor or block. This design separates directed access, bidirectional access and joint output effects in one connected four-group system.

**Actual $K=1$ profiles.** For each structure and amplitude, Gaussian entropy maximization and regression eliminate the proposal means and block covariances exactly, as proved in Appendix A. BFGS then optimizes the six generating correlations from zero. Thus the nuisance profile covers all densities obeying the specified block and access restrictions, while the outer generating fit receives numerical local stationarity checks. The table compares leading responses with the fitted ratios $\widehat\theta_{ij}/\eta_{ij}$ at $t=.02$.

|Structure|12|13|14|23|24|34|
|---|---:|---:|---:|---:|---:|---:|
|Local product|0.1000 / 0.1000|0.1500 / 0.1500|0.1875 / 0.1875|0.2400 / 0.2400|0.3000 / 0.3000|0.4500 / 0.4500|
|Directed ring|0.1429 / 0.1425|0.1500 / 0.1492|0.2000 / 0.1994|0.3750 / 0.3741|0.3000 / 0.2976|0.6429 / 0.6428|
|Two-way cycle|0.1818 / 0.1811|0.1500 / 0.1460|0.5000 / 0.4984|0.5000 / 0.4989|0.3000 / 0.2944|0.8182 / 0.8180|
|All-input product|0.1818 / 0.1801|0.3333 / 0.3287|0.5000 / 0.4965|0.5000 / 0.4976|0.6667 / 0.6615|0.8182 / 0.8177|
|Local-input blocks|1.0000 / 0.9986|0.1500 / 0.1468|0.1875 / 0.1851|0.2400 / 0.2386|0.3000 / 0.2971|1.0000 / 0.9999|
|All-input blocks|1.0000 / 0.9976|0.3333 / 0.3254|0.5000 / 0.4949|0.5000 / 0.4967|0.6667 / 0.6606|1.0000 / 0.9998|
|Full joint|1.0000 / 1.0000|1.0000 / 1.0000|1.0000 / 1.0000|1.0000 / 1.0000|1.0000 / 1.0000|1.0000 / 1.0000|

Each entry is prediction / fitted response. All seven smallest-amplitude endpoints satisfy the fixed gradient, covariance-interior, positive local-Hessian and response-error criteria. The maximum response discrepancy is $.00791042$, below the $.025$ tolerance. Directed ring access leaves edges 13 and 24 at their local leading responses; bidirectional cycle access raises the other four. With all inputs already present, joining outputs 12 and 34 gives unit leading responses on those two edges while preserving the other four leading responses. The finite-amplitude fits include higher-order changes, visible in the table.

Across the three amplitudes, 20 of 21 endpoints satisfy the generating-gradient threshold $10^{-8}$. The local-input block endpoint at $t=.08$ has maximum gradient $1.3047166\times10^{-8}$ and remains outside that criterion. All 21 covariance matrices are interior and their local profile Hessians are positive definite. These local checks neither locate a global outer optimum nor turn the unqualified endpoint into a qualified stationary point.

**Fixed-$K$ tangent gap curvature.** A separate calculation evaluates the complete negative log mean posterior/proposal ratio along the prescribed Gaussian tangent paths at the independent teacher. The exact observation-likelihood curvature $J$ is separated analytically; the measured gap curvature is compared with $z^\top R_Qz/K$. Six coordinate directions and one connected direction, seven structures and $K\in\{1,2,8\}$ give 147 comparisons. Each uses eight independently scrambled Sobol integrations of 8192 points, with shared inputs for steps $\pm.04$ and $\pm.02$; Richardson extrapolation removes the leading symmetric-difference step error.

|Particle count $K$|Maximum absolute discrepancy among 49 comparisons|Maximum simultaneous interval half-width|
|---:|---:|---:|
|1|.000278012|.001526665|
|2|.002272115|.012243182|
|8|.002782353|.012144070|

All 147 Bonferroni--Student approximate simultaneous 95% intervals ($df=7$) contain the corresponding residual prediction. All comparisons meet the fixed absolute-discrepancy criterion $.004+6\mathrm{SE}$ and half-width criterion $.05$. Maxima in the two columns can occur in different comparisons. The independent units are the eight numerical scrambles; shared inputs do not make the 147 contrasts independent experiments. Their intervals quantify scramble variability and do not cover residual finite-step bias or floating-point error. These are tangent-path gap-curvature checks, rather than fitted $K>1$ profiles or empirical generating-response measurements.
