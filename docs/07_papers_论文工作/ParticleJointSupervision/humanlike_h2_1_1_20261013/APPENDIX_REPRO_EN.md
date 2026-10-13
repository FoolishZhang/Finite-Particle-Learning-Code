# Appendix H. Experimental protocols and reproducibility

## H.1. New-data comparison: risk and the original score

The high-dimensional controlled learning comparison uses the four-active Gaussian-logistic teacher in (A.21), 256 independent fair nuisance observations, and known zero cross-covariance. Active means and unit variances remain fixed; all six active correlations are learned from $C=I$ in $\mathcal E_{.1}^{(4)}$. The experiment draws 16 new datasets of $N=4096$ complete iid observations, with two independent training-prior quartets per dataset. These are fresh realizations of one specified synthetic law, not new domains. The target is complete-observation KL, in nats, without dimension normalization:

$$
\mathcal R(\theta)=\sum_{y\in\{0,1\}^4}P(y)\log\frac{P(y)}{Q_\theta(y)}.
\tag{H.1}
$$

Nuisance factors cancel from (H.1), but not from the random-particle score. Training uses the unchanged official `SCBLoss.compute_concept_loss` [@scbmcode], float32 inputs and autograd, $K=100$, batch size 64, learning rate .02, Adam $(.9,.999)$ and offset $10^{-8}$. Data cycle without reshuffling; there is no head, feature network, precision penalty, clipping, schedule or early stopping. Each observation receives private fresh prior particles at every step. All programs share the generated inputs but have their own parameters and optimizer states.

IN4 averages four K100 gradients at one state before each of 512 updates. OUT4 runs four independent 512-update paths and averages their physical TAIL correlation vectors into one Gaussian model, not a prediction mixture. SEQ4 evaluates each successive gradient after its previous update: four ordered microsteps per coarse step, each at the original learning rate .02. FACT uses bank zero's active samples and the active finite-K score plus $256\log2$; it is a quality reference, not exact likelihood or an equal-budget competitor.

|Program|Scoring calls / quartet|Updates / quartet|Paths|
|---|---:|---:|---:|
|IN4|2048|512|1|
|OUT4|2048|2048|4|
|SEQ4|2048|2048|1|
|FACT|512|512|1|

TAIL averages the last 64 post-update physical states; for SEQ4 these are the last 64 **coarse-step end states**. OUT4 additionally averages its four path outputs. Across 32 quartets each raw strategy makes 65,536 scoring calls. Matching calls does not match per-path optimization time, Adam's time index, parameter displacement or end-to-end cost, and particles are not pooled into K400. Mean risk is .0047217613 / .0019111960 / .0041083895 / .0008839913 for IN4 / OUT4 / SEQ4 / FACT; untrained I has risk .0027110543.

After a proposed correlation update, compute $l=\lambda_{\min}(C^{\rm prop})$ in float64 and use

$$
C^{\rm repaired}=I+\alpha(C^{\rm prop}-I),\qquad
\alpha=1\ (l\ge .1),\quad \alpha=\frac{1-.1001}{1-l}\ (l<.1).
\tag{H.2}
$$

The actual float32 matrix must pass the .1 floor and native Cholesky checks. Adam moments are retained. This is a radial repair with a safety-gap discontinuity, not a nearest-point projection.

The independent outer unit is the dataset. Risks first average its two training outputs. Each dataset also has eight independent evaluation groups of 512 K100 banks; raw P differences average those groups and its two outputs. Evaluation enumerates the 16 active patterns using a qualified numerical truth PMF and canonical fair nuisance labels justified by expectation symmetry. The 16 outer units cover data/training randomness and, for scores, evaluation randomness. For family size $q$, allocation $\alpha_f$ and $L$ independent paired units, the original development interval is

$$
\bar d\ \pm\ t_{L-1,\,1-\alpha_f/(2q)}s_d/\sqrt L,
\qquad s_d^2=(L-1)^{-1}\sum_s(d_s-\bar d)^2.
\tag{H.3}
$$

The risk family uses $q=5,\alpha_f=.025$; the score family $q=2,\alpha_f=.025$; both have $L=16$. Risk SE targets are .0005 for between-program contrasts and .0002 against I; score targets are .0005. These are development Student approximations, not exact coverage certificates, and exclude integration uncertainty. **All seven original contrasts follow in units $10^{-3}$.**

|Contrast|Mean|SE|Original interval|SE pass|
|---|---:|---:|---|---|
|Risk / OUT4 − IN4|-2.810565|0.482752|[-4.396906, -1.224224]|Yes|
|Risk / SEQ4 − OUT4|2.197194|0.443207|[0.740799, 3.653588]|Yes|
|Risk / OUT4 − I|-0.799858|0.129247|[-1.224570, -0.375146]|Yes|
|Risk / SEQ4 − I|1.397335|0.409976|[0.050139, 2.744532]|No|
|Risk / FACT − I|-1.827063|0.111457|[-2.193314, -1.460812]|Yes|
|Raw P / OUT4 − IN4|1.953277|0.219853|[1.329636, 2.576918]|Yes|
|Raw P / SEQ4 − OUT4|-0.808023|0.282956|[-1.610665, -0.005381]|Yes|

All five risk directions pass, but SEQ4−I fails precision: .000409976 exceeds .0002. Both score directions and precision targets pass; the SEQ4−OUT4 score upper endpoint is only −.0000053808. No extra runs were added to close either boundary. OUT4−I supports mean improvement in this task, not uniform or complete recovery. FACT remains better: subtracting the original simultaneous intervals for OUT4−I and FACT−I gives the **post-hoc derived** OUT4−FACT mean .0010272047 and range [.0002362414, .0018181681]. This is not a new narrower test or SE gate.

For the classical identity $F_P=H(P)+\mathcal R+J_P$, the OUT4−IN4 and SEQ4−OUT4 bias differences are +.0047638424 and −.0030052162. Subtracting the original score/risk rectangle endpoints gives [.0025538601, .0069738247] and [−.0052642523, −.0007461801]. No independence is assumed and SEs are not added. This expected-log-bias accounting does not decompose variance or explain Adam causally.

## H.2. Joint active and nuisance scale learning

A separate study draws 16 new datasets from the same mixed teacher. Every program now learns four means, four active standard deviations and six correlations. FIXED holds nuisance scale $\tau=1$; FREE learns one common nuisance scale; ZERO directly evaluates the active K100 score plus $256\log2$. ZERO retains active finite-K bias and is not an exact MLE. NEAR and UNIT differ only in initial active scales .1 and 1. All start at $C=I$; each mean is initialized from the same dataset as $\operatorname{clip}(\operatorname{logit}[(n_i+.5)/(N+1)],-2,2)$. FREE starts at $\tau=1$.

There are 96 private programs, each with 512 steps, the optimizer and input pairing above, physical box constraints $\mu\in[-2,2]^4$, $s\in[.001,2]^4$, $\tau\in[0,2]$, and (H.2). Box clipping and correlation repair retain Adam moments. These are physical-coordinate updates, not softplus scale optimization. Unlike H.1, the primary TAIL averages **means and full covariance**, not the separate scale/correlation factors:

$$
\bar\mu=\frac1{64}\sum_{j=449}^{512}\mu_j,\quad
\bar\Sigma=\frac1{64}\sum_{j=449}^{512}\operatorname{diag}(s_j)C_j\operatorname{diag}(s_j),\quad
\bar s_i=\sqrt{\bar\Sigma_{ii}},\quad
\bar C=\operatorname{diag}(\bar s)^{-1}\bar\Sigma\operatorname{diag}(\bar s)^{-1}.
\tag{H.4}
$$

The nuisance scale is averaged separately. This specifies a single Gaussian output. Convexity of $\Sigma\succeq.1\operatorname{diag}\Sigma$ preserves its correlation floor. No LAST or checkpoint risk integrals were selected afterward in this joint-scale study. The six primary contrasts share $\alpha_f=.05,q=6,L=16$, with SE target .0002; all expected directions are negative. Values below use $10^{-3}$ units.

|Initialization / risk contrast|Mean|SE|Original interval|Effect pass|
|---|---:|---:|---|---|
|NEAR / FREE − FIXED|−.077630|.048803|[−.225811, .070551]|No|
|NEAR / FREE − product|.612683|.119274|[.250534, .974831]|No|
|NEAR / ZERO − product|−.661155|.166211|[−1.165817, −.156492]|Yes|
|UNIT / FREE − FIXED|−.363094|.073796|[−.587160, −.139028]|Yes|
|UNIT / FREE − product|.329162|.155077|[−.141695, .800018]|No|
|UNIT / ZERO − product|−1.147031|.141899|[−1.577875, −.716186]|Yes|

All six precision requirements pass; three effect requirements pass. The true product-marginal baseline has risk .0027110543 and is the minimum over independent observation models. Mean FIXED/FREE/ZERO risks are .003401367/.003323737/.002049900 for NEAR and .003403310/.003040216/.001564023 for UNIT. Mean FREE nuisance scales are .0457988 and .0452287. Small nuisance scale has not established net risk recovery: NEAR FREE is worse than true product, and UNIT FREE is unresolved. ZERO improves on product at both initializations. Derived FREE−ZERO ranges are [.0004070266,.0021406482] and [.0005744919,.0023778931], using the original rectangle, without new tests.

FREE has not lost all dependence. The identity

$$
D(P\Vert\textstyle\prod_i Q_i)=D(P\Vert\prod_i P_i)+\sum_iD(P_i\Vert Q_i)
\tag{H.5}
$$

gives each output's own-product reference. FREE's mean joint log-score gain over its own product is positive: .0000838938 / .0003710605. This descriptive gain is neither mutual information nor a causal share, and does not replace the prespecified true-product comparisons. Training-bank collision is also descriptive; it does not certify entry into a population curvature window. The parameterization caveat in A.9 applies, so these trajectories do not establish a universal trap.

Numerical qualification must be separated from risk inference. Before science, manufactured fixtures passed five of six native32/math64 value-gradient checks: NEAR_FREE's common-$\tau$ gradient difference $4.5489\times10^{-5}$ exceeded $4\times10^{-5}$. That failure was retained and the study evaluated the actual native32 program. The 384 prespecified training-state controls all passed; their largest loss/gradient differences were $3.11738\times10^{-5}$/$3.23815\times10^{-5}$. They do not retroactively pass the manufactured fixture.

## H.3. Finite active-scale activation without fitting

The finite-curvature study uses four fixed teachers with means (A.21): independent $C=I$, the mixed teacher, equicorrelation .4, and signed $C=.15I+.85qq^\top$, $q=(1,1,-1,-1)$. Each uses the direction (A.40), $s=ta$, and $\mu=\operatorname{logit}p$. For the null teacher the analytic $R_Y=I$ uses $v=(1,1,1,1)/2$. Active scale changes the observation model; nuisance extension preserves its risk. Prespecified doses are $t=0,.1,.2,.4$, with primary $t=.2$; $K=100$ and $n=0,4,16,64,256$ share prior inputs. There are no newly sampled observations or fitted parameters.

Sixteen independent prior streams contain 2048 banks each. All 16 active patterns are weighted by the normalized GH40 teacher PMF. Eight primary raw contrasts use (H.3), $q=8,\alpha_f=.025,L=16$, conditional on these fixed numerical teachers; SE targets are .00015. Null contrasts have no directional requirement. The entire family follows in $10^{-3}$ units; KL differences are reference integrals without sampling intervals.

|Teacher|n|Raw difference|Original interval|Risk difference|
|---|---:|---:|---|---:|
|Independent|0|.061887|[.060973, .062801]|.014367|
|Independent|256|3.290798|[3.203203, 3.378392]|.014367|
|Mixed|0|−.151197|[−.152543, −.149851]|−.201177|
|Mixed|256|3.195826|[3.113491, 3.278160]|−.201177|
|Equicorrelation|0|−.790653|[−.793718, −.787588]|−.846886|
|Equicorrelation|256|3.023317|[2.936012, 3.110623]|−.846886|
|Signed|0|−1.775128|[−1.781811, −1.768444]|−1.842596|
|Signed|256|2.686154|[2.601506, 2.770802]|−1.842596|

All six dependent-teacher directions and eight SE targets pass. Independent activation raises risk; its exact second-order risk curvature is zero, so the finite-dose result is not contradictory. Nuisance collision means at the five n values are .010000, .018767, .085491, .394337, .688640. The four nonzero-n contrasts have a separate Hoeffding family, $q=4,\alpha_f=.025$, 32,768 bounded bank values and common half-width .0092879515 (clipped to $[1/K,1]$); n=0 is exactly $1/K$. Mixed/equicorrelation/signed thresholds $1-1/B$ are .0430900/.1524316/.2794412. By n=64 the collision intervals exceed all three. Mapping one interval through (A.41) creates no new test. This bounded-variable family and the development-t family cannot be described together as an exact joint certificate; neither covers numerical PMF error. Other doses and n cells remain descriptive.

## H.4. Original six-correlation learning and direction support

The earlier six-correlation study fixes means/active variances and compares FIXED, FREE with **256 separate softplus nuisance scales**, and FACT. These names must not be confused with H.2's one shared physical scale. Sixteen independent data/prior streams use $N=4096,K=100$, 512 updates and physical TAIL correlation averages. Mean risks are .0063134213/.0014017301/.0009340534; repairs occur 1342/385/0 times. Its original $q=3,\alpha_f=.05,L=16$ training family follows in $10^{-3}$ units.

|TAIL risk contrast|Mean|SE|Joint interval|SE target|
|---|---:|---:|---|---:|
|FIXED − FREE|4.9116912|1.0646554|[2.0437871, 7.7795952]|.2|
|FIXED − FACT|5.3793679|1.0543566|[2.5392060, 8.2195299]|.2|
|FREE − FACT|.4676768|.1330681|[.1092259, .8261276]|.05|

Both improvement directions pass, but all three SE requirements fail; FREE−FACT noninferiority at margin .0002 remains unconfirmed. The positive FREE−FACT interval does not establish an excess above that margin. LAST does not replace TAIL.

The direction study independently uses 32 groups × 512 banks, at I and truth, and nuisance scales one, uniform .2, or factorization. Its 18-contrast family ($\alpha_f=.05$, df31, every SE target .00005) tests six signed level gradients A, six signed scale-one minus FACT gradients B, and six signed scale-.2 minus scale-one gradients C. All 18 direction signs pass and **all 18 precision requirements fail**. Full original intervals, in $10^{-3}$ units, are retained here:

|Contrast / pair|Mean|SE|Joint interval|
|---|---:|---:|---|
|A / 01|−3.8366286|.0649866|[−4.0478385, −3.6254187]|
|A / 02|−2.4232542|.0635255|[−2.6297156, −2.2167928]|
|A / 03|−1.5734998|.0593550|[−1.7664066, −1.3805930]|
|A / 12|−1.5573868|.0722035|[−1.7920521, −1.3227215]|
|A / 13|−2.4231801|.0622794|[−2.6255914, −2.2207687]|
|A / 23|−3.8882056|.0739706|[−4.1286140, −3.6477971]|
|B / 01|−3.6267273|.0708552|[−3.8570107, −3.3964439]|
|B / 02|−2.2855209|.0675505|[−2.5050638, −2.0659781]|
|B / 03|−1.5005665|.0597903|[−1.6948880, −1.3062449]|
|B / 12|−1.4695288|.0716189|[−1.7022941, −1.2367634]|
|B / 13|−2.2626261|.0641267|[−2.4710413, −2.0542109]|
|B / 23|−3.6728456|.0823170|[−3.9403803, −3.4053108]|
|C / 01|2.6737233|.0753470|[2.4288415, 2.9186052]|
|C / 02|1.6774227|.0815887|[1.4122550, 1.9425905]|
|C / 03|1.0962796|.0637099|[.8892191, 1.3033401]|
|C / 12|1.0002334|.0812983|[.7360097, 1.2644571]|
|C / 13|1.6779526|.0824088|[1.4101197, 1.9457856]|
|C / 23|2.6970217|.1003027|[2.3710327, 3.0230107]|

The pair codes in this table are zero-based (01,02,03,12,13,23); the plots use the corresponding one-based labels (12,13,14,23,24,34). The sign convention aligns each coordinate with its true correlation. Uniform .2 is a probe, not FREE's learned vector; Price samples are not optimizer path-gradient samples. With a fixed spin-covariance reference center $h^{\rm ref}_{ij}$, the identity is

$$
g_{ij}=-h^{\rm ref}_{ij}A-\sum_k\pi_k(1-\pi_k)(H_{k,ij}-h^{\rm ref}_{ij}),
\qquad A=1-\sum_k\pi_k^2.
\tag{H.6}
$$

These are full-likelihood weights. Their mean overlap is $1-E(1/\mathrm{ESS})$, not $1-1/E\mathrm{ESS}$. Changing the reference center shifts the two terms oppositely; neither term is a causal percentage. The post-hoc six-correlation squared-error identity also does not partition KL: FIXED/FREE/FACT sample dispersions are .44026324/.08986095/.06273747, while squared sample-mean errors are .01028331/.01177915/.00465239. The latter are not squared population biases.

## H.5. Frozen-panel and fixed-data controls

The frozen-output study conditions on 16 original FIXED/FACT TAIL models and datasets, evaluates 35 candidates with 32 groups × 1024 banks, and separately uses numerical P weights and empirical D frequencies. Its original $q=11,\alpha_f=.05$, df31 family and SE target .0002 follow in $10^{-3}$ units. FACT score denotes an evaluation criterion, distinct from a FACT-trained candidate.

|Score / weights / contrast|Mean|Original interval|SE pass|
|---|---:|---|---|
|Raw / P / FIXED − truth|−.958815|[−1.704038, −.213591]|No|
|Raw / P / FIXED − FACT|−.907856|[−1.568204, −.247507]|No|
|Raw / D / FIXED − truth|−1.337861|[−2.086233, −.589488]|No|
|Raw / D / FIXED − FACT|−.605863|[−1.268919, .057193]|No|
|FACT / P / FIXED − truth|6.075270|[6.061753, 6.088786]|Yes|
|FACT / P / FIXED − FACT|5.174403|[5.162305, 5.186501]|Yes|
|FACT / D / FIXED − truth|5.142748|[5.129907, 5.155590]|Yes|
|FACT / D / FIXED − FACT|5.924402|[5.911604, 5.937199]|Yes|
|Raw / P / high − truth|−2.836999|[−3.629602, −2.044397]|No|
|Raw / P / panel − mean|−.861525|[−1.505467, −.217583]|No|
|FACT / P / panel − mean|5.944371|[5.930723, 5.958020]|Yes|

Ten effect directions and five precision targets pass. All six raw precision failures remain, and raw D FIXED−FACT includes zero. Integrated-risk panel−mean is +.0061704631 versus negative raw P panel−mean. The mean matrix's risk .0001429581 is descriptive, not an equal-cost procedure. Original-interval subtraction gives FIXED−high raw P [.0003403590,.0034160105], showing panel-average improvement room, not each run's global optimization error.

Two subsequent studies fix four old datasets and use 16 new prior columns each. L4 averages four gradients before one update and has the same expected current-batch gradient as L1, at four times the scoring calls. IN4 does the same, while OUT4 averages four independent L1 physical outputs. Their original training and score families each allocate .025. Risk/score rows below use $10^{-3}$ units; parameter rows retain squared-Euclidean units.

|Study / contrast|Mean|Original interval|SE pass|
|---|---:|---|---|
|L4 − L1 / risk|−1.229258|[−2.279813, −.178703]|Yes|
|L4 − L1 / dispersion|−.172395|[−.264123, −.080668]|Yes|
|L4 − L1 / raw D|−1.409279|[−1.786109, −1.032449]|Yes|
|L4 − L1 / raw P|−1.134853|[−1.512167, −.757539]|Yes|
|OUT4 − IN4 / risk|−3.136097|[−4.485575, −1.786619]|Yes|
|OUT4 − IN4 / dispersion|−.115067|[−.231044, .000910]|Yes|
|OUT4 − IN4 / mean error|−.108253|[−.220002, .003496]|Yes|
|OUT4 − IN4 / raw D|1.819468|[1.321781, 2.317156]|Yes|
|OUT4 − IN4 / raw P|1.697666|[1.202711, 2.192621]|Yes|

Risk units are 16 columns averaged over the four fixed datasets (df15), not 64 new datasets. Parameter units are eight disjoint column pairs (df7): half squared distance estimates conditional trace covariance and a truth-centered cross product estimates conditional squared mean error. The latter can be negative. Scores use 32 independent groups × 1024 banks (df31), conditional on realized outputs. Risk/parameter/score SE targets are .0005/.05/.0002. L1/L4 passes both families, and improving both score and risk is an essential counterexample to a universal opposite-ranking claim. IN4/OUT4's two parameter intervals include zero, so neither is an established cause of its KL benefit. The old OUT4 risk .002681300 is close to I, but that study set no I superiority/equivalence gate; H.1 does not backfill one.

## H.6. Matched log-weight variance and earlier boundaries

**Finite original-interface comparison (Figure S3).** A separate study uses 16 independent streams, 4096 complete observations per stream, K100 and fresh per-observation prior particles. The four-active teacher is (A.21); the high candidate has covariance $I+1.75(C_*-I)$. Nuisance extensions share corresponding inputs and leave its integrated KL penalty .001724195 unchanged. The figure reuses two contrasts from the original q=4, alpha=.05, df15 two-sided Bonferroni development-t family: high-minus-truth is +.001424043 with interval [.000825588,.002022499] at n=0 and −.003781125 with interval [−.006451504,−.001110746] at n=256. The family's other contrasts concern the radial-gradient shift and positive nuisance-scale derivative; all four original effect and precision gates pass. Two reversed n256 stream signs remain, so the mean result is not a per-stream guarantee. The reference uses 12 main PMFs and three audit PMFs; native32/stable64 controls qualify this bounded comparison, not every possible input. There is no parameter training, and these data differ from H.1.



A two-active support study compares diffuse, bounded-heterogeneous and concentrated nuisance scales with the same exact observation model and total log-weight variance. At $n=254$, diffuse scales are one; bounded heterogeneity uses .5 and 1.30631561; concentration uses 1/254 for 253 coordinates and 14.2713171 for one. The common variance is 68.9646835. The concentrated family leaves the fixed positive compact scale window, consistent with A.4 rather than refuting it.

The 65,536-bank direction study passes its original 12-contrast empirical-Bernstein family at K64/K256 and true correlations 0/.6. Its intervals condition on numerical reference moments and exclude integration/floating-point error. A separate shared-bank learning study uses 16 base datasets, $N=4096,n=254,K=256,B=128$, and 801 prescribed candidates. Two teacher worlds share inputs and are not 32 independent units. At truth 0, mean diffuse/concentrated risk is .008417575/.000126907; at truth .6 it is .000347575/.000149114. The original three training comparisons pass: risk differences have intervals [.008193944,.008387391] and [.000094344,.000302577], and the pooled discrepancy from same-data MLE interval is [.228358106,.294443131]. This development-t family is separate from the 12-contrast family. All 128 grid/interpolation qualifications pass, maximum $1.90189\times10^{-5}$ against $3\times10^{-5}$, but do not certify the continuous population optimum or floating-point error. Closeness to MLE is descriptive, without a new noninferiority gate.

Earlier radial FREE-versus-FACT noninferiority and precision failures are also retained in their source studies. Later conditional evaluation of frozen radial models does not repair them, nor does a successful fixed-slice score improvement establish joint-training risk improvement. Finite calibration and dual-budget results remain supporting theory for specified procedures, not a universal algorithm or runtime claim.

## H.7. Numerical reference, accounting and verification limits

All empirical intervals use the original independent units, families and numerical weights. Source packages bind protocols and code before new draws and retain failures. They do not claim exact interval coverage or rigorous integration-error certificates. The records preserve the original data roles, random inputs and decisions. Additional active-only learning and selection protocols are included in Appendix I of this manuscript.

|Study|New observations|Optimizations|Main / audit PMFs|
|---|---:|---:|---:|
|Original-API finite comparison|65,536|0|12 / 3|
|Original six-correlation + direction|65,536|48|291 / 9|
|Frozen six-correlation evaluation|0|0|3 / 1|
|L1/L4|0|128|408 / 16|
|IN4/OUT4 fixed data|0|320|408 / 16|
|New-data IN4/OUT4/SEQ4|65,536|224|384 / 16|
|Finite active activation|0|0|48 / 16|
|Joint scale learning|65,536|96|288 / 32|

Every optimization count refers to controlled parameters, with zero neural-network training. Shared random inputs are counted once. L1/L4 and fixed-data IN4/OUT4 each use 218,103,808,000 training Gaussian scalars and 851,968,000 evaluation scalars; the new-data study uses 109,051,904,000 and 1,703,936,000. Finite activation uses 32,768 banks, 3,276,800 prior vectors and 851,968,000 Gaussian scalars. Joint learning uses 52,428,800 prior vectors and 13,631,488,000 Gaussian scalars, with 49,152 main and 384 qualification API calls. Teacher integration reuse and analytic product probabilities are not new PMFs.

The six-correlation references use GH24/32/40 (32 primary); finite activation uses GH24/32/40 (40 primary), 63,045,632 nodes, with four analytic product PMFs; joint learning uses GH32/40/48 (40 primary), 856,031,232 nodes. Their separate symmetric-root audits use order36, 36, and44, respectively; joint learning adds 119,939,072 audit nodes. All declared PMF qualifications pass. Positivity, normalization and cross-order agreement are numerical checks, not certified remainder bounds. Original truth PMFs and later normalized GH40 teachers retain their own distinct contracts.

Audits replay all input hashes, saved output/statistical arithmetic and, where applicable, saved Adam/repair updates. Bottom-level reconstruction is bounded: 24 original six-correlation training states and 128 direction banks; 143,360 frozen-evaluation cells; 80 L1/L4 training bank calls and 282,624 evaluation cells; 80 fixed-data aggregate training states, 128 bank calls and 282,624 evaluation cells; 160 new-data training states, 208 bank calls and 81,920 evaluation cells; 32 finite-activation prefix banks plus 4096 native API calls; and 384 joint-scale states. Joint-scale audit rechecks all 49,152 saved updates but does not recompute every underlying gradient. Full input replay is not full retraining or full score replay.

The fixed-data IN4/OUT4 audit's filename overwrite left only data0-repeat1 confirmed complete; a separately frozen software repair completed the prescribed checks on the same inputs. Joint-scale records retain output-format failures and the absence of the original failed development source. These repairs did not resample scientific inputs or extend budgets. Local large NPZ files are not all in Git, so public JSON, RNG specifications and hashes do not replace the binary data.

\clearpage

![Matched total log-weight variance does not fix the scoring direction or learned risk. This figure reuses the frozen variance-matched study; its direction and learning families remain separate.](figures/representation_structure.png){width=100%}

![Original correlation-direction decomposition. Uniform nuisance scale .2 is a fixed probe, not a learned FREE output. All 18 directional requirements pass, but all 18 precision requirements fail; the two components are a reference-dependent identity rather than causal shares.](figures/overlap_coupling.png){width=100%}
