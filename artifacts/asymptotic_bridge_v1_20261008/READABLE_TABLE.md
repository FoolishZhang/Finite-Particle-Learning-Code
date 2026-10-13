# AsymptoticBridge-v1: fixed full-panel readout

99% conditional joint intervals for 26 fixed targets; ideal iid complete banks and the stated numerical qualification; frozen numeric references. 131072 same banks across all n, not independent replicates. Native floating-point error and exact coefficient enclosures are not certified.

Every cell below is mean [joint-family interval]. All four n values and all six predeclared paired contrasts are retained. Numerical qualification is not a strict enclosure of native arithmetic or exact reference coefficients.

Primary gap and single-coordinate gradient

| n | g = F(I)-F(Cb) | A = g/(w gapV) | D12 at rho=.45 | Bgrad = -D12/(m_H w) |
|---:|---:|---:|---:|---:|
| 64 | -0.00448185566 [-0.00506564113, -0.0038980702] | -0.142730334 [-0.161321717, -0.124138951] | 0.00172259379 [0.00143977373, 0.00200541385] | -0.148594141 [-0.172990724, -0.124197557] |
| 256 | 0.00530496451 [0.0047205826, 0.00588934643] | 0.337886542 [0.300665787, 0.375107297] | -0.00190094564 [-0.0021847013, -0.00161718998] | 0.327958206 [0.279003625, 0.376912787] |
| 1024 | 0.00511321392 [0.00457247121, 0.00565395663] | 0.651346928 [0.5824644, 0.720229457] | -0.00188881694 [-0.00214969791, -0.00162793597] | 0.651731434 [0.561715126, 0.741747742] |
| 4096 | 0.00325468235 [0.00276135742, 0.00374800728] | 0.829195643 [0.703511218, 0.954880069] | -0.00120397332 [-0.00144171572, -0.000966230922] | 0.830855804 [0.666790996, 0.994920613] |

Residual decomposition on one fixed absolute scale

| n | overlap/w | R = D12/w+m_H | O = m_H(1-overlap/w) | J = -coupling/w |
|---:|---:|---:|---:|---:|
| 64 | 0.991604762 [0.987413299, 0.995796226] | 0.022134878 [0.0216647246, 0.0226050313] | 0.000161786968 [8.10121037e-05, 0.000242561832] | 0.021973091 [0.0214642779, 0.0224819041] |
| 256 | 1.02633236 [1.01697882, 1.0356859] | 0.0129511048 [0.0120076874, 0.0138945222] | -0.000507458266 [-0.000687712875, -0.000327203656] | 0.0134585631 [0.0124601525, 0.0144569737] |
| 1024 | 1.01815037 [1.00267579, 1.03362494] | 0.00671158065 [0.00497685128, 0.00844631002] | -0.000349780821 [-0.000647995653, -5.15659894e-05] | 0.00706136147 [0.00528073696, 0.00884198598] |
| 4096 | 1.00821993 [0.984512751, 1.03192712] | 0.00325962496 [9.78862861e-05, 0.00642136363] | -0.000158408628 [-0.000615276349, 0.000298459093] | 0.00341803359 [0.000223696567, 0.00661237061] |

R=O+J holds bankwise up to the declared numerical qualification. Individual R/O/J intervals reuse their raw target events; their endpoints must not be summed as if independent.

Secondary direct loss gap and fixed implementation diagnostics

| n | direct gap, T8 truncation and tail allowance | direct minus path mean | paired diagnostic SE | max bank identity residual |
|---:|---:|---:|---:|---:|
| 64 | -0.00452243934 [-0.0112638865, 0.00221900783] | -4.05836794e-05 | 0.000337460157 | 5.20417043e-17 |
| 256 | 0.0055723925 [-0.00225116325, 0.0133959482] | 0.000267427984 | 0.000596560087 | 5.55111512e-17 |
| 1024 | 0.00613213357 [-0.00237853943, 0.0146428066] | 0.00101891966 | 0.00074527077 | 4.51028104e-17 |
| 4096 | 0.00365793433 [-0.00522307418, 0.0125389428] | 0.000403251984 | 0.000821364219 | 4.16333634e-17 |

The direct-minus-path mean and SE are diagnostics, with no additional confidence claim or outcome-based pass threshold. Only the direct gap is clipped when any active |Z|>8; primary g and D12 are not clipped.

All six adjacent paired contrasts: next n minus previous n

| quantity | n transition | difference [joint-family interval] | raw same-bank covariance | paired sample variance | independence-only variance, unused |
|---|---:|---:|---:|---:|---:|
| A | 64 to 256 | 0.480616876 [0.430928336, 0.530305416] | -3.29377312e-06 | 2.33422982 | 2.32086785 |
| A | 256 to 1024 | 0.313460386 [0.219621541, 0.407299232] | 1.56684499e-05 | 6.70530513 | 6.95955624 |
| A | 1024 to 4096 | 0.177848715 [0.00305139839, 0.352646031] | 9.03013173e-06 | 16.7667694 | 17.3528953 |
| Bgrad | 64 to 256 | 0.476552347 [0.411165852, 0.541938841] | 5.84345685e-07 | 3.81373164 | 3.83112435 |
| Bgrad | 256 to 1024 | 0.323773228 [0.200002452, 0.447544004] | 1.30637002e-06 | 10.9981727 | 11.1537061 |
| Bgrad | 1024 to 4096 | 0.17912437 [-0.0508921163, 0.409140857] | 6.74467506e-07 | 26.6002077 | 26.9214099 |

All covariance entries use the same complete banks and ddof=1. Paired contrast variance uses covariance; the independence-only variance is diagnostic and not used for intervals.
The complete 20-target and 26-target covariance matrices, plus the R/O/J covariance matrices, are saved in AUTHOR_READOUT.json.

Frozen numeric references: v=0.271514502, J100=2.50759364, gapV=0.052200051, m_H45=0.0192712789.

| n | w_n | leading gap w_n gapV | leading single-coordinate gradient -m_H45 w_n |
|---:|---:|---:|---:|
| 64 | 0.601548512 | 0.031400863 | -0.0115926091 |
| 256 | 0.300774256 | 0.0157004315 | -0.00579630456 |
| 1024 | 0.150387128 | 0.00785021575 | -0.00289815228 |
| 4096 | 0.075193564 | 0.00392510788 | -0.00144907614 |

This readout extracts fixed fields only and selects no author claim. Signs, intervals spanning reference levels, and nonmonotone sequences remain in the tables and figure.

Fixed full-panel ratios and residuals (historical reference not distributed)

Input hashes and producer source hash are recorded in AUTHOR_READOUT.json.
