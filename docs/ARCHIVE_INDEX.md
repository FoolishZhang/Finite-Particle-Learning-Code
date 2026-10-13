# Scientific archive

This index distinguishes evidence supporting the current h2.2 paper from numerical context, historical controls and source-only programs. Inclusion is not a claim of independent repetition. Consult `REPRODUCING.md` for checks actually run on this release, and the individual saved objects for their numerical and statistical conditions.

Original scientific filenames and small numerical objects are retained. Unavailable links in old scientific notes are rendered as historical source labels; a historical label is not a promised downloadable input. Exact reference values are not replaced by regenerated data, and failed or unresolved conditions are not relabeled as passes.

## Current paper and retained numerical context

|Scientific role|Where to read|Scope|
|---|---|---|
|Theorem 3: teacher probabilities, full Hessian, stationary root and risk|`script/certified_teacher_*`, `script/certified_f8_gradient_*`, `script/certified_f8_hessian_*`, `script/local_witness_feasibility_*`; corresponding `artifacts/` folders|One population root, strict local minimum modulo common rotations; saved finite sums and analytical error budgets|
|Same-root fixed-marginal angle conflict (R01)|`script/angular_witness_*`; `artifacts/angular_witness_v1_20261010/`|Specified angle path and sufficiently small nonzero steps; no numerical angle-step radius or specified finite-step gain|
|Finite-scale correction and deterministic sign (R02)|[R02 conclusions](08_research_研究工作/R02Remainder_v1_20261010/CONCLUSIONS.md); `script/r02_*`|Fixed model and coordinate: sign certified at n=4096; the budgets at n=64, 256 and 1024 remain unresolved|
|Local teacher-family persistence (R03)|[R03 conclusions](08_research_研究工作/R03TeacherFamily_v1_20261011/CONCLUSIONS.md); `script/r03_*`|The continuous family with half-width .001 is certified; the wider .01 box remains unresolved, not disproved|
|Fair-coin control, local curvature and finite-program bound|`script/nuisance_contrast_*`, `script/local_geometry_*`, `script/finite_selection_*`, `script/failure_probability_*`|Saved matched comparison and specified audit families; conditional statistical guarantees and numerical allowances remain essential|
|Overlap versus coupling and the four-scale numerical panel|`script/asymptotic_bridge_*`; `artifacts/asymptotic_bridge_v1_20261008/`|Prespecified numerical targets and qualified intervals, including sign reversal and unresolved precision/increment comparisons|
|Learning, rank-one stationary points and K11/K12 probes|`script/joint_dependence_*`, `script/rankone_*`, `script/noncollinear_probe_*`, `script/direction_validation_*`|Numerical discovery and controls; these do not certify optimizer attraction or a universal critical particle count|

## Studies outside the h2.2 PDF

The six bilingual files named `APPENDIX_REPRO_*`, `APPENDIX_NEW_*`, and `APPENDIX_PREDICTION_*` in the h2.1.1 source directory retain the original H, I and L lettering. Their definitions, qualifications and original outcomes are preserved. Their studies are not newly established by h2.2, and should not be read as broader consequences of its theorems. The h2.2 H/I lettering refers instead to the former J/K.

**Historical averaging attribution.** The archived H.1/H.2 studies use late-iterate TAIL averages and independent-path OUT4 averages. An earlier description that these appendices reported only LAST was incorrect. Zinkevich et al. (2010) and Izmailov et al. (2018), retained in the archived bibliography, provide the averaging precedents; their use here does not imply an Adam guarantee. The original numerical results are unchanged.

|Research family|Available scientific source and evidence|What the archive preserves|
|---|---|---|
|Finite learning, unknown scales and mean freedoms|`finite_learning_*`, `scale_learning_*`, `jointscale_*`, `outer_learning_*`, `activevar_*`|Fixed experimental panels, changes under freed parameters, numerical gates and retained failures|
|Weight structure, averaging and budget allocation|`wstructure_*`, `scbmavg_*`, `scbmalloc_*`, `bankvar_*`|Variance-matching controls, independent-path/late-iterate averages, and comparisons of inner/outer particle allocation|
|SCBM concept-loss interface, multicorrelation training and frozen evaluation|`scbminv_*`, `scbmopt_*`, `scbm6_*`, `scbm6eval_*`, `scbmeval_*`, `scbmseq_*`|Specified API-level and synthetic-model comparisons; not an official end-to-end SCBM training or checkpoint reproduction|
|REC recovery, readout and budget controls|`rec0_*`, `rec1_*`, `recscale_*`, `reck_*`, `reclearn_*`, `reccross_*`, `recreadout_*`|Saved endpoint diagnoses, implementable independent baselines, helpful changes and failed prespecified changes|
|Natural binary-image selection (USE)|`uselect_*`, `udigits_*`, `usemargin_*`; `artifacts/use_digits_v3_20261005/RESULTS.json`|All 32 Optdigits workflows selected L001 under five evaluation methods; no evaluation-induced candidate change was observed. This is not positive evidence for broad selection harm|
|Curvature-based particle-budget development|[Development conclusions](08_research_研究工作/ProspectiveDiagnosticDev_v1_20261007/CONCLUSIONS.md); `prospective_diagnostic_*`|The 12-unit development prototype did not improve mean risk over the simpler reference-only rule and cost more than always using the high particle count. The planned larger confirmatory stage was not run|
|Explanation of the budget prototype|[Explanation conclusions](08_research_研究工作/ProspectiveExplanation_v1_20261007/CONCLUSIONS.md); `prospective_explanation_*`|Fixed empirical directions can change curvature sign under population evaluation; radial/bias channels and endpoint risk accounting do not identify one training-history cause|
|Calibration, regularization and covariance/constraint boundaries|`fincal_*`, `curvcal_*`, `reglearn_*`, `compensation_learning_*`, `covariance_boundary_*`, `constraint_boundary_*`, `weak_emission_*`|Positive controls, mixed development gates, failed noninferiority conditions, and boundaries of proposed corrections|
|Earlier structure/query theory and nonexchangeability|`rc_*`, `structure_learning_*`, `v2_*`, `v3_*`, `objective_*`, `nuisance_*`|Historical analytic and numerical investigations, with their own conditions rather than the current-paper guarantee|

## Known source-only limitations

The earlier default checkers for RC quantification, structure learning, v2 synthesis and v3 selection depended on old Git objects that are no longer present. Those private-seal wrappers are not a public reproduction interface. Their scientific functions and selected results remain available for reading.

The historical `dependence_boundary_*` application programs require a removed CBM/GoEmotions environment and data/model inputs that are not bundled. Their presence is a source archive, not a claim that their training can now be resumed. Unrelated clinical or business material is not part of this scientific release.

Some scientific drivers originally bound internal execution records. Those records are not needed to read a mathematical function, but their absence can prevent an original archival command from running. Use the public production adapters described in the reproduction guide for supported computations; do not disable assertions or overwrite reference evidence to make an old command pass.

The SCBM text snapshot is third-party source at upstream commit `58c3e602cb11fb4ea9106226e0f9e58f50d5d10a`, with its Apache-2.0 license preserved. It contains selected source/configuration files, not external datasets, pretrained weights or a validation of the upstream training procedure.

## Package-by-package saved-object map

This table lists selected numerical-object folders, not a claim that every original array or operational record is bundled. Each row is paired with scientific source families in the tables above; complete replay status belongs to the reproduction guide.

|Package|Role|Selected numeric objects|
|---|---|---:|
|[active_scale_boundary](../artifacts/active_scale_boundary/)|historical science outside the PDF|21|
|[angular_witness_v1_20261010](../artifacts/angular_witness_v1_20261010/)|current-paper support|7|
|[asymptotic_bridge_v1_20261008](../artifacts/asymptotic_bridge_v1_20261008/)|current-paper support|2057|
|[bank_precision](../artifacts/bank_precision/)|historical science outside the PDF|10|
|[calibration_precision](../artifacts/calibration_precision/)|historical science outside the PDF|10|
|[certified_f8_gradient_v1_20261008](../artifacts/certified_f8_gradient_v1_20261008/)|current-paper support|50|
|[certified_f8_hessian_v1_20261008](../artifacts/certified_f8_hessian_v1_20261008/)|current-paper support|50|
|[certified_teacher_v1_20261008](../artifacts/certified_teacher_v1_20261008/)|current-paper support|23|
|[compensation_learning](../artifacts/compensation_learning/)|historical science outside the PDF|23|
|[constraint_boundary_v2](../artifacts/constraint_boundary_v2/)|historical science outside the PDF|6|
|[covariance_boundary](../artifacts/covariance_boundary/)|historical science outside the PDF|6|
|[dependence_boundary_v1](../artifacts/dependence_boundary_v1/)|historical science outside the PDF|17|
|[direction_validation_v1_20261007](../artifacts/direction_validation_v1_20261007/)|retained numerical context|6|
|[failure_probability_v1_20261006](../artifacts/failure_probability_v1_20261006/)|current-paper support|19|
|[finite_calibration](../artifacts/finite_calibration/)|historical science outside the PDF|16|
|[finite_learning](../artifacts/finite_learning/)|historical science outside the PDF|10|
|[finite_selection_v1_20261006](../artifacts/finite_selection_v1_20261006/)|current-paper support|12|
|[joint_dependence_v1_20261007](../artifacts/joint_dependence_v1_20261007/)|retained numerical context|12|
|[joint_risk_refinement_v1_20261006](../artifacts/joint_risk_refinement_v1_20261006/)|current-paper support|3|
|[joint_scale_learning](../artifacts/joint_scale_learning/)|historical science outside the PDF|2|
|[local_geometry_v1_20261006](../artifacts/local_geometry_v1_20261006/)|current-paper support|7|
|[local_witness_feasibility_v1_20261008](../artifacts/local_witness_feasibility_v1_20261008/)|current-paper support|13|
|[mean_robustness_v1_20261006](../artifacts/mean_robustness_v1_20261006/)|current-paper support|7|
|[noncollinear_probe_v1_20261007](../artifacts/noncollinear_probe_v1_20261007/)|retained numerical context|3|
|[nuisance_contrast_v1_20261006](../artifacts/nuisance_contrast_v1_20261006/)|current-paper support|8|
|[nuisance_dimensions](../artifacts/nuisance_dimensions/)|historical science outside the PDF|14|
|[nuisance_learning](../artifacts/nuisance_learning/)|historical science outside the PDF|19|
|[nuisance_structure](../artifacts/nuisance_structure/)|historical science outside the PDF|5|
|[numerical_sign_budget_v1_20261006](../artifacts/numerical_sign_budget_v1_20261006/)|historical science outside the PDF|2|
|[objective_strengthening](../artifacts/objective_strengthening/)|historical science outside the PDF|8|
|[positive_control_v1](../artifacts/positive_control_v1/)|historical science outside the PDF|3|
|[prospective_design_v2_20261007](../artifacts/prospective_design_v2_20261007/)|historical science outside the PDF|1|
|[prospective_diagnostic_dev_v1_20261007](../artifacts/prospective_diagnostic_dev_v1_20261007/)|historical science outside the PDF|33|
|[prospective_explanation_v1_20261007](../artifacts/prospective_explanation_v1_20261007/)|historical science outside the PDF|5|
|[r02_finite_scale_v1_20261010](../artifacts/r02_finite_scale_v1_20261010/)|current-paper support|8|
|[r02_remainder_v1_20261010](../artifacts/r02_remainder_v1_20261010/)|current-paper support|8|
|[r03_teacher_family_v1_20261011](../artifacts/r03_teacher_family_v1_20261011/)|current-paper support|81|
|[rankone_continuation_v1_20261007](../artifacts/rankone_continuation_v1_20261007/)|retained numerical context|14|
|[rankone_stability_v1_20261007](../artifacts/rankone_stability_v1_20261007/)|retained numerical context|5|
|[rankone_stationary_v1_20261007](../artifacts/rankone_stationary_v1_20261007/)|retained numerical context|4|
|[rc_quant](../artifacts/rc_quant/)|historical science outside the PDF|12|
|[rec_cross_v5_20261005](../artifacts/rec_cross_v5_20261005/)|historical science outside the PDF|4|
|[rec_k_v3_20261005](../artifacts/rec_k_v3_20261005/)|historical science outside the PDF|4|
|[rec_learning_v4_20261005](../artifacts/rec_learning_v4_20261005/)|historical science outside the PDF|67|
|[rec_readout_v6_20261005](../artifacts/rec_readout_v6_20261005/)|historical science outside the PDF|3|
|[rec_scale_v2_20261005](../artifacts/rec_scale_v2_20261005/)|historical science outside the PDF|4|
|[rec_v1_20261004](../artifacts/rec_v1_20261004/)|historical science outside the PDF|2|
|[regularization_learning](../artifacts/regularization_learning/)|historical science outside the PDF|7|
|[scale_learning_v2](../artifacts/scale_learning_v2/)|historical science outside the PDF|3|
|[scbm_bank_averaging](../artifacts/scbm_bank_averaging/)|historical science outside the PDF|68|
|[scbm_budget_allocation](../artifacts/scbm_budget_allocation/)|historical science outside the PDF|69|
|[scbm_frozen_evaluation](../artifacts/scbm_frozen_evaluation/)|historical science outside the PDF|6|
|[scbm_invariance](../artifacts/scbm_invariance/)|historical science outside the PDF|7|
|[scbm_multicorrelation](../artifacts/scbm_multicorrelation/)|historical science outside the PDF|6|
|[scbm_scale_optimization](../artifacts/scbm_scale_optimization/)|historical science outside the PDF|5|
|[scbm_sequential_generalization](../artifacts/scbm_sequential_generalization/)|historical science outside the PDF|4|
|[scbm_six_frozen_targets](../artifacts/scbm_six_frozen_targets/)|historical science outside the PDF|5|
|[structure_learning_v2](../artifacts/structure_learning_v2/)|historical science outside the PDF|13|
|[structure_tradeoff_v2](../artifacts/structure_tradeoff_v2/)|historical science outside the PDF|11|
|[use_digits_v3_20261005](../artifacts/use_digits_v3_20261005/)|historical science outside the PDF|35|
|[use_margin_v4_20261005](../artifacts/use_margin_v4_20261005/)|historical science outside the PDF|3|
|[use_selection_v2_20261005](../artifacts/use_selection_v2_20261005/)|historical science outside the PDF|100|
|[use_v1_20261004](../artifacts/use_v1_20261004/)|historical science outside the PDF|2|
|[v2_synthesis_20260930](../artifacts/v2_synthesis_20260930/)|historical science outside the PDF|3|
|[v3_query](../artifacts/v3_query/)|historical science outside the PDF|7|
|[v3_select](../artifacts/v3_select/)|historical science outside the PDF|10|
|[weak_emission](../artifacts/weak_emission/)|historical science outside the PDF|6|

## Selected mathematical and result documents

The following folders contain selected proof, scope or result notes. Operational readmes and private review discussions are not necessary prerequisites for reading these scientific notes.

- [AngularWitness_v1_20261010](08_research_研究工作/AngularWitness_v1_20261010/): 3 selected scientific notes.
- [AsymptoticBridge_v1_20261008](08_research_研究工作/AsymptoticBridge_v1_20261008/): 2 selected scientific notes.
- [CertifiedF8Gradient_v1_20261008](08_research_研究工作/CertifiedF8Gradient_v1_20261008/): 1 selected scientific notes.
- [CertifiedF8Hessian_v1_20261008](08_research_研究工作/CertifiedF8Hessian_v1_20261008/): 2 selected scientific notes.
- [CertifiedTeacher_v1_20261008](08_research_研究工作/CertifiedTeacher_v1_20261008/): 1 selected scientific notes.
- [DirectionValidation_v1_20261007](08_research_研究工作/DirectionValidation_v1_20261007/): 1 selected scientific notes.
- [FailureProbability_v1_20261006](08_research_研究工作/FailureProbability_v1_20261006/): 1 selected scientific notes.
- [FiniteSelection_v1_20261006](08_research_研究工作/FiniteSelection_v1_20261006/): 2 selected scientific notes.
- [JointDependence_v1_20261007](08_research_研究工作/JointDependence_v1_20261007/): 6 selected scientific notes.
- [JointRiskRefinement_v1_20261006](08_research_研究工作/JointRiskRefinement_v1_20261006/): 1 selected scientific notes.
- [LocalGeometry_v1_20261006](08_research_研究工作/LocalGeometry_v1_20261006/): 5 selected scientific notes.
- [LocalWitnessFeasibility_v1_20261008](08_research_研究工作/LocalWitnessFeasibility_v1_20261008/): 2 selected scientific notes.
- [MeanRobustness_v1_20261006](08_research_研究工作/MeanRobustness_v1_20261006/): 4 selected scientific notes.
- [NoncollinearProbe_v1_20261007](08_research_研究工作/NoncollinearProbe_v1_20261007/): 2 selected scientific notes.
- [NuisanceContrast_v1_20261006](08_research_研究工作/NuisanceContrast_v1_20261006/): 1 selected scientific notes.
- [ProspectiveDiagnosticDev_v1_20261007](08_research_研究工作/ProspectiveDiagnosticDev_v1_20261007/): 2 selected scientific notes.
- [ProspectiveExplanation_v1_20261007](08_research_研究工作/ProspectiveExplanation_v1_20261007/): 1 selected scientific notes.
- [PublicVerification_v1_20261009](08_research_研究工作/PublicVerification_v1_20261009/): 1 selected scientific notes.
- [R02FiniteScale_v1_20261010](08_research_研究工作/R02FiniteScale_v1_20261010/): 3 selected scientific notes.
- [R02Remainder_v1_20261010](08_research_研究工作/R02Remainder_v1_20261010/): 4 selected scientific notes.
- [R03TeacherFamily_v1_20261011](08_research_研究工作/R03TeacherFamily_v1_20261011/): 10 selected scientific notes.
- [RC定量界与多块查询_20261001](08_research_研究工作/RC定量界与多块查询_20261001/): 2 selected scientific notes.
- [REC_v1_20261004](08_research_研究工作/REC_v1_20261004/): 2 selected scientific notes.
- [REC_v2_20261005](08_research_研究工作/REC_v2_20261005/): 2 selected scientific notes.
- [REC_v3_20261005](08_research_研究工作/REC_v3_20261005/): 2 selected scientific notes.
- [REC_v5_cross_20261005](08_research_研究工作/REC_v5_cross_20261005/): 1 selected scientific notes.
- [REC_v6_readout_20261005](08_research_研究工作/REC_v6_readout_20261005/): 1 selected scientific notes.
- [RankOneContinuation_v1_20261007](08_research_研究工作/RankOneContinuation_v1_20261007/): 2 selected scientific notes.
- [RankOneStability_v1_20261007](08_research_研究工作/RankOneStability_v1_20261007/): 1 selected scientific notes.
- [RankOneStationary_v1_20261007](08_research_研究工作/RankOneStationary_v1_20261007/): 2 selected scientific notes.
- [USE_v1_20261004](08_research_研究工作/USE_v1_20261004/): 1 selected scientific notes.
- [USE_v2_20261005](08_research_研究工作/USE_v2_20261005/): 1 selected scientific notes.
- [USE_v3_20261005](08_research_研究工作/USE_v3_20261005/): 1 selected scientific notes.
- [USE_v4_margin_20261005](08_research_研究工作/USE_v4_margin_20261005/): 1 selected scientific notes.
- [v2收束与v3方向_20260930](08_research_研究工作/v2收束与v3方向_20260930/): 1 selected scientific notes.
- [v3结构选择与查询诊断_20260930](08_research_研究工作/v3结构选择与查询诊断_20260930/): 2 selected scientific notes.
- [v3表示与条件查询_20260930](08_research_研究工作/v3表示与条件查询_20260930/): 2 selected scientific notes.
- [两组尺度共同学习与恢复边界_20261004](08_research_研究工作/两组尺度共同学习与恢复边界_20261004/): 5 selected scientific notes.
- [全局约束与采样门槛_20260929](08_research_研究工作/全局约束与采样门槛_20260929/): 2 selected scientific notes.
- [六相关冻结输出与有限目标_20261003](08_research_研究工作/六相关冻结输出与有限目标_20261003/): 4 selected scientific notes.
- [六相关自由学习与方向机制_20261003](08_research_研究工作/六相关自由学习与方向机制_20261003/): 4 selected scientific notes.
- [冻结表示的条件与总体目标_20261003](08_research_研究工作/冻结表示的条件与总体目标_20261003/): 3 selected scientific notes.
- [可学习无关尺度与优化路径_20261003](08_research_研究工作/可学习无关尺度与优化路径_20261003/): 5 selected scientific notes.
- [同目标下的粒子梯度平均_20261003](08_research_研究工作/同目标下的粒子梯度平均_20261003/): 4 selected scientific notes.
- [外层精度与有限样本伪相关_20261001](08_research_研究工作/外层精度与有限样本伪相关_20261001/): 7 selected scientific notes.
- [弱发射展开与总体偏好_20261001](08_research_研究工作/弱发射展开与总体偏好_20261001/): 7 selected scientific notes.
- [总体目标塌缩与同模型修复_20261001](08_research_研究工作/总体目标塌缩与同模型修复_20261001/): 3 selected scientific notes.
- [新数据与连续更新对照_20261004](08_research_研究工作/新数据与连续更新对照_20261004/): 4 selected scientific notes.
- [无关权重结构与方差匹配_20261002](08_research_研究工作/无关权重结构与方差匹配_20261002/): 7 selected scientific notes.
- [无关维度与固定噪声总体偏好_20261001](08_research_研究工作/无关维度与固定噪声总体偏好_20261001/): 5 selected scientific notes.
- [有效尺度自由度与有限曲率_20261004](08_research_研究工作/有效尺度自由度与有限曲率_20261004/): 5 selected scientific notes.
- [有限校准与真实样本训练_20261002](08_research_研究工作/有限校准与真实样本训练_20261002/): 4 selected scientific notes.
- [有限样本学习与均值自由度_20261001](08_research_研究工作/有限样本学习与均值自由度_20261001/): 4 selected scientific notes.
- [有限粒子尺度学习_v2_20260930](08_research_研究工作/有限粒子尺度学习_v2_20260930/): 2 selected scientific notes.
- [校准精度与跨尺度学习_20261002](08_research_研究工作/校准精度与跨尺度学习_20261002/): 6 selected scientific notes.
- [正族同分布控制_20260930](08_research_研究工作/正族同分布控制_20260930/): 1 selected scientific notes.
- [相关结构补偿与发射噪声_20261001](08_research_研究工作/相关结构补偿与发射噪声_20261001/): 6 selected scientific notes.
- [真实联合评分接口与无关扩展_20261003](08_research_研究工作/真实联合评分接口与无关扩展_20261003/): 3 selected scientific notes.
- [粒子精度与局部校准_20261002](08_research_研究工作/粒子精度与局部校准_20261002/): 5 selected scientific notes.
- [结构先验释放与分组学习_v2_20260930](08_research_研究工作/结构先验释放与分组学习_v2_20260930/): 4 selected scientific notes.
- [结构预算与对称化代价_v2_20260930](08_research_研究工作/结构预算与对称化代价_v2_20260930/): 2 selected scientific notes.
- [统一正则与真实相关学习_20261002](08_research_研究工作/统一正则与真实相关学习_20261002/): 3 selected scientific notes.
- [补偿学习与连续目标_20261001](08_research_研究工作/补偿学习与连续目标_20261001/): 5 selected scientific notes.
- [计算匹配的路径与梯度平均_20261003](08_research_研究工作/计算匹配的路径与梯度平均_20261003/): 4 selected scientific notes.
- [边际依赖接口_20260929](08_research_研究工作/边际依赖接口_20260929/): 5 selected scientific notes.
- [非交换适用域_v2_20260930](08_research_研究工作/非交换适用域_v2_20260930/): 2 selected scientific notes.
