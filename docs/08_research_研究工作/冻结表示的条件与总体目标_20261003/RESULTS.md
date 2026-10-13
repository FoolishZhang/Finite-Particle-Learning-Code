# 冻结表示独立评估的全部主要读数

固定16个旧FREE模型及原数据；独立单位为32个全新MC组，每组512个K100 bank。P表示冻结数值生成权重，D表示各自旧数据频数；区间条件于这一panel，不是新的训练算法总体区间。

## 四项预定主比较

|比较|平均值|SE|区间下端|区间上端|df|效应|精度|
|---|---:|---:|---:|---:|---:|---|---|
|P1_population_wrong_preference|-0.0001095233|0.0000201490|-0.0001629567|-0.0000560900|31|True|True|
|P2_nuisance_excess_tilt|-0.0003147385|0.0000204780|-0.0003690443|-0.0002604326|31|True|True|
|P3_half_scale_upward_shift|0.0002611983|0.0000169305|0.0002163001|0.0003060965|31|True|True|
|P4_LAST_conditional_selected_improvement|0.0000488999|0.0000032834|0.0000395862|0.0000582136|15|True|True|

P1/P2要求上端<0，P3/P4要求下端>0；全部SE门5e−5。P1…P3使用32组，P4只用后16组验证前16组选出的比较点。一个四项Bonferroni双侧t开发近似家族；不称精确覆盖，也不把P1…P3的区间说成条件于选点。

## 同一候选对的曲线读数

|表示|P中t14−truth|D中t14−truth|P的truth处Price|D的truth处Price|
|---|---:|---:|---:|---:|
|TAIL learned|-0.0001095233|-0.0001099245|-0.0009499382|-0.0009556176|
|TAIL half|0.0001516750|0.0001517468|-0.0003033384|-0.0003077957|
|TAIL FACT|0.0002052152|0.0002052871|-0.0001704502|-0.0001746033|
|LAST learned|-0.0000891687|-0.0000895251|-0.0008995287|-0.0009050562|
|LAST FACT|0.0002052152|0.0002052871|-0.0001704502|-0.0001746033|

t14实际保存值为0.30000000000000004，表示潜相关幅度乘1.3；其参考joint KL为0.000266063632。表中梯度为描述读数，完整7480点同时带另存STATISTICS/RESULTS；其family与四主门分开，不合称总报告95%。Price与path是期望相同的不同估计量，不能把本评估Price方差当原训练路径梯度噪声。

## 原共同区间的派生解释

由c_half=P1+P3，半尺度候选差均值0.0001516750，保守区间[0.0000533434,0.0002500066]。这是事后相加原共同区间的逻辑推论，继承原开发近似事件，不是新增检验。只描述t14与truth的这对候选，不宣称全曲线或极小点恢复。

同理e_half=P2+P3，均值-0.0000535402，保守区间[-0.0001527443,0.0000456639]。没有从P3正值自动推断所有参数点的绝对偏差变小。

## LAST选点及独立验证

|旧model|实际LAST t|诊断t|后16组平均D改进|原LAST参考KL|诊断参考KL|
|---:|---:|---:|---:|---:|---:|
|0|0.15387015|-0.07500000|0.0001240985|0.0000692255|0.0000161765|
|1|0.42656171|0.30000000|0.0000334150|0.0005432197|0.0002660636|
|2|-0.09510437|-0.15000000|0.0000089597|0.0000259745|0.0000643669|
|3|0.37576163|0.30000000|0.0000138315|0.0004198646|0.0002660636|
|4|0.44836581|0.60000000|0.0000311157|0.0006012070|0.0010898239|
|5|0.37825292|0.15000000|0.0001260407|0.0004255330|0.0000657681|
|6|0.36105832|0.30000000|0.0000133571|0.0003872065|0.0002660636|
|7|0.23107411|0.15000000|0.0000200805|0.0001570252|0.0000657681|
|8|0.46373335|0.75000000|0.0002298054|0.0006439090|0.0017241950|
|9|0.30814674|0.30814674|0.0000000000|0.0002808856|0.0002808856|
|10|0.09586260|0.15000000|-0.0000036466|0.0000267545|0.0000657681|
|11|0.07378919|-0.07500000|0.0000669598|0.0000158265|0.0000161765|
|12|0.44888979|0.45000000|-0.0000001147|0.0006026380|0.0006056758|
|13|0.03458224|-0.15000000|0.0000731882|0.0000034663|0.0000643669|
|14|-0.15786989|0.00000000|0.0000389337|0.0000712593|0.0000000000|
|15|0.39109176|0.52500000|0.0000063735|0.0004553664|0.0008293389|

负的逐model验证均值为[10, 12]；保留全部反向与零值。P4只保证panel平均条件切片改进，不是每个model都改善，也不是连续全局优化证书。

风险列为事后描述：原LAST平均0.0002955851，诊断候选平均0.0003554063，7改善/8变差/1相同。它复用已合格PMF，不新增拟合/积分/风险门，不是继续联合训练FREE的结果或因果预测。条件目标分数更低与真实观察风险更低必须分别核查。

## 数值资格与账目

API数值资格：{"core_cases": 128, "core_cases_passed": 128, "weighted_score_checks": 256, "weighted_score_passed": 256, "weighted_path_checks": 256, "weighted_path_passed": 256, "maximum_weighted_score_difference": 1.3677088257613428e-05, "maximum_weighted_path_difference": 3.386284055589573e-07, "canonical_cases": 128, "canonical_cases_passed": 128, "maximum_canonical_score_difference": 3.0517578125e-05, "maximum_canonical_path_difference": 1.862645149230957e-06}。

新评估账目：{"independent_evaluation_replicates": 32, "new_base_banks": 16384, "new_prior_vectors": 1638400, "Gaussian_scalars": 425984000, "active_Gaussian_scalars": 6553600, "nuisance_Gaussian_scalars": 419430400, "new_observations": 0, "new_fits": 0, "new_model_integrals": 0, "shared_active_grid_terms_per_chunk": 21, "actual_t_terms_per_chunk": 32, "common_FACT_grid_scores_per_chunk": 21, "learned_or_half_grid_scores_per_chunk": 1008, "actual_t_score_cases_per_chunk": 80, "score_kernel_calls": 283904, "reused_bank_pattern_score_cases": 290717696, "original_API_pattern_control_calls": 2048, "original_API_canonical_control_calls": 256, "original_API_control_calls": 2304}。

参考账目：{"new_PMF_integrations": 54, "new_scalar_pattern_integrals": 864, "new_Gaussian_nodes": 70926336, "reused_grid_queries": 3, "reused_actual_queries": 32, "new_random_draws": 0, "new_fits": 0}。主评估791.523s，参考89.445s；独立审计成本另列。

0新观测、0原训练更新、0NN；P4产生16个诊断候选选择，已单列。新曲线是理想数学准则的数值MC评价；canonical/API32对照仅为前缀数值资格，不提供原API32全域精度或Gaussian参考严格余项证书。

报告 (historical reference not distributed) · 设计 (historical reference not distributed) · 理论 (historical reference not distributed) · 审查 (historical reference not distributed)
