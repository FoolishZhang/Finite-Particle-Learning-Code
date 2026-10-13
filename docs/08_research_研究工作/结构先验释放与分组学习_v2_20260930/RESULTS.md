# 结构先验释放与分组学习：A/B/C完整结果

本页只整理已经完成并封存的A/B/C，0新样本、拟合、参考积分或统计家族。A有1024次、B有640次、C有576次有限数据拟合，共2240次；完整表全部保留。

A在固定尺度比例下给错分组，S实际部署完整对称化S(Q)；B释放两个尺度，T只用总count估参后部署假定组下的Q，保留所有镜像；C再释放组信息，由带坐标pair学习分组。A/B共用A观察，C另生数据；A复用旧bank、C复用B bank，不能叫全部独立新确认。

## 原判定与完整失败账本

A：40/40精度、4/4主门通过。B：32项有限数据精度29通过、8/8固定总体精度通过；人口固定比较主门通过，有限数据主门未通过。C：36项精度35通过、3/3主门通过。

B的r0/N65536 G10−J虽然区间为正，SE=.00101459565略高于原.001门，因此原主门false保持。C有不同问题、不同家族及SE门.002，其通过不能追认B。

|package|r|comparison|N|primary_pass|mean|SE|lower|upper|precision_pass|kind|passed|profile|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|A|0|G10-J|65536|通过|0.03045892022|0.0005829815732|0.02814856839|0.03276927204|通过||||
|A|1|G10-J|65536|通过|0.03074436477|0.0004857873258|0.02881919306|0.03266953648|通过||||
|A|2|G10-J|65536|通过|0.03183718143|0.0008612394584|0.02842409559|0.03525026728|通过||||
|A|4|G10-J|65536|通过|0.0338236183|0.0007734084407|0.03075860615|0.03688863044|通过||||
|B|0||||-0.09001432997|0.002247234656|-0.09715402885|-0.08287463108|通过|population_fixed_witness|通过||
|B|0||65536||0.01908153528|0.001014595651|0.0151717232|0.02299134737|未通过|finite_data_excess|未通过||
|C|||||0.02827636007||0.02539166362|0.03116105652||finite_particle_bias_after_learned_group|通过|null|
|C|||||0.01956916494||0.0153440372|0.02379429268||finite_particle_bias_after_learned_group|通过|contrast|
|C|||||-0.001914031102||-0.001922285189|-0.001905777015||labelled_pair_below_count_only_bound|通过|contrast|

全部4项精度失败如下，均保持原规则：

|package|source|r|N|comparison|mean|SE|lower|upper|precision_pass|profile|
|---|---|---|---|---|---|---|---|---|---|---|
|B|FINITE_RISK_INTERVALS|0|4096|G10-J|0.01868167262|0.001310828798|0.01363030632|0.02373303891|False||
|B|FINITE_RISK_INTERVALS|0|65536|G10-J|0.01908153528|0.001014595651|0.0151717232|0.02299134737|False||
|B|FINITE_RISK_INTERVALS|1|65536|G10-J|0.01824843076|0.001044113461|0.01422486982|0.0222719917|False||
|C|RISK_INTERVALS||256|GLOBAL_P-COUNT_ONLY_BOUND|0.01078091969|0.002083993678|0.002629823324|0.01893201606|False|contrast|

## 错误分组造成的结构下界

分组projection floor (historical reference not distributed)

|r|projection_floor|A_best_J_KL|B_best_J_KL|
|---|---|---|---|
|0|0|0|1.776356839e-15|
|1|0.0008459873596|0.0009701967247|0.0009374807575|
|2|0.00144490053|0.001937041356|0.001483959573|
|4|0.001920832971|0.003801294862|0.001942173575|

所有risk均减原真实教师H(P)。floor是错误组内轨道投影损失；A/B的J最佳值分别只在各自预定网格内最优，不是连续全局最优。释放比例降低了额外模型/网格损失，但不消除错误分组floor。

## 二维目标、镜像歧义与连续诊断

二维目标 (historical reference not distributed)

每个panel减自己的最小值，P、T、J/G分别使用pair、count、完整向量单位；各列色标独立，不比较它们原始训练目标的大小。G10仅为16流聚合bank目标估计，不能称精确总体最优。T的mirror具有严格共享的total-count概率。

已知两组但尺度朝向未知时，任意数量count-only训练数据的两点minimax joint风险下界为JS(P6,2,P2,6)，本数值为0.001907818192 nats。T按小u选择镜像只是预定tie-break实例；全部132个最小等价类和各方向风险另表完整保留。

r4的连续精确pair oracle为(2.926155585,2.926155585)，joint KL=0.001920976075；网格选择[3.0, 3.0]，pair网格gap=6.075264825e-08。连续解不加入361拟合候选，也不把离散对角选择当成一般定理。

B四个r的人口J/P当前网格点相同，不能据此宣称发现了P相对J的错设优势或劣势。

## 未知组学习的全部有限风险

未知组风险 (historical reference not distributed)

C的contrast count-only prior-average joint风险下界为0.001920899443 nats，与B的两方向JS下界是不同信息合同。图中每个点为一次拟合的部署KL，短横线仅为描述均值；水平界约束count-only程序的prior-average风险，不是每个数据集的逐点界。

ORACLE_J/P已知真组；GLOBAL_P穷举全mask/pair候选，ROW_P使用行分数恢复组。GLOBAL_G10/G100先沿用GLOBAL_P选组再按B有限粒子曲面拟合尺度；并未穷举未知组的完整joint likelihood，也不是新的独立bank确认。

contrast全部N与六程序汇总如下。N256的边界尺度和较高损失保留，不只展示大N成功：

|profile|N|method|mean_KL|mean_u|mean_v|identifiable_groups|exact_high_groups|mean_swaps_if_identifiable|
|---|---|---|---|---|---|---|---|---|
|contrast|256|ORACLE_J|0.005148684247|7.09375|2.28125|16|14|1|
|contrast|256|ORACLE_P|0.004500360812|7.03125|2.09375|16|14|1|
|contrast|256|GLOBAL_P|0.01270181914|9|1.21875|16|0|3.3125|
|contrast|256|ROW_P|0.009085879774|9|1.4375|16|0|3.125|
|contrast|256|GLOBAL_G10|0.02394322261|2.0625|0.9375|16|0|3.3125|
|contrast|256|GLOBAL_G100|0.01157451191|5|1.375|16|0|3.3125|
|contrast|4096|ORACLE_J|0.0002489989573|5.96875|2.09375|16|16|0|
|contrast|4096|ORACLE_P|0.0003353905561|6.1875|2.0625|16|16|0|
|contrast|4096|GLOBAL_P|0.001410493037|6.8125|1.96875|16|5|1.1875|
|contrast|4096|ROW_P|0.001325536379|6.78125|1.96875|16|5|1.125|
|contrast|4096|GLOBAL_G10|0.02035661415|1.5625|1.28125|8|3|1.125|
|contrast|4096|GLOBAL_G100|0.002184620011|3.125|2.03125|16|5|1.1875|
|contrast|65536|ORACLE_J|8.265908121e-06|6.03125|2|16|16|0|
|contrast|65536|ORACLE_P|6.868340813e-06|6.03125|2|16|16|0|
|contrast|65536|GLOBAL_P|6.868340813e-06|6.03125|2|16|16|0|
|contrast|65536|ROW_P|6.868340813e-06|6.03125|2|16|16|0|
|contrast|65536|GLOBAL_G10|0.01957603328|1.53125|1.34375|6|6|0|
|contrast|65536|GLOBAL_G100|0.001970117291|3|2|16|16|0|

## 组恢复与概率界

组恢复 (historical reference not distributed)

contrast的GLOBAL_P与ROW_P在N65536各16/16恢复正确高尺度组，只是本次16个配对数据流的经验结果；不同N是前缀，不能当48个独立重复，也不能说普遍100%。null的组本来不可识别，不列为恢复成功。

右图为预定lambda网格上的ROW Chernoff与64对union上界；小N界等于1属于无信息，不是预测一定失败。该界条件于qualified数值教师，数值余量并非严格Gaussian积分证书。

|N|primary_union_failure_upper|numerical_allowance_union_failure_upper|
|---|---|---|
|256|1|1|
|4096|1|1|
|65536|0.0001782838325|0.0001782850012|

## 完整可读表

以下包含全部2240次拟合、116项原统计（A40/B32+8/C36）、所有人口/参考曲线以及T等价类。未增加置信家族；各原家族分开解释。

- A_FINITE_FITS (historical reference not distributed)
- A_FINITE_RISK_INTERVALS (historical reference not distributed)
- A_POPULATION_FITS (historical reference not distributed)
- A_POPULATION_CURVES (historical reference not distributed)
- B_FINITE_FITS (historical reference not distributed)
- B_FINITE_RISK_INTERVALS (historical reference not distributed)
- B_POPULATION_FITS (historical reference not distributed)
- B_POPULATION_FIXED_COMPARISONS (historical reference not distributed)
- B_REFERENCE_CURVES (historical reference not distributed)
- B_GRID (historical reference not distributed)
- C_FITS (historical reference not distributed)
- C_RISK_INTERVALS (historical reference not distributed)
- C_SUMMARY (historical reference not distributed)
- B_T_EQUIVALENCE_CLASSES (historical reference not distributed)
- ALL_PRIMARY (historical reference not distributed)
- PRECISION_FAILURES (historical reference not distributed)

## 来源与生成

图表来源使用csv.DictReader，字符串null保留为真实profile。每个科学包的原FREEZE与RESULTS_FREEZE逐hash核对后才作图。SVG行尾空格及末尾换行规范化后再登记figures/FIGURE_MANIFEST.json。原科学freeze、失败与统计门不改。
