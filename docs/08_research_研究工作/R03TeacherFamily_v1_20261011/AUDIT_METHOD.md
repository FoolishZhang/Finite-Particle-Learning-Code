# R03 内部独立核验：教师 GH 与三个候选点标签

日期：2026-10-11。实现为 [r03_independent_audit.py](../../../script/r03_independent_audit.py)。这是作者内部实现/数值核验，不是用户安排的外部科研审核。本模块在导入时不评价目标、不写文件；正式入口 `produce_audit(inputs=None, wall_seconds=120)` 必须由主执行器冻结本模块、输入、相关方法及来源文件后调用。

本文编写阶段只运行7项 codec/shape/门槛制造检查，并只读核对保存 schema、原点身份与256个新旧严格 PMF 区间相交。**尚未运行本模块的 GH80/GH100 教师目标或三个候选点标签。** 后续实际状态以冻结执行的审计 JSON 为准。

## 1. 固定核验对象及来源

教师是原零偏置、八行长度4/5、角度(-.7,-.5,-.3,-.1,.1,.3,.5,.7)的精确输入模型，在 lambda=0 只扰动第一行角度。新严格参考为 `artifacts/r03_teacher_family_v1_20261011/TEACHER_CERTIFICATE.json`，其中字段是 `q_true_component_intervals` 与 `prime_true_component_intervals`，每项保存 exact dyadic 端点。

候选点始终是旧精确 dyadic a0，不是移动根、不使用宽 a 盒。它从同批 `BOX_INPUTS.json` 的 `boxes.h01.a0_exact` 取得，并核对另外两个盒的中心、`DEVELOPMENT.json.initial_a_exact` 完全一致。

独立保存数值参考是 `DEVELOPMENT.json.points[0].evaluations.D.result`。实际读到并逐项校验的结构为：

|字段|保存形状/身份|实际用途|
|---|---|---|
|K_values|[8,"inf"]|选择第0项 K8|
|patterns|256x8，整数标签最高位为第一观察量|核标签顺序|
|label_radial_gradient|2x256x8|比较新 g_a|
|label_bias_gradient|2x256x8|保存参考，不计入比较|
|label_H_ba|2x256x16x16，b_then_a|取左上 Hbb、右下 Haa 比较|
|label_Hperp|2x256x8x8|比较新 Hperp|

D 规则身份同时核对：独立 Legendre384、Gaussian限制11、log-u 范围[-32,64]、步长1/8。保存 b 为零；保存 a 的每个 binary64 数按其**精确浮点有理值**与旧 a0 比较，实际一致。检查不是用打印小数或容差冒认同一点。

旧严格教师参考是 `artifacts/certified_teacher_v1_20261008/TEACHER_CERTIFICATE.json`，实际字段为 `true_PMF_enclosures`，长度256；schema 为 `certified_teacher_true_PMF_v1`。模块保存这四份输入的路径、SHA256和字节数；主 wrapper 的冻结锁负责约束这些哈希不能在正式运行前改变。

## 2. 教师 GH80/GH100：直接完整 likelihood 乘 score

使用 `scipy.special.roots_hermitenorm` 生成标准 Gaussian 规则，其原始权重只除理论常数 sqrt(2 pi)，不除所生成权重的实测质量，也不对输出 PMF 重归一。两档二维节点分别为80^2=6400与100^2=10000，共16400。

每个完整标签 s 直接计算

\[
 A_s(z)=\prod_{i=1}^8p_i(z)^{s_i}(1-p_i(z))^{1-s_i},
 \qquad
 \partial_\lambda A_s(z)=A_s(z)(s_1-p_1(z))\dot\eta_1(z),
\]
\[
 \dot\eta_1(z)=\tfrac45[-\sin(-7/10)z_1+\cos(-7/10)z_2].
\tag{A1}
\]

数值实现逐标签构造全部八个 Bernoulli 因子再作乘积，没有调用主教师模块、第一因子导数积、标签前缀树、反序折叠、互补平均或旧缓存。A1 与主实现的“第一 Bernoulli 因子导数乘其余七因子”代数等价，但计算路径独立；Gaussian 规则也不同。

每档保存全部256个 q 与256个 q' 原始 binary64 数、Gaussian质量、q总和、q'总和及节点数。它们是普通数值求积，角度/三角函数/累加均有原生浮点误差；这里没有为 GH 截断或浮点运算另建第二套严格误差证书。

对每档的 q/q' 向量，逐分量与新严格区间比较；另比较两档规则的向量 L1 差。四个“单规则与严格参考”的 L1 门、两个“两规则”的 L1 门均为1e-11。输出保留每个分量的距离，而非只保留是否通过。

## 3. 三个固定标签的候选点重新评价

固定标签为0、85、127，分别对应全零、交替与一零七一的最高位标签。新点入口复用已经冻结的 `r03_candidate_interval.precompute(a0,config)` 与 `finite_label_block(cache,[0,85,127])`，但输入是八个精确中心有理数，不能替换成已生产的宽盒。

配置严格与本轮主冻结 coarse 规则相同：K8、192bit、h_z=1/32、L=10、d_z=1/6；h_v=1/4、v范围[-28,56]、d_v=1、T=4、128饱和上界、互补代表模式。内层641节点，外层337节点，每标签216017个 Laplace 因子，三标签648051个；不是所有128或256标签重算。

每个标签保存新有限和的完整八维 g_a 以及三个8x8矩阵 Hbb、Haa、Hperp 的原生区间；同一点的解析预算用 `analytic_error_bounds(a0,config, teacher_error_L1_upper="0", teacher_mass_upper="1")` 得到。此处是单标签条件量，没有教师重加权误差。

逐分量/逐矩阵元素扩入对应的统一真实积分误差：g_a 使用 `uniform_gradient_L2`；Hbb/Haa 使用 `uniform_Hba_operator`；Hperp 使用 `uniform_Hperp_operator`。向量 L2 与矩阵 operator 误差分别控制任一坐标/任一元素，故这种逐元素扩包合法。门槛只按单个元素距离判断，没有把逐元素误差当作矩阵 operator 误差。

然后与已保存 D 规则的相应原始标签值比较，不重跑旧 GL kernel。每标签实际比较8+3x64=200个分量，共600个，每项门为1e-8。

新候选 API 没有生产单标签 g_b 或 mixed Hba。**单标签的这两者通常不为零**，不能借教师互补平均后消失的结论将其补成零矩阵。审计因此仅声称 g_a、Hbb、Haa、Hperp 的比较；同时保存对应旧单标签 g_b 与完整16x16 Hba 供定位，但明确标为未比较范围。这里的独立性来自已有独立 GL 数值实现与新严格点评价的对照，不声称新点代码本身独立于本轮主区间实现。

## 4. 距离和门槛的精确解释

保存 binary64 数 x 按 `Fraction.from_float(x)` 转成精确有理值；严格区间端点 l,u 按 dyadic 表示读入。模块同时报告

\[
 \operatorname{gap}(x,[l,u])=\max(0,l-x,x-u),\qquad
 e_{\max}=\max(|x-l|,|x-u|).
\tag{A2}
\]

gap 是到区间集合的距离；e_max 控制 x 到区间内任何真实值的绝对差。本审计采用更保守的 e_max 通过门槛，因此不能用一个极宽、恰好覆盖普通数值的区间获得虚假的零距离通过。

教师规则与严格参考的门使用 sum e_max<1e-11；候选门使用 max_entry e_max<1e-8。两档 GH 的差直接对两个保存 binary64 数的精确有理差求 L1。所有门用 Fraction 比较；显示小数只方便阅读，完整每分量有理差、最大值与L1和进入结果。

通过这些门支持实现与原保存证据的一致性，不将普通 GH/GL 自动升级成各自独立的严格证书，也不替代连续参数盒的根包含、完整稳定性与风险负号证明。

## 5. 新旧严格 PMF 的纯保存比较

对每个标签比较本轮 q_true 区间与旧 `true_PMF_enclosures` 是否相交，使用 exact dyadic 端点；输出256项相交布尔值、区间间 gap 和新减旧的区间。这里不新增积分，也不重建旧网格。编写阶段的只读比较已见256项全部相交，正式冻结审计会再次针对其绑定的输入保存该结果。

可选旧 Hessian `rep_00.json` 的实际 `original_labels` 已只读确认是[0,255]，且保存的是带教师 W/M/S 权重的类矩阵。由于从带权有限矩阵除以 W 后，还需明确传播其相应单标签解析误差，本轮不把这一项自动算入已完成核验；模块输入显式记 `old_hessian_rep00_comparison=False`。未修改旧来源或扩大核验范围。

## 6. 预算、部分结果与实际制造范围

默认墙钟预算120秒，从读取证据前开始计时；GH 在标签边界检查，候选点使用其原有节点/外层边界 deadline 检查。超时保存已完成阶段和原始部分结果在返回对象中，状态为 `budget_stopped_*`；不自动更换规则、减少标签、延长预算或重试。异常也返回已产生的证据、错误类型/文本和实际计数。源模块不写文件，持久保存由冻结 wrapper 负责。

实际计数分开报告普通 GH 节点、规则标签和、点 Laplace 因子；预定16400与648051只记在 planned_counts，actual_counts由返回结果累加，不能在部分失败时冒认计划已全部执行。没有随机抽样、优化、旧 NPZ 或旧全量 kernel 重放。

`--self-test` 仅检查 exact dyadic codec、端点扩包、普通数值转精确有理比较、宽区间不能仅凭覆盖通过门、矩形/不规则 shape 和计划计数四则运算；0 Gaussian节点、0候选节点、0科学文件读取。编写阶段另行只读核验了上述实际保存 schema 与旧/新 PMF 相交。正式目标执行不属于这些制造检查。

```python
spec = prepared_inputs()  # 只生成固定规格，不读取或评价目标
# 主执行器先冻结 source、spec、所有来源哈希和本方法文件
result = produce_audit(spec, wall_seconds=120)
# 主执行器 exclusive 保存 result；module 本身不写文件
```
