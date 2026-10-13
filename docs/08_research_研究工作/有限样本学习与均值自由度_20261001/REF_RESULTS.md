# REF完成：正族同Q私噪声边际化数值参考

REF在首次科学积分前冻结。d=16/64、c=.9、axis=[1,1.5,2,3,4,6,8]及49个lex(u,v)候选全部完成，STATE=complete、QUALIFICATION.all_passed=true；未改阶数或门。0科学随机观察、0粒子集、0拟合。统一RUN随后只能读取这些已封存表。

## 1. 每粒子和完整PMF的对象

col单粒子为给定一个公共标准Gaussian z后，逐坐标私噪声被积分的product-Bernoulli。两个组使用各自h_{d,u}(z)、h_{d,v}(z)，仍保持每个候选原Gaussian-logistic积分Q，不是对原latent粒子做group count-RB。

同组标签仅为计算加速压缩成两个count。公开REFERENCE_d16/64.json各有49个完整双count PMF、完整模式log概率（已扣两个组合数）、三个参考变体，以及中央/双尾事件概率和风险。中央及双尾严格按整型count的划分定义，没有新增事件抽样。

## 2. lookup原门与实际误差

两个私噪声GH阶数固定1024/2048，主阶2048；z∈[-12,12]、4097等距节点，记录h和h'后用cubic Hermite求值。以下均为全节点/全尺度最大值。

| 项目 | 原门 | d16 | d64 |
| --- | ---: | ---: | ---: |
| h阶差 | 1e−9 | 3.84637e−12 | 6.87816e−12 |
| h'阶差 | 1e−8 | 6.18557e−11 | 5.26917e−11 |
| log归一化误差 | 1e−12 | 1.35482e−15 | 1.33227e−15 |
| h反号/导数偶对称误差 | 1e−12 | 1.33227e−15 | 1.33227e−15 |
| 两插值器的全域向量logscore稳定性界 | 1e−6 | 3.96246e−8 | 5.60041e−9 |
| exact-node插值向量logscore余项 | 1e−7 | 1.61267e−9 | 5.44270e−12 |

两维lookup的正性及Hermite单调性充分条件均通过；最小节点概率/补概率分别为0.0015897131、0.0794840220。运行时common超出[-12,12]会停止，不clip、不外推。

精确节点的理论界来自
$$
|h''''|\le\sqrt{24}\{\rho/(1-\rho)\}^2,\quad
e_{\rm interp}\le\Delta z^4\sup|h''''|/384,\quad
h,1-h\ge m_d=\tfrac12\Phi[-12\sqrt{\rho/(1-\rho)}].
$$
两维e_interp分别为6.12670e−14和3.15851e−15，m_d分别为0.0006078574和0.0371404437；向量logscore界为d e_interp/(m_d−e_interp)。这项证明只覆盖exact-node插值，**不包含GH节点误差**。所展示的常数用float计算，并非纯有理区间认证。

## 3. 独立quad与完整PMF资格

独立quad固定每个d/尺度的九个z点−12,−8,−4,−1,0,1,4,8,12，E范围[-12,12]，epsabs=epsrel=1e−12、limit400。所有378个标量积分成功，实际函数点评价123396次。

| 项目 | 原门 | d16 | d64 |
| --- | ---: | ---: | ---: |
| quad与主GH的log概率差 | 1e−9 | 3.37508e−14 | 3.99681e−14 |
| quad与主GH的h'差 | 1e−10 | 1.09635e−14 | 4.93356e−15 |
| quad最大报告误差 | 仅记录 | 9.14697e−13 | 9.50453e−13 |

原完整PMF三变体为(private1024,outer512)、(private2048,outer256)、(private2048,outer512)，主值第三项。outer节点超出lookup域时直接独立计算lp/lq的logsumexp，不用1−h相减，也不外推插值。每个原始PMF先记录质量误差，再固定除以其已记录正质量；没有删负值或clip。

| 完整49格参考检查 | 原门 | d16 | d64 |
| --- | ---: | ---: | ---: |
| 原始PMF质量差 | 1e−12 | 1.66534e−14 | 3.75256e−14 |
| 变体PMF L1差 | 1e−8 | 1.69625e−14 | 2.20461e−14 |
| 全cell logPMF差 | 1e−6 | 1.15464e−13 | 5.25802e−13 |
| u/v交换transpose误差 | 1e−12 | 2.77556e−17 | 6.93890e−18 |
| 全bit complement误差 | 1e−12 | 1.45717e−16 | 1.31839e−16 |
| teacher×candidate joint KL敏感性 | 1e−7 | 2.44597e−15 | 2.90567e−15 |
| 三事件概率敏感性 | 1e−7 | 8.65974e−15 | 1.13243e−14 |
| 三事件Brier/logloss风险敏感性 | 1e−7 | 1.31493e−15 | 1.47799e−15 |

全部完整PMF严格正。以上门是确定性方法一致性资格，不是严格Gaussian求积余项证书；不能将它们升级为每个候选真实KL的严格机器区间。

## 4. 实际计算与交付

- 98个主PMF、294个变体PMF，是确定性参考计算，不是98/294个拟合。
- lookup的私噪声logit—GH节点组合评价176203776个；reference outer位置上的私噪声节点组合29360128个；独立quad固定探针的对照GH节点组合258048个，合计205821952。这是计算原语，不是样本或逐函数调用次数。
- 固定代数QA使用三次多项式与d4人工两product混合；插值误差2.22e−16、pattern/count恒等差2.78e−17。开发与freeze后各执行一次，0科学积分/RNG，不计拟合。
- lookup两维耗时18.805/19.279秒；PMF参考两维耗时2.175/2.049秒。START至RESULTS_FREEZE约42.754秒，包含quad、QA和I/O；不是与raw端到端训练的同成本比较。
- 两个lookup NPZ共1471739字节。完整PMF及其所有变体为公开JSON。

只读API已核对：log_h(common[2,3],d)返回两个[7,2,3]数组；reference(d)返回pmf/log_pattern为[49,9,9]或[49,33,33]，variants_pmf为[3,49,...]，parameters/axis顺序显式匹配。API仅读已完成REF，不调用积分器或随机数。源码在REF冻结后未改。

FREEZE SHA256为26d0ed78c867fc9880e5346448bf7d54e01ca80454be34004280f79c0f6b7a44；RESULTS_FREEZE SHA256为598362a09c35f26de45312a78a0d44c6342a2e7a66381cf300e4531547acac72；源码SHA256为4528f6d87633336ca3bb596e792081b22314f2964bbd7f2fc9fa12c10024135d。

只读检查已通过：
~~~text
.venv/Scripts/python script/finite_learning_col.py check --workspace
~~~
该命令检查源、公开参考及lookup数组字节，不重算科学积分。
