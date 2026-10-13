# 理论层次与不可合并的边界

|对象|已建立的结论|条件与范围|
|---|---|---|
|有界异质nuisance|所有确定尺度数组在固定正紧区间内，错误相关矩之和方向统一成立；总体实际min最终端点|固定active尺度1、已知相关对、固定R<1、2≤K≤C n^p；近优误差须o(J_K/s_n)|
|同方差集中族|总体raw减nlog2后统一逼近精确active交叉熵，K=n+2下全部η→0近优点恢复θ|n−1尺度1/n、一尺度A_n=Θ(√n)；EW²和对数尾双重控制，不能只有二阶矩|
|一参数实际学习|同数据/K=n+2/B=n/网格，diffuse有正极限joint风险、集中风险趋0；集中自身B≥1即可|固定内点真参数；事前网格含端点、大小O(n^1/4)、网宽→0、N任意预定发散；精确ERM或各侧规定的近优误差，非SGD|
|noise-only权重|集中ESS/K→p1/2、max权重→p0；分散/有界异质ESS→p1、max权重→p1|集中K也趋∞；分散2≤K≤C n^p。不给增长K下EESS→1，也不把1/2赋给完整active权重|
|一般固定active模型|有限字母表、统一正概率和相对log矩条件下，集中族实际网格ERM的完整观察KL恢复|K=n+2、固定active大小、候选网格m=o√n且加密，N→∞，精确ERM或优化误差趋零；覆盖紧SPD Gaussian-logistic类，潜参数可不辨识|
|四变量六相关负侧|R<1/3的完整相关盒，真相关六坐标均非零内点时，regret=o(w_n)的raw总体近优趋同号顶点且joint KL正|此处负侧只有总体目标；没有另证多参数共享bank学习预算，不与正侧拼成多参数实际同B分离|

具体证明见[有界异质](HETEROGENEOUS_THEORY.md)、[集中恢复](CONCENTRATED_THEORY.md)、同预算学习 (historical reference not distributed)、[权重几何](WEIGHT_GEOMETRY.md)、一般active (historical reference not distributed)。作者和独立审读范围分开记录于审读一 (historical reference not distributed)、审读二 (historical reference not distributed)。审读通过不等于形式化证明或全领域首创认证。

集中族改变旧统一正噪声窗口：虽然每个有限n的尺度严格正，许多尺度趋零，一个趋∞。因此新对照不能推翻旧固定噪声定理。它保持完整统计任务和总log方差，却改变随机权重形状；有界异质族则说明旧方向并非必须逐坐标同分布。

风险始终是积分观察模型Q的总joint KL。raw随机K预测器本身具有另一种预测合同；上述比较不声称对数评分失去适当性。按总维数归一化会稀释固定active块的风险。

数值实验对应双精度尺度、固定K/n/N/B和801点，不能直接作为渐近网格/全域数值误差合同的验证。新MC门、训练门、旧科学失败分开保留。当前没有未知active、增长active维数、任意连续优化器或真实SCBM训练结论。
