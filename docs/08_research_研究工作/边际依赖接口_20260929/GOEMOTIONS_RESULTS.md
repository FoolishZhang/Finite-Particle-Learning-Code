# GoEmotions：局部边界得到复核，默认初始化下没有实质风险收益

GN/GE已按独立冻结完成：复用旧word、word_char两个TF-IDF概念均值模型，0新均值/任务头；拟合48个协方差终点（96个训练阶段），随后才评价历史test的5622条评论/3595个作者—thread—文本连通组。全部16主对比通过数值门，**0项通过预定“同时区间下界>0.005 nats”的实质收益门**。全部16同时区间都落在±0.005内是数据事实；原GE没有登记`equivalence_pass`门，不能回写成事前等效性确认。8项EQ区间均含0；word_char/association的两项NP差显著为负、量值很小，保留为antithetic略差的结果。

## 完整16项主表

差值为iid风险减antithetic风险，正值有利于antithetic；单位为**10⁻⁶ nats**（实质尺度0.005 nats=5000）。区间采用10000次原连通组bootstrap、16项Bonferroni同时95%开发区间；先平均三训练seed，条件于固定模型。NP是不反馈联合风险；EQ为一次二值事件反馈后的两个partner联合风险。

| 旧均值 | 协方差角色 | K | 评分 | 差值 | 同时区间 |
|---|---|---:|---|---:|---|
| word | train | 10 | NP | −3.003 | [−88.196, 90.428] |
| word | train | 10 | EQ | 43.895 | [−51.359, 148.433] |
| word | train | 100 | NP | −5.120 | [−17.409, 7.912] |
| word | train | 100 | EQ | 1.280 | [−13.027, 17.377] |
| word | association | 10 | NP | −132.259 | [−370.531, 113.989] |
| word | association | 10 | EQ | 43.617 | [−225.615, 319.079] |
| word | association | 100 | NP | −30.588 | [−72.299, 13.197] |
| word | association | 100 | EQ | 1.771 | [−46.031, 52.267] |
| word_char | train | 10 | NP | −38.278 | [−108.711, 33.861] |
| word_char | train | 10 | EQ | 0.679 | [−77.429, 84.249] |
| word_char | train | 100 | NP | −10.200 | [−19.618, 0.157] |
| word_char | train | 100 | EQ | −6.031 | [−17.758, 7.118] |
| word_char | association | 10 | NP | −229.731 | [−423.297, −22.356] |
| word_char | association | 10 | EQ | −67.649 | [−278.493, 169.657] |
| word_char | association | 100 | NP | −44.748 | [−81.046, −5.936] |
| word_char | association | 100 | EQ | −10.661 | [−52.775, 34.647] |

## 哪些机制事实仍成立

在未缩放logits、rater0经验目标上，word的train/association零协方差临界K为10.696/3.452，word_char为15.545/3.636。因此K10时两个训练角色均无PSD下降方向、两个训练外角色均有；K16/K100四者均有，不能将K100称为collapse。rater1/2敏感性使训练角色临界K降至6.863–9.023，K10二分不推广到所有评审目标。

从默认标准差1初始化，K10下antithetic使最终协方差迹增加：word train为4.942→5.273、association为6.839→7.783；word_char为4.891→5.204、6.811→7.749。**局部谱和训练迹变化没有转化为本评价合同下的实质风险收益。** 局部零协方差理论不认证有限步优化、全局最优或外部风险。M是未中心化残差二阶矩，均值失配也进入谱，不能把谱量直接称为真实语义依赖。

## 数据、数值与解释边界

四个官方来源文件按旧SHA恢复到GDATA新目录；全体56451评论的文本哈希、连通组、角色哈希均重建匹配。标签为27情绪的评审意见，按既有哈希选三位明确作答者；主目标仅rater0。GN仅用train33395及association_fit5577拟合，两角色不是相同总数据预算；主比较是同角色、同K的采样器。GE单独获准读取历史test，GDATA原不评价test的合同保持。

两个旧joblib哈希均与原封存相同，隔离环境使用匹配的scikit-learn1.9.0，手工TF-IDF/coef预测核验通过。旧word_char/TRAINING.json的5938人数来自`ids`被前一模型的policy_validation循环覆盖；按训练源码，实际fit的x和train.labels仍为33395条。本轮记录此metadata错误，未改旧收据或重训。

每head各自按最高边际熵选query，partners只由train rater0的MI选择，同head所有协方差比较使用相同查询与目标。**不同head可能评分不同partner对，不直接比较两head绝对pair NLL。** 没有AG、Hard或其它幅值校准，没有多查询/任务头，因此不主张普遍最佳反馈法或完整CBM改进。

主风险先pool四流×1024节点的概率再取log；前512节点为敏感性。16主差的最大MC influence SE为2.573×10⁻⁷，最大配对节点敏感性9.265×10⁻⁷；预定极端子集CDF2049/4097的最大真目标NLL误差7.852×10⁻⁶，所有真目标floor为0。独立从public逐评论差重建区间误差为0，从各流NLL还原NP/EQ概率pool最大误差8.882×10⁻¹⁶。这些是预定数值稳健性核验，不是完整有限MC误差上界。

证据：[GN模型核验](../../../artifacts/dependence_boundary_v1/GN/AUDIT.json) · GE完整主表 (historical reference not distributed) · 数值收据 (historical reference not distributed) · 独立重算 (historical reference not distributed) · GDATA谱 (historical reference not distributed) · GN冻结 (historical reference not distributed) · GE冻结 (historical reference not distributed)。全部计算结束，无追加模型、seed或节点队列。
