# USE0 结果与范围

日期：2026-10-04。结论：**单候选资格完成，未通过进入 USE1 的完整门槛；停止，不训练。**

## 已获得的判断

1. 官方 hard head 与 precision 的确有标准流程依据。固定源码32份 manifest记录全部哈希通过，本轮从固定提交读取的6份关键文本与本地快照逐字节相同。源码默认值：hard、linear、joint、straight-through=True、amortized、L1 precision；不是所有可能配置的必要条件。
2. 在独立零均值 nuisance 的同维块模型中，其二元分布为公平硬币。任意固定 hard head——包括 nuisance 权重非零——都不会打破尺度变换的精确概念/预测分布身份。该身份只关于模型分布与前向采样边缘，不认证 ST surrogate 梯度或有限训练轨迹。
3. 上述同一个已知结构使 nuisance 概念似然可以解析积掉，有限 active 概念评分加 `α n log2` 即可。可以保留同260维head及同维正则、相同概念模型类；替代不是只能由知道真实 active 教师的 oracle 实行。
4. 原 precision 不惩罚对角；块独立下其值只包含 active precision 的非对角，除以 `260·259`。恢复该项不会直接阻止 nuisance 尺度改变。
5. 完整标准类允许学习每维均值、全部协方差与 head；不强制身份所需的结构。当前未建立“标准可学全类仍有精确任务身份，且必须保留随机因子”的同一工作流，也未另冻完整输入/标签风险合同。

因此新增知识是用途边界，而不是把旧概念反例包装成完整 SCBM 的实际失败。完整推导与严格条件在 QUALIFICATION (historical reference not distributed)。

## 本轮实际验证与预算

| 项目 | 结果 |
|---|---|
| 冻结合同 | 内容哈希与源码资格前一致 |
| 固定本地来源 | 32/32 manifest记录字节数与SHA256匹配 |
| 关键官方远端文本 | 6/6读取成功且与固定本地字节相同 |
| 原论文 | 官方作者 arXiv v2，核对§3.1/§3.2；§3.3只定位用途范围 |
| 新科学RNG/数据处理 | 0 / 0 |
| 新评分/API前向/积分 | 0 / 0 |
| 新拟合/训练/统计推断 | 0 / 0 / 0 |
| 外部消息、权重、数据下载 | 0 |
| USE1 | 未启动 |

上述计数是执行过程记录；来源核验脚本仅使用标准库文本/哈希/HTTP操作，没有导入或运行科研模型。[SOURCE_AUDIT](../../../artifacts/use_v1_20261004/SOURCE_AUDIT.json) 保存来源记录。该首份收据由保留的 `DEVELOPMENT/verify_sources_initial.py` 产生；后续脚本安全修订没有覆盖收据，见 DEVELOPMENT (historical reference not distributed)。

只读复核命令：

```powershell
.venv/Scripts/python.exe artifacts/use_v1_20261004/verify_sources.py --check
```

该入口只比对本地哈希和既有收据，不联网、不写文件。`--run` 是新收据生成入口，在当前已有收据时直接拒绝覆盖，不应用作验收命令。

## 尚未获得的判断

- 没有证明所有 SCBM 使用场景不合格，或 dense 标准流程无法公平控制。
- 没有证明解析替代使完整训练/目标风险成功，或使 ST 尺度敏感性消失。
- 没有数值认证官方 precision 的浮点抵消、hard/ST前向逐位身份或任何新 head 实现。
- 没有从有限数据识别正确块，也没有把未知结构的估计成本忽略为零。
- 没有用新教师、网络、自然数据或新候选替换本次候选。
