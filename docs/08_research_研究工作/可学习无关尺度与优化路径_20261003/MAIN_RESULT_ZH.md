# 可学习尺度改善风险但非劣恢复尚未确认

本轮真正更新了原SCBM概念评分下的模型参数。四active非零均值fixture、256独立公平无关位、K100保持；16个独立流各有4096固定完整观测，三程序配对同数据与每步新prior，以512步Adam训练。FIXED固定无关尺度1，FREE学256个各自softplus尺度，FACT正确分解。所有程序均从真实t=0开始；它是低维受控优化，未训练完整神经网络。

预定主输出为末64步物理参数平均，其平均完整joint KL分别为：FIXED .0012547105，FREE .0003273879，FACT .0001626322。FIXED−FREE与FIXED−FACT的原改善检验通过。FREE的平均尺度降至.202875，平均t由FIXED的.586363减至.259911；FACT平均t为.018322。这表明允许尺度学习的程序实际改善了观察风险。

恢复程度仍有限定：FREE−FACT的非劣区间上端.000337731高于预定.0002，SE .000064214也超过.00005。主家族效应2/3、精度2/3；区间是16流t开发近似。非劣失败不等于证实FREE更差；不以LAST或追加训练追认。FACT仍有active有限K及数据/优化误差。

新增解析推导显示，小尺度处总体梯度趋零而物理路径噪声不消失；零floor或明确的ε≪τ≪1中间尺度下，softplus的平均raw梯度比其标准差衰减更快，固定正floor需另一极限。这是固定点性质，不是Adam收敛定理或本次风险差的因果分解。官方precision项在当前独立块的实算术表达中排除所有无关对角，不会直接阻止尺度收缩；浮点抵消和释放cross covariance后的情形需另查。

192数值对照和全部连续输出参考资格通过；没有fresh API assessment，不能把训练loss当总体目标恢复的证据。下一步优先用独立评分区分残余目标方向与优化误差，再决定是否需要方差控制或释放结构。v5.1及旧结果保持。

报告 (historical reference not distributed) · [全部读数](RESULTS.md) · [理论](SCALE_DYNAMICS_THEORY.md) · 审查 (historical reference not distributed) · 下一步 (historical reference not distributed)
