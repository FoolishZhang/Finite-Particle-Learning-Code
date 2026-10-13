# 同一多项式训练预算下：raw失真，校准恢复

本推论由本包[预算定理](BUDGET_THEORY.md)与旧[无关维度一致极限](../无关维度与固定噪声总体偏好_20261001/POLYNOMIAL_K_THEORY.md)组合而来，零新随机、零模型积分。它把充分预算写成一个明确例子，不主张这些预算为必要或最优阶。

固定a=1、0<R<1、已知两位active相关块及真实teacher θ∈(−R,R)。令n→∞，采用

\[
K_n=n+2,\qquad B_n=n,\qquad C_n=\lceil\sqrt n\rceil,
\qquad N_n\longrightarrow\infty.
\]

观测为正确指定Q_θ的iid完整行。raw与校准程序使用**同一批B_n个完整共享训练bank**；校准另外使用C_n个独立bank，以精确已知模型函数r构造连续Price估计并积分。采用预算定理的实算术有限实现：校准导数分段线性插值后积分，校准与训练网格均细化到最大间距≤1/n且包含端点，训练在共同网格上完整枚举最小值。允许额外目标近优误差η_n=o_p(w_n)，但没有SGD保证。

记 \(w_n=J_{K_n}/\sqrt{n\operatorname{Var}(\log\sigma Z)}\)。Gaussian最大值满足 \(J_{n+2}^2\sim2\log n\)，故

\[
B_nw_n^2\asymp\log n\to\infty,
\qquad C_nw_n\asymp\sqrt{\log n}\to\infty.
\]

预算定理因此保证训练MC振幅、校准误差及实算术网格误差都相对w_n消失。校准学习器满足

\[
\widehat\rho_{cal,n}\to_p\theta,\qquad
E\,\mathrm{KL}(Q_\theta\|Q_{\widehat\rho_{cal,n}})\to0.
\]

同一批训练bank的未校准raw则有相反结果。令 \(H(\rho)=\int_0^\rho r(u)du\)、\(t_N=N^{-1}\sum_jY_{j1}Y_{j2}\)。旧统一极限与本包MC振幅界给

\[
\sup_{|\rho|\le R}\left|
\frac{\widehat F_{raw,n}(\rho)-\widehat F_{raw,n}(0)}{w_n}
+\tfrac14\{H(\rho)+r(\theta)\rho\}
\right|\to_p0.
\]

这里先条件于任意固定完整数据使用MC界，再由t_N→r(θ)转为新iid数据结论；不是同一bank实现同时对所有数据集成立。H为偶函数且H″=r′>0，所以极限函数严格凹。θ>0时两端点中R具有严格更低的值，θ<0时−R更低；θ=0时两个端点同为最小点。紧区间上的统一收敛及o_p(w_n)近优误差遂给

\[
\widehat\rho_{raw,n}\to_p\operatorname{sign}(\theta)R
\quad(\theta\ne0),\qquad
\operatorname{dist}(\widehat\rho_{raw,n},\{-R,R\})\to_p0
\quad(\theta=0).
\]

严格识别和紧区间内KL的有界连续性进一步给

\[
E\,\mathrm{KL}(Q_\theta\|Q_{\widehat\rho_{raw,n}})
\longrightarrow
\begin{cases}
\mathrm{KL}(Q_\theta\|Q_{\operatorname{sign}(\theta)R})>0,&\theta\ne0,\\
\mathrm{KL}(Q_0\|Q_R)>0,&\theta=0.
\end{cases}
\]

零teacher情形利用Q_0到±R的KL相同，故不需让raw的随机符号收敛。本结论只给raw参数趋向端点，不宣称有限n的数值程序恰好选择端点；θ=±R也不在此正风险失败声明内。

这排除了把该例子的学习失败完全归因于“训练bank不够精确”：同一B_n已足以使相对MC误差消失，raw仍趋向错误参数，而加上独立有限校准可以恢复。新增校准bank只是训练bank的约1/√n，但同bank不等于同总计算成本；每bank有K_n(n+2)个Gaussian标量，校准网格也有O(n)个评分点。数据N_n可任意发散，本文未声明总运行时间或位复杂度最优。

该构造使用已知active结构和精确已知r点值。数值r、浮点误差和任何其它实现误差均须另行控制为o_p(w_n)。本轮实际实验使用固定101/801节点及有限预算；数值资格不能直接宣称已验证这条渐近构造或严格有限风险界。边际化与MLE仍是必要的强基线，推论不证明校准在本模型中比它们高效。
