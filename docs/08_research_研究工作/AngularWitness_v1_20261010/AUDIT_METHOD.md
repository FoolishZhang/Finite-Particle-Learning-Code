# R01 作者内部独立角路径表达核对

本文件说明 `script/angular_witness_path_audit.py` 的数学对象、解析界和实现隔离范围。它是作者内部定向核对，不是外部科研审核。目标输入、源码冻结、一次性正式执行和最终结果由本轮主合同与 driver 管理；本模块没有目标科学执行命令，不读取旧数组或旧候选积分。

## 路径二阶导数

固定零偏置、精确有理行长向量 $a$，整数向量 $u\ne0$ 且 $\sum_i u_i=0$。记 $S=\sum_i u_i^2$、$d_i=u_i/\sqrt S$。目标为 $m=8$；模块允许较小的 $m$ 以进行制造输入测试。取独立标准正态 $Z_1,Z_2$，完整保持行长的角路径为

$$x_i(t)=a_i\{Z_1\cos(td_i)+Z_2\sin(td_i)\}.$$

因此 $x'_i(0)=a_id_iZ_2$，$x''_i(0)=-a_id_i^2Z_1$。给定标签 $y$，令 $p_i=\sigma(a_i z)$、$r_i=y_i-p_i$、$A_y(z)=\prod_i p_i^{y_i}(1-p_i)^{1-y_i}$。对条件概率乘积求导并解析使用 $E Z_2^2=1$，得到

$$Q_y''(0)=E\!\left[A_y(Z_1)\left\{\left(\sum_i a_id_ir_i\right)^2-\sum_i a_i^2d_i^2p_i(1-p_i)-Z_1\sum_i a_id_i^2r_i\right\}\right].$$

最后一项来自路径加速度，不能删去。本实现使用整数 $u$，将花括号中三个未归一化项共同除以 $S$；不近似计算 $\sqrt S$。保存 $N_{\rm path,y}=-Q_y''(0)$ 及独立计算的 $Q_y$。$Q_y'(0)=0$ 来自 $E Z_2=0$，没有进行二维有限网格求积。

在真 Gaussian 积分上，可通过分部积分将此式联系到主表达的配对式。该恒等式不意味着两个被积函数或有限网格和相等。本核对必须分别扩张两种表达的解析误差，再检查对应真积分区间是否相交；不得要求两个原始有限和逐节点或无误差地相等。

## 独立解析误差预算

以下界限定 $|a_i|\le1$、$\|d\|_2=1$、解析带 $|\operatorname{Im}z|\le1$。实现显式检查行长上界。设 $c=\sec(1/2)$、$C=c^m$。在边界 $z=x\pm i$，sigmoid 的分母无零点，且

$$|A_y|\le C,\quad |r_i|\le c,\quad |p_i(1-p_i)|\le c^2/4.$$

若 $T=\sum_i a_id_ir_i$、$D=\sum_i a_i^2d_i^2p_i(1-p_i)$、$B=\sum_i a_id_i^2r_i$，Cauchy–Schwarz 与 $\sum d_i^2=1$ 给出 $|T|\le\sqrt m c$、$|D|\le c^2/4$、$|B|\le c$。又有 $|\phi(x\pm i)|=e^{1/2}\phi(x)$ 以及 $E|Z\pm i|\le\sqrt2$，故每一侧边界的绝对积分可取

$$M_Q=e^{1/2}C,\qquad M_N=e^{1/2}C\{mc^2+c^2/4+c\sqrt2\}.$$

无限梯形和的解析带别名误差采用

$$e_{Q,\rm alias}=\frac{2M_Q}{e^{2\pi/h}-1},\qquad e_{N,\rm alias}=\frac{2M_N}{e^{2\pi/h}-1}.$$

实轴上 $|A_y|\le1$，未加 Gaussian 权的路径式绝对值至多 $m+1/4+|x|$。设 $L=Nh\ge1$，使用 $\phi(x)$ 和 $x\phi(x)$ 在相应正尾上的单调性，将遗漏的 full-weight 格点和从 $L$ 开始用积分上界控制，得到

$$e_{Q,\rm tail}\le\frac{2\phi(L)}L,\qquad e_{N,\rm tail}\le\left(m+\frac14\right)\frac{2\phi(L)}L+2\phi(L).$$

两端保留节点均取完整权重。目标 $h=1/8,N=88,L=11$，共 177 点；没有半端点、重归一化或后验调整。`path_bounds` 用独立标量 Arb 算术保存上述常数、alias、tail 和 total 的精确二进端点。有限和的 Arb 算术半径另存于其球中；比较时增加解析 total 的上界，不能把显示小数当区间端点。

## 实现边界及接口

- `finite_path_audit(a,u,step='1/8',N=88,bits=192,hard_seconds=600)` 返回 `angular_witness_path_finite_sum_v1`。正式 driver 必须锁定 $m=8$、256 标签、177 节点及精度。状态为 `completed` 或 `budget_stopped`；后者的部分和不得提升为真积分证书。
- `Q_finite_sum` 和 `N_path_finite_sum` 按整数标签 $0,\ldots,2^m-1$ 排列，第 0 个观察位为 MSB。每个标签重新按其整数位顺序乘入各 Bernoulli 因子；不调用主模块的动态标签乘积或配对核。
- `path_bounds(a,u,...)` 返回 `errors.Q_total` 和 `errors.N_path_total`，分别是标签统一的 $Q$ 和 $-Q''$ 解析误差。计算预算只做标量算术，不新增积分。
- 输入拒绝 float 行长、float 方向项、非整数/零/非零和方向以及 $|a_i|>1$。精确有理行长可用 `Fraction`、有理字符串或整数。所有函数临时设置所需 `ctx.prec` 并在退出时恢复。
- 编解码由本文件自行实现，字段为 `lower_exact_dyadic`、`upper_exact_dyadic`（整数 mantissa 与二的指数），仅兼容旧公开格式。没有从主积分模块或旧项目核导入函数。共享的可信算术库是公开 `python-flint 0.8.0` 与底层 Arb；因此这是不同表达/不同标签相乘实现的核对，不是不同算术库的核对。
- 计数记录完成节点、sigmoid/Gaussian 次数、逐标签概率乘法、路径行贡献与两个积分累积。时间预算在完整节点之间及最后一次节点后检查；单节点可能使实耗略过预算，此时状态仍为 `budget_stopped`。序列化精确端点的时间不属于该节点循环计时。

## 制造输入测试范围

`--self-test` 仅取 $m=3$、$a=(1/3,-1/2,2/5)$、$u=(1,2,-3)$。用精确有理 Bernoulli 因子导数与乘积二阶 jet 独立推导全部 8 标签的路径值，核对解析对数导数表达及 Arb 包含；额外检查 $a=0$ 的常数路径、标签总质量的二阶导数为零、精确端点往返、输入拒绝和精度恢复。数值 API 测试只执行 $h=1/2,N=1$ 的 3 节点制造有限和，并检查 MSB 标签互补对称和 Gaussian 有限质量；预算停止测试在首节点前停止。解析界测试只做制造参数的标量算术。

上述测试不执行目标 177 节点科学积分，不认证目标角见证的符号。正式全 256 标签交叉核对及最终判定仍由主 driver 在合同冻结后保存。
