# LocalWitnessFeasibility-v1：解析截断尾界

本文件是作者的局部可行性推导与制造测试说明，不是外部审稿。它只给标准 Gaussian 银行、完整标签概率及固定 scale=1 的导数截断误差外包；不认证旧 Gauss–Legendre 规则，不执行真实中心积分，不给驻点存在性或局部极小证明。所有矩阵界均为谱范数，梯度界为 Euclidean 范数。

## 对象、坐标与尺度

固定标签 $y\in\{0,1\}^m$，令

\[
A_k=f_y(z_k)=\prod_j\sigma(b_j+a_jz_k)^{y_j}
(1-\sigma(b_j+a_jz_k))^{1-y_j}>0,\quad S=\sum_{k=1}^K A_k,
\quad w_k=A_k/S.
\]

银行损失是 $-\log(S/K)$。按 $[b,a]$ 排列，$r_k=y-\sigma(b+az_k)$、$G_k=(r_k,z_kr_k)$、$g=\sum_k w_kG_k$、$B_k=\operatorname{diag}_j\{\sigma'_j(1,z_k)(1,z_k)^T\}$，故

\[
\nabla\ell=-g,\qquad H=\sum_kw_kB_k-\sum_kw_kG_kG_k^T+gg^T.
\]

这些与旧 `rankone_stationary_kernel_test.py:direct_bank` 的直接 K2 全导数一致；旧 `rankone_stationary_audit.py:independent_label` 另外保留 distinct-particle 的 $K(K-1)$ 项。无 $1/K$ 遗失：损失中的 $\log K$ 对参数导数为零，Laplace 指数是 $e^{-uS}$。其一粒子分解含 $K\Psi^{K-1}$ 与 $K(K-1)\Psi^{K-2}$，与直接银行表达式同一。

**旧 kernel 的 v 尺度需要显式处理。** `negative_ux=-exp(v+logA-logscale)`，故物理 $u=\exp(v)/\mathrm{scale}_y$。默认 scale 是内层求积 evidence/normal_mass；固定截断 v 时，必须把 scale 看作固定坐标变换，不能随参数对它求导。本 API 固定 `likelihood_scale=1`，于是 $u=e^v$，Frullani subtraction 是 $e^{-Ku}$。这里没有额外 $u/K$。若继续采用旧默认 scale，必须另带 label scale 的上下外包，分别修改上下 u 尾，不能直接套本 API。

## 全域包络

设 $M=\max_k|z_k|$。因 $|r_{kj}|\le1$、$0\le\sigma'_j\le1/4$，

\[
\|G_k\|^2\le m(1+M^2),\quad
\|B_k\|\le(1+M^2)/4,
\quad\|\nabla\ell\|\le\sqrt m(1+M^2),
\quad\|H\|\le(2m+1/4)(1+M^2).
\]

横向坐标是加载矩阵第二列 $W_{j2}$，不是角度。解析积分独立 $z_{2k}\sim N(0,1)$ 后，

\[
H_\perp=\operatorname{diag}\sum_k w_k\sigma'_k-
\sum_k w_k(1-w_k)r_kr_k^T,
\qquad \|H_\perp\|\le m+1/4.
\]

这个包络不带 $M^2$。角度 Hessian 需要额外加载因子和 Ward/径向项；本 API 不输出其误差界。以上包络也适用于有限 u 区间的导数积分：d、same、cross 系数皆非负，各总质量不超过 1；横向 c 总质量不超过 1。因此，下面的 Gaussian/teacher 误差能按三角不等式与 u 尾叠加，而非要求旧内层规则准确。

## 上 u 尾

对固定银行，梯度余项精确为

\[
R_G(U)=-g e^{-US}.
\]

固定物理 U 后对参数求导，径向 Hessian 余项为

\[
R_H(U)=(H+USgg^T)e^{-US},\quad
\|R_H(U)\|\le[(2m+1/4)+mUS](1+M^2)e^{-US}.
\]

用 $s e^{-s/2}\le2/e$，得到 $C_H=2m+1/4+2m/e$ 乘 $E[(1+M^2)e^{-US/2}]$。梯度可保守用 $\sqrt m$ 乘同一加权量。

横向的 d 尾为 $w_ke^{-US}$，c 尾为 $w_k(1-w_k)(1+US)e^{-US}$，从而

\[
\|R_\perp(U)\|\le[1/4+m(1+US)]e^{-US}
\le(m+1/4+2m/e)e^{-US/2}.
\]

若至少一粒子 $|z_k|\le T$，则 $S\ge\eta$，其中对参数 box 的保守统一下界

\[
\eta=2^{-m}\exp(-\|b\|_1-T\|a\|_1)
\]

由 $\sigma(\pm x)\ge\tfrac12e^{-|x|}$ 得到。输入的两项 l1 值必须是实际 box 的**上界**。

令 $p(x)=P(|Z|>x)$、$t_2(x)=E[Z^2\mathbf1_{|Z|>x}]$。代码使用 Mills 外包

\[
\bar p(x)=2\phi(x)/x,\qquad
\bar t_2(x)=2\phi(x)(x+1/x),\qquad x>0.
\]

即便 $\bar p>1$，仍是合法但较松的上界；未通过裁剪制造精度。独立银行全部超 T 的事件满足

\[
E[(1+M^2)\mathbf1_{\mathrm{all}}]
\le\bar p(T)^K+K\bar t_2(T)\bar p(T)^{K-1}.
\]

于是定义

\[
D_w=(1+K)e^{-U\eta/2}+\bar p(T)^K+
K\bar t_2(T)\bar p(T)^{K-1},\quad
D_0=e^{-U\eta/2}+\bar p(T)^K.
\]

上 u 尾三界依次为 $\sqrt mD_w$、$C_HD_w$、$(m+1/4+2m/e)D_0$。这里 $E(1+M^2)\le1+K$，故 good 事件的加权部分可用 $1+K$；不需要把 K 放入 eta。

## 下 u 尾

令 $u_0=e^{v_{\min}}$、$Q=\sum_k A_kG_k$。梯度积分密度为 $-Qe^{-uS}$，径向 Hessian 密度为

\[
\left[\sum_k A_k(B_k-G_kG_k^T)+uQQ^T\right]e^{-uS}.
\]

用 $A_k\le1$、$E(1+Z^2)=2$、$E\|Q\|^2\le K\sum_k E\|G_k\|^2\le2mK^2$，得

\[
\|R_G(0,u_0)\|\le K\sqrt{2m}\,u_0,
\quad\|R_H(0,u_0)\|\le2K(m+1/4)u_0+mK^2u_0^2.
\]

横向密度是 $\operatorname{diag}\sum_k A_k\sigma'_ke^{-uS}
-u\sum_k A_k(S-A_k)r_kr_k^Te^{-uS}$，因此

\[
\|R_\perp(0,u_0)\|\le Ku_0/4+mK(K-1)u_0^2/2.
\]

此处只预算导数尾，不预算完整 loss 的 Frullani 下尾。

## Gaussian 内层与 teacher 截断

把各银行 z1 限制在 $[-L,L]$ 的原始 measure，不重归一化。令 bad 表示至少一粒子超 L。并集界与独立性给

\[
E[(1+M^2)\mathbf1_{bad}]\le K[\bar t_2(L)+K\bar p(L)],
\qquad P(bad)\le K\bar p(L).
\]

径向 gradient/Hessian 分别乘 $\sqrt m$、$2m+1/4$；横向乘 $m+1/4$ 和未加权并集概率。它们是完整银行 measure 截断的界，不能认证一个 finite-order 内层 quadrature。

若 teacher 是二维标准 Gaussian，截到 $[-L_t,L_t]^2$，完整标签分布的 L1 遗漏质量至多 $2\bar p(L_t)$。原因是非负 complete label conditional probabilities 在每个 teacher 点之和为 1；**没有 $2^m$ 或 256 倍因子**。通过银行全域期望包络传导，gradient/Hessian 乘 $\sqrt m(1+K)$、$(2m+1/4)(1+K)$，横向乘 $m+1/4$。只有 teacher 的 measure 截断受此界控制；teacher PMF 的有限规则积分误差另计。

## 严格标量实现与验收

`script/local_witness_feasibility_bounds.py:bound_box(m,K,b_l1,a_l1,normalL,teacherL,vmin,vmax,T)` 使用已有私有 python-flint，默认 192-bit Arb。整数必须是实际整数，拒绝 bool 与整数形式浮点；标量输入只接受 exact rational 字符串/int/Fraction，拒绝 float。JSON 保存输入有理数、版本、精度、所有 bound 的精确 dyadic 上下端点；十进制 Arb 字符串只供阅读。巨大二进制指数不展开成天文长度的分母。`targets` 比较用整数交叉乘与位数比较完成，是 dyadic 上端点与 rational target 的严格比较。

每类 `total_truncation` 是上/下 u、银行 Gaussian、teacher Gaussian 四项和，未含任何有限区间求积误差。制造测试选 $m=1,b=a=0,y=1$、$K=1,2,5$，likelihood 恒为 1/2，Gaussian moments 可直接解析。它检查上下 Laplace 尾的精确表达式（含可能为负的横向上尾）被包络覆盖；另核 Gaussian erfc/moment 外包、实际整数输入、precision 恢复与巨大 dyadic 指数的 rational 比较。不读科学数组，不抽随机数，不调用 `acb.integral`。

运行：`.venv/Scripts/python.exe script/local_witness_feasibility_bounds_test.py --write-receipt`。收据以 exclusive-create 保存到 `artifacts/local_witness_feasibility_v1_20261008/BOUNDS_MANUFACTURED_CHECK.json`，已存在时只运行无写入测试。

仍未闭合：真实 box/中心的固定预算、有限区间积分误差、邻域 Hessian 变化、严格驻点存在性及局部极小 witness。本尾界单独通过不会把旧数值签名升级为严格证书。
