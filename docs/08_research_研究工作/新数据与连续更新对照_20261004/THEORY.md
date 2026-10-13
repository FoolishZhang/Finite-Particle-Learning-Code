# 新数据外层的风险、目标与连续更新：理论和统计合同

2026-10-04。本合同只作解析/统计设计，不新增随机数、训练或模型积分。配套 DESIGN_REVIEW.md (historical reference not distributed)。不同 d 为相同固定生成律下的新数据实现，非新分布或领域。生产随机输入与实际资格以冻结 PLAN/CONFIG/源码为准。

## 1. 模型和三种程序

完整相关矩阵类为 E_.1={C=Cᵀ,diag(C)=1,C⪰.1I}，物理六相关向量rho线性决定active4的C。单位无关尺度、均值/方差/块位置保持；无关256位公平且独立，所有候选的精确无关观察PMF相同。因此完整观察KL等于active16模式的KL，但raw有限粒子全联合评分仍含无关随机权重。

令g_K(rho;B,Z)是一份K100输入的原raw路径梯度。固定状态和批次、四独立输入有

$$\bar g_4=\frac14\sum_{j=0}^3g_K(rho;B,Z_j),\quad E\bar g_4=\nabla F_{B,K}(rho),\quad\operatorname{Cov}(\bar g_4)=\Sigma_B(rho)/4.\tag{1}$$

在固定K/维数和SPD紧类下，Cholesky导数有界，log-sigmoid导数有界，梯度有有限二阶矩；式(1)为实算术身份。批次固定循环，冻结一个参数后对全日程批次平均才给完整F_D梯度；不能称每个固定批次都是完整数据的无偏随机梯度。浮点cast与原API误差由数值资格另核。

IN4在同状态计算四个原梯度，平均后一次Adam/修复。OUT4四成员各自私有状态/Adam/修复，每宏步一份bank，末端平均四个物理TAIL。SEQ4每宏步依次bank0..3四次更新，每次lr=.02，在当前新状态评分。新bank相对于该路径先前输入独立，所以每个微步的条件平均梯度仍对应当前批次原目标；但Adam一二阶矩、偏差修正及修复非线性，不能据此说各程序具有相同期望更新。

同样的目标与输入预算没有给出相同访问状态、优化误差、输出分布或风险。SEQ单路径2048步，IN单路径512步，OUT每成员512步；比较的是预算使用方式，无优化器收敛或全局最优定理。

## 2. 物理TAIL和OUT身份

令rho_A,dr,t是第d数据、第r四元组、策略A宏步t末态。主TAIL为

$$\widehat\rho_{A,dr}=\frac1{64}\sum_{t=449}^{512}\rho_{A,dr,t}.\tag{2}$$

SEQ的rho_t专指第4微步末态，不包括前3微步。OUT成员分别按(2)构造，再取

$$\widehat\rho_{OUT,dr}=\frac14\sum_{j=0}^3\widehat\rho_{M_j,dr}.\tag{3}$$

相关矩阵类凸，故实算术下(2)(3)可行；实际float64平均与谱边界仍需检查。没有额外投影、成员选择或预测混合。

给定D，四成员的完整prior流独立同分布。设单成员条件均值mu_D、协方差Sigma_D，则

$$E(\widehat\rho_{OUT}\mid D)=\mu_D,\qquad\operatorname{Cov}(\widehat\rho_{OUT}\mid D)=\Sigma_D/4.\tag{4}$$

与IN/SEQ共享输入不破坏成员间条件独立，因为成员状态没有从其他程序接受反馈。式(4)只相对单成员成立，非OUT相对IN或SEQ的身份，也非KL分解。Q_Cbar一般不等于四Q_Cj的预测混合，不能套用混合PMF的KL凸性保证OUT风险降低。

## 3. 外层随机性与主风险

D_d，d=0..15，为新 iid 完整N4096样本；给定D_d，Z_d0、Z_d1为两个独立训练四元组。各d的全部输入独立，各策略在一个quartet内按合同共享输入配对。令C_A(D,Z)为既定部署输出，定义

$$\theta_R^{A-B}=E_D E_Z[R(C_A(D,Z))-R(C_B(D,Z))],\quad R(C)=D(P\Vert Q_C).\tag{5}$$

I参照与D/Z无关，代入固定R(I)。每外层单位为

$$U^R_d=\frac12\sum_{r=0}^1[R(C_{Adr})-R(C_{Bdr})],\quad E U^R_d=\theta_R^{A-B}.\tag{6}$$

U_d iid，样本量16、df15。两重复彼此条件独立，但共享同一随机D而非无条件独立数据单元。对一个差值X(D,Z)，外层方差可写

$$\operatorname{Var}(U^R)=\operatorname{Var}_D(E[X\mid D])+\frac12E_D\operatorname{Var}_Z(X\mid D).\tag{7}$$

式(7)解释R=2怎样减少条件训练方差；不从16实现估计每个来源的因果贡献。主对象是单个既定模型的期望风险，不是两个输出再平均参数的另一个部署器。

## 4. 独立raw P评价

对固定C，F_P(C)=E_Y~P E_Z~prior[-log Qhat_C,K(Y;Z)]。评价用16个active模式的truth数值PMF作为权重，公平无关块利用独立prior期望的标签对称性，保留raw联合内层估计。模型真P权重的数值近似误差另核，不由下面的概率区间消除。

给每d独立W_d，与所有D/Z独立；W_d含G=8组各512独立bank，K100，每d共享于10本地候选和raw/FACT profile。设f_dg(C)是组评分，E_W f_dg(C)=F_P(C)。主目标为

$$\theta_F^{A-B}=E_D E_Z[F_P(C_A(D,Z))-F_P(C_B(D,Z))].\tag{8}$$

实现单位

$$U^F_d=\frac1{2G}\sum_{r=0}^1\sum_{g=1}^{G}[f_{dg}(C_{Adr})-f_{dg}(C_{Bdr})],\quad E U^F_d=\theta_F^{A-B}.\tag{9}$$

不同d的W独立，故U^F_d也iid，df15。d内候选/重复的评分相关是配对设计的一部分，由该单位的方差捕获；组数或候选数不能代替外层样本量。

令T_d=E[U^F_d\mid D_d,Z_d0,Z_d1]，则

$$\operatorname{Var}(U^F)=\operatorname{Var}(T)+E\operatorname{Var}(U^F\mid D,Z_0,Z_1).\tag{10}$$

样本外层SE已覆盖两项，不另把条件MC SE加一次。共享跨d评价池会引入跨外层依赖；单个单位的全方差身份(10)仍成立，但不能据此直接使用简单iid t，故本合同使用独立d池。组/512bank仅是执行和诊断单元，不构成64个训练总体重复。

raw D把权重换为本数据的active经验频数，d内同池算术有效；只描述，没有新主门。FACT评价profile的分解正确性不令其等于精确似然，它仍有active有限K。

## 5. 家族、精度和范围

对16单位U定义barU=mean(U)，s²=sum(U−barU)²/15，SE=s/4。每个q项家族用双侧区间

$$[\bar U-t_{15}(1-\alpha/(2q))SE,\;\bar U+t_{15}(1-\alpha/(2q))SE].\tag{11}$$

风险家族alpha=.025/q5：OUT−IN、SEQ−OUT（SE门.0005），OUT−I、SEQ−I、FACT−I（SE门.0002）。raw P家族alpha=.025/q2：OUT−IN、SEQ−OUT（SE门.0005）。区间正/负/跨0分别为升高/降低/未分辨，精度门独列。Bonferroni不要求家族内独立；t本身为开发近似，合计alpha=.05不是严格有限样本覆盖认证。

较低风险与较高raw P若同时分辨，可支持这个固定生成律、固定程序下的平均排序分离；不证明逐数据/逐输出同时发生、不证明充分优化原目标、不说明更准确优化必然更坏。相对I分辨改善也不证明与truth等效或有效恢复全部依赖。无效、等效、非劣都不能从CI含0推出。

科学范围预定为一次SEQ针对性对照和新数据证据；主门不支持、SE失败或反向结果均保留，不更换lr、TAIL、统计单位或追加预算。

## 6. 精确Jensen账与派生区间

对每Y，Qhat是正且无偏的有限prior似然估计。定义

$$J_P(C)=E_{Y\sim P}[\log Q_C(Y)-E_Z\log\widehat Q_{C,K}(Y;Z)]\ge0.\tag{12}$$

Gaussian/log-sigmoid有限K估计具所需对数可积性，Jensen给非负。在本模型，true及候选无关边际都相同，因此full风险可由active积分；完整H(P)保留共同无关熵。严格有

$$F_P(C)=H(P)+R(C)+J_P(C),\quad\theta_F^{A-B}=\theta_R^{A-B}+\theta_J^{A-B}.\tag{13}$$

新输出逐d可构造U^J_d=U^F_d−U^R_d，估计theta_J。但二者通过训练输出相关，Var(U^J)=Var(U^F)+Var(U^R)−2Cov(U^F,U^R)，不能独立相加SE。本合同不新增J检验；只给点值和逐d描述。

若风险共同界为[L_R,U_R]、评分共同界[L_F,U_F]，同时成立时

$$\theta_J\in[L_F-U_R,\;U_F-L_R].\tag{14}$$

(14)继承既有两家族联合的开发近似，无新增alpha或主门。不能把两个随机区间用同向端点相减，也不能用评分SE单独给新训练总体J区间。J由完整随机似然分布决定，非单一方差、MC标准误或优化动力学因果份额。

旧冻结面板给ΔF=.00169766620173、固定数值ΔR=−.00313609713761，故ΔJ=.00483376333934。原P区间减去固定面板风险得[.00433880849126,.00532871818742]；这是不同于(14)的新旧对象区分：旧区间条件于实际面板，不覆盖重训或积分误差，不能升级其外推范围。

## 7. 可选参数描述，非机制确认

对每d、每A的两个条件独立输出x/y，真实物理向量rho*给

$$E[\tfrac12\|x-y\|^2\mid D]=\operatorname{tr}\operatorname{Cov}(\widehat\rho_A\mid D),\quad E[(x-\rho_*)^\top(y-\rho_*)\mid D]=\|E(\widehat\rho_A\mid D)-\rho_*\|^2.\tag{15}$$

逐实现二者相加为两输出参数平方误差均值。可平均16新D作描述，负交叉积保留；无参数主假设，R=2不提供可靠逐数据中心估计。式(15)非KL身份，非目标/数据/Adam误差因果分摊。

## 8. 资格与停止边界

积分采用事前固定阶数/容差，原API与稳定math64对照单列各bank及SEQ各微步真实状态。实算术身份、浮点资格、t开发区间与严格积分证书互不替代。生产和独审范围在科学输入前固定，异常保留；底层限定子集不称全流程重跑。

本理论合同不主张finite-F全局凹、Adam近优、谱下界定理已在当前512/2048步实现、OUT普遍预算最优或通用恢复。固定生成律新数据证据和一次连续更新对照完成后收口，下一稿可把它纳入实现层章节，不以持续扩展方法排名延迟整稿。
