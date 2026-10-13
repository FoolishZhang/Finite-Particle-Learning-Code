# 完整Hessian与局部见证结果表

Arb显示值仅供阅读；严格判定用对应JSON的exact dyadic端点及冻结有理门。下界常数的精确意义见WITNESS_ARGUMENT。

|对象|外包/上界表达式|
|---|---|
|finite_matrix_to_saved_Frobenius_upper/Hba|[1.4554511792358950518763505411757851410092769496712871230e-13 +/- 5.67e-69]|
|finite_matrix_to_saved_Frobenius_upper/Hperp|[2.6623073951579798537572613265177177950406096431247386301e-14 +/- 5.76e-70]|
|true_matrix_to_saved_operator_upper/Hba|[1.13772174001590904824213664179732751164088658335332062991e-8 +/- 3.68e-65]|
|true_matrix_to_saved_operator_upper/Hperp|[2.53053999724766321330879772872262397101310519162618054293e-9 +/- 4.03e-66]|
|radial_curvature_lower|[0.0419655000000000000000000000000000000000000000000000000000 +/- 5.35e-59]|
|transverse_curvature_lower|[0.0059655000000000000000000000000000000000000000000000000000 +/- 1.14e-59]|
|outward_capacity|[2.09827500000000000000000000000000000000000000000000000000e-8 +/- 4.12e-65]|
|root_distance_upper|[1.80433927869321228151696036029595739357329234728527004325e-8 +/- 2.49e-65]|
|root_plus_step_distance_upper|[1.18043392786932122815169603602959573935732923472852700432e-7 +/- 5.34e-64]|
|F8_increase_lower_bound|[2.09827500000000000000000000000000000000000000000000000000e-16 +/- 5.26e-73]|
|risk_decrease_lower_bound|[2.01723000000000000000000000000000000000000000000000000000e-9 +/- 2.39e-66]|
|root_KL_strict_lower_bound|[2.01723000000000000000000000000000000000000000000000000000e-9 +/- 2.39e-66]|

|已核前提|通过|
|---|---|
|saved_floors_verified|True|
|Hessian_error_gates|True|
|predecessor_gradient_below_fixed_cap|True|
|radial_margin_positive|True|
|transverse_margin_positive|True|
|outward_root_condition|True|
|fixed_slice_transverse|True|
|risk_uniform_cap|True|
|fixed_step_stays_in_ball|True|

结论：真根存在、模共同旋转严格局部极小、正观察风险、固定小步F8上升/R下降，均通过。未重新优化或求出真根精确坐标。

operator距离<1.138e−8/<2.531e−9，根距中心<1.805e−8，经exact Fraction端点舍入核验。冻结有理式给F8增量≥2.098275e−16及风险降幅>2.01723e−9；这些是保守下界，不是实际效应大小估计。

独立复算为2/41类、384矩阵项，全部区间重叠；不称全量独立重算。
