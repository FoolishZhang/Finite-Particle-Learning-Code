# Same-target outer averaging improves conditional risk and dispersion

On four fixed old datasets, sixteen fresh prior-repeat columns compare one K100-bank gradient with the average of four independent K100-bank gradients before one unchanged Adam/feasibility update. The expected fixed-parameter, current-batch criterion is preserved. L4 uses four times the sampling/scoring per update.

Mean integrated KL decreases from 0.007042179 to 0.005812921; the difference is −0.001229258 with simultaneous development interval [−0.002279813,−0.000178703]. Conditional parameter dispersion, estimated with eight disjoint repeat-column pairs, decreases from 0.383091588 to 0.210696251, with difference interval [−0.264122502,−0.080668172]. Both prespecified precision targets are met.

A separate independent MC family places the realized-panel raw D and P score differences below zero and meets its precision targets. These intervals condition on the trained matrices and are not training-prior population score intervals.

Prespecified descriptive estimates move differently: debiased conditional mean-parameter error and KL at each sixteen-output mean matrix increase on all four datasets. No new interval confirms that center shift. The intervention is not target correction, a pure variance-only dynamical change, or a causal allocation of the old FREE gain. Next priority is a new-prior, computation-matched comparison with four independent L1 trajectories averaged at output; fixed preconditioning is a subsequent mechanism control. v7 remains immutable. See the report (historical reference not distributed).
