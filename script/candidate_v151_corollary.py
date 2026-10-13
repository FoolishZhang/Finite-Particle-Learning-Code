"""Deterministic scalar check for the v1.5.1 shared-event region corollary.

Reads only public text/JSON and manifest entries. No scientific arrays, RNG,
score evaluation, optimization or numerical integration is used. --write
creates one new authoring receipt; --check recomputes it without modifying it.
"""
from pathlib import Path
from fractions import Fraction
import argparse
import hashlib
import json
import math
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/submission_candidate_v1_5_1_20261006'
TARGET=OUT/'SHARED_EVENT_REGION.json'
HANDOFF_SHA='701c9e70987266c1ad9f52750461fe30afa99eefd00ae438150c7d3ad702500e'
REVIEWED_COMMIT='8ff1111b1a8078848225b96101450de291381741'
JROOT='artifacts/joint_risk_refinement_v1_20261006'
FROOT='artifacts/failure_probability_v1_20261006'
OLD_PAPER='docs/07_papers_论文工作/ParticleJointSupervision/submission_candidate_v1_5_20261006'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def exact_record(value):
    q=Fraction.from_float(float(value))
    return dict(value=float(value),hex=float(value).hex(),numerator=str(q.numerator),denominator=str(q.denominator))


def decimal_outward(q,places=21):
    q=Fraction(q);scale=10**places
    low=q.numerator*scale//q.denominator
    high=-((-q.numerator*scale)//q.denominator)
    def fmt(n):
        sign='-' if n<0 else '';n=abs(n)
        return f'{sign}{n//scale}.{n%scale:0{places}d}'
    return dict(lower=fmt(low),upper=fmt(high),places=places)


def compute():
    private=ROOT/'.cache/finite_selection_tools'
    if str(private) not in sys.path:sys.path.insert(0,str(private))
    from flint import arb,ctx
    paths=[OUT/'HANDOFF_INPUT.md',ROOT/f'{JROOT}/RESULTS.json',ROOT/f'{JROOT}/DELIVERY_MANIFEST.json',
           ROOT/f'{FROOT}/RESULTS.json',ROOT/f'{FROOT}/FREEZE.json',ROOT/f'{FROOT}/DELIVERY_MANIFEST.json',
           ROOT/'docs/08_research_研究工作/JointRiskRefinement_v1_20261006/PROOF.md',
           ROOT/'docs/08_research_研究工作/FailureProbability_v1_20261006/PROOF.md',
           ROOT/f'{OLD_PAPER}/APPENDIX_FINITE_EN.md',ROOT/f'{OLD_PAPER}/APPENDIX_PROBABILITY_EN.md']
    source_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    assert source_hashes[(OUT/'HANDOFF_INPUT.md').relative_to(ROOT).as_posix()]==HANDOFF_SHA
    joint=read(ROOT/f'{JROOT}/RESULTS.json');fresh=read(ROOT/f'{FROOT}/RESULTS.json')
    bindings=[]
    for folder in (JROOT,FROOT):
        manifest=read(ROOT/f'{folder}/DELIVERY_MANIFEST.json')
        name=f'{folder}/RESULTS.json'
        assert manifest['tracked_evidence'][name]['sha256']==source_hashes[name]
        bindings.append(dict(source=name,manifest=f'{folder}/DELIVERY_MANIFEST.json',
                             source_sha256=source_hashes[name],manifest_sha256=source_hashes[f'{folder}/DELIVERY_MANIFEST.json']))
    freeze=read(ROOT/f'{FROOT}/FREEZE.json')
    assert (freeze['K'],freeze['L'],freeze['T'],freeze['alpha'])==(100,262144,6,.01)
    assert fresh['inference']['audit_alpha']==.01 and fresh['inference']['family_components']==3
    assert fresh['inference']['g_from_old_family_used'] is False
    assert joint['claims']['original_passed_gates']==6 and joint['claims']['original_total_gates']==7
    assert joint['claims']['post_hoc_analytic_refinement'] is True
    old_e=(ROOT/f'{OLD_PAPER}/APPENDIX_FINITE_EN.md').read_text(encoding='utf-8')
    old_h=(ROOT/f'{OLD_PAPER}/APPENDIX_PROBABILITY_EN.md').read_text(encoding='utf-8')
    assert r'F_n(C)&=\mathbb E\,8^{-1}\sum_y\phi' in old_e
    assert r'g=\mathbb E d(Y,W)' in old_h and r'd(y,w)&=\phi(I;y,w)-\phi(C_b;y,w)' in old_h
    assert r'a_k(W)=\frac{e^{S_k}}{\sum_{l=1}^{K}e^{S_l}}' in old_h
    g_lower=fresh['inference']['g_lower']
    cost_record=joint['constants']['new_population_cost']
    cost_upper=cost_record['upper']
    g_exact=Fraction.from_float(g_lower);c_exact=Fraction.from_float(cost_upper)
    difference_exact=g_exact-c_exact
    strict_threshold=Fraction(67,50000)
    assert difference_exact>strict_threshold
    previous=ctx.prec;ctx.prec=256
    try:
        def enclosure(value):
            return dict(lower=math.nextafter(float(value.lower()),-math.inf),
                        upper=math.nextafter(float(value.upper()),math.inf),arb=str(value))
        r0=arb(1)/10**6;K=100
        beta=(-arb(1)/2).exp()/4
        radius=2*(1+(8*r0).sqrt())*r0.sqrt()/beta
        lam=1-radius
        bK=arb(8)/3*(arb(K).log()+3*arb(2).log())
        curvature=13*bK/(16*lam)+(3*bK).sqrt()/(4*lam*lam.sqrt())
        cost=curvature*radius**2/2
        assert arb(cost_record['lower'])<=cost<=arb(cost_record['upper'])
        recomputed_cost=enclosure(cost)
        intersection=[max(cost_record['lower'],recomputed_cost['lower']),min(cost_record['upper'],recomputed_cost['upper'])]
        assert intersection[0]<=intersection[1]
        assert arb(intersection[0])<=cost<=arb(intersection[1])
        margin=arb(g_lower)-arb(cost_upper)
        exact_margin=arb(difference_exact.numerator)/difference_exact.denominator
        assert margin.contains(exact_margin)
        assert margin>arb(67)/50000
        margin_record=enclosure(margin)
        assert Fraction.from_float(margin_record['lower'])<=difference_exact<=Fraction.from_float(margin_record['upper'])
        # The scientific g_L already targets the untruncated g: its published
        # lower endpoint subtracts both the original numerical allowance and
        # the analytical first-moment tail correction. Reconstruct only this
        # scalar relationship to confirm that T=6 is not a changed target.
        scalar_mean=arb(fresh['scalar_statistics']['X']['mean']['enclosure'])
        epsilon=arb(fresh['inference']['constants']['epsilon_X']['enclosure'])
        halfwidth=arb(fresh['inference']['confidence_components']['X']['EB_halfwidth']['enclosure'])
        tail=arb(fresh['inference']['constants']['tau1']['enclosure'])
        g_expression=scalar_mean-epsilon-halfwidth-tail
        assert arb(g_lower)<=g_expression
        numeric=dict(bits=256,sealed_g_lower=exact_record(g_lower),
                     sealed_population_cost_upper=exact_record(cost_upper),
                     sealed_population_cost_interval=[cost_record['lower'],cost_record['upper']],
                     recomputed_analytic_radius=enclosure(radius),recomputed_analytic_cost=recomputed_cost,
                     cost_intersection=intersection,recomputed_cost_within_sealed_interval=True,
                     formal_margin_uses_sealed_cost_upper=True,
                     shared_event_margin=margin_record,
                     exact_margin_fraction=dict(numerator=str(difference_exact.numerator),denominator=str(difference_exact.denominator)),
                     exact_margin_decimal_enclosure=decimal_outward(difference_exact),
                     strict_threshold='0.00134',strictly_above_threshold=True,
                     threshold_slack_decimal_enclosure=decimal_outward(difference_exact-strict_threshold),
                     tail_corrected_g_expression_check=enclosure(g_expression),
                     g_lower_targets_untruncated_population_gap=True)
    finally:ctx.prec=previous
    return dict(schema_version='1.0',kind='deterministic_authoring_corollary_check',
                reviewed_commit=REVIEWED_COMMIT,handoff_sha256=HANDOFF_SHA,handoff_use='reference',
                handoff_input=(OUT/'HANDOFF_INPUT.md').relative_to(ROOT).as_posix(),
                source_hashes=source_hashes,source_manifest_bindings=bindings,
                script_sha256=sha(Path(__file__)),
                model_identity_author_checks=dict(m=3,known_mean=[0,0,0],known_unit_marginal_variances=True,
                    teacher_correlation='I',K=100,n=256,correlation_spectral_floor='1/10',
                    anchor_offdiagonal='9/10',teacher_labels='uniform over eight signs',
                    common_target='g = E_uniform_Y E_W[phi(I;Y,W)-phi(C_b;Y,W)] = F_256(I)-F_256(C_b)',
                    active_nuisance_independence_same=True,normalized_nuisance_weights_same=True,
                    positive_symmetric_candidate_root_same=True,candidate_independent_normalization_cancels=True,
                    T6_is_audit_truncation_not_target_change=True,mean_tail_correction_already_in_g_lower=True,
                    population_cost_uses_deterministic_bK_not_empirical_U=True,
                    cost_needs_no_old_random_coverage_event=True,
                    identity_is_author_analysis_not_automated_proof=True),
                numeric=numeric,
                probability_scope=dict(same_new_audit_event=True,audit_confidence=.99,
                    audit_family_components=3,new_alpha_spent=0,
                    same_event_supports_main_12_region_and_main_13_probability=True,
                    mathematical_program_probability_lower_unchanged=fresh['inference']['mathematical_program']['probability_lower'],
                    four_family_joint_coverage_unchanged=.96,five_family_joint_coverage_unchanged=.95,
                    continuous_optimizer_fresh_run_probability_proved=False,
                    original_input_and_sampling_qualifications_retained=True),
                history_preserved=dict(old_region_lower=joint['constants']['new_population_region_lower']['lower'],
                    original_passed_gates=6,original_total_gates=7,old_scientific_results_rewritten=False,
                    reviewer_decimal_used_as_formal_bound=False),
                resources=dict(new_random_draws=0,new_banks=0,new_observations=0,new_score_calls=0,
                    new_fits=0,new_integrals=0,new_quadrature_nodes=0,new_statistical_comparisons=0,
                    new_alpha_spent=0,large_arrays_read=0),
                manuscript_insertion=dict(section='D.5',equation='D.14',deterministic_source_appendix='C',
                    probability_source_appendix='D',main_region_equation='12',main_probability_equation='13'))


def main(write):
    record=compute()
    if write:
        OUT.mkdir(parents=True,exist_ok=True)
        with TARGET.open('x',encoding='utf-8',newline='\n') as f:
            json.dump(record,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    else:
        assert read(TARGET)==record,'saved scalar receipt differs from read-only recomputation'
    print(json.dumps(dict(passed=True,written=write,record=TARGET.relative_to(ROOT).as_posix(),
                         record_sha256=sha(TARGET),margin=record['numeric']['exact_margin_decimal_enclosure'],
                         strictly_above_point00134=True,new_alpha_spent=0,large_arrays_read=0),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--write',action='store_true');mode.add_argument('--check',action='store_true')
    args=parser.parse_args();main(args.write)
