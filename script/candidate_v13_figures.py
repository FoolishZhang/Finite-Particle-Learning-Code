"""Plot saved contrasts only; no scientific sampling or revised intervals."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/submission_candidate_v1_3_20261006'
DOC=ROOT/'docs/07_papers_论文工作/ParticleJointSupervision/submission_candidate_v1_3_20261006'
SOURCE=ROOT/'artifacts/nuisance_contrast_v1_20261006/POPULATION.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (OUT/'VERSION_LOCK.json').exists()
p=json.loads(SOURCE.read_text('utf-8'))
plt.rcParams.update({'font.size':11,'font.family':'DejaVu Sans','svg.hashsalt':'candidate-v1.3'})
fig,ax=plt.subplots(figsize=(8.1,3.4),layout='constrained')
for y,r in zip([2,1,0],p['results']):
 mean=r['mean']*1000;lo,hi=[v*1000 for v in r['interval']]
 ax.errorbar(mean,y,xerr=[[mean-lo],[hi-mean]],fmt='o',color=['#32699b','#b65329','#387966'][2-y],capsize=5,lw=2,ms=6)
ax.axvline(0,color='#666666',lw=1)
cost=p['region_cost']['mid']*1000
ax.vlines(cost,.65,1.35,color='#b65329',linestyles='--',lw=2)
ax.annotate('Region cost 4.362\n(for nuisance gap only)',xy=(cost,1.35),xytext=(13,1.85),arrowprops={'arrowstyle':'-','color':'#b65329'},color='#8b3a1d',fontsize=10)
ax.set_yticks([2,1,0],['Uniform gap $g_0$','Nuisance gap $g_{256}$','Paired interaction'])
ax.set_xlabel('Score contrast (in $10^{-3}$ units); positive favors correlated anchor')
ax.set_ylim(-.45,2.6);ax.set_xlim(-38,46);ax.grid(axis='x',color='#e4e7eb',lw=.7)
ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
for ext in ('png','svg'):
 kw={'dpi':240} if ext=='png' else {'metadata':{'Date':None}}
 fig.savefig(DOC/f'figures/nuisance_contrast.{ext}',**kw)
plt.close(fig)
record={'outputs':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted((DOC/'figures').glob('*')) if p.is_file()},'new_scientific_computation':False,'source_data':SOURCE.relative_to(ROOT).as_posix(),'source_sha256':sha(SOURCE),'figure_script_sha256':sha(Path(__file__)),'main_figures':['preference_geometry','learning_risk'],'supplementary_figures':['representation_structure','overlap_coupling','readout_and_selection','nuisance_contrast'],'source':'Existing figure bytes preserved; new plot only displays frozen population contrasts and fixed curvature cost'}
(OUT/'FIGURES.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
print('Saved contrast figure and hashes')
