# Finite Particle Learning

## Current h3 code and evidence (v1.1.0)

The [h3 material entry](h3/README.md) supports **Importance Proposals Can Reverse Model Rankings and Induce Local Correlation Shrinkage**. It publishes the fixed-K scientific implementations, proof sources, 96 observed validation datasets, saved fitted outputs and risks, all numerical diagnostics, paired proposal-score differences, and figure sources. This is a named research release, not an anonymous submission or an acceptance claim.

Start with [h3/README.md](h3/README.md) for standalone saved-value checks and exact reproduction levels. The supported commands verify stored arithmetic, import the original kernels, and redraw the three figures; they do not rerun model integration or fitting. No large data download is required for those h3 checks. [v1.1.0 validation](release/V1_1_0_VALIDATION.json) records actual work and limitations.

The v1.1.0 update is additive. The v1.0.0 source tag, data-v1.0.0 assets and original h2.2 supplementary files are retained without replacement. h3's public material does not relabel the earlier anonymous package as reproducing h3.

## Archived h2.2 line


This repository accompanies **Finite Prior-Particle Likelihood Objectives Can Reward Risk-Increasing Dependence**, manuscript h2.2. It studies the expected negative logarithm of a likelihood average over independent prior particles, and compares its dependence preferences with the KL risk of the integrated model.

At fixed particle count, improving the precision of an objective estimate does not change the population objective being estimated. The paper establishes two results in controlled Gaussian--logistic models: a uniform, shrinking-scale dependence preference with fixed active marginals and an independent unit-scale appended block; and a computer-assisted local conflict in an eight-item, two-factor model with eight particles, correct observation marginals, and positive joint risk. The latter minimum is strict modulo common latent rotations. These are objective-selection and local-geometry results, with the conditions and numerical qualifications stated in the paper.

## Choose an entry point

- **Read the paper:** [English PDF](output/pdf/finite_particle_h2_2_en.pdf), [Chinese PDF](output/pdf/finite_particle_h2_2_zh.pdf), and [reader sources](docs/07_papers_论文工作/ParticleJointSupervision/humanlike_h2_2_20261013/).
- **Check saved numerical evidence:** [h2.2 supplementary materials](supplementary/h2_2/README.md). This low-cost route rebuilds saved arithmetic and plots; it does not regenerate all production quadrature, random banks, or training runs.
- **Inspect implementations and research history:** [reproduction guide](REPRODUCING.md) and [scientific archive index](docs/ARCHIVE_INDEX.md). The guide records the actual validation performed on this release. The index includes positive, negative and unresolved historical results.

Scientific implementations retain their original filenames under `script/`; selected saved numerical objects are under `artifacts/`. Source availability, saved-arithmetic verification, small production tests and full production replay are separate levels. A passing cheap check is not a claim that a complete training or integration experiment has been replayed.

## Quick start (Python 3.11)

Create a new virtual environment and install `requirements.txt`. On Windows use its `Scripts/python.exe`; on Linux/macOS use `bin/python`. Windows was actually tested; other platforms were not tested for this release.

```text
python -m venv ../fpl-env
../fpl-env/Scripts/python.exe -I -m pip --isolated install -r requirements.txt
../fpl-env/Scripts/python.exe -I -B supplementary/h2_2/check.py --level all
../fpl-env/Scripts/python.exe -I -B tools/production_replay.py --mode small --target all --output-root ../fpl-small-v1
```

Choose a fresh output directory. The final small suite exercised 16 targets in about six seconds; saved-arithmetic checks took about one minute on the tested machine. [Validation](docs/VALIDATION.json) and the [result-to-code map](docs/CORE_RESULT_MAP.json) state the exact scope. Full paper-scale quadrature and training were not rerun.

`python -B tools/verify_release.py` verifies release file hashes only. `python -B tools/download_asset.py --list` lists optional data and sizes without downloading it; the downloader requires an explicit asset, checks bytes and hashes, and extracts only into a new directory. [ASSETS.json](assets/ASSETS.json) records availability. Historical local objects not distributed in this version are summarized in [LOCAL_DATA_STATUS.json](assets/LOCAL_DATA_STATUS.json).

Source release **v1.0.0** accompanies paper **h2.2**; optional data uses the separate **data-v1.0.0** release. The source tree is an independently initialized public history. The original anonymous h2.2 low-cost package is kept byte-for-byte; this named repository as a whole is not an anonymous review submission.

## Data and historical programs

The repository contains selected small configurations, exact endpoints, numerical summaries and plots. Large synthetic banks and checkpoints are not automatically downloaded by a clone. Consult the reproduction guide and asset manifest for their actual availability; a stored hash is not the file itself. External datasets and official SCBM model weights are not bundled.

Some historical programs are source-only. Their original commands can require unavailable arrays, removed application environments, or management checks tied to unavailable old Git objects. These limitations are recorded rather than bypassed. Archived findings do not expand the conditions or claims of h2.2.

The former h2.1.1 appendices H, I and L remain available as scientific archives with their original lettering. They are outside the h2.2 PDF; they are not new h2.2 results. In h2.2, the retained former J and K are numbered H and I.

## Attribution and licensing

Author: **ZHANG** (provisional public display name, to be completed by the author). This is a research manuscript; no acceptance status or DOI is implied. Cite the paper and the specific release commit or tag used for a computation.

Self-authored code uses MIT; self-authored scientific text, figures and data use CC BY 4.0. The selected third-party SCBM snapshot retains its original Apache-2.0 license and upstream attribution. See [LICENSE](LICENSE), [LICENSE-DATA](LICENSE-DATA) and [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md), including the MIT attribution for the upstream minimizer adapted inside SCBM.

Optional paper rebuilding: [paper/README.md](paper/README.md). Scientific code does not require TeX or font installation.
