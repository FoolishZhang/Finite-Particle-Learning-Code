# Third-party materials

The root MIT license covers original project software, including own scientific code snapshots. Original project prose, figures and author-owned synthetic data are CC BY 4.0 under LICENSE-DATA. These terms do not replace third-party licenses.

## SCBM

`reproducibility_snapshots/cbm_public_sources/SCBM/58c3e602cb11fb4ea9106226e0f9e58f50d5d10a/` retains its original Apache-2.0 LICENSE.md and copyright notices (Moritz Vandenhirtz, 2024). The source snapshot is unchanged. It supports historical interface studies. Its CUB/CIFAR configuration references do not grant rights to redistribute those datasets. The SCBM `utils/minimize_constraint.py` header credits an adaptation from Reuben Feinman's pytorch-minimize. The upstream MIT notice (Copyright 2021 Reuben Feinman) is separately preserved in `third_party/pytorch-minimize-LICENSE.txt`; this adapted code is not attributed as original ZHANG code. The upstream license was checked for this release; a particular historical upstream commit of the adaptation is not claimed.

## Paper styles and fonts

The ICML 2026 style bundle is not redistributed. Some files state LPPL terms; the local algorithm/algorithmic and main style copies did not provide a distinct redistribution license. `tools/fetch_paper_style.py` retrieves the original bundle separately and checks its recorded hashes: https://media.icml.cc/Conferences/ICML2026/Styles/icml2026.zip . Retain the upstream notices. This is a format preview, not a submission or acceptance claim.

No font binaries are included. The Chinese wrapper names Times New Roman, SimSun and Microsoft YaHei. Install fonts you are licensed to use or explicitly adapt a local wrapper; layout can change.

## Data and dependencies

The five optional data assets contain author-generated synthetic arrays. Historical Optdigits work uses Optical Recognition of Handwritten Digits, E. Alpaydin and C. Kaynak (1998), https://doi.org/10.24432/C50P49, under the upstream CC BY 4.0 terms. These five downloads do not include that natural dataset. Obtain it from https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits and retain its attribution. Historical computed summaries are identified as such.

Python packages, TeX tools and fonts are dependencies, not vendored or relicensed here. Consult their upstream terms.
