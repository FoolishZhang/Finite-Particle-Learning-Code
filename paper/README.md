# Paper h2.2

The English and Chinese PDFs are unchanged h2.2 authoring snapshots: 97 and 92 pages, not submitted. The English main text is eight pages. They retain their anonymous preview author line; CITATION.cff supplies the repository author's provisional name ZHANG. This code release creates no new paper version.

Sources: `docs/07_papers_论文工作/ParticleJointSupervision/humanlike_h2_2_20261013/`. Appendices A--I are retained: former J became H, former K became I. Former H/I/L are archived separately with historical numbering. The 259 numbered equations and scientific scope are unchanged.

For optional rebuilding, install requirements-paper.txt, Pandoc and Tectonic. Fetch the original style into the source's `templates/icml2026/` with `tools/fetch_paper_style.py --destination <new-directory>`. Install the fonts named in the wrappers under their own licenses. From the repository root:

```text
python -B script/paper_delivery.py init --profile delivery/profiles/h2.2-public.json --workspace ../paper-rebuild-001
python -B script/paper_delivery.py build --profile delivery/profiles/h2.2-public.json --workspace ../paper-rebuild-001 --pandoc <pandoc> --tectonic <tectonic>
python -B script/paper_delivery.py check --profile delivery/profiles/h2.2-public.json --workspace ../paper-rebuild-001
```

Choose a new workspace. No TeX installation is needed for scientific checks. Public rebuilding omits private historical delivery gates. Font/tool changes require visual inspection; bitwise-identical PDFs are not promised.
