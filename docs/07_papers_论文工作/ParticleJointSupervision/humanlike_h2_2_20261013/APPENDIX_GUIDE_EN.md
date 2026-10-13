# Appendix reading guide

The retained appendices supply the proofs and evidence for the main-text results. The table maps each main-text item to its support, with continuous appendix lettering.

|Main text|Where it is established|Type of support|
|---|---|---|
|Figure 1, fair-coin control (Section 1)|C.1 matched selection; C.2 gap audit; C.3 analytic curvature at $n=0$; C.4 curvature audit at $n=256$|Analytic bound at $n=0$; 99% audit families with stated numerical allowances; one saved matched run|
|Model, decomposition and representation facts (Section 2)|A.1|Analytic|
|Selection scale, Theorem 1 (Section 3.1--3.2)|A.2; scale heuristic and its relation to particle-filter collapse in B.2|Uniform asymptotic proof|
|Conditional-variance reading of $V$|B.1|Exact identity|
|Theorem 2 and the four-variable example|A.3 (A.21--A.22)|Analytic|
|Other parameter regimes|A.8 (free active scales), A.9 (free nuisance scale); further boundaries in A.4--A.7|Analytic|
|Figure 2, overlap versus coupling (Section 4.1)|C.5--C.6|Conditional 99% simultaneous intervals, 26 targets|
|[Fixed-$K$ correction](#fixed-k-proposition) and [finite-scale corollary](#finite-direction-corollary) (Section 4.2)|[C.7 onward](#fixed-k-correction): proof C.8, budgets C.9--C.12, constants and synthesis C.13, risk derivative C.14|Asymptotic proof; deterministic enclosure at $n=4096$|
|Finite-program consequences (Section 4.3)|[C.15](#finite-evidence-summary); D (fresh-run bound); E (saved-input arithmetic)|99% audit event, ideal iid sampling, declared numerical conditions|
|Model (13) and weak-loading heuristic (Sections 5.1, 5.3)|[I.1](#dense-model)|Expansion with qualitative remainder|
|Theorem 3 (Section 5.2)|[I.2](#full-hessian-criterion) criterion ([full derivatives](#full-derivatives)); [I.3](#laplace-representation) representation; [I.8](#proof-theorem-3) root and [full-space statement](#full-local-statement); [I.10](#same-root-angle) same-root angular path|Computer-assisted proof|
|[Quantitative persistence corollary](#quantitative-persistence-corollary) (Section 5.2)|[I.11](#quantitative-teacher-family) and [full family statement](#teacher-family-full-statement)|Computer-assisted proof over $\lvert\lambda\rvert\le.001$|
|Qualitative persistence (Section 5.2)|[I.9](#local-persistence)|Implicit-function argument; neighbourhood not quantified|
|Numerical learning and K11/K12 probes (Section 5.3)|I.4--I.7|Numerical, with stated quadrature qualifications|

**Material reported only in the appendices.** Known-mean robustness and observation-equivalent scales (F); a two-particle counterexample showing that matched means need not preserve finite-particle stationarity (G); and the earlier finite selection and its negative control (Appendix H).

A.2 retains the full uniform proof. Saved-endpoint and certificate-composition checks do not recompute probability, gradient or Hessian grids.

\clearpage
