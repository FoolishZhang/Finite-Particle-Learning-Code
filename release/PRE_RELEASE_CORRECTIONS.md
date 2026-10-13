# Preparation corrections for source v1.0.0

The first preparation commit projected historical Markdown links and machine-path metadata too broadly: eight Markdown files contained TeX expressions misclassified as links or drive paths. The authoring sources, PDFs, scientific modules and numerical evidence were unchanged. This was an export error, not a change to the scientific results.

Before the stable source tag, the exporter was made aware of mathematical and code spans, and machine-path matching was restricted to actual environment directories. Every transformed Markdown mathematical span is checked against its source, and all current h2.2 reader inputs match their original bytes. The correction is preserved as a normal public commit; early preparation commits should not be used as the released scientific reference. Use the v1.0.0 source tag and its file manifest.
