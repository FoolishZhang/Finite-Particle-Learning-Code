"""Read-only delivery checks and explicit, one-time authoring seals.

These checks cover source identity, bilingual structure, output identity and
reader navigation. They do not prove scientific statements. A renderer receipt
is not a visual review: sealing additionally requires actual review coverage or
exact, all-page pixel equivalence to an already reviewed reference.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re

import fitz
from PIL import Image

from .core import read_json, record_file, require, sha256, verify_preserved_sources


def _identity(ctx, row, label):
    require(row.get("engine") == ctx.engine_identity(), label + ": engine identity differs")
    require(row.get("configuration_sha256") == ctx.config_sha256, label + ": configuration identity differs")
    if "version" in row:
        require(row["version"] == ctx.config["release"]["version"], label + ": release differs")


def _verify_file(path, row, label):
    require(path.is_file(), label + ": missing file")
    actual = record_file(path)
    require(actual["sha256"] == row.get("sha256"), label + ": digest differs")
    require(actual["bytes"] == row.get("bytes"), label + ": byte count differs")


def _snapshot(ctx):
    row = read_json(ctx.records / "INPUTS.json")
    require(row.get("schema") == "paper_delivery_inputs_v1", "Unsupported source snapshot")
    _identity(ctx, row, "Source snapshot")
    require(row.get("source_document") == ctx.config["source"]["document"], "Snapshot source directory differs")
    files = row["files"]
    require(bool(files), "Empty source snapshot")
    require(len({r["source"] for r in files}) == len(files), "Duplicate source input")
    require(len({r["snapshot"] for r in files}) == len(files), "Duplicate snapshot destination")
    for item in files:
        _verify_file(ctx.source(item["source"]), item, "Repository source " + item["source"])
        snapshot = ctx.output(item["snapshot"])
        require(snapshot.is_relative_to(ctx.document), "Snapshot outside document directory")
        expected_source = ctx.source_doc / snapshot.relative_to(ctx.document)
        require(ctx.source(item["source"]) == expected_source.resolve(), "Source and snapshot paths differ")
        _verify_file(snapshot, item, "Snapshot " + item["snapshot"])
    return row


def pdf_navigation(pdf_path, labels, target_ids, rules):
    """Validate generated PDF destinations and actual incoming annotations."""
    require(set(labels) == set(target_ids), "Semantic label set differs from contract")
    require(set(rules) == set(target_ids), "Semantic destination rules differ from contract")
    result = {}
    with fitz.open(pdf_path) as pdf:
        destinations = pdf.resolve_names()
        links = [(i + 1, item) for i, page in enumerate(pdf) for item in page.get_links()]
        for key in target_ids:
            row = labels[key]
            page_number = row["page"]
            require(isinstance(page_number, int) and 1 <= page_number <= len(pdf), "Semantic destination page outside PDF: " + key)
            require(row["destination"] in destinations, "PDF named destination missing: " + key)
            require(destinations[row["destination"]]["page"] + 1 == page_number, "Named destination page differs: " + key)
            text = pdf[page_number - 1].get_text()
            rule = rules[key]
            if "section" in rule:
                require(set(rule) == {"section"} and rule["section"] in text, "Target section heading absent: " + key)
            else:
                terms = {"proposition": ("proposition", "命题"), "theorem": ("theorem", "定理"),
                         "corollary": ("corollary", "推论")}
                require(set(rule) == {"kind"} and rule["kind"] in terms, "Unknown semantic target rule: " + key)
                require(any(term in text.lower() for term in terms[rule["kind"]]), "Semantic statement kind absent: " + key)
            incoming = [(i, item) for i, item in links if item.get("nameddest") == row["destination"]]
            require(bool(incoming), "No incoming clickable semantic link: " + key)
            require(all(item.get("page") == page_number - 1 for _, item in incoming), "Incoming semantic link resolves to a different page: " + key)
            result[key] = dict(page=page_number, destination=row["destination"],
                               incoming_from_pages=sorted({i for i, _ in incoming}), annotation_count=len(incoming))
    return result


def _pdf_structure(ctx, lang, built, path):
    with fitz.open(path) as pdf:
        require(len(pdf) == built["total_pages"], "PDF page count differs: " + lang)
        require(0 < built["main_body_pages"] <= len(pdf), "Invalid main-body page count")
        dimensions = [[round(p.rect.width, 3), round(p.rect.height, 3)] for p in pdf]
        recorded = built.get("page_dimensions_points")
        require(recorded == dimensions, "PDF per-page sizes differ: " + lang)
        unique_sizes = [list(size) for size in sorted({tuple(size) for size in dimensions})]
        require(built["page_sizes_points"] == unique_sizes, "PDF page-size set differs: " + lang)
        if lang == "en":
            require(all(abs(w - 612) < .1 and abs(h - 792) < .1 for w, h in dimensions), "English PDF must use Letter format")
        texts = [page.get_text() for page in pdf]
        text = "\n".join(texts)
        require(all(token not in text for token in ("operatorname", "qquad", "\ufffd", "\uffff", "|---|", "Finite-Particle-Learning")), "Unresolved content in PDF: " + lang)
        require(not re.search(r"(?<![A-Za-z0-9._])@[A-Za-z][A-Za-z0-9_-]*", text), "Unresolved PDF citation: " + lang)
    if ctx.config["mode"] == "reference_replay":
        reference = ctx.source(ctx.config["reference"]["languages"][lang]["pdf"])
        with fitz.open(reference) as pdf:
            require(len(pdf) == len(texts), "Reference PDF page count differs: " + lang)
            require([[round(p.rect.width, 3), round(p.rect.height, 3)] for p in pdf] == dimensions, "Reference PDF page sizes differ: " + lang)
            require([page.get_text() for page in pdf] == texts, "Reference PDF page text differs: " + lang)
    return {"main": built["main_body_pages"], "total": built["total_pages"], "pdf_sha256": sha256(path)}


def _render(ctx, lang, built, verify_images):
    path = ctx.records / f"RENDER_{lang.upper()}.json"
    if not path.exists():
        return None
    row = read_json(path)
    _identity(ctx, row, "Render " + lang)
    require(row["language"] == lang, "Render language differs")
    require(row["pdf_sha256"] == built["outputs"]["pdf"]["sha256"], "Render PDF binding differs")
    require(row["build_sha256"] == sha256(ctx.records / f"BUILD_{lang.upper()}.json"), "Render build binding differs")
    expected = list(range(1, built["total_pages"] + 1))
    require(row["rendered_all_pages"] and row["pages"] == len(expected), "Incomplete rendering")
    require([r["page"] for r in row["page_details"]] == expected, "Render page coverage differs")
    require(not row["source_chinese_bare_citation_boundary_issues"], "Unsafe Chinese citation boundary")
    dimensions = built["page_dimensions_points"]
    for item, size in zip(row["page_details"], dimensions):
        require(abs(item["width_points"] - size[0]) < .001 and abs(item["height_points"] - size[1]) < .001, "Rendered PDF page size differs")
        require(re.fullmatch(r"[0-9a-f]{64}", item["pixel_sha256"]) is not None, "Invalid rendered pixel digest")
        if verify_images:
            image_path = ctx.output(item["image"])
            require(sha256(image_path) == item["image_sha256"], "Rendered image digest differs")
            with Image.open(image_path) as image:
                pixel_sha = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
            require(pixel_sha == item["pixel_sha256"], "Rendered pixel digest differs")
    if verify_images:
        for item in row.get("contact_sheet_details", []):
            require(sha256(ctx.output(item["path"])) == item["sha256"], "Contact sheet differs")
        if "extracted_text" in row:
            require(sha256(ctx.output(row["extracted_text"])) == row["extracted_text_sha256"], "Extracted text differs")
    return row


def _reference_equivalence(ctx, renders):
    from .renderer import validate_reference_reuse
    require(ctx.config["mode"] == "reference_replay", "Automatic visual reuse is restricted to reference replay")
    result = {"schema": "paper_delivery_reference_equivalence_v1", "engine": ctx.engine_identity(),
              "configuration_sha256": ctx.config_sha256, "version": ctx.config["release"]["version"],
              "new_human_visual_review_claimed": False, "languages": {}}
    for lang in ("en", "zh"):
        require(lang in renders and renders[lang] is not None, "Both complete render receipts are required")
        reuse = validate_reference_reuse(ctx, lang, renders[lang])
        require(reuse["all_pages_pixel_identical_at_same_positions"], "Reference pixels are not identical on every page: " + lang)
        require(not reuse["current_pages_requiring_visual_review"], "New visual review is needed: " + lang)
        result["languages"][lang] = {"render_sha256": sha256(ctx.records / f"RENDER_{lang.upper()}.json"),
                                      "pdf_sha256": renders[lang]["pdf_sha256"], "reference": reuse}
    return result


def record_reference_equivalence(ctx):
    """Record exact reuse; never turn geometric screening into human review."""
    ctx.assert_writable()
    check(ctx, verify_images=True)
    _source_gate(ctx, required=True)
    renders = {lang: read_json(ctx.records / f"RENDER_{lang.upper()}.json") for lang in ("en", "zh")}
    result = _reference_equivalence(ctx, renders)
    path = ctx.records / "REFERENCE_EQUIVALENCE.json"
    with path.open("x", encoding="utf8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return result


def _visual_review(ctx, renders, required):
    path = ctx.records / "VISUAL_REVIEW.json"
    equivalent = ctx.records / "REFERENCE_EQUIVALENCE.json"
    visual = False
    if path.exists():
        row = read_json(path)
        _identity(ctx, row, "Visual review")
        require(row.get("passed") is True and bool(str(row.get("reviewer", "")).strip()), "Visual review incomplete")
        require(row.get("completed_utc"), "Visual review completion time missing")
        for lang in ("en", "zh"):
            render = renders[lang]
            require(render is not None, "Visual review lacks render receipt")
            qa = row["languages"][lang]
            expected = list(range(1, render["pages"] + 1))
            require(qa["pdf_sha256"] == render["pdf_sha256"] and qa["render_sha256"] == sha256(ctx.records / f"RENDER_{lang.upper()}.json"), "Visual review output binding differs")
            require(qa["covered_pages"] == expected and not qa["blocking_issues"], "Visual review page coverage incomplete")
            primary = qa.get("primary_visual_pages", [])
            delegated = qa.get("delegated_visual_pages", [])
            require(sorted(set(primary) | set(delegated)) == expected, "Actual visual review coverage incomplete")
            for key in ("overlap_review", "warning_review"):
                disposition = qa[key]
                require(disposition["reviewed"] and not disposition["blocking_issues"] and disposition["note"], "Visual issue disposition incomplete: " + key)
        visual = True
    if equivalent.exists():
        require(read_json(equivalent) == _reference_equivalence(ctx, renders), "Reference equivalence receipt differs")
        visual = True
    require(not required or visual, "Seal requires actual complete VISUAL_REVIEW.json or verified REFERENCE_EQUIVALENCE.json")
    return {"accepted": visual, "human_review_receipt": path.exists(), "reference_equivalence_receipt": equivalent.exists()}


def published_files(ctx):
    """The sealed delivery excludes disposable work files and page PNGs."""
    manifest = ctx.records / "DELIVERY_MANIFEST.json"
    found = {}
    for directory in (ctx.document, ctx.records, ctx.pdf_dir):
        require(directory.is_dir(), "Missing delivery directory: " + str(directory))
        for path in directory.rglob("*"):
            require(not path.is_symlink(), "Symlinks are not sealed delivery files")
            if path.is_file() and path != manifest:
                found[ctx.rel_output(path)] = record_file(path)
    return dict(sorted(found.items()))


def _manifest(ctx, files):
    return {"schema": "paper_delivery_manifest_v1", "engine": ctx.engine_identity(),
            "configuration_sha256": ctx.config_sha256, "version": ctx.config["release"]["version"],
            "run_identity": record_file(ctx.workspace / "RUN.json"),
            "configuration": record_file(ctx.workspace / "PROFILE.json"),
            "mode": ctx.config["mode"], "files": files, "scientific_proofs_verified": False,
            "scientific_computation_executed": False, "disposable_render_images_required_for_check": False}


def _source_gate(ctx, required):
    """Bind the existing release's scientific/specific checker when configured."""
    gate = ctx.config.get("source_gate")
    if gate is None:
        return {"configured": False, "passed": None, "scientific_proofs_verified_by_common_engine": False}
    path = ctx.records / "SOURCE_CHECK.json"
    if not path.exists():
        require(not required, "Configured source checker receipt is required before visual equivalence/sealing")
        return {"configured": True, "passed": False, "status": "missing"}
    row = read_json(path)
    require(row["checker"] == gate["script"] and row["checker_sha256"] == gate["sha256"], "Source checker identity differs")
    require(sha256(ctx.source(gate["script"])) == gate["sha256"], "Source checker changed")
    require(row["configuration_sha256"] == ctx.config_sha256, "Source checker configuration differs")
    require(row["returncode"] == 0 and row["passed"] is True, "Source checker did not pass")
    for key in ("stdout", "stderr"):
        log = row[key]
        target = ctx.output(log["path"])
        require(target.is_relative_to(ctx.records), "Source checker log outside permanent records")
        require(sha256(target) == log["sha256"], "Source checker log changed: " + key)
    return {"configured": True, "passed": True, "receipt_sha256": sha256(path),
            "checker": gate["script"], "scientific_proofs_verified_by_common_engine": False}


def check(ctx, require_visual=False, verify_images=False):
    """Inspect only; all receipts must already exist. Never repair or re-seal."""
    from . import builder
    ctx.assert_identity()
    _identity(ctx, read_json(ctx.workspace / "RUN.json"), "Run")
    sealed = ctx.records / "DELIVERY_MANIFEST.json"
    if sealed.exists():
        manifest = read_json(sealed)
        common = _manifest(ctx, published_files(ctx))
        require(all(manifest.get(key) == value for key, value in common.items()), "Sealed delivery membership, hash or engine changed")
        require_visual = True
    source_gate = _source_gate(ctx, required=require_visual)
    verify_preserved_sources(ctx)
    snapshot = _snapshot(ctx)
    contract = builder.check_sources(ctx, doc=ctx.document)
    require(contract["input_file_count"] == len(snapshot["files"]), "Snapshot input membership differs")
    pages, navigation, renders = {}, {}, {}
    for lang in ("en", "zh"):
        built = read_json(ctx.records / f"BUILD_{lang.upper()}.json")
        _identity(ctx, built, "Build " + lang)
        require(built["language"] == lang, "Build language differs")
        require(built["source_contract"] == contract, "Build source contract differs")
        require(built["numbered_equations"] == contract["tags"], "Build equation numbering differs")
        require(built["snapshot_inputs"] == snapshot["files"], "Build source snapshot differs")
        expected_inputs = {r["source"]: {k: r[k] for k in ("sha256", "bytes")} for r in snapshot["files"]}
        require(built["inputs"] == expected_inputs, "Build input set differs")
        receipt = built["input_receipt"]
        require(ctx.output(receipt["path"]) == ctx.records / "INPUTS.json", "Unexpected input receipt path")
        _verify_file(ctx.output(receipt["path"]), receipt, "Build input receipt")
        for log in built["logs"]:
            path = ctx.output(log["path"])
            require(path.is_relative_to(ctx.records), "Build log outside permanent records")
            _verify_file(path, log, "Build log")
        parts, _, _ = builder.assemble(ctx, lang, ctx.document)
        output_paths = {}
        for kind in ("manuscript", "tex", "pdf"):
            item = built["outputs"][kind]
            path = ctx.output(item["path"])
            require(path.is_relative_to(ctx.pdf_dir if kind == "pdf" else ctx.document), "Output outside designated directory")
            _verify_file(path, item, "Generated " + kind + " " + lang)
            output_paths[kind] = path
            if ctx.config["mode"] == "reference_replay" and kind != "pdf":
                ref = ctx.source(ctx.config["reference"]["languages"][lang][kind])
                require(path.read_bytes() == ref.read_bytes(), "Reference " + kind + " bytes differ: " + lang)
        canonical = "\n".join(line.rstrip() for line in parts["manuscript"].splitlines()).strip()
        require(output_paths["manuscript"].read_text("utf8").strip() == canonical, "Assembled manuscript is stale")
        require(output_paths["tex"].read_bytes().splitlines()[0].decode("utf8") == ctx.config["release"]["identity_header"], "Generated TeX release identity differs")
        targets = ctx.config["contract"]["semantic_targets"]
        for key in targets:
            require("{#" + key + "}" in canonical or "{#" + key + " " in canonical, "Semantic source target missing: " + key)
            require("](#" + key + ")" in canonical, "Semantic source link missing: " + key)
        require(not re.search(r"(?:artifacts/|submission_candidate|\.\./|Finite-Particle-Learning)", canonical), "Repository path in reader manuscript")
        pages[lang] = _pdf_structure(ctx, lang, built, output_paths["pdf"])
        navigation[lang] = pdf_navigation(output_paths["pdf"], built["semantic_labels"], targets,
                                         ctx.config["contract"]["navigation_rules"])
        renders[lang] = _render(ctx, lang, built, verify_images)
    visual = _visual_review(ctx, renders, require_visual)
    return {"passed": True, "sealed": sealed.exists(), "version": ctx.config["release"]["version"],
            "pages": pages, "source_contract": contract, "navigation": navigation,
            "rendered_languages": sorted(lang for lang, row in renders.items() if row is not None),
            "visual": visual, "source_gate": source_gate,
            "scientific_proofs_verified": False, "scientific_computation_executed": False}


def seal(ctx):
    ctx.assert_writable()
    path = ctx.records / "DELIVERY_MANIFEST.json"
    require(not path.exists(), "Delivery is already sealed")
    require(not list(ctx.workspace.rglob("VERSION_LOCK.json")), "Existing version lock")
    result = check(ctx, require_visual=True, verify_images=True)
    manifest = _manifest(ctx, published_files(ctx))
    manifest["created_utc"] = datetime.now(timezone.utc).isoformat()
    with path.open("x", encoding="utf8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    result["sealed"] = True
    result["delivery_manifest"] = ctx.rel_output(path)
    return result
