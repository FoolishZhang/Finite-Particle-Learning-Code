"""All-page rendering and evidence for a caller-owned delivery workspace.

This module never imports a historical release's build/render implementation.
Geometry warnings and matching archived pixels are evidence, not new human QA.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import traceback

import fitz
from PIL import Image, ImageDraw

from .core import DeliveryContext, read_json, relative_path, require, sha256, write_json, write_text


def _output_record(ctx: DeliveryContext, path: Path) -> dict:
    return {"path": ctx.rel_output(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def _source_record(ctx: DeliveryContext, path: Path) -> dict:
    return {"origin": "source", "path": path.resolve().relative_to(ctx.root.resolve()).as_posix(),
            "sha256": sha256(path), "bytes": path.stat().st_size}


def _checked_output(ctx: DeliveryContext, record: dict, label: str) -> Path:
    path = ctx.output(record["path"])
    require(path.is_file(), f"Missing {label}: {record['path']}")
    require(sha256(path) == record["sha256"], f"Changed {label}: {record['path']}")
    require(path.stat().st_size == record["bytes"], f"Changed {label} length")
    return path


def resolve_poppler(explicit: str | Path | None = None) -> Path:
    """An explicit executable is authoritative; never silently fall back."""
    if explicit is not None:
        supplied = Path(explicit).expanduser()
        candidate = supplied if supplied.is_file() else Path(shutil.which(str(explicit)) or str(supplied))
        require(candidate.is_file(), f"Explicit pdftoppm executable does not exist: {explicit}")
        return candidate.resolve()
    located = shutil.which("pdftoppm")
    require(located is not None, "pdftoppm is not on PATH; provide an explicit Poppler executable")
    return Path(located).resolve()


def inspect_chinese_citation_boundaries(ctx: DeliveryContext) -> list[dict]:
    """Inspect bare Pandoc citations; bracketed citation groups remain allowed."""
    issues = []
    for prefix in ctx.config["source"]["source_prefixes"]:
        path = ctx.document / f"{prefix}_ZH.md"
        require(path.is_file(), f"Missing Chinese citation-boundary input: {path.name}")
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for match in re.finditer(r"@[A-Za-z][A-Za-z0-9_-]*", line):
                if line.rfind("[", 0, match.start()) > line.rfind("]", 0, match.start()):
                    continue
                before = line[match.start() - 1] if match.start() else ""
                after = line[match.end()] if match.end() < len(line) else ""
                if any(char and not char.isascii() and not char.isspace() for char in (before, after)):
                    issues.append({"source": ctx.rel_output(path), "origin": "run", "line": line_number,
                                   "citation": match.group(), "context": line})
    return issues


def overlap_candidates(spans: list[dict]) -> list[dict]:
    """Screen comparable glyph boxes; candidates always need visual judgment."""
    spans = sorted((span for span in spans if span["text"].strip()), key=lambda span: span["bbox"][1])
    found = []
    for index, left in enumerate(spans):
        a = left["bbox"]
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        if area_a == 0:
            continue
        for right in spans[index + 1:]:
            b = right["bbox"]
            if b[1] >= a[3]:
                break
            if abs(left["size"] - right["size"]) > 0.5:
                continue
            dx, dy = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
            area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
            if dx <= 1 or dy <= 0 or area_b == 0:
                continue
            fraction = dx * dy / min(area_a, area_b)
            if fraction >= 0.20:
                found.append({"left": left["text"], "right": right["text"], "left_bbox": a,
                              "right_bbox": b, "intersection_area": dx * dy,
                              "minimum_box_fraction": fraction, "font_sizes": [left["size"], right["size"]]})
    return found


def _page_ids(values: list, count: int, label: str, *, complete: bool = False) -> set[int]:
    require(isinstance(values, list), f"{label} must be a list")
    require(all(type(value) is int and 1 <= value <= count for value in values), f"{label} has invalid page IDs")
    require(len(values) == len(set(values)), f"{label} has duplicate pages")
    if complete:
        require(values == list(range(1, count + 1)), f"{label} is not a complete ordered page list")
    return set(values)


def validate_reference_reuse(ctx: DeliveryContext, lang: str, render_receipt: dict) -> dict:
    """Validate archived QA bindings and report eligible exact-pixel matches.

    The archived QA is evidence already recorded by its reviewer. This does not
    re-perform that review, claim review of changed pages, or mark a new QA pass.
    """
    refs = ctx.config.get("reference", {})
    require(lang in ("en", "zh"), "Invalid render language")
    current = render_receipt["page_details"]
    count = render_receipt["pages"]
    _page_ids([row["page"] for row in current], count, "Current rendered pages", complete=True)
    if not refs:
        return {"reference_available": False, "exact_page_mapping": {}, "eligible_reuse_pages": [],
                "current_pages_requiring_visual_review": list(range(1, count + 1)),
                "all_pages_pixel_identical_at_same_positions": False, "no_new_page_visual_review_claimed": True}
    config = refs["languages"][lang]
    paths = {key: ctx.source(config[key]) for key in ("build", "render", "pdf")}
    paths["visual"] = ctx.source(refs["visual"])
    old, build, qa = read_json(paths["render"]), read_json(paths["build"]), read_json(paths["visual"])
    old_count = old["pages"]
    require(type(old_count) is int and old_count > 0, "Invalid reference page count")
    require(old["language"] == lang and build["language"] == lang, "Reference language mismatch")
    require(old["rendered_all_pages"] is True, "Reference did not render every page")
    require(old["build_sha256"] == sha256(paths["build"]), "Reference render/build hash mismatch")
    actual_pdf_hash = sha256(paths["pdf"])
    if "outputs" in build:
        build_pdf = build["outputs"]["pdf"]
        require(bool(refs.get("package_root")), "Native reference BUILD requires reference.package_root")
        package = ctx.source(refs["package_root"])
        require(package.is_dir(), "Reference package root is not a directory")
        relative_path(build_pdf["path"], "Reference build PDF path")
        resolved_pdf = (package / build_pdf["path"]).resolve()
        require(resolved_pdf.is_relative_to(package.resolve()), "Reference build PDF escapes its package")
        require(resolved_pdf == paths["pdf"].resolve(), "Reference build PDF path mismatch")
        require(paths["pdf"].stat().st_size == build_pdf["bytes"], "Reference build PDF byte count mismatch")
        build_pdf_hash = build_pdf["sha256"]
    else:
        require(build["pdf"] == config["pdf"], "Reference build PDF path mismatch")
        build_pdf_hash = build["pdf_sha256"]
    require(old["pdf_sha256"] == build_pdf_hash == actual_pdf_hash, "Reference PDF/build/render mismatch")
    require(build["total_pages"] == old_count, "Reference build/render page count mismatch")
    require(qa["passed"] is True and bool(qa.get("reviewer")), "Reference visual review did not pass")
    language_qa = qa["languages"][lang]
    require(language_qa["pdf_sha256"] == actual_pdf_hash, "Reference QA/PDF hash mismatch")
    require(language_qa["render_sha256"] == sha256(paths["render"]), "Reference QA/render hash mismatch")
    require(language_qa.get("blocking_issues") == [], "Reference visual QA has undisposed blockers")
    reviewed = _page_ids(language_qa["covered_pages"], old_count, "Reference QA coverage", complete=True)
    actual = set()
    for key in ("primary_visual_pages", "delegated_visual_pages", "reused_pixel_identical_pages"):
        actual.update(_page_ids(language_qa.get(key, []), old_count, "Reference " + key))
    require(actual == reviewed, "Reference QA coverage has no actual review/reuse evidence")
    for key in ("overlap_review", "warning_review"):
        disposition = language_qa[key]
        require(disposition.get("reviewed") is True and disposition.get("blocking_issues") == []
                and bool(disposition.get("note")), "Reference " + key + " is incomplete")
    _page_ids([row["page"] for row in old["page_details"]], old_count, "Reference rendered pages", complete=True)
    old_by_page = {row["page"]: row for row in old["page_details"]}
    with fitz.open(paths["pdf"]) as pdf:
        require(len(pdf) == old_count, "Reference actual PDF page count mismatch")
        for page_id, page in enumerate(pdf, 1):
            row = old_by_page[page_id]
            require(abs(page.rect.width - row["width_points"]) < 0.001
                    and abs(page.rect.height - row["height_points"]) < 0.001,
                    "Reference page geometry mismatch")
            require(bool(re.fullmatch(r"[0-9a-f]{64}", row["pixel_sha256"])), "Malformed reference pixel hash")
    same_dpi = old["render_dpi"] == render_receipt["render_dpi"]
    by_pixel: dict[str, list[int]] = {}
    if same_dpi:
        for row in old["page_details"]:
            if row["page"] in reviewed:
                by_pixel.setdefault(row["pixel_sha256"], []).append(row["page"])
    mapping = {}
    for row in current:
        matches = [page for page in by_pixel.get(row["pixel_sha256"], [])
                   if old_by_page[page]["width_points"] == row["width_points"]
                   and old_by_page[page]["height_points"] == row["height_points"]]
        if matches:
            mapping[str(row["page"])] = row["page"] if row["page"] in matches else matches[0]
    return {"reference_available": True,
            "reference_files": {key: _source_record(ctx, path) for key, path in paths.items()},
            "reference_dpi": old["render_dpi"], "same_dpi": same_dpi,
            "reference_coverage_verified": True, "reference_blockers_cleared": True,
            "exact_page_mapping": mapping, "eligible_reuse_pages": sorted(map(int, mapping)),
            "current_pages_requiring_visual_review": [row["page"] for row in current if str(row["page"]) not in mapping],
            "all_pages_pixel_identical_at_same_positions": old_count == count and len(mapping) == count
            and all(mapping.get(str(page)) == page for page in range(1, count + 1)),
            "no_new_page_visual_review_claimed": True}


def _tex_issues(ctx: DeliveryContext, build: dict) -> list[str]:
    issues = []
    logs = build.get("logs", [])
    if isinstance(logs, dict):
        logs = list(logs.values())
    for record in logs:
        path = _checked_output(ctx, record, "build log")
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if any(word in line for word in ("Overfull", "Missing character", "Undefined control", "undefined references")):
                if line not in issues:
                    issues.append(line)
    return issues


def _render_impl(ctx: DeliveryContext, lang: str, poppler: str | Path | None, dpi: int) -> dict:
    require(lang in ("en", "zh"), "Invalid render language")
    require(type(dpi) is int and 36 <= dpi <= 600, "Render DPI must be an integer from 36 to 600")
    receipt_path = ctx.records / f"RENDER_{lang.upper()}.json"
    require(not receipt_path.exists(), "Render receipt already exists; start a new run")
    build_path = ctx.records / f"BUILD_{lang.upper()}.json"
    build = read_json(build_path)
    version = ctx.config["release"]["version"]
    require(build["version"] == version and build["language"] == lang, "Render version/language identity drift")
    require(build["configuration_sha256"] == ctx.config_sha256, "Render build/config mismatch")
    require(build["engine"] == ctx.engine_identity(), "Render build/engine mismatch")
    pdf = _checked_output(ctx, build["outputs"]["pdf"], "built PDF")
    tex = _checked_output(ctx, build["outputs"]["tex"], "built TeX")
    _checked_output(ctx, build["outputs"]["manuscript"], "built manuscript")
    require(tex.read_text(encoding="utf-8").splitlines()[0] == ctx.config["release"]["identity_header"],
            "Render TeX identity drift")
    executable = resolve_poppler(poppler)
    executable_hash = sha256(executable)
    destination = ctx.work / lang / ("render_" + sha256(pdf)[:16])
    ctx.output(ctx.rel_output(destination))
    require(not destination.exists(), "Render directory already exists; start a new run")
    destination.mkdir(parents=True, exist_ok=False)
    logpath = ctx.records / f"RENDER_{lang.upper()}.log"
    require(not logpath.exists(), "Render log already exists; start a new run")
    process = subprocess.run([str(executable), "-r", str(dpi), "-png", str(pdf), str(destination / "page")],
                             capture_output=True, text=True, encoding="utf-8", errors="replace")
    write_text(logpath, process.stdout + process.stderr)
    require(process.returncode == 0, f"Poppler failed ({process.returncode}): {process.stderr}")
    require(sha256(executable) == executable_hash, "Poppler executable changed during rendering")
    pngs = {}
    for path in destination.glob("page-*.png"):
        match = re.fullmatch(r"page-(\d+)\.png", path.name)
        require(match is not None, "Unexpected Poppler page filename")
        number = int(match.group(1))
        require(number not in pngs, "Duplicate rendered page number")
        pngs[number] = path
    pages, warnings, starts, overlaps, alltext, contacts = [], [], {}, [], [], []
    with fitz.open(pdf) as document:
        require(len(document) == build["total_pages"], "Built PDF page count mismatch")
        require(sorted(pngs) == list(range(1, len(document) + 1)), "Poppler did not render exactly every page")
        for index, page in enumerate(document, 1):
            text = page.get_text()
            alltext.append(text)
            issues = []
            for detected, name in ((len(text.strip()) < 50, "nearly_empty_text_page"),
                    ("operatorname" in text, "literal_math_operator_command"),
                    ("qquad" in text, "literal_math_spacing_command"),
                    (any(c in text for c in ("\ufffd", "\uffff")), "replacement_glyph"),
                    ("Œ" in text or "œ" in text, "unexpected_OE_ligature_possible_T1_multiplication_substitution"),
                    ("[@" in text or bool(re.search(r"(?<![A-Za-z0-9._])@[A-Za-z][A-Za-z0-9_-]*", text)), "unresolved_citation")):
                if detected:
                    issues.append(name)
            blocks = page.get_text("dict")["blocks"]
            spans = [span for block in blocks for line in block.get("lines", []) for span in line["spans"]]
            ordinary = [span for span in spans if not (lang == "en" and re.fullmatch(r"\d{3,}", span["text"].strip())
                        and span["bbox"][0] < 32 and span["color"] == 11776947)]
            boundary = [{"text": span["text"], "bbox": span["bbox"]} for span in ordinary if span["text"].strip()
                        and (span["bbox"][0] < 32 or span["bbox"][2] > page.rect.width - 32
                             or span["bbox"][1] < 18 or span["bbox"][3] > page.rect.height - 18)]
            if issues or boundary:
                warnings.append({"page": index, "issues": issues, "boundary_spans": boundary})
            candidates = overlap_candidates(ordinary)
            if candidates:
                overlaps.append({"page": index, "candidates": candidates})
            for block in blocks:
                for line in block.get("lines", []):
                    content = "".join(span["text"] for span in line["spans"]).strip()
                    if line["spans"] and (max(span["size"] for span in line["spans"]) >= 10
                                          or any(span["flags"] & 16 for span in line["spans"])):
                        if re.match(r"^(?:Appendix\s+[A-Z]\.|[A-Z]\.\d+(?:[.\s]|$))", content) \
                                or re.match(r"^附录\s*[A-Z]", content) or content in ("References", "参考文献"):
                            starts.setdefault(content, index)
            with Image.open(pngs[index]) as image:
                rgb = image.convert("RGB")
                pixel_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
                dimensions = [rgb.width, rgb.height]
            pages.append({"page": index, "image": ctx.rel_output(pngs[index]), "image_sha256": sha256(pngs[index]),
                          "pixel_sha256": pixel_hash, "pixel_dimensions": dimensions, "text_characters": len(text),
                          "width_points": page.rect.width, "height_points": page.rect.height})
        for first in range(1, len(document) + 1, 4):
            sheet = Image.new("RGB", (1400, 1940), "#e8ebee")
            draw = ImageDraw.Draw(sheet)
            for offset in range(4):
                page_id = first + offset
                if page_id > len(document):
                    break
                with Image.open(pngs[page_id]) as original:
                    image = original.convert("RGB")
                image.thumbnail((676, 916))
                x, y = (offset % 2) * 700 + (700 - image.width) // 2, (offset // 2) * 970 + 30
                sheet.paste(image, (x, y))
                draw.text((x, y - 21), f"{version} {lang.upper()} / page {page_id}", fill="black")
            contact = destination / f"contact_{first:02}.png"
            sheet.save(contact)
            contacts.append(contact)
    text_path = destination / "extracted_text.txt"
    write_text(text_path, "\n\f\n".join(alltext))
    result = {"schema": "paper_delivery_render_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "version": version, "language": lang, "engine": ctx.engine_identity(), "configuration_sha256": ctx.config_sha256,
              "pdf_sha256": sha256(pdf), "build_sha256": sha256(build_path), "pages": len(pages),
              "page_details": pages, "section_start_candidates": starts, "boundary_or_text_warnings": warnings,
              "TeX_issues": _tex_issues(ctx, build), "contact_sheets": [ctx.rel_output(path) for path in contacts],
              "contact_sheet_details": [_output_record(ctx, path) for path in contacts],
              "overlap_warnings": overlaps, "overlap_candidates_are_not_automatic_layout_failures": True,
              "overlap_candidate_count": sum(len(row["candidates"]) for row in overlaps),
              "extracted_text": ctx.rel_output(text_path), "extracted_text_sha256": sha256(text_path),
              "render_directory": ctx.rel_output(destination), "render_log": _output_record(ctx, logpath),
              "renderer": str(executable), "renderer_sha256": executable_hash, "render_dpi": dpi,
              "rendered_all_pages": True, "main_body_pages": build["main_body_pages"],
              "source_chinese_bare_citation_boundary_issues": inspect_chinese_citation_boundaries(ctx),
              "visual_review_not_automated": True, "human_visual_review_pending": True}
    result["pixel_reuse_candidates"] = validate_reference_reuse(ctx, lang, result)
    write_json(receipt_path, result)
    return result


def render(ctx: DeliveryContext, lang: str, poppler: str | Path | None = None, dpi: int = 110) -> dict:
    """Render once; save failure details only inside an unsealed run workspace."""
    ctx.assert_writable()
    try:
        return _render_impl(ctx, lang, poppler, dpi)
    except BaseException as error:
        ctx.assert_writable()
        serial = 1
        label = lang.upper() if lang in ("en", "zh") else "INVALID"
        while (ctx.records / f"RENDER_FAILURE_{label}_{serial:03}.json").exists():
            serial += 1
        ctx.records.mkdir(parents=True, exist_ok=True)
        write_json(ctx.records / f"RENDER_FAILURE_{label}_{serial:03}.json",
                   {"schema": "paper_delivery_render_failure_v1", "failed_utc": datetime.now(timezone.utc).isoformat(),
                    "language": lang, "engine": ctx.engine_identity(), "configuration_sha256": ctx.config_sha256,
                    "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(),
                    "scientific_failure": False, "new_integrals": 0, "new_random_draws": 0, "new_fits": 0})
        raise
