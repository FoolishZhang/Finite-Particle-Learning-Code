"""Configuration-driven bilingual manuscript assembly and isolated PDF builds.

This module reads authoring inputs, not scientific arrays or historical delivery
code. A build has one immutable input snapshot and never writes to the source
repository. Its receipts describe this build only; they are not scientific or
human visual-review certificates.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import traceback

from .core import DeliveryContext, require, sha256, record_file, read_json, write_json, write_text


LANGUAGES = ("en", "zh")
FRAGMENTS = ("abstract", "main", "impact", "appendices")
SCIENTIFIC_EXTENSIONS = {".npz", ".npy", ".pt", ".pth", ".pkl", ".ckpt", ".h5", ".hdf5"}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _document(ctx, doc):
    path = ctx.source_doc if doc is None else Path(doc)
    allowed = {ctx.source_doc.resolve()}
    snapshot = getattr(ctx, "document", None)
    if snapshot is not None:
        allowed.add(snapshot.resolve())
    require(path.resolve() in allowed,
            "Only the configured source document or its run snapshot may be assembled")
    return path


def _safe_file(base, relative):
    path = base / relative
    require(path.resolve().is_relative_to(base.resolve()), "Authoring input escapes document: " + str(relative))
    require(path.is_file(), "Missing authoring input: " + str(path))
    require(path.suffix.lower() not in SCIENTIFIC_EXTENSIONS, "Scientific data is not an authoring input")
    return path


def _displays(text):
    return [re.sub(r"\s+", "", block) for block in re.findall(r"\$\$(.*?)\$\$", text, re.S)]


def assemble(ctx: DeliveryContext, lang: str, doc=None):
    """Return assembled fragments, their source paths, and ordered equation tags."""
    require(lang in LANGUAGES, "Language must be en or zh")
    base = _document(ctx, doc)
    source = ctx.config["source"]
    names = [f"{prefix}_{lang.upper()}.md" for prefix in
             ("BODY", "RELATED_WORK", *("APPENDIX_" + p for p in source["appendices"]))]
    paths = [_safe_file(base, name) for name in names]
    parts = {name: path.read_text("utf-8") for name, path in zip(names, paths)}
    body = parts[names[0]]
    require(body.count("<!-- RELATED_WORK -->") == body.count("<!-- APPENDICES -->") == 1,
            "BODY must contain exactly one related-work and one appendix placeholder")
    body = body.replace("<!-- RELATED_WORK -->", parts[names[1]].strip())
    before, after = body.split("<!-- APPENDICES -->")
    require(not after.strip(), "Appendix placeholder must end BODY")
    main = re.sub(r"\A---\s*\n.*?\n---\s*\n", "", before, flags=re.S).strip()
    abstract_match = re.match(r"#\s+(?:Abstract|摘要)\s*\n(.*?)(?=\n#\s)", main, re.S)
    require(abstract_match is not None, "BODY must start with Abstract/摘要 followed by a section")
    abstract_md = abstract_match.group(1).strip()
    main_md = main[abstract_match.end():].strip()
    appendix_md = "\n\n\\clearpage\n\n".join(parts[name].strip() for name in names[2:])
    guide = _safe_file(base, f"APPENDIX_GUIDE_{lang.upper()}.md")
    impact = _safe_file(base, f"IMPACT_{lang.upper()}.md")
    appendix_md = guide.read_text("utf-8").strip() + "\n\n" + appendix_md
    impact_md = impact.read_text("utf-8").strip()
    manuscript = body.replace("<!-- APPENDICES -->", impact_md + "\n\n" + appendix_md)
    require(not any(ord(c) < 32 and c not in "\n\r\t" for c in manuscript), "Control character in manuscript")
    require(manuscript.count("$$") % 2 == 0, "Unpaired display-math delimiter")
    require(not re.search(r"^\$$", manuscript, re.M), "Standalone single-dollar delimiter")
    tags = re.findall(r"\\tag\{([^}]+)\}", manuscript)
    display_tags = re.findall(r"\\tag\{([^}]+)\}", "".join(re.findall(r"\$\$(.*?)\$\$", manuscript, re.S)))
    require(display_tags == tags, "Numbered equation outside display math")
    require(len(tags) == len(set(tags)), "Duplicate numbered equation tags")
    order = source["appendix_order"]
    require(len(order) == len(source["appendices"]) and len(set(order)) == len(order), "Invalid appendix order")
    require(all(re.fullmatch(r"[A-Z]", letter) for letter in order), "Appendix identifiers must be single capital letters")
    pattern = r"(?:[" + "".join(order) + r"]\.)?[1-9][0-9]*"
    require(all(re.fullmatch(pattern, tag) for tag in tags), "Unexpected equation tag format")
    for prefix in ("", *(letter + "." for letter in order)):
        group = [tag for tag in tags if (tag.startswith(prefix) if prefix else tag.isdigit())]
        require(group == [prefix + str(i) for i in range(1, len(group) + 1)], "Noncontiguous equation tags: " + prefix)
    return dict(manuscript=manuscript, abstract=abstract_md, main=main_md,
                impact=impact_md, appendices=appendix_md), paths + [impact, guide], tags


def _input_files(ctx, doc=None):
    """Whitelist manuscript inputs; never include old manifests or receipts."""
    base = _document(ctx, doc)
    prefixes = ctx.config["source"]["source_prefixes"]
    expected = {"BODY", "RELATED_WORK", "IMPACT", "APPENDIX_GUIDE",
                *("APPENDIX_" + value for value in ctx.config["source"]["appendices"])}
    require(len(prefixes) == len(set(prefixes)) and set(prefixes) == expected,
            "source_prefixes must exactly describe the assembly inputs")
    files = [_safe_file(base, f"{prefix}_{lang.upper()}.md") for lang in LANGUAGES for prefix in prefixes]
    files += [_safe_file(base, f"templates/wrapper_{lang}.tex") for lang in LANGUAGES]
    files.append(_safe_file(base, "references.bib"))
    for directory in ("templates/icml2026", "figures"):
        folder = base / directory
        require(folder.is_dir(), "Missing authoring directory: " + str(folder))
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                files.append(_safe_file(base, path.relative_to(base)))
    require(len(files) == len(set(files)), "Duplicate authoring input")
    return sorted(files, key=lambda path: path.relative_to(base).as_posix())


def check_sources(ctx: DeliveryContext, doc=None):
    """Check bilingual mathematical structure, citations, identity, and assets."""
    base = _document(ctx, doc)
    inputs = _input_files(ctx, base)
    en, _, tags = assemble(ctx, "en", base)
    zh, _, zh_tags = assemble(ctx, "zh", base)
    contract = ctx.config["contract"]
    require(tags == zh_tags, "Bilingual numbered-equation tags differ")
    require(_displays(en["manuscript"]) == _displays(zh["manuscript"]), "Bilingual display-math content differs")
    require(len(tags) == contract["numbered_equations"], "Numbered-equation count differs from release contract")
    require([tag for tag in tags if tag.isdigit()] == [str(i) for i in range(1, contract["main_equations"] + 1)],
            "Main equation numbering differs from release contract")
    citations = lambda text: sorted(set(re.findall(r"@([A-Za-z][A-Za-z0-9_-]*)", text)))
    cites = citations(en["manuscript"])
    require(cites == citations(zh["manuscript"]), "Bilingual citation sets differ")
    require(len(cites) == contract["citation_count"], "Citation count differs from release contract")
    bib_keys = re.findall(r"@\w+\s*\{\s*([^,\s]+)", (base / "references.bib").read_text("utf-8"))
    require(len(bib_keys) == len(set(bib_keys)), "Duplicate bibliography keys")
    require(len(bib_keys) == contract["bibliography_count"], "Bibliography count differs from release contract")
    require(set(cites) <= set(bib_keys), "Missing bibliography keys: " + str(sorted(set(cites) - set(bib_keys))))
    figures = {path.relative_to(base / "figures").as_posix(): record_file(path)
               for path in inputs if path.is_relative_to(base / "figures")}
    require(len(figures) == contract["figure_count"], "Figure count differs from release contract")
    identity = ctx.config["release"]["identity_header"]
    for lang in LANGUAGES:
        wrapper = (base / f"templates/wrapper_{lang}.tex").read_text("utf-8")
        require(wrapper.splitlines()[0] == identity, "Wrapper release identity differs from configuration")
        for name in FRAGMENTS:
            require(wrapper.count("% CANDIDATE_" + name.upper()) == 1, "Wrapper fragment marker missing or duplicated")
        if lang == "en":
            notices = re.findall(r"\\renewcommand\{\\Notice@String\}\{([^{}]*)\}", wrapper)
            require(notices == [ctx.config["release"]["english_notice"]], "English author-preview notice differs")
    return dict(tags=tags, numbered_equations=len(tags), citations=cites, bib_keys=bib_keys,
                figures=figures, figure_hashes={name: row["sha256"] for name, row in figures.items()},
                appendix_order=ctx.config["source"]["appendix_order"],
                display_equations=len(_displays(en["manuscript"])), input_file_count=len(inputs),
                identity_header=identity, proof_automatically_verified=False)


def snapshot_inputs(ctx: DeliveryContext):
    """Create or verify the single authoring snapshot used by both languages."""
    ctx.assert_writable()
    original_files = _input_files(ctx)
    rows = []
    for path in original_files:
        source_name = path.relative_to(ctx.root).as_posix()
        require(ctx.source(source_name).resolve() == path.resolve(), "Unsafe source input")
        snapshot = ctx.document / path.relative_to(ctx.source_doc)
        rows.append(dict(source=source_name, snapshot=ctx.rel_output(snapshot), **record_file(path)))
    receipt = ctx.records / "INPUTS.json"
    expected = dict(schema="paper_delivery_inputs_v1", engine=ctx.engine_identity(),
                    configuration_sha256=ctx.config_sha256,
                    source_document=ctx.config["source"]["document"], files=rows)
    if receipt.exists():
        require(read_json(receipt) == expected, "Source or engine changed since the input snapshot")
        for row in rows:
            require(record_file(ctx.output(row["snapshot"])) == {key: row[key] for key in ("sha256", "bytes")},
                    "Snapshot input changed: " + row["snapshot"])
        actual_names = {path.relative_to(ctx.document).as_posix() for path in _input_files(ctx, ctx.document)}
        require(actual_names == {path.relative_to(ctx.source_doc).as_posix() for path in original_files},
                "Snapshot input tree differs")
        return expected
    require(not ctx.document.exists() or not any(ctx.document.iterdir()), "Unregistered document snapshot already exists")
    for path, row in zip(original_files, rows):
        destination = ctx.output(row["snapshot"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        require(record_file(destination) == {key: row[key] for key in ("sha256", "bytes")}, "Input changed while snapshotting")
    for path, row in zip(original_files, rows):
        require(record_file(path) == {key: row[key] for key in ("sha256", "bytes")}, "Source changed while snapshotting")
    write_json(receipt, expected)
    return expected


def main_tables(tex):
    """Convert short Pandoc main-text longtables without changing cell bytes."""
    audits = []
    pattern = re.compile(r"\\begin\{longtable\}(?:\[[^\]]*\])?(\{.*?\})\n(\\toprule.*?)(?:\\end\{longtable\})", re.S)

    def convert(match):
        spec, body = match.groups()
        require("\\caption" not in body, "Use a raw table/table* for captioned main tables")
        first = body.find("\\endhead")
        if first >= 0:
            header = body[:first].replace("\\endfirsthead", "")
            rest = body[first + len("\\endhead"):]
        else:
            header, rest = "", body
        rest = re.sub(r"\A.*?\\endlastfoot", "", rest, flags=re.S)
        rest = rest.replace("\\endfoot", "").replace("\\endlastfoot", "")
        if "\\bottomrule" not in rest:
            rest = rest.rstrip() + "\n\\bottomrule\n"
        result = "\\begin{center}\n\\begin{tabular}" + spec + "\n" + header + rest + "\\end{tabular}\n\\end{center}"
        audits.append(dict(original_sha256=hashlib.sha256(match.group().encode()).hexdigest(),
                           result_sha256=hashlib.sha256(result.encode()).hexdigest(), column_spec=spec))
        return result

    result = pattern.sub(convert, tex)
    require("\\begin{longtable}" not in result, "Unconverted main-text longtable; inspect Pandoc format")
    return result, audits


def _tool(name, explicit=None):
    selected = explicit if explicit is not None else os.environ.get("FPL_" + name.upper())
    if selected is not None:
        selected = str(selected)
        require(bool(selected.strip()), "Requested " + name + " path is empty")
        located = shutil.which(selected)
        path = Path(located or selected).expanduser()
        require(path.is_file(), "Requested " + name + " is unavailable: " + selected)
        return path.resolve()
    located = shutil.which(name)
    if located:
        return Path(located).resolve()
    if name == "tectonic" and os.environ.get("LOCALAPPDATA"):
        path = Path(os.environ["LOCALAPPDATA"]) / "CodexPaperTools/tectonic-0.17.0/tectonic.exe"
        if path.is_file():
            return path.resolve()
    raise RuntimeError(name + " is unavailable; specify its executable or FPL_" + name.upper())


def _tool_record(path):
    result = subprocess.run([str(path), "--version"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    require(result.returncode == 0, "Cannot identify build tool: " + str(path))
    return dict(path=str(path), version=(result.stdout or result.stderr).strip(), **record_file(path))


def _log_records(ctx, paths):
    return [dict(kind=("pandoc" if path.name.startswith("PANDOC_") else
                       "tectonic_engine" if path.name.startswith("TECTONIC_ENGINE_") else "tectonic_stdout"),
                 path=ctx.rel_output(path), **record_file(path)) for path in paths]


def _pdf_details(ctx, lang, pdf, work, tex):
    import fitz

    boundaries = {}
    source = ctx.config["source"]
    expected_headings = {}
    for letter, prefix in zip(source["appendix_order"], source["appendices"]):
        text = (ctx.document / f"APPENDIX_{prefix}_{lang.upper()}.md").read_text("utf-8")
        match = re.search(r"(?m)^#\s+.+$", text)
        if match:
            expected_headings[re.sub(r"\s+", "", match.group(0).lstrip("# "))] = letter
    with fitz.open(pdf) as document:
        pages = len(document)
        page_dimensions = [[round(page.rect.width, 3), round(page.rect.height, 3)] for page in document]
        sizes = sorted(set((round(page.rect.width, 3), round(page.rect.height, 3)) for page in document))
        fonts = sorted({font[3] for page in document for font in page.get_fonts()})
        for number, page in enumerate(document, 1):
            for line in page.get_text().splitlines():
                line = line.strip()
                if line in ("References", "参考文献"):
                    boundaries.setdefault("references", number)
                if line in ("Impact Statement", "影响声明"):
                    boundaries.setdefault("impact_statement", number)
                heading = re.sub(r"\s+", "", line)
                if heading in expected_headings:
                    boundaries.setdefault("appendix_" + expected_headings[heading], number)
    aux = (work / (tex.stem + ".aux")).read_text("utf-8", errors="replace")
    match = re.search(r"\\newlabel\{candidateMainEnd\}\{\{[^}]*\}\{([0-9]+)\}", aux)
    require(match is not None, "Missing main end page label")
    semantic_labels = {}
    targets = ctx.config["contract"]["semantic_targets"]
    require(len(targets) == len(set(targets)), "Duplicate semantic targets")
    for label in targets:
        pattern = r"^\\newlabel\{" + re.escape(label) + r"\}\{\{[^\n]*?\}\{(\d+)\}\{[^\n]*?\}\{([^{}]+)\}\{\}\}$"
        entries = re.findall(pattern, aux, re.M)
        require(len(entries) == 1, "Missing or duplicate semantic target: " + label)
        page, destination = entries[0]
        require(1 <= int(page) <= pages, "Semantic target page outside PDF")
        semantic_labels[label] = dict(page=int(page), destination=destination)
    require(lang != "en" or sizes == [(612.0, 792.0)], "English preview output must be US Letter")
    bbl = work / (tex.stem + ".bbl")
    require(bbl.exists(), "Compiler did not produce natbib/BibTeX references")
    blg = work / (tex.stem + ".blg")
    warnings = [line for line in blg.read_text("utf-8", errors="replace").splitlines() if line.startswith("Warning")] if blg.exists() else []
    return dict(total_pages=pages, main_body_pages=int(match.group(1)), page_sizes=sizes,
                page_sizes_points=sizes, page_dimensions_points=page_dimensions,
                section_start_pages=boundaries, embedded_pdf_font_names=fonts,
                bibliography_warnings=warnings, bbl_sha256=sha256(bbl), semantic_labels=semantic_labels)


def build(ctx: DeliveryContext, lang: str, pandoc=None, tectonic=None):
    """Build one language into an unsealed isolated run, refusing overwrites."""
    ctx.assert_writable()
    require(lang in LANGUAGES, "Language must be en or zh")
    record_path = ctx.records / f"BUILD_{lang.upper()}.json"
    manuscript = ctx.document / f"MANUSCRIPT_{lang.upper()}.md"
    tex = ctx.document / f"manuscript_{lang}.tex"
    pdf = ctx.pdf_dir / f"{ctx.config['release']['pdf_prefix']}_{lang}.pdf"
    require(not any(path.exists() for path in (record_path, manuscript, tex, pdf)), "This language already has build outputs; use a new run")
    started, commands, log_paths = _now(), [], []
    try:
        source_contract = check_sources(ctx)
        inputs = snapshot_inputs(ctx)
        require(check_sources(ctx, ctx.document) == source_contract, "Snapshot source contract differs")
        parts, _, tags = assemble(ctx, lang, ctx.document)
        tools = {"pandoc": _tool_record(_tool("pandoc", pandoc)), "tectonic": _tool_record(_tool("tectonic", tectonic))}
        work = ctx.work / lang
        require(not work.exists(), "A build attempt already occupies this language work directory; use a new run")
        work.mkdir(parents=True)
        ctx.assert_writable()
        write_text(manuscript, parts["manuscript"])
        fragment_tex = {}
        pandoc_logs = []
        for name in FRAGMENTS:
            md, frag = work / f"{name}.md", work / f"{name}.tex"
            write_text(md, parts[name])
            command = [tools["pandoc"]["path"], str(md), "--from=markdown+tex_math_single_backslash", "--natbib",
                       "--bibliography=references.bib", "--to=latex", "--output=" + str(frag), "-M", "lang=en-US"]
            commands.append(command)
            result = subprocess.run(command, cwd=ctx.document, capture_output=True, text=True, encoding="utf-8", errors="replace")
            pandoc_logs.append(name + "\n" + result.stdout + result.stderr)
            pandoc_log = ctx.records / f"PANDOC_{lang.upper()}.log"
            write_text(pandoc_log, "\n".join(pandoc_logs))
            if pandoc_log not in log_paths:
                log_paths.append(pandoc_log)
            require(result.returncode == 0 and "not found" not in result.stderr.lower(), "Pandoc failed: " + result.stderr[-9000:])
            fragment_tex[name] = frag.read_text("utf-8")
        fragment_tex["appendices"], heading_count = re.subn(r"\\(section|subsection|subsubsection)\{", r"\\\1*{", fragment_tex["appendices"])
        fragment_tex["impact"] = re.sub(r"\\section\{", r"\\section*{", fragment_tex["impact"])
        transformations, figure_count = [], 0
        if lang == "en":
            fragment_tex["main"], transformations = main_tables(fragment_tex["main"])
            figure_count = fragment_tex["main"].count(r"\begin{figure}")
            fragment_tex["main"] = fragment_tex["main"].replace(r"\begin{figure}", r"\begin{figure*}[t]").replace(r"\end{figure}", r"\end{figure*}")
        result_tex = (ctx.document / f"templates/wrapper_{lang}.tex").read_text("utf-8")
        for name in FRAGMENTS:
            marker = "% CANDIDATE_" + name.upper()
            require(result_tex.count(marker) == 1, "Wrapper marker changed during build")
            result_tex = result_tex.replace(marker, fragment_tex[name].rstrip())
        require(result_tex.splitlines()[0] == ctx.config["release"]["identity_header"], "Generated TeX release identity differs")
        write_text(tex, result_tex)
        shutil.copyfile(tex, work / tex.name)
        style = ctx.document / "templates/icml2026"
        for path in sorted(style.rglob("*")):
            if path.is_file() and path.suffix in (".sty", ".bst"):
                shutil.copyfile(path, work / path.name)
        shutil.copyfile(ctx.document / "references.bib", work / "references.bib")
        env = os.environ.copy()
        env["TEXINPUTS"] = str(style) + os.pathsep + str(ctx.document) + os.pathsep + env.get("TEXINPUTS", "")
        env["BSTINPUTS"] = str(style) + os.pathsep + env.get("BSTINPUTS", "")
        command = [tools["tectonic"]["path"], str(tex), "--outdir", str(work), "--keep-logs", "--keep-intermediates",
                   "-Z", "search-path=" + str(style)]
        commands.append(command)
        result = subprocess.run(command, cwd=ctx.document, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        tectonic_log = ctx.records / f"TECTONIC_{lang.upper()}.log"
        write_text(tectonic_log, result.stdout + result.stderr)
        log_paths.append(tectonic_log)
        engine_log = work / (tex.stem + ".log")
        if engine_log.exists():
            permanent_log = ctx.records / f"TECTONIC_ENGINE_{lang.upper()}.log"
            shutil.copyfile(engine_log, permanent_log)
            log_paths.append(permanent_log)
        require(result.returncode == 0, "Tectonic failed: " + result.stderr[-9000:])
        ctx.assert_writable()
        require(snapshot_inputs(ctx) == inputs, "Input snapshot changed during build")
        require(check_sources(ctx, ctx.document) == source_contract, "Source contract changed during build")
        for tool in tools.values():
            require(record_file(Path(tool["path"])) == {key: tool[key] for key in ("sha256", "bytes")}, "Build tool changed during execution")
        built_pdf = work / (tex.stem + ".pdf")
        details = _pdf_details(ctx, lang, built_pdf, work, tex)
        pdf.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(built_pdf, pdf)
        outputs = {name: dict(path=ctx.rel_output(path), **record_file(path))
                   for name, path in (("manuscript", manuscript), ("tex", tex), ("pdf", pdf))}
        receipt = dict(schema="paper_delivery_build_v1", created_utc=started, finished_utc=_now(),
                       version=ctx.config["release"]["version"], language=lang, engine=ctx.engine_identity(),
                       configuration_sha256=ctx.config_sha256,
                       inputs={row["source"]: {key: row[key] for key in ("sha256", "bytes")} for row in inputs["files"]},
                       snapshot_inputs=inputs["files"], input_receipt=dict(path=ctx.rel_output(ctx.records / "INPUTS.json"), **record_file(ctx.records / "INPUTS.json")),
                       outputs=outputs, numbered_equations=tags, source_contract=source_contract, tools=tools, commands=commands,
                       logs=_log_records(ctx, log_paths),
                       main_table_transformations=transformations, main_figures_converted_to_double_column_float=figure_count,
                       appendix_headings_using_starred_commands=heading_count, visual_review_pending=True,
                       official_ICML_submission_claimed=False, submission_compliance_not_claimed=True,
                       scientific_operations=dict(scope="This authoring builder only; no assertion about prior research validation",
                           scientific_array_reads=0, new_integrals=0, new_random_draws=0, new_fits=0), **details)
        write_json(record_path, receipt)
        return receipt
    except Exception as error:
        # Never write even a failure receipt into a sealed or drifted run.
        ctx.assert_writable()
        ctx.records.mkdir(parents=True, exist_ok=True)
        serial = 1
        while (ctx.records / f"BUILD_FAILURE_{lang.upper()}_{serial:03}.json").exists():
            serial += 1
        write_json(ctx.records / f"BUILD_FAILURE_{lang.upper()}_{serial:03}.json",
                   dict(schema="paper_delivery_build_failure_v1", started_utc=started, failed_utc=_now(),
                        language=lang, engine=ctx.engine_identity(), commands=commands,
                        logs=_log_records(ctx, log_paths),
                        error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                        scope="Authoring failure in isolated run; original science and source files were not modified"))
        raise
