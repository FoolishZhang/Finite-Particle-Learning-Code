"""Paths, configuration and immutable run identity for the common paper tools."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any

ENGINE = "paper_delivery_v1"
ENGINE_FILES = (
    "__init__.py", "core.py", "builder.py", "renderer.py", "verification.py",
)
FORBIDDEN_SUFFIXES = {".npz", ".npy", ".pt", ".pth", ".pkl"}
MUTABLE_PATHS = {
    "AGENTS.md", "ARGUMENT_FRAMEWORK.md", "HANDOFF.md", "REVIEW_BRIEF.md",
    "docs/00_project_项目管理/START_HERE_当前工作入口.md",
    "docs/00_project_项目管理/CURRENT_WORK_STATE.json",
    "docs/00_project_项目管理/NEXT_STEPS.md",
    "docs/00_project_项目管理/PROJECT_CONTEXT.md",
    "docs/00_project_项目管理/PROJECT_HISTORY.md",
}


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_file(path: Path | str) -> dict:
    path = Path(path)
    return {"sha256": sha256(path), "bytes": path.stat().st_size}


def read_json(path: Path | str) -> Any:
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          ValueError("Non-finite JSON number: " + value)))


def write_text(path: Path | str, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(line.rstrip() for line in text.splitlines()).rstrip() + "\n",
                    encoding="utf-8", newline="\n")


def write_json(path: Path | str, value: Any) -> None:
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def relative_path(value: str, label: str) -> str:
    require(isinstance(value, str) and bool(value), label + " must be a nonempty path")
    require("\\" not in value and ":" not in value, label + " must use relative POSIX paths")
    require(not value.startswith("/") and all(p not in ("", ".", "..") for p in value.split("/")),
            label + " must be canonical and relative")
    return value


def inside(base: Path, relative: str, label: str) -> Path:
    relative_path(relative, label)
    path = (base / relative).resolve()
    require(path.is_relative_to(base), label + " escapes its root")
    return path


def validate_config(config: dict) -> None:
    require(isinstance(config, dict), "Configuration must be an object")
    permitted = {"schema_version", "engine", "mode", "release", "source", "contract",
                 "reference", "render", "source_gate", "notes"}
    require(not set(config) - permitted, "Unknown configuration fields: " + str(set(config) - permitted))
    require(config.get("schema_version") == 1 and config.get("engine") == ENGINE,
            "Unsupported configuration or engine generation")
    require(config.get("mode") in ("reference_replay", "authoring"), "Unknown delivery mode")
    release = config.get("release", {})
    require(isinstance(release, dict) and set(release) == {
        "version", "date", "identity_header", "pdf_prefix", "english_notice",
    }, "Unexpected release configuration fields")
    for field in ("version", "date", "identity_header", "pdf_prefix", "english_notice"):
        require(isinstance(release.get(field), str) and release[field], "Missing release." + field)
    require(re.fullmatch(r"[A-Za-z0-9_.-]+", release["pdf_prefix"]), "Unsafe PDF prefix")
    require(re.fullmatch(r"\d{4}-\d{2}-\d{2}", release["date"]), "Identity date must be YYYY-MM-DD")
    require(release["identity_header"].startswith("% ") and "\n" not in release["identity_header"]
            and "\r" not in release["identity_header"], "Identity header must be one TeX comment line")
    source = config.get("source", {})
    require(isinstance(source, dict) and not set(source) - {
        "document", "source_prefixes", "appendices", "appendix_order", "preserved_manifests",
    }, "Unexpected source configuration fields")
    relative_path(source.get("document"), "source.document")
    require(source["document"].startswith("docs/07_papers_论文工作/"), "Expected a paper source directory")
    for key in ("source_prefixes", "appendices", "appendix_order"):
        values = source.get(key)
        require(isinstance(values, list) and values and len(set(values)) == len(values),
                "Invalid source." + key)
        require(all(isinstance(v, str) and re.fullmatch(r"[A-Z_]+", v) for v in values),
                "Unsafe source member name")
    require(source["source_prefixes"] == ["BODY", "RELATED_WORK", "IMPACT", "APPENDIX_GUIDE",
            *("APPENDIX_" + name for name in source["appendices"])], "Source roles must be explicit")
    require(source["appendix_order"] == [chr(65 + i) for i in range(len(source["appendices"]))]
            and len(source["appendices"]) <= 26, "Appendix order must be A onward")
    contract = config.get("contract", {})
    require(isinstance(contract, dict) and not set(contract) - {
        "numbered_equations", "main_equations", "citation_count", "bibliography_count",
        "figure_count", "semantic_targets", "navigation_rules",
    }, "Unexpected contract configuration fields")
    for key in ("numbered_equations", "main_equations", "citation_count", "bibliography_count", "figure_count"):
        require(type(contract.get(key)) is int and contract[key] >= 0, "Invalid contract." + key)
    targets = contract.get("semantic_targets")
    require(isinstance(targets, list) and targets and len(set(targets)) == len(targets)
            and all(isinstance(x, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", x) for x in targets),
            "Invalid semantic navigation targets")
    rules = contract.get("navigation_rules", {})
    require(isinstance(rules, dict) and set(rules) == set(targets), "Each navigation target needs a rule")
    for rule in rules.values():
        require(isinstance(rule, dict) and len(rule) == 1, "Each navigation rule needs one criterion")
        require(("section" in rule and isinstance(rule["section"], str) and bool(rule["section"]))
                or ("kind" in rule and rule["kind"] in ("theorem", "proposition", "corollary")),
                "Unknown navigation rule")
    require(isinstance(source.get("preserved_manifests", []), list), "Manifest pins must be a list")
    for row in source.get("preserved_manifests", []):
        require(isinstance(row, dict), "Invalid preserved manifest")
        require(not set(row) - {"path", "sha256", "mapping", "count", "base"},
                "Unknown preserved manifest fields")
        relative_path(row.get("path"), "preserved manifest")
        if "base" in row:
            relative_path(row["base"], "preserved package base")
        require(re.fullmatch(r"[a-f0-9]{64}", row.get("sha256", "")), "Pin each preserved manifest")
        require(row.get("mapping", "tracked_evidence") in ("tracked_evidence", "files"),
                "Unsupported manifest file map")
        if "count" in row:
            require(type(row["count"]) is int and row["count"] > 0, "Invalid manifest count")
    if config["mode"] == "reference_replay":
        require(config.get("reference"), "Replay references required")
        require(source.get("preserved_manifests"), "Replay must pin its original sealed package")
        require(config.get("source_gate"), "Replay must retain the version-specific source checker")
    if "reference" in config:
        reference = config.get("reference", {})
        require(isinstance(reference, dict) and {"languages", "visual"} <= set(reference)
                and not set(reference) - {"languages", "visual", "package_root"},
                "Unexpected reference configuration fields")
        if "package_root" in reference:
            relative_path(reference["package_root"], "reference package root")
        require(set(reference.get("languages", {})) == {"en", "zh"}, "Both replay references required")
        for row in reference["languages"].values():
            require(isinstance(row, dict) and set(row) == {"manuscript", "tex", "pdf", "build", "render"},
                    "Unexpected language reference fields")
            for key in ("manuscript", "tex", "pdf", "build", "render"):
                relative_path(row.get(key), "reference." + key)
        relative_path(reference.get("visual"), "reference.visual")
    render = config.get("render", {})
    require(isinstance(render, dict) and not set(render) - {"dpi"}, "Unexpected render configuration fields")
    dpi = render.get("dpi", 130)
    require(type(dpi) is int and 72 <= dpi <= 300, "Render DPI outside supported range")
    gate = config.get("source_gate")
    if gate is not None:
        require(set(gate) == {"script", "sha256", "args"}, "Unexpected source gate fields")
        relative_path(gate["script"], "source gate")
        require(gate["script"].startswith("script/") and gate["script"].endswith("_delivery.py"),
                "Source gate must be a repository delivery checker")
        require(re.fullmatch(r"[a-f0-9]{64}", gate["sha256"]), "Source checker must be pinned")
        require(gate["args"] == ["--check", "--public-only"], "Only read-only public source gates allowed")


class DeliveryContext:
    """One read-only source repository and one separate output run."""

    def __init__(self, root: Path | str, config_path: Path | str,
                 workspace: Path | str | None = None):
        require(sys.flags.optimize == 0, "Run without -O/-OO")
        self.root = Path(root).resolve(strict=True)
        require(self.root.is_dir(), "Repository root must be a directory")
        self.config_path = Path(config_path).resolve(strict=True)
        self.config = read_json(self.config_path)
        validate_config(self.config)
        self.config_sha256 = sha256(self.config_path)
        self.source_doc = self.source(self.config["source"]["document"])
        require(self.source_doc.is_dir(), "Paper source directory missing")
        self.workspace = Path(workspace).resolve() if workspace is not None else None
        if self.workspace is not None:
            require(not self.workspace.is_relative_to(self.root)
                    and not self.root.is_relative_to(self.workspace),
                    "Output workspace must be separate from, and outside, the source repository")
            self.document = self.workspace / "document"
            self.records = self.workspace / "records"
            self.work = self.workspace / "work"
            self.pdf_dir = self.workspace / "pdf"

    def source(self, relative: str) -> Path:
        path = inside(self.root, relative, "Source path")
        require(path.suffix.lower() not in FORBIDDEN_SUFFIXES, "Scientific/binary model arrays are excluded")
        require(not path.is_relative_to(self.root / "tmp")
                and not path.is_relative_to(self.root / ".git"), "Temporary/Git internals are not source inputs")
        return path

    def output(self, relative: str) -> Path:
        require(self.workspace is not None, "This command needs a workspace")
        path = inside(self.workspace, relative, "Output path")
        require(path.suffix.lower() not in FORBIDDEN_SUFFIXES, "Scientific arrays are excluded")
        return path

    def rel_output(self, path: Path | str) -> str:
        require(self.workspace is not None, "No workspace")
        result = Path(path).resolve()
        require(result.is_relative_to(self.workspace), "Output path escapes run")
        return result.relative_to(self.workspace).as_posix()

    def rel_source(self, path: Path | str) -> str:
        result = Path(path).resolve()
        require(result.is_relative_to(self.root), "Source path escapes repository")
        return result.relative_to(self.root).as_posix()

    def engine_identity(self) -> dict:
        base = Path(__file__).resolve().parent
        paths = [(f"script/{ENGINE}/{name}", base / name) for name in ENGINE_FILES]
        paths.append(("script/paper_delivery.py", base.parent / "paper_delivery.py"))
        require(all(p.is_file() for _, p in paths), "Incomplete common engine installation")
        return {"id": ENGINE, "sources": {name: record_file(path) for name, path in paths}}

    def initialize(self) -> dict:
        require(self.workspace is not None, "A new workspace is required")
        require(not self.workspace.exists(), "Output workspace already exists; choose a fresh directory")
        # Validate everything that can fail before creating the new run.
        engine = self.engine_identity()
        source_result = verify_preserved_sources(self)
        marker = {"schema": "paper_delivery_run_v1", "version": self.config["release"]["version"],
                  "mode": self.config["mode"], "configuration_sha256": self.config_sha256,
                  "engine": engine, "source_root": str(self.root), "preserved_sources": source_result}
        self.workspace.mkdir(parents=True, exist_ok=False)
        for path in (self.document, self.records, self.work, self.pdf_dir):
            path.mkdir()
        # The original configuration bytes are retained, not regenerated JSON.
        (self.workspace / "PROFILE.json").write_bytes(self.config_path.read_bytes())
        write_json(self.workspace / "RUN.json", marker)
        return marker

    def assert_identity(self) -> dict:
        require(self.workspace is not None and self.workspace.is_dir(), "Initialize a fresh run first")
        self.assert_output_layout()
        require(sha256(self.config_path) == self.config_sha256, "Configuration changed while running")
        require(sha256(self.output("PROFILE.json")) == self.config_sha256, "Run configuration differs")
        marker = read_json(self.output("RUN.json"))
        require(marker["schema"] == "paper_delivery_run_v1"
                and marker["configuration_sha256"] == self.config_sha256
                and marker["version"] == self.config["release"]["version"]
                and marker["mode"] == self.config["mode"], "Run identity differs")
        require(marker["engine"] == self.engine_identity(), "Common engine changed since initialization")
        # source_root is provenance, not a machine-specific dependency.
        # Actual source bytes are revalidated from the caller's root by check().
        return marker

    def assert_output_layout(self, require_work: bool = False) -> None:
        """Reject replaced directories before any direct pathlib/copyfile write.

        Path.is_symlink alone misses Windows junctions. A run does not need
        reparse points at any depth; rejecting them also protects nested tool
        output directories rather than merely checking the four top-level paths.
        """
        require(self.workspace is not None and self.workspace.resolve() == self.workspace,
                "Workspace was redirected")
        required = [self.document, self.records, self.pdf_dir]
        if require_work or self.work.exists():
            required.append(self.work)
        for path in required:
            require(path.is_dir() and path.resolve() == path,
                    "Output directory was replaced or redirected: " + path.name)
        pending = [self.workspace]
        while pending:
            directory = pending.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    info = entry.stat(follow_symlinks=False)
                    require(not entry.is_symlink()
                            and not (getattr(info, "st_file_attributes", 0)
                                     & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)),
                            "Reparse points/symlinks are forbidden in output runs: " + entry.name)
                    path = Path(entry.path)
                    require(path.resolve().is_relative_to(self.workspace), "Output path escaped run")
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(path)

    def assert_writable(self) -> None:
        self.assert_identity()
        self.assert_output_layout(require_work=True)
        for name in ("DELIVERY_MANIFEST.json", "VERSION_LOCK.json",
                     "records/DELIVERY_MANIFEST.json", "records/VERSION_LOCK.json",
                     "document/VERSION_LOCK.json"):
            require(not self.output(name).exists(), "Run is sealed or locked: " + name)


def verify_preserved_sources(ctx: DeliveryContext) -> dict:
    """Check pinned public records without importing old modules or computing science."""
    counts = {}
    for item in ctx.config["source"].get("preserved_manifests", []):
        path = ctx.source(item["path"])
        require(sha256(path) == item["sha256"], "Preserved manifest changed: " + item["path"])
        saved = read_json(path)
        mapping = saved[item.get("mapping", "tracked_evidence")]
        require(isinstance(mapping, dict) and mapping, "Empty preserved file map")
        if "count" in item:
            require(len(mapping) == item["count"], "Preserved manifest count differs")
        for name, row in mapping.items():
            require(name not in MUTABLE_PATHS, "Mutable navigation cannot be a frozen dependency: " + name)
            relative_path(name, "Manifest member")
            logical = item["base"] + "/" + name if "base" in item else name
            target = ctx.source(logical)
            digest = row if isinstance(row, str) else row["sha256"]
            require(sha256(target) == digest, "Preserved source changed: " + name)
            if isinstance(row, dict) and "bytes" in row:
                require(target.stat().st_size == row["bytes"], "Preserved size changed: " + name)
        counts[item["path"]] = len(mapping)
    return {"manifests": counts, "files_verified_with_repetition": sum(counts.values()),
            "scientific_array_reads": 0}
