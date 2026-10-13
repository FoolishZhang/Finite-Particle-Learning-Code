"""Run the pinned common authoring pipeline without writing to sealed papers."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from paper_delivery_v1.core import (
    DeliveryContext, read_json, require, sha256, verify_preserved_sources,
    write_json, write_text,
)


def source_check(ctx):
    gate = ctx.config.get("source_gate")
    require(gate is not None, "This profile has no archived source checker")
    script = ctx.source(gate["script"])
    require(sha256(script) == gate["sha256"], "Source checker identity changed")
    if ctx.workspace is not None:
        ctx.assert_writable()
        require(not (ctx.records / "SOURCE_CHECK.json").exists(), "Source check already recorded")
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [sys.executable, "-B", str(script), *gate["args"]]
    result = subprocess.run(command, cwd=ctx.root, env=env, capture_output=True,
                            text=True, encoding="utf-8", errors="replace")
    receipt = {
        "schema": "paper_delivery_source_check_v1",
        "checker": gate["script"], "checker_sha256": gate["sha256"],
        "returncode": result.returncode, "passed": result.returncode == 0,
        "configuration_sha256": ctx.config_sha256,
        "scope": "Archived version-specific checker; not a new scientific computation",
    }
    if ctx.workspace is not None:
        for suffix, content in (("stdout", result.stdout), ("stderr", result.stderr)):
            path = ctx.records / f"SOURCE_CHECK.{suffix}.log"
            write_text(path, content)
            receipt[suffix] = {"path": ctx.rel_output(path), "sha256": sha256(path)}
        write_json(ctx.records / "SOURCE_CHECK.json", receipt)
    require(result.returncode == 0, "Archived source check failed: " + result.stderr[-3000:])
    return receipt


def main(argv=None):
    # JSON output has an explicit encoding even when Windows pipes default to
    # a legacy code page. Source paths and receipts can contain Chinese names.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(
        "inspect", "init", "source-check", "build", "render",
        "reference-equivalence", "check", "seal",
    ))
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--language", choices=("en", "zh", "both"), default="both")
    parser.add_argument("--pandoc")
    parser.add_argument("--tectonic")
    parser.add_argument("--poppler")
    parser.add_argument("--dpi", type=int)
    parser.add_argument("--images", action="store_true", help="Also verify stored render image contents")
    parser.add_argument("--require-visual", action="store_true")
    args = parser.parse_args(argv)
    try:
        ctx = DeliveryContext(args.root, args.profile, args.workspace)
        languages = ("en", "zh") if args.language == "both" else (args.language,)
        if args.command == "inspect":
            from paper_delivery_v1.builder import check_sources
            result = {"passed": True, "preserved_sources": verify_preserved_sources(ctx),
                      "source_contract": check_sources(ctx)}
        elif args.command == "init":
            result = {"initialized": True, "run": ctx.initialize()}
        elif args.command == "source-check":
            result = source_check(ctx)
        else:
            require(ctx.workspace is not None, "--workspace is required for this command")
            ctx.assert_identity()
            if args.command == "build":
                from paper_delivery_v1.builder import build
                result = {"built": {lang: build(ctx, lang, pandoc=args.pandoc, tectonic=args.tectonic)
                                    for lang in languages}}
            elif args.command == "render":
                from paper_delivery_v1.renderer import render
                dpi = args.dpi if args.dpi is not None else ctx.config.get("render", {}).get("dpi", 130)
                result = {"rendered": {lang: render(ctx, lang, poppler=args.poppler, dpi=dpi)
                                       for lang in languages}}
            else:
                from paper_delivery_v1.verification import check, seal, record_reference_equivalence
                if args.command == "reference-equivalence":
                    result = record_reference_equivalence(ctx)
                elif args.command == "check":
                    result = check(ctx, require_visual=args.require_visual, verify_images=args.images)
                else:
                    result = seal(ctx)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, AssertionError) as error:
        print(json.dumps({"passed": False, "error_type": type(error).__name__,
                          "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
