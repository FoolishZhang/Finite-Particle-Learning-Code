# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ZHANG
"""Redraw the adaptation figure from saved points, without coefficient fitting."""
from pathlib import Path
import argparse
import hashlib
import json
from drawing import plot

HERE=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output-dir",type=Path,required=True);args=p.parse_args()
    out=args.output_dir.resolve()
    if out.is_relative_to(HERE.parents[2]):raise ValueError("fresh output must be outside the source checkout")
    out.mkdir(parents=True,exist_ok=False);source=HERE/"data/adaptation.json";data=json.loads(source.read_text())
    plot(data,out);png=out/"adaptation_response.png"
    result=dict(schema="h34_public_redraw_v1",saved_point_count=30,coefficient_fit=False,new_scientific_calls=0,
        input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),png_sha256=hashlib.sha256(png.read_bytes()).hexdigest())
    (out/"REDRAW.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf8");print(json.dumps(result,indent=2))


if __name__=="__main__":main()
