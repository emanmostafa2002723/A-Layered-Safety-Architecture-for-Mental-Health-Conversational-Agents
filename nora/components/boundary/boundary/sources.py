"""Download the public Aegis source with an immutable revision and license card.

Downloading does NOT map generic safety annotations to project boundary labels.
The audit lists native labels, annotation sources and advice category counts.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def download_aegis(output):
    from huggingface_hub import HfApi, snapshot_download
    repo="nvidia/Aegis-AI-Content-Safety-Dataset-2.0"
    info=HfApi().dataset_info(repo)
    revision=info.sha
    destination=Path(output)
    destination.mkdir(parents=True,exist_ok=True)
    snapshot_download(repo,repo_type="dataset",revision=revision,local_dir=destination,
                      allow_patterns=["README.md","train.json","validation.json","test.json","*.parquet","**/*.parquet","*.jsonl","**/*.jsonl","LICENSE*"])
    files={str(p.relative_to(destination)):sha256(p) for p in destination.rglob("*") if p.is_file() and ".cache" not in p.parts and p.name!="source_manifest.json"}
    manifest={"repository":repo,"revision":revision,"files":files,
              "license":"CC-BY-4.0 according to source dataset card",
              "native_prompt_labels":"human annotated generic safety, not clinician boundary labels",
              "used_for_boundary_training":False}
    (destination/"source_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return audit_aegis(destination)


def audit_aegis(directory):
    import pyarrow.parquet as pq
    report={"files":{},"used_for_boundary_training":False,
            "mapping_decision":"Do not equate generic unsafe/unauthorized-advice labels with this project's boundary policy. No automatic import.",
            "source_manifest":json.loads((Path(directory)/"source_manifest.json").read_text(encoding="utf-8"))}
    paths=sorted(Path(directory).rglob("*.parquet"))+[p for p in sorted(Path(directory).glob("*.json")) if p.name!="source_manifest.json"]
    for path in paths:
        if path.suffix==".parquet": rows=pq.read_table(path).to_pylist()
        else:
            raw=path.read_text(encoding="utf-8")
            try: rows=json.loads(raw)
            except json.JSONDecodeError: rows=[json.loads(line) for line in raw.splitlines() if line.strip()]
            if isinstance(rows,dict): rows=rows.get("data",[])
        counts=Counter(str(r.get("prompt_label")) for r in rows)
        provenance=Counter(str(r.get("prompt_label_source")) for r in rows)
        advice=[r for r in rows if "advice" in str(r.get("violated_categories","")).lower()]
        report["files"][str(path.relative_to(directory))]={"rows":len(rows),"columns":list(rows[0]) if rows else [],"prompt_labels":dict(counts),"prompt_annotation_source":dict(provenance),"advice_category_rows":len(advice),"advice_native_labels":dict(Counter(str(r.get("prompt_label")) for r in advice))}
    if not report["files"]: raise ValueError("No data files downloaded; refusing to report an empty audit as success")
    Path("evidence").mkdir(exist_ok=True)
    Path("evidence/aegis_source_audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--download-aegis",action="store_true")
    parser.add_argument("--directory",default="data/raw/aegis2")
    args=parser.parse_args()
    print(json.dumps(download_aegis(args.directory) if args.download_aegis else audit_aegis(args.directory),indent=2))

