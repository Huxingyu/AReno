"""Generate one Kaggle kernel folder per role from kernel_template.py."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
USER = "x1ngyu"
ROLES = ["build", "lane0", "lane1", "lane2", "lane3"]
tpl = (HERE / "kernel_template.py").read_text()
out = HERE / "kernels"
for role in ROLES:
    d = out / role; d.mkdir(parents=True, exist_ok=True)
    (d / "run.py").write_text(tpl.replace('"__LANE__"', json.dumps(role)))
    meta = {"id": f"{USER}/areno-kaggle-{role}", "title": f"areno-kaggle-{role}", "code_file": "run.py",
            "language": "python", "kernel_type": "script", "is_private": "true", "enable_gpu": "true",
            "enable_tpu": "false", "enable_internet": "true", "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [], "competition_sources": [], "model_sources": [],
            "kernel_sources": [] if role == "build" else [f"{USER}/areno-kaggle-build"]}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(role, "->", d)
