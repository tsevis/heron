"""Fetch the optional Layer A models a fresh clone needs (`--ai`).

Heron renders without any of these: the classical path is always available and
every instrument must work with ``--no-ai`` (CLAUDE.md §2.6). These buy real
depth and a clean subject matte, which is the difference between plausible form
and a pseudo-depth prior.

Deliberately a **separate, explicit step**. The engine never downloads anything
mid-render — its loaders pass ``local_files_only=True`` — because an offline,
deterministic engine that quietly reaches for the network is neither. So this
script exists, it asks before spending bandwidth, and it skips anything already
present.

    python scripts/fetch_models.py            # report, then ask
    python scripts/fetch_models.py --check    # report only, exit 1 if missing
    python scripts/fetch_models.py --yes      # no prompt (CI, or you already read it)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from heron.scene.ai import registry  # noqa: E402


def report() -> list:
    print("Layer A models\n")
    for spec, where in registry.status():
        if where is None:
            gate = " [gated]" if spec.gated else ""
            print(f"  ✗ {spec.role:6s} MISSING   {spec.approx_mb:>5} MB{gate}  {spec.purpose}")
            if spec.note:
                print(f"      {spec.note}")
        else:
            origin = "hugging face cache" if where == spec.hf_id else where
            print(f"  ✓ {spec.role:6s} present   {origin}")
    missing = registry.missing()
    print()
    if not missing:
        print("Everything needed for --ai is already here. Nothing to download.")
    else:
        total = sum(s.approx_mb for s in missing)
        print(f"{len(missing)} model(s) missing, about {total} MB total.")
        print("Heron still runs without them — the classical path needs no models —")
        print("but --ai will fall back and the result will be visibly coarser.")
    return missing


def fetch(missing: list) -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("\nhuggingface_hub is not installed. Install the AI extras first:")
        print("    .venv/bin/pip install -r requirements-ai.txt")
        return 1

    target_root = REPO / "models"
    target_root.mkdir(exist_ok=True)
    failed = []
    for spec in missing:
        target = target_root / spec.dir_name
        print(f"\n→ {spec.role}: {spec.hf_id} (~{spec.approx_mb} MB)")
        try:
            kwargs = {"repo_id": spec.hf_id, "local_dir": str(target)}
            # Some repos ship their own python package (SAM 3), so an allow-list
            # of weight extensions would fetch something that cannot load. Those
            # use an ignore-list instead.
            if spec.patterns:
                kwargs["allow_patterns"] = list(spec.patterns)
            if spec.ignore:
                kwargs["ignore_patterns"] = list(spec.ignore)
            snapshot_download(**kwargs)
            print(f"  installed -> models/{spec.dir_name}")
        except Exception as e:
            print(f"  FAILED: {e}")
            if spec.gated:
                print(f"      {spec.hf_id} is gated. Accept the licence at")
                print(f"      https://huggingface.co/{spec.hf_id} then run")
                print("      `huggingface-cli login` and try again.")
            failed.append(spec.role)

    if failed:
        print(f"\nCould not fetch: {', '.join(failed)}")
        print("Heron still works on the classical path (--no-ai).")
        return 1
    print("\nDone. `--ai` is now available.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="report only; exit 1 if anything is missing")
    ap.add_argument("--yes", "-y", action="store_true", help="download without asking")
    args = ap.parse_args()

    missing = report()
    if not missing:
        return 0
    if args.check:
        return 1

    if not args.yes:
        total = sum(s.approx_mb for s in missing)
        answer = input(f"\nDownload {len(missing)} model(s), ~{total} MB? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Skipped. Run this again any time, or use --no-ai.")
            return 0

    return fetch(missing)


if __name__ == "__main__":
    raise SystemExit(main())
