from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "configs" / "artifact_bundle.example.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether a local artifact bundle has the files needed for CACC reproduction rows."
    )
    parser.add_argument(
        "--bundle",
        default=DEFAULT_BUNDLE,
        type=Path,
        help="Artifact bundle manifest JSON.",
    )
    parser.add_argument(
        "--root",
        default=ROOT,
        type=Path,
        help="Root directory used to resolve relative artifact paths.",
    )
    parser.add_argument(
        "--row",
        action="append",
        default=[],
        help="Row id to check, for example compmath/cacc_spp. Repeat to check multiple rows.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional machine-readable output path.",
    )
    parser.add_argument(
        "--jsonl-sample-lines",
        default=20,
        type=int,
        help="Number of initial JSONL lines to parse for jsonl artifacts.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if any selected artifact is missing or invalid.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = root / path
    return path


def selected_for_rows(artifact: dict[str, Any], rows: set[str]) -> bool:
    if not rows:
        return True
    required_for = set(artifact.get("required_for", []))
    if "all" in required_for:
        return True
    return bool(required_for & rows)


def check_json(path: Path) -> tuple[str, str]:
    try:
        load_json(path)
    except Exception as exc:  # noqa: BLE001
        return "INVALID", f"json parse failed: {exc}"
    return "OK", "json parsed"


def check_jsonl(path: Path, sample_lines: int) -> tuple[str, str]:
    parsed = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if line_no > sample_lines:
                    break
                if not line.strip():
                    continue
                json.loads(line)
                parsed += 1
    except Exception as exc:  # noqa: BLE001
        return "INVALID", f"jsonl parse failed near sampled line {parsed + 1}: {exc}"
    return "OK", f"sampled {parsed} jsonl lines"


def check_metric_summary(path: Path) -> tuple[str, str]:
    try:
        data = load_json(path)
    except Exception as exc:  # noqa: BLE001
        return "INVALID", f"metric summary json parse failed: {exc}"
    required = [
        ("oracle_coverage",),
        ("selection_efficiency_given_oracle", "verifier"),
        ("verifier_accuracy",),
    ]
    missing: list[str] = []
    for key_path in required:
        cur: Any = data
        for part in key_path:
            if not isinstance(cur, dict) or part not in cur:
                missing.append(".".join(key_path))
                break
            cur = cur[part]
        else:
            if not isinstance(cur, (int, float)):
                missing.append(".".join(key_path))
    if missing:
        return "INVALID", "missing numeric fields: " + ", ".join(missing)
    return "OK", "metric fields present"


def check_model_dir(path: Path) -> tuple[str, str]:
    if not path.is_dir():
        return "MISSING", "model directory not found"
    config = path / "config.json"
    tokenizer = path / "tokenizer_config.json"
    markers = [marker.name for marker in (config, tokenizer) if marker.exists()]
    if markers:
        return "OK", "model markers: " + ", ".join(markers)
    return "OK", "directory exists; model marker files not checked"


def check_artifact(artifact: dict[str, Any], root: Path, sample_lines: int) -> dict[str, Any]:
    path = resolve_path(root, artifact["path"])
    kind = artifact.get("kind", "file")
    result: dict[str, Any] = {
        "id": artifact["id"],
        "kind": kind,
        "path": str(path),
        "required_for": artifact.get("required_for", []),
    }
    if kind in {"dir", "model_dir"}:
        if not path.exists():
            result.update({"status": "MISSING", "detail": "path does not exist"})
            return result
        if kind == "model_dir":
            status, detail = check_model_dir(path)
        else:
            status, detail = ("OK", "directory exists") if path.is_dir() else ("INVALID", "expected directory")
        result.update({"status": status, "detail": detail})
        return result

    if not path.exists():
        result.update({"status": "MISSING", "detail": "path does not exist"})
        return result
    if not path.is_file():
        result.update({"status": "INVALID", "detail": "expected file"})
        return result

    if kind == "json":
        status, detail = check_json(path)
    elif kind == "jsonl":
        status, detail = check_jsonl(path, sample_lines)
    elif kind == "metric_summary":
        status, detail = check_metric_summary(path)
    else:
        status, detail = "OK", "file exists"
    result.update({"status": status, "detail": detail, "size_bytes": path.stat().st_size})
    return result


def print_results(results: list[dict[str, Any]]) -> None:
    print("| Artifact | Kind | Status | Detail | Path |")
    print("| --- | --- | --- | --- | --- |")
    for result in results:
        print(
            "| {id} | {kind} | {status} | {detail} | {path} |".format(
                id=result["id"],
                kind=result["kind"],
                status=result["status"],
                detail=result["detail"],
                path=result["path"],
            )
        )


def main() -> int:
    args = parse_args()
    bundle = load_json(args.bundle)
    rows = set(args.row)
    artifacts = [
        artifact
        for artifact in bundle.get("artifacts", [])
        if selected_for_rows(artifact, rows)
    ]
    results = [
        check_artifact(artifact, args.root, args.jsonl_sample_lines)
        for artifact in artifacts
    ]
    print_results(results)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    failed = any(result["status"] != "OK" for result in results)
    return 1 if args.strict and failed else 0


if __name__ == "__main__":
    sys.exit(main())
