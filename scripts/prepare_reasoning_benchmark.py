from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
import sys
from urllib.parse import quote
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import Request
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
VENDOR = ROOT / "vendor"
if VENDOR.exists() and str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from src.eval.evaluate_predictions import canonicalize_numeric_token
from src.utils.io_utils import write_jsonl
from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError:
    pq = None


PROJECT_ROOT = ROOT.parents[1]
SOURCE_ROOT = PROJECT_ROOT / "Experiment" / "datasets" / "raw" / "source" / "hf_benchmarks"


DATASET_SPECS = {
    "competition_math_numeric": {
        "dataset": "jeggers/competition_math",
        "config": "numeric",
        "default_split": "test",
        "source_file_template": "numeric/{split}-00000-of-00001.parquet",
        "source_format": "parquet",
    },
    "mmlu_pro": {
        "dataset": "TIGER-Lab/MMLU-Pro",
        "config": "default",
        "default_split": "test",
        "source_file_template": "data/{split}-00000-of-00001.parquet",
        "source_format": "parquet",
    },
    "gpqa_diamond": {
        "dataset": "johnsonafool/gpqa",
        "config": "gpqa_diamond",
        "default_split": "train",
        "source_file_template": "gpqa_diamond.csv",
        "source_format": "csv",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare benchmark-ready JSONL records for reasoning benchmarks.")
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=sorted(DATASET_SPECS),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--start-offset", default=0, type=int)
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Sample a reproducible shuffled subset instead of taking a contiguous slice.",
    )
    parser.add_argument(
        "--request-sleep",
        default=0.75,
        type=float,
        help="Sleep interval in seconds between row-block requests to avoid dataset-server rate limits.",
    )
    parser.add_argument(
        "--max-attempts",
        default=8,
        type=int,
        help="Maximum retries per metadata/row request.",
    )
    return parser.parse_args()


def _fetch_json(url: str, max_attempts: int = 8) -> tuple[dict, dict]:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            request = Request(url, headers={"User-Agent": "dr-claw-benchmark-prep/1.0"})
            with urlopen(request, timeout=120) as response:
                payload = response.read().decode("utf-8")
                headers = dict(response.info())
            return json.loads(payload), headers
        except (HTTPError, URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == max_attempts:
                break
            if isinstance(exc, HTTPError) and exc.code == 429:
                time.sleep(max(10.0, 8.0 * attempt))
            else:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"Failed to fetch dataset metadata after {max_attempts} attempts: {url}") from last_error


def _download_repo_source(dataset: str, filename: str) -> tuple[Path, str | None]:
    repo_info = HfApi().dataset_info(dataset)
    local_dir = SOURCE_ROOT / dataset.replace("/", "__")
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = hf_hub_download(
        repo_id=dataset,
        repo_type="dataset",
        filename=filename,
        local_dir=local_dir,
    )
    return Path(local_path), repo_info.sha


def _load_rows_from_repo_source(spec: dict, split: str) -> tuple[list[dict], str | None]:
    filename_template = spec.get("source_file_template")
    source_format = spec.get("source_format")
    if not filename_template or not source_format:
        raise ValueError("Missing repo-source metadata in dataset spec.")
    filename = filename_template.format(split=split)
    local_path, revision = _download_repo_source(spec["dataset"], filename)
    if source_format == "csv":
        with local_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return rows, revision
    if source_format == "parquet":
        if pq is None:
            raise RuntimeError("pyarrow is required for parquet-backed benchmark preparation.")
        table = pq.read_table(local_path)
        return table.to_pylist(), revision
    raise ValueError(f"Unsupported source format: {source_format}")


def _build_rows_url(dataset: str, config: str, split: str, offset: int, length: int) -> str:
    return (
        "https://datasets-server.huggingface.co/rows"
        f"?dataset={quote(dataset, safe='')}"
        f"&config={quote(config, safe='')}"
        f"&split={quote(split, safe='')}"
        f"&offset={offset}&length={length}"
    )


def _fetch_rows(dataset: str, config: str, split: str, offset: int, length: int, max_attempts: int) -> tuple[list[dict], str | None]:
    payload, headers = _fetch_json(_build_rows_url(dataset, config, split, offset, length), max_attempts=max_attempts)
    rows = [entry["row"] for entry in payload["rows"]]
    revision = headers.get("x-revision")
    return rows, revision


def _fetch_num_examples(dataset: str, config: str, split: str, max_attempts: int) -> int:
    url = f"https://datasets-server.huggingface.co/splits?dataset={quote(dataset, safe='')}"
    payload, _ = _fetch_json(url, max_attempts=max_attempts)
    for row in payload["splits"]:
        if row["config"] == config and row["split"] == split:
            if "num_examples" in row:
                return int(row["num_examples"])
            break
    info_url = f"https://datasets-server.huggingface.co/info?dataset={quote(dataset, safe='')}"
    payload, _ = _fetch_json(info_url, max_attempts=max_attempts)
    split_info = payload["dataset_info"][config]["splits"][split]
    return int(split_info["num_examples"])


def _select_indices(total: int, start_offset: int, max_examples: int | None, seed: int, shuffle: bool) -> list[int]:
    if total <= 0:
        return []
    if max_examples is None:
        end = total
    else:
        end = min(total, start_offset + max_examples)
    if not shuffle:
        return list(range(start_offset, end))
    population = list(range(total))
    rng = random.Random(seed)
    count = total if max_examples is None else min(total, max_examples)
    sampled = rng.sample(population, k=count)
    sampled.sort()
    return sampled


def _fetch_selected_rows(
    dataset: str,
    config: str,
    split: str,
    indices: list[int],
    max_attempts: int,
    request_sleep: float,
) -> tuple[list[tuple[int, dict]], str | None]:
    if not indices:
        return [], None
    indexed_rows: list[tuple[int, dict]] = []
    revision: str | None = None
    block_start = indices[0]
    block_indices = [indices[0]]
    for index in indices[1:]:
        if index == block_indices[-1] + 1 and len(block_indices) < 100:
            block_indices.append(index)
            continue
        rows, current_revision = _fetch_rows(
            dataset,
            config,
            split,
            block_start,
            len(block_indices),
            max_attempts=max_attempts,
        )
        revision = revision or current_revision
        indexed_rows.extend(zip(block_indices, rows))
        if request_sleep > 0:
            time.sleep(request_sleep)
        block_start = index
        block_indices = [index]
    rows, current_revision = _fetch_rows(
        dataset,
        config,
        split,
        block_start,
        len(block_indices),
        max_attempts=max_attempts,
    )
    revision = revision or current_revision
    indexed_rows.extend(zip(block_indices, rows))
    return indexed_rows, revision


def _format_options(options: list[str]) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "\n".join(f"{letters[idx]}. {option}" for idx, option in enumerate(options))


def _build_competition_math_record(index: int, split: str, row: dict, revision: str | None) -> dict:
    gold = canonicalize_numeric_token(str(row["extracted_solution"]))
    if gold is None:
        raise ValueError(f"Could not normalize competition_math numeric answer: {row['extracted_solution']!r}")
    return {
        "example_id": f"competition-math-numeric-{split}-{index:05d}",
        "dataset": "competition_math_numeric",
        "problem": str(row["problem"]),
        "gold_answer": gold,
        "answer_mode": "numeric",
        "choices": [],
        "metadata": {
            "level": str(row.get("level", "")),
            "type": str(row.get("type", "")),
            "split": split,
            "source_index": index,
            "dataset_revision": revision,
        },
    }


def _build_mmlu_pro_record(index: int, split: str, row: dict, revision: str | None) -> dict:
    options = [str(option).strip() for option in row["options"]]
    problem = f"{row['question']}\n\nOptions:\n{_format_options(options)}"
    answer = str(row.get("answer") or "")
    if not answer and row.get("answer_index") is not None:
        answer = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[int(row["answer_index"])]
    return {
        "example_id": f"mmlu-pro-{split}-{int(row['question_id']):05d}",
        "dataset": "mmlu_pro",
        "problem": problem,
        "gold_answer": answer,
        "answer_mode": "choice_letter",
        "choices": options,
        "metadata": {
            "question_id": int(row["question_id"]),
            "category": str(row.get("category", "")),
            "src": str(row.get("src", "")),
            "split": split,
            "source_index": index,
            "dataset_revision": revision,
        },
    }


def _build_gpqa_record(index: int, split: str, row: dict, seed: int, revision: str | None) -> dict:
    raw_options = [
        str(row["Correct Answer"]).strip(),
        str(row["Incorrect Answer 1"]).strip(),
        str(row["Incorrect Answer 2"]).strip(),
        str(row["Incorrect Answer 3"]).strip(),
    ]
    rng = random.Random(f"{seed}:{split}:{row['Record ID']}")
    permuted = list(raw_options)
    rng.shuffle(permuted)
    gold_answer = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[permuted.index(str(row["Correct Answer"]).strip())]
    problem = f"{row['Question']}\n\nOptions:\n{_format_options(permuted)}"
    return {
        "example_id": f"gpqa-diamond-{split}-{row['Record ID']}",
        "dataset": "gpqa_diamond",
        "problem": problem,
        "gold_answer": gold_answer,
        "answer_mode": "choice_letter",
        "choices": permuted,
        "metadata": {
            "record_id": str(row["Record ID"]),
            "high_level_domain": str(row.get("High-level domain", "")),
            "subdomain": str(row.get("Subdomain", "")),
            "split": split,
            "source_index": index,
            "dataset_revision": revision,
        },
    }


def _build_record(benchmark: str, index: int, split: str, row: dict, seed: int, revision: str | None) -> dict:
    if benchmark == "competition_math_numeric":
        return _build_competition_math_record(index, split, row, revision)
    if benchmark == "mmlu_pro":
        return _build_mmlu_pro_record(index, split, row, revision)
    if benchmark == "gpqa_diamond":
        return _build_gpqa_record(index, split, row, seed, revision)
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def main() -> None:
    args = parse_args()
    spec = DATASET_SPECS[args.benchmark]
    split = args.split or spec["default_split"]
    try:
        source_rows, revision = _load_rows_from_repo_source(spec, split)
        indices = _select_indices(
            total=len(source_rows),
            start_offset=args.start_offset,
            max_examples=args.max_examples,
            seed=args.seed,
            shuffle=args.shuffle,
        )
        indexed_rows = [(index, source_rows[index]) for index in indices]
    except Exception:
        total_examples = _fetch_num_examples(
            spec["dataset"],
            spec["config"],
            split,
            max_attempts=args.max_attempts,
        )
        indices = _select_indices(
            total=total_examples,
            start_offset=args.start_offset,
            max_examples=args.max_examples,
            seed=args.seed,
            shuffle=args.shuffle,
        )
        indexed_rows, revision = _fetch_selected_rows(
            spec["dataset"],
            spec["config"],
            split,
            indices,
            max_attempts=args.max_attempts,
            request_sleep=args.request_sleep,
        )

    records = [
        _build_record(args.benchmark, index, split, row, args.seed, revision)
        for index, row in indexed_rows
    ]
    write_jsonl(args.output, records)


if __name__ == "__main__":
    main()
