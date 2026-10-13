#!/usr/bin/env python3
"""
Validates every data file in this kit before it reaches a training pipeline.
Run: python3 validate_dataset.py
"""
import json
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _search_dirs():
    dirs = []
    try:
        from zivawhite.config import get_config
        dirs.append(str(get_config().datasets_dir / "_eval"))
    except Exception:  # noqa: BLE001
        pass
    return dirs + [DATA_DIR]

SCHEMAS = {
    "feedback_log.jsonl": {
        "required": {"interaction_id", "timestamp", "category", "input",
                      "model_output", "feedback_type"},
        "valid_feedback_types": {"accept", "reject", "edit", "implicit_positive", "implicit_negative"},
    },
    "eval_benchmark.jsonl": {
        "required": {"eval_id", "category", "input", "reference_output", "acceptance_criteria"},
    },
    "replay_buffer_sample.jsonl": {
        "required": {"buffer_id", "source", "original_id", "category", "messages", "weight"},
    },
    "drift_monitor_log.jsonl": {
        "required": {"batch_id", "timestamp", "n_examples", "category_distribution"},
    },
    "version_registry.jsonl": {
        "required": {"version_id", "batch_id", "timestamp", "promoted", "decision_basis"},
    },
}


def validate_file(filename, spec):
    path = next((os.path.join(d, filename) for d in _search_dirs() if os.path.exists(os.path.join(d, filename))), None)
    if path is None:
        return None, [f"{filename}: file not found"]

    errors = []
    seen_ids = set()
    id_field = next((f for f in spec["required"] if f.endswith("_id")), None)
    count = 0

    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            count += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"{filename}:{lineno} invalid JSON ({e})")
                continue

            missing = spec["required"] - obj.keys()
            if missing:
                errors.append(f"{filename}:{lineno} missing keys {missing}")

            if id_field and id_field in obj:
                if obj[id_field] in seen_ids:
                    errors.append(f"{filename}:{lineno} duplicate {id_field} '{obj[id_field]}'")
                seen_ids.add(obj[id_field])

            if "valid_feedback_types" in spec and obj.get("feedback_type") not in spec["valid_feedback_types"]:
                errors.append(f"{filename}:{lineno} invalid feedback_type '{obj.get('feedback_type')}'")

            if filename == "feedback_log.jsonl" and obj.get("feedback_type") == "edit" and not obj.get("corrected_output"):
                errors.append(f"{filename}:{lineno} feedback_type 'edit' requires non-empty corrected_output")

    return count, errors


def main():
    total_errors = []
    for filename, spec in SCHEMAS.items():
        count, errors = validate_file(filename, spec)
        if count is None:
            print(f"{filename:32s} SKIPPED (not present)")
            continue
        status = "OK" if not errors else f"{len(errors)} ISSUE(S)"
        print(f"{filename:32s} {count:4d} records  [{status}]")
        total_errors.extend(errors)

    if total_errors:
        print(f"\n{len(total_errors)} error(s):")
        for e in total_errors:
            print(f"  - {e}")
        sys.exit(1)
    print("\nAll present files passed validation.")


if __name__ == "__main__":
    main()
