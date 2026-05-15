from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any


def read_corpus_ids(csv_path: Path) -> set[str]:
    ids: set[str] = set()
    if not csv_path.exists():
        return ids
    for line in csv_path.read_text(encoding='utf-8').splitlines():
        row = line.strip()
        if not row:
            continue
        ids.add(row.split('|', 1)[0])
    return ids


def first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def collect_anchor_entries(node: Any) -> list[dict[str, Any]]:
    if isinstance(node, list):
        entries: list[dict[str, Any]] = []
        for item in node:
            entries.extend(collect_anchor_entries(item))
        return entries

    if not isinstance(node, dict):
        return []

    if any(key in node for key in ('sample_id', 'id', 'sequence_id')) and any(
        key in node for key in ('frame_idx', 'frame', 'time_idx')
    ):
        return [node]

    entries: list[dict[str, Any]] = []
    for key in ('eligible_anchors', 'anchors', 'eligible'):
        if key in node:
            entries.extend(collect_anchor_entries(node[key]))
    return entries


def load_lookup(path: Path) -> dict[str, int]:
    if path.suffix == '.pkl':
        with path.open('rb') as handle:
            loaded = pickle.load(handle)
        return {str(key): int(value) for key, value in loaded.items()}

    data = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(data, dict) and 'vocab' in data:
        data = data['vocab']
    if not isinstance(data, dict):
        raise SystemExit(f'unsupported lookup format: {path}')
    return {str(key): int(value) for key, value in data.items()}


def validate_subset_artifact(
    artifact: dict[str, Any],
    word_to_id: dict[str, int],
    train_csv: Path,
    valid_csv: Path,
    test_csv: Path,
) -> dict[str, Any]:
    train_ids = read_corpus_ids(train_csv)
    valid_ids = read_corpus_ids(valid_csv)
    test_ids = read_corpus_ids(test_csv)
    forbidden_ids = valid_ids | test_ids
    vocab_size = len(word_to_id)

    loaded_anchor_map: dict[str, list[tuple[int, int]]] = {}
    skipped_unknown_tokens = 0
    skipped_invalid = 0
    skipped_non_train = 0

    raw_entries = collect_anchor_entries(artifact)
    top_level_eligible_true_count = sum(1 for entry in raw_entries if entry.get('eligible', True) is not False)

    for entry in raw_entries:
        if entry.get('eligible') is False:
            continue

        sample_id = first_present(entry, ('sample_id', 'id', 'sequence_id'))
        frame_idx = first_present(entry, ('frame_idx', 'frame', 'time_idx'))
        token_id = entry.get('token_id')

        if token_id is None:
            token_text = first_present(entry, ('token', 'token_text'))
            if token_text not in word_to_id:
                skipped_unknown_tokens += 1
                continue
            token_id = word_to_id[str(token_text)]

        if sample_id is None or frame_idx is None:
            skipped_invalid += 1
            continue

        try:
            sample_id = str(sample_id)
            frame_idx = int(frame_idx)
            token_id = int(token_id)
        except (TypeError, ValueError):
            skipped_invalid += 1
            continue

        if token_id < 0 or token_id >= vocab_size or frame_idx < 0:
            skipped_invalid += 1
            continue

        if sample_id in forbidden_ids:
            raise SystemExit(f'artifact contains validation/test sample id: {sample_id}')
        if train_ids and sample_id not in train_ids:
            skipped_non_train += 1
            continue

        loaded_anchor_map.setdefault(sample_id, []).append((frame_idx, token_id))

    for sample_anchors in loaded_anchor_map.values():
        sample_anchors.sort(key=lambda item: item[0])

    loaded_anchor_count = sum(len(sample_anchors) for sample_anchors in loaded_anchor_map.values())
    unique_token_ids = sorted({token_id for sample_anchors in loaded_anchor_map.values() for _, token_id in sample_anchors})
    min_anchor_frame = min((frame for sample_anchors in loaded_anchor_map.values() for frame, _ in sample_anchors), default=None)
    max_anchor_frame = max((frame for sample_anchors in loaded_anchor_map.values() for frame, _ in sample_anchors), default=None)

    artifact_overview = artifact.get('artifact_overview', {}) if isinstance(artifact, dict) else {}
    sample_anchor_examples = {
        sample_id: len(sample_anchors)
        for sample_id, sample_anchors in sorted(loaded_anchor_map.items())[:10]
    }
    sample_anchor_map = {
        sample_id: sample_anchors
        for sample_id, sample_anchors in sorted(loaded_anchor_map.items())[:10]
    }

    return {
        'raw_entry_count': len(raw_entries),
        'top_level_eligible_true_count': top_level_eligible_true_count,
        'loaded_anchor_count': loaded_anchor_count,
        'unique_sample_count': len(loaded_anchor_map),
        'unique_token_count': len(unique_token_ids),
        'loaded_counts_match_artifact': {
            'eligible_anchor_count': loaded_anchor_count == artifact_overview.get('eligible_anchor_count'),
            'unique_anchor_sample_count': len(loaded_anchor_map) == artifact_overview.get('unique_anchor_sample_count'),
            'unique_anchor_token_count': len(unique_token_ids) == artifact_overview.get('unique_anchor_token_count'),
        },
        'skipped_unknown_tokens': skipped_unknown_tokens,
        'skipped_invalid': skipped_invalid,
        'skipped_non_train': skipped_non_train,
        'min_anchor_frame': min_anchor_frame,
        'max_anchor_frame': max_anchor_frame,
        'min_anchors_in_one_sample': min((len(v) for v in loaded_anchor_map.values()), default=0),
        'max_anchors_in_one_sample': max((len(v) for v in loaded_anchor_map.values()), default=0),
        'sample_anchor_examples': sample_anchor_examples,
        'sample_anchor_map': sample_anchor_map,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Dry-run validate a dense pseudo-align subset artifact against train.py anchor loading expectations.'
    )
    parser.add_argument('--artifact_json', required=True)
    parser.add_argument('--lookup_json', required=True)
    parser.add_argument('--train_corpus_csv', required=True)
    parser.add_argument('--valid_corpus_csv', required=True)
    parser.add_argument('--test_corpus_csv', required=True)
    args = parser.parse_args()

    artifact_path = Path(args.artifact_json)
    lookup_path = Path(args.lookup_json)
    train_csv = Path(args.train_corpus_csv)
    valid_csv = Path(args.valid_corpus_csv)
    test_csv = Path(args.test_corpus_csv)

    if not artifact_path.exists():
        raise SystemExit(f'missing artifact json: {artifact_path}')
    if not lookup_path.exists():
        raise SystemExit(f'missing lookup json: {lookup_path}')
    if not train_csv.exists():
        raise SystemExit(f'missing train corpus csv: {train_csv}')

    artifact = json.loads(artifact_path.read_text(encoding='utf-8'))
    word_to_id = load_lookup(lookup_path)
    validation = validate_subset_artifact(
        artifact=artifact,
        word_to_id=word_to_id,
        train_csv=train_csv,
        valid_csv=valid_csv,
        test_csv=test_csv,
    )

    payload = {
        'ok': True,
        'artifact_json': str(artifact_path),
        **validation,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
