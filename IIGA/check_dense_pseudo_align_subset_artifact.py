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

    with csv_path.open('r', encoding='utf-8') as handle:
        for line in handle:
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


def validate_subset_artifact(
    artifact: dict[str, Any],
    word_to_id: dict[str, int],
    vocab_size: int,
    train_csv: Path,
    valid_csv: Path,
    test_csv: Path,
) -> dict[str, Any]:
    train_ids = read_corpus_ids(train_csv)
    valid_ids = read_corpus_ids(valid_csv)
    test_ids = read_corpus_ids(test_csv)
    forbidden_ids = valid_ids | test_ids

    anchor_map: dict[str, list[tuple[int, int]]] = {}
    skipped_unknown_tokens = 0
    skipped_invalid = 0
    skipped_non_train = 0

    raw_entries = collect_anchor_entries(artifact)
    eligible_true_count = 0
    top_level_eligible_anchors = artifact.get('eligible_anchors', [])
    if isinstance(top_level_eligible_anchors, list):
        eligible_true_count = sum(1 for entry in top_level_eligible_anchors if isinstance(entry, dict) and entry.get('eligible') is not False)

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
            token_id = word_to_id[token_text]

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
            raise ValueError(f'Artifact contains validation/test sample id {sample_id}')
        if train_ids and sample_id not in train_ids:
            skipped_non_train += 1
            continue

        anchor_map.setdefault(sample_id, []).append((frame_idx, token_id))

    for sample_anchors in anchor_map.values():
        sample_anchors.sort(key=lambda item: item[0])

    total_anchors = sum(len(sample_anchors) for sample_anchors in anchor_map.values())
    unique_tokens = {token_id for sample_anchors in anchor_map.values() for _, token_id in sample_anchors}

    if not total_anchors:
        raise ValueError('No usable eligible anchors loaded from subset artifact')

    sample_anchor_counts = Counter({sample_id: len(sample_anchors) for sample_id, sample_anchors in anchor_map.items()})
    frame_indices = [frame_idx for sample_anchors in anchor_map.values() for frame_idx, _ in sample_anchors]

    expected_count = artifact.get('artifact_overview', {}).get('eligible_anchor_count')
    expected_unique_samples = artifact.get('artifact_overview', {}).get('unique_anchor_sample_count')
    expected_unique_tokens = artifact.get('artifact_overview', {}).get('unique_anchor_token_count')

    summary = {
        'raw_entry_count': len(raw_entries),
        'top_level_eligible_true_count': eligible_true_count,
        'loaded_anchor_count': total_anchors,
        'unique_sample_count': len(anchor_map),
        'unique_token_count': len(unique_tokens),
        'min_anchor_frame': min(frame_indices),
        'max_anchor_frame': max(frame_indices),
        'max_anchors_in_one_sample': max(sample_anchor_counts.values()),
        'min_anchors_in_one_sample': min(sample_anchor_counts.values()),
        'loaded_counts_match_artifact': {
            'eligible_anchor_count': expected_count == total_anchors,
            'unique_anchor_sample_count': expected_unique_samples == len(anchor_map),
            'unique_anchor_token_count': expected_unique_tokens == len(unique_tokens),
        },
        'skipped_unknown_tokens': skipped_unknown_tokens,
        'skipped_invalid': skipped_invalid,
        'skipped_non_train': skipped_non_train,
        'sample_anchor_examples': dict(sample_anchor_counts.most_common(10)),
        'sample_anchor_map': {
            sample_id: [{'frame_idx': frame_idx, 'token_id': token_id} for frame_idx, token_id in anchors]
            for sample_id, anchors in sorted(anchor_map.items())
        },
    }

    return summary


def load_lookup(path: Path) -> dict[str, int]:
    if path.suffix == '.pkl':
        with path.open('rb') as handle:
            vocab = pickle.load(handle)
    else:
        payload = json.loads(path.read_text(encoding='utf-8'))
        vocab = payload.get('vocab', payload)

    return {str(key): int(value) for key, value in vocab.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Dry-run validate a subset pseudo-alignment artifact against train.py anchor expectations.'
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
    if not valid_csv.exists():
        raise SystemExit(f'missing valid corpus csv: {valid_csv}')
    if not test_csv.exists():
        raise SystemExit(f'missing test corpus csv: {test_csv}')

    artifact = json.loads(artifact_path.read_text(encoding='utf-8'))
    word_to_id = load_lookup(lookup_path)

    summary = validate_subset_artifact(
        artifact=artifact,
        word_to_id=word_to_id,
        vocab_size=len(word_to_id),
        train_csv=train_csv,
        valid_csv=valid_csv,
        test_csv=test_csv,
    )

    payload = {
        'ok': True,
        'artifact_json': str(artifact_path),
        'lookup_json': str(lookup_path),
        'train_corpus_csv': str(train_csv),
        'valid_corpus_csv': str(valid_csv),
        'test_corpus_csv': str(test_csv),
        **summary,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
