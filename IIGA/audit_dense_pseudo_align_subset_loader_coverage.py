from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

from dataloader import loader


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def load_lookup(path: Path) -> dict[str, int]:
    with path.open('rb') as handle:
        loaded = pickle.load(handle)
    return {str(key): int(value) for key, value in loaded.items()}


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


def load_anchor_audit_local(
    anchor_audit_json: Path,
    word_to_id: dict[str, int],
    vocab_size: int,
    train_csv: Path,
    valid_csv: Path,
    test_csv: Path,
) -> dict[str, list[tuple[int, int]]]:
    audit = load_json(anchor_audit_json)

    for leakage_key in ('validation_ids_in_anchors', 'valid_ids_in_anchors', 'test_ids_in_anchors'):
        leaked_ids = audit.get(leakage_key, []) if isinstance(audit, dict) else []
        if leaked_ids:
            raise SystemExit(f'anchor audit reports leakage in {leakage_key}: {leaked_ids}')

    train_ids = read_corpus_ids(train_csv)
    valid_ids = read_corpus_ids(valid_csv)
    test_ids = read_corpus_ids(test_csv)
    forbidden_ids = valid_ids | test_ids

    anchor_map: dict[str, list[tuple[int, int]]] = {}
    skipped_unknown_tokens = 0
    skipped_invalid = 0
    skipped_non_train = 0

    for entry in collect_anchor_entries(audit):
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
            raise SystemExit(f'anchor audit contains validation/test sample id {sample_id}')
        if train_ids and sample_id not in train_ids:
            skipped_non_train += 1
            continue

        anchor_map.setdefault(sample_id, []).append((frame_idx, token_id))

    for sample_anchors in anchor_map.values():
        sample_anchors.sort(key=lambda item: item[0])

    total_anchors = sum(len(entries) for entries in anchor_map.values())
    unique_tokens = {token_id for entries in anchor_map.values() for _, token_id in entries}
    if not total_anchors:
        raise SystemExit(f'no usable eligible anchors loaded from {anchor_audit_json}')

    print(
        f'Loaded {total_anchors} eligible anchors for {len(unique_tokens)} unique tokens across {len(anchor_map)} samples'
    )
    if skipped_unknown_tokens or skipped_invalid or skipped_non_train:
        print(
            'Skipped anchors - unknown tokens: %d, invalid: %d, non-train: %d'
            % (skipped_unknown_tokens, skipped_invalid, skipped_non_train)
        )

    return anchor_map


def audit_loader_coverage(
    artifact_json: Path,
    lookup_pkl: Path,
    train_csv: Path,
    valid_csv: Path,
    test_csv: Path,
    train_root: Path,
    train_segment_root: Path,
    sample_probe_limit: int,
) -> dict[str, Any]:
    word_to_id = load_lookup(lookup_pkl)
    vocab_size = len(word_to_id)
    anchor_map = load_anchor_audit_local(
        artifact_json,
        word_to_id,
        vocab_size,
        train_csv,
        valid_csv,
        test_csv,
    )

    dataloader, _ = loader(
        csv_file=str(train_csv),
        root_dir=str(train_root),
        segment_path=str(train_segment_root),
        lookup=str(lookup_pkl),
        rescale=224,
        batch_size=1,
        num_workers=0,
        random_drop=0.0,
        uniform_drop=None,
        show_sample=False,
        istrain=False,
        fixed_padding=None,
        hand_dir=None,
        data_stats=None,
        hand_stats=None,
        channels=3,
        return_sample_ids=True,
    )

    target_sample_ids = sorted(anchor_map.keys())
    target_sample_id_set = set(target_sample_ids)
    seen_target_sample_ids: set[str] = set()
    sample_probe_examples: list[dict[str, Any]] = []
    anchor_hits_by_sample: Counter[str] = Counter()
    in_bounds_anchor_hits_by_sample: Counter[str] = Counter()
    out_of_bounds_by_sample: Counter[str] = Counter()
    loader_batches_seen = 0
    loader_unique_sample_ids: set[str] = set()
    logged_step_hits: list[dict[str, Any]] = []
    logging_interval = 100

    for batch in dataloader:
        x, x_lengths, y, y_lengths, hand_regions, _, sample_ids = batch
        loader_batches_seen += 1
        step_index = loader_batches_seen - 1

        for batch_idx, sample_id in enumerate(sample_ids):
            sample_id_str = str(sample_id)
            loader_unique_sample_ids.add(sample_id_str)
            if sample_id_str not in anchor_map:
                continue

            seen_target_sample_ids.add(sample_id_str)
            raw_length = int(x_lengths[batch_idx]) if not hasattr(x_lengths[batch_idx], 'item') else int(x_lengths[batch_idx].item())
            anchors = anchor_map[sample_id_str]
            valid_anchor_count = 0
            invalid_anchor_count = 0
            for frame_idx, token_id in anchors:
                anchor_hits_by_sample[sample_id_str] += 1
                if frame_idx < raw_length:
                    valid_anchor_count += 1
                    in_bounds_anchor_hits_by_sample[sample_id_str] += 1
                else:
                    invalid_anchor_count += 1
                    out_of_bounds_by_sample[sample_id_str] += 1

            if len(sample_probe_examples) < sample_probe_limit:
                sample_probe_examples.append(
                    {
                        'sample_id': sample_id_str,
                        'step_index': step_index,
                        'raw_length': raw_length,
                        'anchors': anchors,
                        'in_bounds_anchor_count': valid_anchor_count,
                        'out_of_bounds_anchor_count': invalid_anchor_count,
                    }
                )

            if step_index % logging_interval == 0:
                logged_step_hits.append(
                    {
                        'step_index': step_index,
                        'sample_id': sample_id_str,
                        'raw_length': raw_length,
                        'anchor_count': len(anchors),
                        'in_bounds_anchor_count': valid_anchor_count,
                    }
                )

        if seen_target_sample_ids == target_sample_id_set:
            break

    loaded_anchor_count = sum(len(entries) for entries in anchor_map.values())
    unique_token_ids = sorted({token_id for entries in anchor_map.values() for _, token_id in entries})
    min_anchors = min((len(entries) for entries in anchor_map.values()), default=0)
    max_anchors = max((len(entries) for entries in anchor_map.values()), default=0)
    total_in_bounds = sum(in_bounds_anchor_hits_by_sample.values())
    total_out_of_bounds = sum(out_of_bounds_by_sample.values())

    return {
        'loaded_anchor_count': loaded_anchor_count,
        'unique_sample_count': len(anchor_map),
        'unique_token_count': len(unique_token_ids),
        'loaded_counts_match_artifact': {
            'loaded_anchor_count': loaded_anchor_count == 18,
            'unique_sample_count': len(anchor_map) == 6,
            'unique_token_count': len(unique_token_ids) == 10,
        },
        'min_anchors_in_one_sample': min_anchors,
        'max_anchors_in_one_sample': max_anchors,
        'loader_batches_seen': loader_batches_seen,
        'loader_unique_sample_ids': len(loader_unique_sample_ids),
        'loader_target_sample_ids': target_sample_ids,
        'loader_seen_target_sample_ids': sorted(seen_target_sample_ids),
        'loader_saw_all_target_sample_ids': seen_target_sample_ids == target_sample_id_set,
        'sample_probe_examples': sample_probe_examples,
        'logging_interval': logging_interval,
        'logged_step_hits': logged_step_hits,
        'logged_step_hit_count': len(logged_step_hits),
        'total_anchor_hits_seen': sum(anchor_hits_by_sample.values()),
        'total_in_bounds_anchor_hits_seen': total_in_bounds,
        'total_out_of_bounds_anchor_hits_seen': total_out_of_bounds,
        'samples_with_any_out_of_bounds': sorted(sample_id for sample_id, count in out_of_bounds_by_sample.items() if count > 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Measure whether stable subset anchor-bearing samples naturally hit the train.py logging windows and whether their anchors stay in-bounds in a shuffled train loader.'
    )
    parser.add_argument('--artifact_json', required=True)
    parser.add_argument('--lookup_pkl', required=True)
    parser.add_argument('--train_corpus_csv', required=True)
    parser.add_argument('--valid_corpus_csv', required=True)
    parser.add_argument('--test_corpus_csv', required=True)
    parser.add_argument('--train_root', required=True)
    parser.add_argument('--train_segment_root', required=True)
    parser.add_argument('--sample_probe_limit', type=int, default=10)
    args = parser.parse_args()

    payload = audit_loader_coverage(
        artifact_json=Path(args.artifact_json),
        lookup_pkl=Path(args.lookup_pkl),
        train_csv=Path(args.train_corpus_csv),
        valid_csv=Path(args.valid_corpus_csv),
        test_csv=Path(args.test_corpus_csv),
        train_root=Path(args.train_root),
        train_segment_root=Path(args.train_segment_root),
        sample_probe_limit=args.sample_probe_limit,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
