from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEV_IDS = ['600', '1174', '706', '1224', '199', '1030', '1319', '1341']
HARD_IDS = ['600', '1174', '1224', '1319', '1341']
EASY_IDS = ['706', '199', '1030']


def read_corpus_rows(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open('r', encoding='utf-8') as handle:
        reader = csv.reader(handle, delimiter='|')
        for row in reader:
            if not row:
                continue
            sample_id = str(row[0])
            target = str(row[1]) if len(row) > 1 else ''
            rows.append({'sample_id': sample_id, 'target': target, 'tokens': target.split(), 'raw_row': row})
    return rows


def write_corpus_rows(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle, delimiter='|')
        for row in rows:
            writer.writerow(row['raw_row'])


def reset_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        for child in path.iterdir():
            reset_path(child)
        path.rmdir()


def ensure_symlink(target: Path, link_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        reset_path(link_path)
    link_path.symlink_to(target)


def weighted_train_rows(dev_by_id: dict[str, dict[str, Any]], hard_weight: int, easy_weight: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_id in HARD_IDS:
        rows.extend([dev_by_id[sample_id]] * hard_weight)
    for sample_id in EASY_IDS:
        rows.extend([dev_by_id[sample_id]] * easy_weight)
    return rows


def build_surface(prepared_root: Path, output_root: Path, hard_weight: int, easy_weight: int) -> dict[str, Any]:
    dev_rows = read_corpus_rows(prepared_root / 'annotations' / 'manual' / 'dev.corpus.csv')
    dev_by_id = {row['sample_id']: row for row in dev_rows}

    missing_ids = [sample_id for sample_id in DEV_IDS if sample_id not in dev_by_id]
    if missing_ids:
        raise SystemExit(f'missing dev ids in prepared root: {missing_ids}')

    selected_rows = [dev_by_id[sample_id] for sample_id in DEV_IDS]
    train_rows = weighted_train_rows(dev_by_id, hard_weight, easy_weight)

    if output_root.exists():
        reset_path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    annotations_root = output_root / 'annotations' / 'manual'
    write_corpus_rows(train_rows, annotations_root / 'train.corpus.csv')
    write_corpus_rows(selected_rows, annotations_root / 'dev.corpus.csv')
    write_corpus_rows(selected_rows, annotations_root / 'test.corpus.csv')

    ensure_symlink(prepared_root / 'lookup', output_root / 'lookup')

    source_feature_root = prepared_root / 'features' / 'fullFrame-210x260px' / 'dev'
    source_seg_root = prepared_root / 'segmentation' / 'val_segmentation'

    for split in ['train', 'dev', 'test']:
        split_feature_root = output_root / 'features' / 'fullFrame-210x260px' / split
        split_seg_root = output_root / 'segmentation' / (
            'train_segmentation' if split == 'train' else 'val_segmentation' if split == 'dev' else 'test_segmentation'
        )
        split_feature_root.mkdir(parents=True, exist_ok=True)
        split_seg_root.mkdir(parents=True, exist_ok=True)
        for sample_id in DEV_IDS:
            ensure_symlink(source_feature_root / sample_id, split_feature_root / sample_id)
            ensure_symlink(source_seg_root / sample_id, split_seg_root / sample_id)

    train_counts = Counter(row['sample_id'] for row in train_rows)
    payload = {
        'ok': True,
        'name': 'normalized_quarantine_gate_phase_b_v1_dev8_replay',
        'source_prepared_root': str(prepared_root),
        'output_root': str(output_root),
        'selected_sample_ids': DEV_IDS,
        'selected_targets': [row['target'] for row in selected_rows],
        'hard_ids': HARD_IDS,
        'easy_ids': EASY_IDS,
        'hard_weight': hard_weight,
        'easy_weight': easy_weight,
        'train_replay_counts': {sample_id: train_counts[sample_id] for sample_id in DEV_IDS},
        'counts': {
            'train': len(train_rows),
            'dev': len(selected_rows),
            'test': len(selected_rows),
        },
        'feature_source_split': 'dev',
        'segmentation_source_split': 'val_segmentation',
        'lookup_reused': True,
    }
    (output_root / 'surface_manifest.json').write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Build a weighted replay prepared view over the exact 8 Phase B dev clips, reusing the same feature/segmentation roots while changing train replay counts.'
    )
    parser.add_argument('--prepared_root', required=True)
    parser.add_argument('--output_root', required=True)
    parser.add_argument('--hard_weight', type=int, default=4)
    parser.add_argument('--easy_weight', type=int, default=2)
    args = parser.parse_args()

    if args.hard_weight < 1 or args.easy_weight < 1:
        raise SystemExit('hard_weight and easy_weight must both be >= 1')

    prepared_root = Path(args.prepared_root)
    output_root = Path(args.output_root)
    if not prepared_root.exists():
        raise SystemExit(f'missing prepared root: {prepared_root}')

    payload = build_surface(prepared_root, output_root, args.hard_weight, args.easy_weight)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
