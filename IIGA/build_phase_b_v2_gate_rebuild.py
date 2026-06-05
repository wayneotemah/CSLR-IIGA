from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


INVALID_DEV8_IDS = ['600', '1174', '706', '1224', '199', '1030', '1319', '1341']

REPLACEMENT_DEV_IDS = ['1251', '1065', '1346', '936', '1209', '1387', '1331', '104']

KEEP_TEST_IDS = ['3157', '1121', '1219', '84', '1356', '1001', '1284', '1206']


def read_corpus_rows(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open('r', encoding='utf-8') as handle:
        reader = csv.reader(handle, delimiter='|')
        for row in reader:
            if not row:
                continue
            sample_id = str(row[0])
            target = str(row[1]) if len(row) > 1 else ''
            rows.append({
                'sample_id': sample_id,
                'target': target,
                'tokens': target.split(),
                'raw_row': row,
            })
    return rows


def write_corpus_rows(rows: list[dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle, delimiter='|')
        for row in rows:
            writer.writerow(row['raw_row'])


def reset_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def ensure_symlink(target: Path, link_path: Path) -> None:
    reset_path(link_path)
    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(target)


def build_surface(prepared_root: Path, output_root: Path) -> dict[str, Any]:
    prepared_root = prepared_root.resolve()
    output_root = output_root.resolve()

    broader_root = prepared_root.parent / 'normalized_quarantine_gate_v1'

    train_rows = read_corpus_rows(prepared_root / 'annotations' / 'manual' / 'train.corpus.csv')
    dev_rows = read_corpus_rows(broader_root / 'annotations' / 'manual' / 'dev.corpus.csv')
    test_rows = read_corpus_rows(broader_root / 'annotations' / 'manual' / 'test.corpus.csv')

    dev_by_id = {row['sample_id']: row for row in dev_rows}
    test_by_id = {row['sample_id']: row for row in test_rows}

    rebuilt_dev_rows = [dev_by_id[sample_id] for sample_id in REPLACEMENT_DEV_IDS]
    rebuilt_test_rows = [test_by_id[sample_id] for sample_id in KEEP_TEST_IDS]

    annotations_root = output_root / 'annotations' / 'manual'
    write_corpus_rows(train_rows, annotations_root / 'train.corpus.csv')
    write_corpus_rows(rebuilt_dev_rows, annotations_root / 'dev.corpus.csv')
    write_corpus_rows(rebuilt_test_rows, annotations_root / 'test.corpus.csv')

    ensure_symlink(prepared_root / 'lookup', output_root / 'lookup')

    feature_source_roots = {
        'train': prepared_root / 'features' / 'fullFrame-210x260px' / 'train',
        'dev': broader_root / 'features' / 'fullFrame-210x260px' / 'dev',
        'test': broader_root / 'features' / 'fullFrame-210x260px' / 'test',
    }
    seg_source_roots = {
        'train': prepared_root / 'segmentation' / 'train_segmentation',
        'dev': broader_root / 'segmentation' / 'val_segmentation',
        'test': broader_root / 'segmentation' / 'test_segmentation',
    }

    for split in ('train', 'dev', 'test'):
        reset_path(output_root / 'features' / 'fullFrame-210x260px' / split)
        reset_path(output_root / 'segmentation' / {'train': 'train_segmentation', 'dev': 'val_segmentation', 'test': 'test_segmentation'}[split])

    rebuilt_rows_by_split = {
        'train': train_rows,
        'dev': rebuilt_dev_rows,
        'test': rebuilt_test_rows,
    }

    for split, rows in rebuilt_rows_by_split.items():
        feature_root = output_root / 'features' / 'fullFrame-210x260px' / split
        seg_root = output_root / 'segmentation' / {'train': 'train_segmentation', 'dev': 'val_segmentation', 'test': 'test_segmentation'}[split]
        feature_root.mkdir(parents=True, exist_ok=True)
        seg_root.mkdir(parents=True, exist_ok=True)
        for row in rows:
            sample_id = row['sample_id']
            if split == 'train':
                feature_target = feature_source_roots['train'] / sample_id
                seg_target = seg_source_roots['train'] / sample_id
            elif split == 'dev':
                feature_target = feature_source_roots['dev'] / sample_id
                seg_target = seg_source_roots['dev'] / sample_id
            else:
                feature_target = feature_source_roots['test'] / sample_id
                seg_target = seg_source_roots['test'] / sample_id
            ensure_symlink(feature_target, feature_root / sample_id)
            ensure_symlink(seg_target, seg_root / sample_id)

    manifest = {
        'ok': True,
        'name': 'normalized_quarantine_gate_phase_b_v2_rebuild',
        'source_prepared_root': str(prepared_root),
        'broader_prepared_root': str(broader_root),
        'output_root': str(output_root),
        'invalid_dev8_ids': INVALID_DEV8_IDS,
        'replacement_dev_ids': REPLACEMENT_DEV_IDS,
        'preserved_test_ids': KEEP_TEST_IDS,
        'replacement_dev_targets': [row['target'] for row in rebuilt_dev_rows],
        'preserved_test_targets': [row['target'] for row in rebuilt_test_rows],
        'counts': {
            'train': len(train_rows),
            'dev': len(rebuilt_dev_rows),
            'test': len(rebuilt_test_rows),
        },
        'lookup_reused': True,
        'feature_source_roots': {split: str(path) for split, path in feature_source_roots.items()},
        'segmentation_source_roots': {split: str(path) for split, path in seg_source_roots.items()},
    }

    manifest_path = output_root / 'surface_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8')
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description='Build a rebuilt Phase B v2 gate that keeps the full train split but replaces the invalid dev8 gate with better-supported rows.')
    parser.add_argument('--prepared_root', required=True)
    parser.add_argument('--output_root', required=True)
    args = parser.parse_args()

    manifest = build_surface(Path(args.prepared_root), Path(args.output_root))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
