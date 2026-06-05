from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import _pickle as pickle


DEV_IDS = ['600', '1174', '706', '1224', '199', '1030', '1319', '1341']


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


def read_lookup(lookup_path: Path) -> dict[str, int]:
    with lookup_path.open('rb') as handle:
        return pickle.load(handle)


def list_feature_frames(feature_dir: Path) -> list[str]:
    if not feature_dir.exists():
        return []
    return sorted(
        file_name.name
        for file_name in feature_dir.iterdir()
        if file_name.is_file() and file_name.suffix.lower() in {'.png', '.jpg', '.jpeg'}
    )


def list_segmentation_frames(seg_dir: Path) -> list[str]:
    if not seg_dir.exists():
        return []
    names = []
    for file_name in seg_dir.iterdir():
        if not file_name.is_file() or not file_name.name.endswith('.npy.gz'):
            continue
        names.append(file_name.name[:-7] + '.png')
    return sorted(names)


def split_dirs(prepared_root: Path, split: str, sample_id: str) -> tuple[Path, Path]:
    feature_root = prepared_root / 'features' / 'fullFrame-210x260px' / split / sample_id / '1'
    if split == 'train':
        seg_split = 'train_segmentation'
    elif split == 'dev':
        seg_split = 'val_segmentation'
    else:
        seg_split = 'test_segmentation'
    seg_root = prepared_root / 'segmentation' / seg_split / sample_id
    return feature_root, seg_root


def inspect_clip(prepared_root: Path, lookup: dict[str, int], split_rows: dict[str, dict[str, dict[str, Any]]], sample_id: str) -> dict[str, Any]:
    split_targets = {
        split: split_rows.get(split, {}).get(sample_id, {}).get('target')
        for split in ('train', 'dev', 'test')
    }
    canonical_target = split_targets['dev'] or split_targets['train'] or split_targets['test'] or ''
    tokens = canonical_target.split()
    missing_lookup_tokens = [token for token in tokens if token not in lookup]

    split_checks = {}
    frame_name_consistency = True
    for split in ('train', 'dev', 'test'):
        feature_dir, seg_dir = split_dirs(prepared_root, split, sample_id)
        feature_frames = list_feature_frames(feature_dir)
        seg_frames = list_segmentation_frames(seg_dir)
        feature_set = set(feature_frames)
        seg_set = set(seg_frames)
        matches = feature_frames == seg_frames
        frame_name_consistency = frame_name_consistency and matches
        split_checks[split] = {
            'target': split_targets[split],
            'feature_dir': str(feature_dir),
            'seg_dir': str(seg_dir),
            'feature_dir_exists': feature_dir.exists(),
            'seg_dir_exists': seg_dir.exists(),
            'feature_is_symlink': feature_dir.parent.is_symlink() if feature_dir.parent.exists() else False,
            'seg_is_symlink': seg_dir.is_symlink() if seg_dir.exists() else False,
            'feature_symlink_target': str(feature_dir.parent.resolve()) if feature_dir.parent.exists() else None,
            'seg_symlink_target': str(seg_dir.resolve()) if seg_dir.exists() else None,
            'feature_frame_count': len(feature_frames),
            'seg_frame_count': len(seg_frames),
            'missing_seg_frames': sorted(feature_set - seg_set)[:5],
            'extra_seg_frames': sorted(seg_set - feature_set)[:5],
            'frame_name_match': matches,
            'first_feature_frames': feature_frames[:5],
            'first_seg_frames': seg_frames[:5],
        }

    same_target = len({target for target in split_targets.values() if target}) <= 1
    feature_counts = {split_checks[split]['feature_frame_count'] for split in ('train', 'dev', 'test')}
    seg_counts = {split_checks[split]['seg_frame_count'] for split in ('train', 'dev', 'test')}

    ok = (
        canonical_target != ''
        and same_target
        and not missing_lookup_tokens
        and frame_name_consistency
        and all(split_checks[split]['feature_frame_count'] > 0 for split in ('train', 'dev', 'test'))
        and all(split_checks[split]['seg_frame_count'] > 0 for split in ('train', 'dev', 'test'))
        and len(feature_counts) == 1
        and len(seg_counts) == 1
    )

    return {
        'sample_id': sample_id,
        'target': canonical_target,
        'tokens': tokens,
        'same_target_across_splits': same_target,
        'missing_lookup_tokens': missing_lookup_tokens,
        'frame_name_consistency': frame_name_consistency,
        'split_checks': split_checks,
        'ok': ok,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Build a per-clip integrity manifest for the 8 Phase B dev oracle-overfit clips.')
    parser.add_argument('--prepared_root', required=True)
    parser.add_argument('--output_json', required=True)
    args = parser.parse_args()

    prepared_root = Path(args.prepared_root)
    lookup = read_lookup(prepared_root / 'lookup' / 'json_lookup.pkl')
    split_rows = {
        split: {
            row['sample_id']: row
            for row in read_corpus_rows(prepared_root / 'annotations' / 'manual' / f'{split}.corpus.csv')
        }
        for split in ('train', 'dev', 'test')
    }

    rows = [inspect_clip(prepared_root, lookup, split_rows, sample_id) for sample_id in DEV_IDS]
    ok_rows = [row['sample_id'] for row in rows if row['ok']]
    bad_rows = [row['sample_id'] for row in rows if not row['ok']]

    payload = {
        'ok': len(bad_rows) == 0,
        'prepared_root': str(prepared_root),
        'sample_count': len(rows),
        'ok_rows': ok_rows,
        'bad_rows': bad_rows,
        'rows': rows,
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
