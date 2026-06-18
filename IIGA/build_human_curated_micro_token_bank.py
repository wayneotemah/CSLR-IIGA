from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_PREPARED_ROOT = Path(
    '/home/cloudsurfer/projects/ishara-ai/Sign2Text/local_runs/prepared/normalized_quarantine_gate_v1'
)
DEFAULT_OUTPUT_ROOT = Path(
    '/home/cloudsurfer/projects/ishara-ai/Sign2Text/local_runs/prepared/normalized_quarantine_gate_v1_human_curated_micro_token_bank_v1'
)
DEFAULT_QUARANTINE_JSON = Path(
    '/home/cloudsurfer/projects/ishara-ai/odocs/research/label_gloss_audit_2026-05-07/quarantine_candidates.json'
)

SELECTED_TOKENS = [
    'hear',
    'talk',
    'circumcision',
    'continue',
    'must',
    'habit',
    'money',
    'remember',
    'there',
    'inside',
    'word',
    'learn',
]

ISOLATED_TRAIN_ROWS = [
    ('891', 'circumcision'),
    ('1095', 'continue'),
    ('1124', 'continue'),
    ('1086', 'continue'),
    ('1153', 'habit'),
    ('749', 'hear'),
    ('1305', 'inside'),
    ('585', 'learn'),
    ('1003', 'money'),
    ('1223', 'must'),
    ('1102', 'remember'),
    ('701', 'remember'),
    ('151', 'talk'),
    ('751', 'talk'),
    ('1247', 'there'),
    ('146', 'word'),
]

SHORT_TRAIN_ROWS = [
    ('1255', 'talk must there'),
    ('1235', 'hear remember talk'),
    ('1109', 'continue bad habit'),
    ('1399', 'hear that inside'),
    ('1234', 'word perfect'),
    ('832', 'visitors circumcision'),
    ('309', 'spend money'),
    ('586', 'learn hard try'),
    ('1227', 'must fall continue'),
    ('1333', 'university word'),
    ('176', 'there challenges'),
    ('192', 'talking remember'),
    ('762', 'sale money'),
    ('1126', 'odd habit'),
    ('1368', 'inside festival'),
    ('835', 'hospital circumcision'),
    ('1258', 'here fall learn'),
]

DEV_ROWS = [
    ('113', 'continue forget'),
    ('765', 'farm there'),
    ('830', 'hear parents'),
    ('878', 'inside cold'),
    ('965', 'continue time finish'),
    ('1346', 'hear your know'),
    ('1342', 'learn fell please'),
    ('1139', 'same odd habit'),
    ('1273', 'talk about day'),
    ('874', 'time circumcision ceremony'),
    ('1331', 'time talk why'),
]

TEST_ROWS = [
    ('834', 'circumcision us'),
    ('310', 'money spend'),
    ('1118', 'remember about'),
    ('96', 'word question'),
    ('876', 'no must circumcision'),
    ('1283', 'day same talk'),
    ('286', 'example hear corruption'),
]

EXCLUDED_TOKENS = {'love', 'god', 'much'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build the fallback local human-curated micro token bank gate.'
    )
    parser.add_argument('--source_prepared_root', default=str(DEFAULT_SOURCE_PREPARED_ROOT))
    parser.add_argument('--output_root', default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument('--quarantine_json', default=str(DEFAULT_QUARANTINE_JSON))
    return parser.parse_args()


def read_pipe_rows(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open('r', encoding='utf-8') as handle:
        reader = csv.reader(handle, delimiter='|')
        for row in reader:
            if not row:
                continue
            sample_id = str(row[0])
            target = str(row[1]) if len(row) > 1 else ''
            rows.append(
                {
                    'sample_id': sample_id,
                    'target': target,
                    'tokens': target.split(),
                    'raw_row': row,
                }
            )
    return rows


def write_pipe_rows(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle, delimiter='|')
        for row in rows:
            writer.writerow(row['raw_row'])


def load_quarantine_keys(path: Path) -> set[tuple[str, str]]:
    raw = json.loads(path.read_text(encoding='utf-8'))
    return {(str(entry['split']), str(entry['id'])) for entry in raw}


def reset_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def ensure_symlink(target: Path, link_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    reset_path(link_path)
    link_path.symlink_to(target)


def key_map(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row['sample_id'], row['target']): row for row in rows}


def build_selected_rows(
    expected: list[tuple[str, str]],
    source_rows: list[dict[str, Any]],
    split_name: str,
    quarantine_keys: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    lookup = key_map(source_rows)
    selected: list[dict[str, Any]] = []
    for sample_id, target in expected:
        row = lookup.get((sample_id, target))
        if row is None:
            raise SystemExit(f'missing {split_name} row in source prepared root: {sample_id} -> {target}')
        if (split_name, sample_id) in quarantine_keys:
            raise SystemExit(f'quarantined row selected for {split_name}: {sample_id} -> {target}')
        if any(token in EXCLUDED_TOKENS for token in row['tokens']):
            raise SystemExit(f'excluded token family present in {split_name} row: {sample_id} -> {target}')
        selected.append(
            {
                'sample_id': sample_id,
                'target': target,
                'tokens': row['tokens'],
                'phrase_len': len(row['tokens']),
                'source_split': {'train': 'train', 'eval': 'dev', 'test': 'test'}[split_name],
                'original_split': split_name,
                'raw_row': [sample_id, target, target],
            }
        )
    return selected


def validate_selected_tokens(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    examples = Counter()
    contexts: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for token in set(row['tokens']) & set(SELECTED_TOKENS):
            examples[token] += 1
            contexts[token].add(row['target'])
    support = {
        token: {
            'example_count_in_gate': examples[token],
            'context_count_in_gate': len(contexts[token]),
        }
        for token in SELECTED_TOKENS
    }
    missing = {
        token: counts
        for token, counts in support.items()
        if counts['example_count_in_gate'] < 3 or counts['context_count_in_gate'] < 3
    }
    if missing:
        raise SystemExit(f'micro token support too weak: {json.dumps(missing, sort_keys=True)}')
    return support


def build_surface(
    source_prepared_root: Path,
    output_root: Path,
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    token_support: dict[str, dict[str, int]],
) -> dict[str, Any]:
    if output_root.exists():
        reset_path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    annotations_root = output_root / 'annotations' / 'manual'
    write_pipe_rows(train_rows, annotations_root / 'train.corpus.csv')
    write_pipe_rows(dev_rows, annotations_root / 'dev.corpus.csv')
    write_pipe_rows(test_rows, annotations_root / 'test.corpus.csv')

    ensure_symlink(source_prepared_root / 'lookup', output_root / 'lookup')

    feature_source_roots = {
        'train': source_prepared_root / 'features' / 'fullFrame-210x260px' / 'train',
        'dev': source_prepared_root / 'features' / 'fullFrame-210x260px' / 'dev',
        'test': source_prepared_root / 'features' / 'fullFrame-210x260px' / 'test',
    }
    segmentation_source_roots = {
        'train': source_prepared_root / 'segmentation' / 'train_segmentation',
        'dev': source_prepared_root / 'segmentation' / 'val_segmentation',
        'test': source_prepared_root / 'segmentation' / 'test_segmentation',
    }

    for split_name in ('train', 'dev', 'test'):
        reset_path(output_root / 'features' / 'fullFrame-210x260px' / split_name)
        reset_path(
            output_root
            / 'segmentation'
            / {'train': 'train_segmentation', 'dev': 'val_segmentation', 'test': 'test_segmentation'}[split_name]
        )

    rows_by_split = {'train': train_rows, 'dev': dev_rows, 'test': test_rows}
    for split_name, rows in rows_by_split.items():
        feature_root = output_root / 'features' / 'fullFrame-210x260px' / split_name
        seg_root = output_root / 'segmentation' / {'train': 'train_segmentation', 'dev': 'val_segmentation', 'test': 'test_segmentation'}[split_name]
        feature_root.mkdir(parents=True, exist_ok=True)
        seg_root.mkdir(parents=True, exist_ok=True)
        for row in rows:
            source_split = row['source_split']
            ensure_symlink(feature_source_roots[source_split] / row['sample_id'], feature_root / row['sample_id'])
            ensure_symlink(segmentation_source_roots[source_split] / row['sample_id'], seg_root / row['sample_id'])

    train_targets = {row['target'] for row in train_rows}
    heldout_rows = dev_rows + test_rows
    heldout_duplicate_targets = sorted({row['target'] for row in heldout_rows if row['target'] in train_targets})

    manifest = {
        'ok': True,
        'name': 'normalized_quarantine_gate_v1_human_curated_micro_token_bank_v1',
        'source_prepared_root': str(source_prepared_root),
        'output_root': str(output_root),
        'counts': {
            'train': len(train_rows),
            'dev': len(dev_rows),
            'test': len(test_rows),
            'train_isolated_rows': len(ISOLATED_TRAIN_ROWS),
            'train_short_rows': len(SHORT_TRAIN_ROWS),
        },
        'selected_tokens': SELECTED_TOKENS,
        'selected_token_count': len(SELECTED_TOKENS),
        'heldout_total': len(heldout_rows),
        'heldout_duplicate_train_phrases': heldout_duplicate_targets,
        'lookup_reused': True,
        'token_support': token_support,
        'selected_dev_rows': [
            {
                'sample_id': row['sample_id'],
                'target': row['target'],
                'tokens': row['tokens'],
                'phrase_len': row['phrase_len'],
            }
            for row in dev_rows
        ],
        'selected_test_rows': [
            {
                'sample_id': row['sample_id'],
                'target': row['target'],
                'tokens': row['tokens'],
                'phrase_len': row['phrase_len'],
            }
            for row in test_rows
        ],
        'selected_train_rows': [
            {
                'sample_id': row['sample_id'],
                'target': row['target'],
                'tokens': row['tokens'],
                'phrase_len': row['phrase_len'],
            }
            for row in train_rows
        ],
    }
    (output_root / 'surface_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8')
    return manifest


def main() -> None:
    args = parse_args()
    source_prepared_root = Path(args.source_prepared_root)
    output_root = Path(args.output_root)
    quarantine_keys = load_quarantine_keys(Path(args.quarantine_json))

    if not source_prepared_root.exists():
        raise SystemExit(f'missing source prepared root: {source_prepared_root}')

    train_source_rows = read_pipe_rows(source_prepared_root / 'annotations' / 'manual' / 'train.corpus.csv')
    dev_source_rows = read_pipe_rows(source_prepared_root / 'annotations' / 'manual' / 'dev.corpus.csv')
    test_source_rows = read_pipe_rows(source_prepared_root / 'annotations' / 'manual' / 'test.corpus.csv')

    isolated_rows = build_selected_rows(ISOLATED_TRAIN_ROWS, train_source_rows, 'train', quarantine_keys)
    short_rows = build_selected_rows(SHORT_TRAIN_ROWS, train_source_rows, 'train', quarantine_keys)
    train_rows = isolated_rows + short_rows
    dev_rows = build_selected_rows(DEV_ROWS, dev_source_rows, 'eval', quarantine_keys)
    test_rows = build_selected_rows(TEST_ROWS, test_source_rows, 'test', quarantine_keys)

    token_support = validate_selected_tokens(train_rows)
    manifest = build_surface(source_prepared_root, output_root, train_rows, dev_rows, test_rows, token_support)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
