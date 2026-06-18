from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_CURRICULUM_ROOT = Path('/tmp/opencode/token_grounding_curriculum_2026_06_05')
DEFAULT_SOURCE_PREPARED_ROOT = Path('/home/cloudsurfer/projects/ishara-ai/Sign2Text/local_runs/prepared/normalized_quarantine_gate_v1')
DEFAULT_OUTPUT_ROOT = Path('/home/cloudsurfer/projects/ishara-ai/Sign2Text/local_runs/prepared/normalized_quarantine_gate_v1_token_grounding_v1')
DEFAULT_QUARANTINE_JSON = Path('/home/cloudsurfer/projects/ishara-ai/odocs/research/label_gloss_audit_2026-05-07/quarantine_candidates.json')

EXCLUDED_TOKENS = {'love', 'god', 'much'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build the first thin prepared token-grounding gate from validated curriculum outputs.'
    )
    parser.add_argument('--curriculum_root', default=str(DEFAULT_CURRICULUM_ROOT))
    parser.add_argument('--source_prepared_root', default=str(DEFAULT_SOURCE_PREPARED_ROOT))
    parser.add_argument('--output_root', default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument('--quarantine_json', default=str(DEFAULT_QUARANTINE_JSON))
    parser.add_argument('--heldout_per_split', type=int, default=8)
    parser.add_argument('--min_examples_per_held_token', type=int, default=2)
    parser.add_argument('--min_contexts_per_held_token', type=int, default=2)
    parser.add_argument('--max_short_train_rows', type=int, default=48)
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


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def parse_tokens(raw: str) -> list[str]:
    value = raw.strip()
    if not value:
        return []
    try:
        parsed = json.loads(value.replace("'", '"'))
    except json.JSONDecodeError:
        return [token for token in value.split() if token]
    if isinstance(parsed, list):
        return [str(token) for token in parsed]
    return [token for token in value.split() if token]


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


def load_quarantine_keys(path: Path) -> set[tuple[str, str]]:
    raw = json.loads(path.read_text(encoding='utf-8'))
    return {(str(entry['split']), str(entry['id'])) for entry in raw}


def load_selected_tokens(curriculum_root: Path) -> set[str]:
    selected = json.loads((curriculum_root / 'selected_tokens.json').read_text(encoding='utf-8'))
    return set(selected.keys())


def load_inventory(curriculum_root: Path) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    with (curriculum_root / 'token_inventory.csv').open('r', encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            inventory[str(row['token'])] = {
                'train_example_count': int(row['train_example_count']),
                'train_isolated_count': int(row['train_isolated_count']),
                'train_short_count': int(row['train_short_count']),
                'train_context_count': int(row['train_context_count']),
                'heldout_short_count': int(row['heldout_short_count']),
                'heldout_context_count': int(row['heldout_context_count']),
            }
    return inventory


def build_row(sample_id: str, target: str) -> dict[str, Any]:
    return {
        'sample_id': sample_id,
        'target': target,
        'tokens': target.split(),
        'raw_row': [sample_id, target, target],
    }


def source_split_for_row(split: str) -> str:
    return {'train': 'train', 'eval': 'dev', 'test': 'test'}[split]


def safe_curriculum_rows(
    curriculum_root: Path,
    quarantine_keys: set[tuple[str, str]],
    selected_tokens: set[str],
    prepared_train_ids: set[str],
    prepared_dev_ids: set[str],
    prepared_test_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    isolated_rows: list[dict[str, Any]] = []
    short_train_rows: list[dict[str, Any]] = []
    heldout_rows: list[dict[str, Any]] = []

    for row in load_csv_rows(curriculum_root / 'isolated_train_candidates.csv'):
        tokens = parse_tokens(row['tokens'])
        if (row['split'], row['sample_id']) in quarantine_keys:
            continue
        if any(token in EXCLUDED_TOKENS for token in tokens):
            continue
        if row['sample_id'] not in prepared_train_ids:
            continue
        isolated_rows.append(
            {
                'source_split': 'train',
                'sample_id': row['sample_id'],
                'target': row['target'],
                'tokens': tokens,
                'phrase_len': int(row['phrase_len']),
                'raw_row': [row['sample_id'], row['target'], row['target']],
            }
        )

    for row in load_csv_rows(curriculum_root / 'short_gloss_train_candidates.csv'):
        tokens = parse_tokens(row['tokens'])
        if (row['split'], row['sample_id']) in quarantine_keys:
            continue
        if any(token in EXCLUDED_TOKENS for token in tokens):
            continue
        if not all(token in selected_tokens for token in tokens):
            continue
        if row['sample_id'] not in prepared_train_ids:
            continue
        short_train_rows.append(
            {
                'source_split': 'train',
                'sample_id': row['sample_id'],
                'target': row['target'],
                'tokens': tokens,
                'phrase_len': int(row['phrase_len']),
                'raw_row': [row['sample_id'], row['target'], row['target']],
            }
        )

    seen_heldout: set[tuple[str, str]] = set()
    for row in load_csv_rows(curriculum_root / 'short_gloss_heldout_candidates.csv'):
        key = (row['split'], row['sample_id'])
        if key in seen_heldout:
            continue
        seen_heldout.add(key)
        tokens = parse_tokens(row['tokens'])
        if key in quarantine_keys:
            continue
        if any(token in EXCLUDED_TOKENS for token in tokens):
            continue
        if str(row['duplicate_train_phrase']).strip().lower() == 'true':
            continue
        if not all(token in selected_tokens for token in tokens):
            continue
        if row['split'] == 'eval' and row['sample_id'] not in prepared_dev_ids:
            continue
        if row['split'] == 'test' and row['sample_id'] not in prepared_test_ids:
            continue
        heldout_rows.append(
            {
                'source_split': source_split_for_row(row['split']),
                'original_split': row['split'],
                'sample_id': row['sample_id'],
                'target': row['target'],
                'tokens': tokens,
                'phrase_len': int(row['phrase_len']),
                'raw_row': [row['sample_id'], row['target'], row['target']],
            }
        )

    isolated_rows.sort(key=lambda row: (row['target'], row['sample_id']))
    short_train_rows.sort(key=lambda row: (row['target'], row['sample_id']))
    heldout_rows.sort(key=lambda row: (row['original_split'], row['target'], row['sample_id']))
    return isolated_rows, short_train_rows, heldout_rows


def choose_heldout_rows(
    heldout_rows: list[dict[str, Any]],
    inventory: dict[str, dict[str, Any]],
    heldout_per_split: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def greedy_pick(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        picked: list[dict[str, Any]] = []
        covered: set[str] = set()
        remaining = rows[:]
        while remaining and len(picked) < heldout_per_split:
            best: dict[str, Any] | None = None
            best_key: tuple[Any, ...] | None = None
            for row in remaining:
                token_set = set(row['tokens'])
                new_tokens = len(token_set - covered)
                avg_contexts = sum(inventory[token]['train_context_count'] for token in row['tokens']) / len(row['tokens'])
                key = (new_tokens, row['phrase_len'], avg_contexts, row['sample_id'])
                if best_key is None or key > best_key:
                    best_key = key
                    best = row
            assert best is not None
            picked.append(best)
            covered |= set(best['tokens'])
            remaining = [row for row in remaining if row is not best]
        return picked

    eval_rows = [row for row in heldout_rows if row['original_split'] == 'eval']
    test_rows = [row for row in heldout_rows if row['original_split'] == 'test']
    return greedy_pick(eval_rows), greedy_pick(test_rows)


def choose_train_rows(
    isolated_rows: list[dict[str, Any]],
    short_train_rows: list[dict[str, Any]],
    heldout_rows: list[dict[str, Any]],
    min_examples_per_held_token: int,
    min_contexts_per_held_token: int,
    max_short_train_rows: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    held_tokens = sorted({token for row in heldout_rows for token in row['tokens']})
    token_examples = Counter()
    token_contexts: dict[str, set[str]] = defaultdict(set)
    selected_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for row in isolated_rows:
        selected_rows.append(row)
        seen_ids.add(row['sample_id'])
        for token in set(row['tokens']):
            if token in held_tokens:
                token_examples[token] += 1
                token_contexts[token].add(row['target'])

    candidates = [row for row in short_train_rows if row['sample_id'] not in seen_ids and any(token in held_tokens for token in row['tokens'])]
    chosen_short_rows: list[dict[str, Any]] = []

    def unmet_tokens() -> set[str]:
        return {
            token
            for token in held_tokens
            if token_examples[token] < min_examples_per_held_token or len(token_contexts[token]) < min_contexts_per_held_token
        }

    while candidates and len(chosen_short_rows) < max_short_train_rows:
        current_unmet = unmet_tokens()
        if not current_unmet:
            break
        best: dict[str, Any] | None = None
        best_key: tuple[Any, ...] | None = None
        for row in candidates:
            row_tokens = set(row['tokens']) & set(held_tokens)
            unmet_overlap = len(row_tokens & current_unmet)
            total_overlap = len(row_tokens)
            new_context_gain = sum(1 for token in row_tokens if row['target'] not in token_contexts[token])
            key = (unmet_overlap, new_context_gain, total_overlap, -row['phrase_len'], row['sample_id'])
            if best_key is None or key > best_key:
                best_key = key
                best = row
        assert best is not None
        chosen_short_rows.append(best)
        selected_rows.append(best)
        seen_ids.add(best['sample_id'])
        for token in set(best['tokens']):
            if token in held_tokens:
                token_examples[token] += 1
                token_contexts[token].add(best['target'])
        candidates = [row for row in candidates if row['sample_id'] != best['sample_id']]

    if unmet_tokens():
        missing = {
            token: {
                'examples': token_examples[token],
                'contexts': len(token_contexts[token]),
            }
            for token in sorted(unmet_tokens())
        }
        raise SystemExit(f'failed to satisfy held-token support within thin gate: {json.dumps(missing, sort_keys=True)}')

    token_support = {
        token: {
            'example_count_in_gate': token_examples[token],
            'context_count_in_gate': len(token_contexts[token]),
        }
        for token in held_tokens
    }
    return selected_rows, token_support


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

    train_targets = [row['target'] for row in train_rows]
    heldout_targets = [row['target'] for row in dev_rows + test_rows]
    train_phrase_set = set(train_targets)
    heldout_duplicate_targets = sorted({target for target in heldout_targets if target in train_phrase_set})

    manifest = {
        'ok': True,
        'name': 'normalized_quarantine_gate_v1_token_grounding_v1',
        'source_prepared_root': str(source_prepared_root),
        'output_root': str(output_root),
        'counts': {
            'train': len(train_rows),
            'dev': len(dev_rows),
            'test': len(test_rows),
        },
        'train_isolated_count': sum(1 for row in train_rows if row['phrase_len'] == 1),
        'train_short_count': sum(1 for row in train_rows if row['phrase_len'] in {2, 3}),
        'heldout_token_count': len(token_support),
        'heldout_duplicate_targets': heldout_duplicate_targets,
        'heldout_duplicate_target_count': len(heldout_duplicate_targets),
        'lookup_reused': True,
        'excluded_tokens': sorted(EXCLUDED_TOKENS),
        'token_support': token_support,
        'selected_rows': {
            'train': [
                {'sample_id': row['sample_id'], 'target': row['target'], 'tokens': row['tokens'], 'phrase_len': row['phrase_len']}
                for row in train_rows
            ],
            'dev': [
                {
                    'sample_id': row['sample_id'],
                    'target': row['target'],
                    'tokens': row['tokens'],
                    'phrase_len': row['phrase_len'],
                    'source_split': row['source_split'],
                    'original_split': row.get('original_split', 'eval'),
                }
                for row in dev_rows
            ],
            'test': [
                {
                    'sample_id': row['sample_id'],
                    'target': row['target'],
                    'tokens': row['tokens'],
                    'phrase_len': row['phrase_len'],
                    'source_split': row['source_split'],
                    'original_split': row.get('original_split', 'test'),
                }
                for row in test_rows
            ],
        },
    }
    (output_root / 'surface_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8')
    return manifest


def main() -> None:
    args = parse_args()
    curriculum_root = Path(args.curriculum_root)
    source_prepared_root = Path(args.source_prepared_root)
    output_root = Path(args.output_root)

    if not curriculum_root.exists():
        raise SystemExit(f'missing curriculum root: {curriculum_root}')
    if not source_prepared_root.exists():
        raise SystemExit(f'missing source prepared root: {source_prepared_root}')

    prepared_train_rows = read_pipe_rows(source_prepared_root / 'annotations' / 'manual' / 'train.corpus.csv')
    prepared_dev_rows = read_pipe_rows(source_prepared_root / 'annotations' / 'manual' / 'dev.corpus.csv')
    prepared_test_rows = read_pipe_rows(source_prepared_root / 'annotations' / 'manual' / 'test.corpus.csv')
    prepared_train_ids = {row['sample_id'] for row in prepared_train_rows}
    prepared_dev_ids = {row['sample_id'] for row in prepared_dev_rows}
    prepared_test_ids = {row['sample_id'] for row in prepared_test_rows}

    quarantine_keys = load_quarantine_keys(Path(args.quarantine_json))
    selected_tokens = load_selected_tokens(curriculum_root)
    inventory = load_inventory(curriculum_root)

    isolated_rows, short_train_rows, heldout_rows = safe_curriculum_rows(
        curriculum_root,
        quarantine_keys,
        selected_tokens,
        prepared_train_ids,
        prepared_dev_ids,
        prepared_test_ids,
    )

    dev_rows, test_rows = choose_heldout_rows(heldout_rows, inventory, heldout_per_split=args.heldout_per_split)
    train_rows, token_support = choose_train_rows(
        isolated_rows,
        short_train_rows,
        dev_rows + test_rows,
        min_examples_per_held_token=args.min_examples_per_held_token,
        min_contexts_per_held_token=args.min_contexts_per_held_token,
        max_short_train_rows=args.max_short_train_rows,
    )

    manifest = build_surface(source_prepared_root, output_root, train_rows, dev_rows, test_rows, token_support)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
