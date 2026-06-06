from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        return [{key: str(value) for key, value in row.items()} for row in reader]


def as_bool(value: str) -> bool:
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y'}


def as_int(value: str) -> int:
    return int(str(value).strip())


def row_tokens(row: dict[str, str]) -> list[str]:
    raw = row.get('tokens', '').strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw.replace("'", '"'))
    except json.JSONDecodeError:
        return [token for token in raw.split() if token]
    if isinstance(parsed, list):
        return [str(token) for token in parsed]
    return [token for token in raw.split() if token]


def main() -> None:
    parser = argparse.ArgumentParser(description='Validate mined token-grounding curriculum outputs before prepared-gate construction.')
    parser.add_argument('--output_root', required=True)
    parser.add_argument('--min_selected_tokens', type=int, default=20)
    parser.add_argument('--min_train_examples', type=int, default=3)
    parser.add_argument('--min_train_contexts', type=int, default=2)
    parser.add_argument('--min_train_short', type=int, default=2)
    parser.add_argument('--require_isolated_or_extra_short', action='store_true', default=True)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    summary_path = output_root / 'summary.json'
    inventory_path = output_root / 'token_inventory.csv'
    selected_path = output_root / 'selected_tokens.json'
    isolated_path = output_root / 'isolated_train_candidates.csv'
    short_train_path = output_root / 'short_gloss_train_candidates.csv'
    short_heldout_path = output_root / 'short_gloss_heldout_candidates.csv'

    required_paths = [summary_path, inventory_path, selected_path, isolated_path, short_train_path, short_heldout_path]
    missing_files = [str(path) for path in required_paths if not path.exists()]

    if missing_files:
        result = {
            'ok': False,
            'reason': 'missing_required_outputs',
            'missing_files': missing_files,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    summary_payload = load_json(summary_path)
    inventory_rows = load_rows(inventory_path)
    isolated_rows = load_rows(isolated_path)
    short_train_rows = load_rows(short_train_path)
    short_heldout_rows = load_rows(short_heldout_path)
    selected_tokens = load_json(selected_path)

    strict_rows = [row for row in inventory_rows if as_bool(row['selected_strict'])]
    borderline_rows = [row for row in inventory_rows if not as_bool(row['selected_strict'])]

    violations: list[dict[str, Any]] = []
    selected_tokens_with_issues: list[str] = []

    for row in strict_rows:
        token = row['token']
        token_violations: list[str] = []
        train_examples = as_int(row['train_example_count'])
        train_contexts = as_int(row['train_context_count'])
        train_short = as_int(row['train_short_count'])
        train_isolated = as_int(row['train_isolated_count'])
        heldout_short = as_int(row['heldout_short_count'])

        if as_bool(row['functionish']):
            token_violations.append('functionish_selected')
        if train_examples < args.min_train_examples:
            token_violations.append('train_examples_below_threshold')
        if train_contexts < args.min_train_contexts:
            token_violations.append('train_contexts_below_threshold')
        if train_short < args.min_train_short:
            token_violations.append('train_short_below_threshold')
        if args.require_isolated_or_extra_short and not (train_isolated >= 1 or train_short >= 3):
            token_violations.append('no_isolated_or_extra_short_support')
        if heldout_short < 1:
            token_violations.append('no_clean_heldout_short_rows')

        token_payload = selected_tokens.get(token, {})
        for bucket_name in ('train_all', 'train_isolated', 'train_short', 'heldout_short'):
            for entry in token_payload.get(bucket_name, []):
                if not entry.get('local_video_exists', False):
                    token_violations.append(f'missing_video_in_{bucket_name}')
                if entry.get('quarantine', False):
                    token_violations.append(f'quarantined_row_in_{bucket_name}')
                if bucket_name == 'heldout_short' and entry.get('duplicate_train_phrase', False):
                    token_violations.append('duplicate_train_phrase_in_heldout')

        if token_violations:
            selected_tokens_with_issues.append(token)
            violations.append({
                'token': token,
                'violations': sorted(set(token_violations)),
                'counts': {
                    'train_example_count': train_examples,
                    'train_isolated_count': train_isolated,
                    'train_short_count': train_short,
                    'train_context_count': train_contexts,
                    'heldout_short_count': heldout_short,
                },
            })

    isolated_token_counts = Counter(row['target'] for row in isolated_rows)
    short_train_token_counts = Counter()
    short_heldout_token_counts = Counter()
    for row in short_train_rows:
        for token in row_tokens(row):
            short_train_token_counts[token] += 1
    for row in short_heldout_rows:
        for token in row_tokens(row):
            short_heldout_token_counts[token] += 1

    ok = (
        len(strict_rows) >= args.min_selected_tokens
        and not missing_files
        and not violations
    )

    result = {
        'ok': ok,
        'output_root': str(output_root),
        'summary_path': str(summary_path),
        'inventory_path': str(inventory_path),
        'selected_tokens_path': str(selected_path),
        'strict_selected_token_count': len(strict_rows),
        'borderline_token_count': len(borderline_rows),
        'isolated_train_candidate_count': len(isolated_rows),
        'short_gloss_train_candidate_count': len(short_train_rows),
        'short_gloss_heldout_candidate_count': len(short_heldout_rows),
        'selected_tokens_with_issues': selected_tokens_with_issues,
        'violations': violations,
        'top_isolated_targets': isolated_token_counts.most_common(30),
        'top_short_train_tokens': short_train_token_counts.most_common(30),
        'top_short_heldout_tokens': short_heldout_token_counts.most_common(30),
        'source_summary': summary_payload.get('summary', {}),
        'policy': summary_payload.get('policy', {}),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
