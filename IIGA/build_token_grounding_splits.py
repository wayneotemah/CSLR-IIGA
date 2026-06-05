from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT_JSONS = {
    'train': Path('/home/cloudsurfer/projects/ishara-ai/video_v1_train.json'),
    'eval': Path('/home/cloudsurfer/projects/ishara-ai/video_v1_eval.json'),
    'test': Path('/home/cloudsurfer/projects/ishara-ai/video_v1_test.json'),
}

DEFAULT_VIDEO_ROOT = Path('/home/cloudsurfer/projects/ishara-ai/videos')
DEFAULT_OUTPUT_ROOT = Path('/home/cloudsurfer/projects/ishara-ai/odocs/research/token_grounding_curriculum_2026-06-05')
DEFAULT_QUARANTINE_JSON = Path('/home/cloudsurfer/projects/ishara-ai/odocs/research/label_gloss_audit_2026-05-07/quarantine_candidates.json')
DEFAULT_POLICY_JSON = Path('/home/cloudsurfer/projects/ishara-ai/odocs/research/label_gloss_audit_2026-05-07/label_policy_decisions.json')
DEFAULT_ROW_RANKING_JSON = Path('/home/cloudsurfer/projects/ishara-ai/odocs/research/normalized_gate_v1_grounding_audit/phase_b_selection/phase_b_row_ranking.json')

FUNCTIONISH_TOKENS = {
    'a', 'about', 'all', 'are', 'do', 'have', 'how', 'i', 'is', 'it', 'me', 'no',
    'please', 'same', 'that', 'the', 'this', 'time', 'to', 'what', 'you', 'your',
}


@dataclass
class Row:
    split: str
    sample_id: str
    video: str
    target: str
    tokens: list[str]
    video_path: Path
    local_video_exists: bool
    quarantine: bool
    quarantine_reason: str
    ranking_score: int | None
    support_ok: bool | None
    video_found_in_ranking: bool | None
    weak_hits: list[str]

    @property
    def phrase_len(self) -> int:
        return len(self.tokens)


def load_json_rows(path: Path, split: str, video_root: Path, quarantined_keys: set[tuple[str, str]], ranking_by_key: dict[tuple[str, str], dict[str, Any]]) -> list[Row]:
    raw = json.loads(path.read_text(encoding='utf-8'))
    rows: list[Row] = []
    for item in raw:
        sample_id = str(item['id'])
        target = str(item['conversations'][-1]['value']).strip().lower()
        video = str(item['video'])
        key = (split, sample_id)
        ranking = ranking_by_key.get(key, {})
        rows.append(Row(
            split=split,
            sample_id=sample_id,
            video=video,
            target=target,
            tokens=target.split(),
            video_path=video_root / video,
            local_video_exists=(video_root / video).exists(),
            quarantine=key in quarantined_keys,
            quarantine_reason='quarantine_candidates.json' if key in quarantined_keys else '',
            ranking_score=int(ranking['score']) if 'score' in ranking else None,
            support_ok=bool(ranking['support_ok']) if 'support_ok' in ranking else None,
            video_found_in_ranking=bool(ranking['video_found']) if 'video_found' in ranking else None,
            weak_hits=list(ranking.get('weak_hits', [])),
        ))
    return rows


def load_quarantined_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    raw = json.loads(path.read_text(encoding='utf-8'))
    keys: set[tuple[str, str]] = set()
    for entry in raw:
        split = str(entry.get('split', ''))
        sample_id = str(entry.get('id', ''))
        if split and sample_id:
            keys.add((split, sample_id))
    return keys


def load_policy(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def load_row_ranking(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding='utf-8'))
    rows = raw.get('rows', [])
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        result[(str(row['split']), str(row['sample_id']))] = row
    return result


def is_exact_train_duplicate(row: Row, train_phrase_set: set[str]) -> bool:
    return row.target in train_phrase_set


def token_stats(rows_by_split: dict[str, list[Row]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    train_rows = rows_by_split['train']
    eval_rows = rows_by_split['eval']
    test_rows = rows_by_split['test']
    heldout_rows = eval_rows + test_rows

    train_phrase_set = {row.target for row in train_rows}

    by_token: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {
        'train_all': [],
        'train_isolated': [],
        'train_short': [],
        'heldout_short': [],
    })
    train_contexts: dict[str, set[str]] = defaultdict(set)
    heldout_contexts: dict[str, set[str]] = defaultdict(set)
    token_counter = Counter()

    for row in train_rows:
        if row.quarantine or not row.local_video_exists:
            continue
        unique_tokens = set(row.tokens)
        for token in unique_tokens:
            token_counter[token] += 1
            train_contexts[token].add(row.target)
            row_dict = row_to_dict(row)
            by_token[token]['train_all'].append(row_dict)
            if row.phrase_len == 1:
                by_token[token]['train_isolated'].append(row_dict)
            if 1 <= row.phrase_len <= 3:
                by_token[token]['train_short'].append(row_dict)

    for row in heldout_rows:
        if row.quarantine or not row.local_video_exists:
            continue
        if row.phrase_len > 3:
            continue
        duplicate = is_exact_train_duplicate(row, train_phrase_set)
        unique_tokens = set(row.tokens)
        for token in unique_tokens:
            heldout_contexts[token].add(row.target)
            row_dict = row_to_dict(row)
            row_dict['duplicate_train_phrase'] = duplicate
            by_token[token]['heldout_short'].append(row_dict)

    inventory: list[dict[str, Any]] = []
    selected_tokens: dict[str, dict[str, list[dict[str, Any]]]] = {}
    borderline_tokens: dict[str, str] = {}

    for token in sorted(set(train_contexts) | set(heldout_contexts)):
        train_all = by_token[token]['train_all']
        train_isolated = by_token[token]['train_isolated']
        train_short = by_token[token]['train_short']
        heldout_short = [row for row in by_token[token]['heldout_short'] if not row['duplicate_train_phrase']]
        item = {
            'token': token,
            'functionish': token in FUNCTIONISH_TOKENS,
            'train_example_count': len(train_all),
            'train_isolated_count': len(train_isolated),
            'train_short_count': len(train_short),
            'train_context_count': len(train_contexts[token]),
            'heldout_short_count': len(heldout_short),
            'heldout_context_count': len({row['target'] for row in heldout_short}),
            'selected_strict': False,
        }
        item['selected_strict'] = (
            not item['functionish']
            and item['train_example_count'] >= 3
            and item['train_context_count'] >= 2
            and item['train_short_count'] >= 2
            and (item['train_isolated_count'] >= 1 or item['train_short_count'] >= 3)
            and item['heldout_short_count'] >= 1
            and item['heldout_context_count'] >= 1
        )
        if item['selected_strict']:
            selected_tokens[token] = {
                'train_all': train_all,
                'train_isolated': train_isolated,
                'train_short': train_short,
                'heldout_short': heldout_short,
            }
        else:
            reasons: list[str] = []
            if item['functionish']:
                reasons.append('functionish')
            if item['train_example_count'] < 3:
                reasons.append('train_examples_lt_3')
            if item['train_context_count'] < 2:
                reasons.append('train_contexts_lt_2')
            if item['train_short_count'] < 2:
                reasons.append('train_short_lt_2')
            if not (item['train_isolated_count'] >= 1 or item['train_short_count'] >= 3):
                reasons.append('no_isolated_or_near_isolated_support')
            if item['heldout_short_count'] < 1:
                reasons.append('no_clean_heldout_short_rows')
            borderline_tokens[token] = ','.join(reasons)
        inventory.append(item)

    summary = {
        'train_rows': len(train_rows),
        'eval_rows': len(eval_rows),
        'test_rows': len(test_rows),
        'selected_token_count': len(selected_tokens),
        'borderline_token_count': len(borderline_tokens),
        'quarantined_row_count': sum(1 for rows in rows_by_split.values() for row in rows if row.quarantine),
        'functionish_selected_count': sum(1 for item in inventory if item['selected_strict'] and item['functionish']),
        'top_train_tokens': token_counter.most_common(50),
    }
    return inventory, selected_tokens, summary


def row_to_dict(row: Row) -> dict[str, Any]:
    return {
        'split': row.split,
        'sample_id': row.sample_id,
        'video': row.video,
        'video_path': str(row.video_path),
        'local_video_exists': row.local_video_exists,
        'target': row.target,
        'tokens': row.tokens,
        'phrase_len': row.phrase_len,
        'quarantine': row.quarantine,
        'quarantine_reason': row.quarantine_reason,
        'ranking_score': row.ranking_score,
        'support_ok': row.support_ok,
        'video_found_in_ranking': row.video_found_in_ranking,
        'weak_hits': row.weak_hits,
    }


def write_inventory_csv(inventory: list[dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'token', 'functionish', 'train_example_count', 'train_isolated_count',
        'train_short_count', 'train_context_count', 'heldout_short_count',
        'heldout_context_count', 'selected_strict',
    ]
    with output_csv.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in inventory:
            writer.writerow({name: row[name] for name in fieldnames})


def write_row_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_csv.write_text('', encoding='utf-8')
        return
    fieldnames = list(rows[0].keys())
    with output_csv.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description='Build token-grounding curriculum candidate manifests from the normalized JSON pool.')
    parser.add_argument('--video_root', default=str(DEFAULT_VIDEO_ROOT))
    parser.add_argument('--output_root', default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument('--quarantine_json', default=str(DEFAULT_QUARANTINE_JSON))
    parser.add_argument('--policy_json', default=str(DEFAULT_POLICY_JSON))
    parser.add_argument('--row_ranking_json', default=str(DEFAULT_ROW_RANKING_JSON))
    args = parser.parse_args()

    video_root = Path(args.video_root)
    output_root = Path(args.output_root)
    quarantined_keys = load_quarantined_keys(Path(args.quarantine_json))
    policy = load_policy(Path(args.policy_json))
    ranking_by_key = load_row_ranking(Path(args.row_ranking_json))

    rows_by_split = {
        split: load_json_rows(path, split, video_root, quarantined_keys, ranking_by_key)
        for split, path in ROOT_JSONS.items()
    }
    inventory, selected_tokens, summary = token_stats(rows_by_split)

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / 'summary.json').write_text(json.dumps({
        'summary': summary,
        'policy': policy.get('policy', {}),
        'source_jsons': {split: str(path) for split, path in ROOT_JSONS.items()},
        'video_root': str(video_root),
    }, indent=2, sort_keys=True), encoding='utf-8')
    (output_root / 'selected_tokens.json').write_text(json.dumps(selected_tokens, indent=2, sort_keys=True), encoding='utf-8')
    write_inventory_csv(inventory, output_root / 'token_inventory.csv')

    isolated_rows = [row for token_rows in selected_tokens.values() for row in token_rows['train_isolated']]
    short_train_rows = [row for token_rows in selected_tokens.values() for row in token_rows['train_short']]
    heldout_short_rows = [row for token_rows in selected_tokens.values() for row in token_rows['heldout_short']]

    write_row_csv(isolated_rows, output_root / 'isolated_train_candidates.csv')
    write_row_csv(short_train_rows, output_root / 'short_gloss_train_candidates.csv')
    write_row_csv(heldout_short_rows, output_root / 'short_gloss_heldout_candidates.csv')

    print(json.dumps({
        'summary': summary,
        'output_root': str(output_root),
        'selected_tokens_json': str(output_root / 'selected_tokens.json'),
        'token_inventory_csv': str(output_root / 'token_inventory.csv'),
        'isolated_train_candidate_count': len(isolated_rows),
        'short_gloss_train_candidate_count': len(short_train_rows),
        'short_gloss_heldout_candidate_count': len(heldout_short_rows),
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
