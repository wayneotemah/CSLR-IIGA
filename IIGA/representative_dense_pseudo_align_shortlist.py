from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_shortlist(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def load_train_rows(corpus_path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with corpus_path.open('r', encoding='utf-8') as handle:
        reader = csv.reader(handle, delimiter='|')
        for row in reader:
            if not row:
                continue
            target = str(row[1]) if len(row) > 1 else ''
            rows[str(row[0])] = {
                'sample_id': str(row[0]),
                'target': target,
                'tokens': target.split(),
            }
    return rows


def canonical_cluster_key(tokens: list[str]) -> str:
    return ' '.join(sorted(tokens))


def deduplicate_shortlist(
    shortlist_data: dict[str, Any],
    train_rows: dict[str, dict[str, Any]],
    top_limit: int,
) -> dict[str, Any]:
    top_candidates = shortlist_data.get('top_candidate_samples')
    if top_candidates is None:
        top_candidates = shortlist_data.get('candidate_samples', [])

    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cluster_token_counter = Counter()

    for item in top_candidates:
        sample_id = str(item['sample_id'])
        target = str(item['target'])
        row = train_rows.get(sample_id, {'tokens': target.split()})
        tokens = list(row.get('tokens', target.split()))
        cluster_key = canonical_cluster_key(tokens)
        candidate = {
            'sample_id': sample_id,
            'target': target,
            'token_count': int(item.get('token_count', len(tokens))),
            'reusable_token_count': int(item.get('reusable_token_count', 0)),
            'dense_token_count': int(item.get('dense_token_count', 0)),
            'mean_reusable_alignment_score': float(item.get('mean_reusable_alignment_score', 0.0)),
            'mean_reusable_entry_count': float(item.get('mean_reusable_entry_count', 0.0)),
            'mean_reusable_sample_count': float(item.get('mean_reusable_sample_count', 0.0)),
            'reusable_tokens': list(item.get('reusable_tokens', [])),
            'dense_tokens': list(item.get('dense_tokens', [])),
        }
        clusters[cluster_key].append(candidate)
        cluster_token_counter.update(tokens)

    representative_rows = []
    cluster_rows = []
    for cluster_key, items in clusters.items():
        items.sort(
            key=lambda item: (
                -item['reusable_token_count'],
                -item['dense_token_count'],
                -item['mean_reusable_alignment_score'],
                -item['mean_reusable_entry_count'],
                item['sample_id'],
            )
        )
        representative = items[0]
        representative_rows.append({
            **representative,
            'cluster_key': cluster_key,
            'cluster_size': len(items),
        })
        cluster_rows.append(
            {
                'cluster_key': cluster_key,
                'cluster_size': len(items),
                'representative_sample_id': representative['sample_id'],
                'representative_target': representative['target'],
                'sample_ids': [item['sample_id'] for item in items],
                'targets': [item['target'] for item in items],
                'mean_reusable_alignment_score': sum(item['mean_reusable_alignment_score'] for item in items) / max(len(items), 1),
            }
        )

    representative_rows.sort(
        key=lambda item: (
            -item['reusable_token_count'],
            -item['dense_token_count'],
            -item['mean_reusable_alignment_score'],
            -item['cluster_size'],
            item['sample_id'],
        )
    )
    cluster_rows.sort(key=lambda item: (-item['cluster_size'], item['cluster_key']))

    rep_token_counter = Counter()
    for item in representative_rows:
        rep_token_counter.update(train_rows.get(item['sample_id'], {'tokens': item['target'].split()})['tokens'])

    return {
        'shortlist_overview': shortlist_data.get('shortlist_overview', {}),
        'representative_overview': {
            'representative_sample_count': len(representative_rows),
            'representative_sample_coverage': len(representative_rows) / max(shortlist_data.get('shortlist_overview', {}).get('candidate_sample_count', 1), 1),
            'representative_unique_token_count': len(rep_token_counter),
        },
        'token_family_cluster_overview': {
            'cluster_count': len(cluster_rows),
            'largest_cluster_size': max((item['cluster_size'] for item in cluster_rows), default=0),
        },
        'top_token_family_clusters': cluster_rows[:50],
        'representative_samples': representative_rows[:top_limit],
        'representative_top_tokens': rep_token_counter.most_common(50),
        'shortlist_top_tokens': cluster_token_counter.most_common(50),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Build a deduplicated representative shortlist from a clustered dense-align shortlist.'
    )
    parser.add_argument('--shortlist_json', required=True)
    parser.add_argument('--train_corpus_csv', required=True)
    parser.add_argument('--top_limit', type=int, default=100)
    args = parser.parse_args()

    shortlist_path = Path(args.shortlist_json)
    if not shortlist_path.exists():
        raise SystemExit(f'missing shortlist json: {shortlist_path}')
    train_corpus_path = Path(args.train_corpus_csv)
    if not train_corpus_path.exists():
        raise SystemExit(f'missing train corpus csv: {train_corpus_path}')

    shortlist_data = load_shortlist(shortlist_path)
    train_rows = load_train_rows(train_corpus_path)
    analysis = deduplicate_shortlist(shortlist_data, train_rows, args.top_limit)
    result = {
        'ok': True,
        'shortlist_path': str(shortlist_path),
        'train_corpus_path': str(train_corpus_path),
        **analysis,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
