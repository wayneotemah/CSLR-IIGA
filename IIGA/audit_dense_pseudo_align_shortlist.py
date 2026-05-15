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


def analyze_shortlist(shortlist_data: dict[str, Any], train_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
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
        key = canonical_cluster_key(tokens)
        clusters[key].append(
            {
                'sample_id': sample_id,
                'target': target,
                'token_count': int(item.get('token_count', len(tokens))),
                'reusable_token_count': int(item.get('reusable_token_count', 0)),
                'dense_token_count': int(item.get('dense_token_count', 0)),
                'mean_reusable_alignment_score': float(item.get('mean_reusable_alignment_score', 0.0)),
            }
        )
        cluster_token_counter.update(tokens)

    cluster_rows = []
    for cluster_key, items in clusters.items():
        cluster_rows.append(
            {
                'cluster_key': cluster_key,
                'cluster_size': len(items),
                'representative_target': items[0]['target'],
                'sample_ids': [item['sample_id'] for item in items],
                'targets': [item['target'] for item in items],
                'mean_reusable_alignment_score': sum(item['mean_reusable_alignment_score'] for item in items) / max(len(items), 1),
            }
        )
    cluster_rows.sort(key=lambda item: (-item['cluster_size'], item['cluster_key']))

    return {
        'shortlist_overview': shortlist_data.get('shortlist_overview', {}),
        'token_family_cluster_overview': {
            'cluster_count': len(cluster_rows),
            'largest_cluster_size': max((item['cluster_size'] for item in cluster_rows), default=0),
        },
        'top_token_family_clusters': cluster_rows[:50],
        'top_shortlist_tokens': cluster_token_counter.most_common(50),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Audit dense-align shortlist clusters to expose repeated phrase families.')
    parser.add_argument('--shortlist_json', required=True)
    parser.add_argument('--train_corpus_csv', required=True)
    args = parser.parse_args()

    shortlist_path = Path(args.shortlist_json)
    if not shortlist_path.exists():
        raise SystemExit(f'missing shortlist json: {shortlist_path}')
    train_corpus_path = Path(args.train_corpus_csv)
    if not train_corpus_path.exists():
        raise SystemExit(f'missing train corpus csv: {train_corpus_path}')

    shortlist_data = load_shortlist(shortlist_path)
    train_rows = load_train_rows(train_corpus_path)
    analysis = analyze_shortlist(shortlist_data, train_rows)
    result = {
        'ok': True,
        'shortlist_path': str(shortlist_path),
        'train_corpus_path': str(train_corpus_path),
        **analysis,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
