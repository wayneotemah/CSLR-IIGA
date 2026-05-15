from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def audit_representatives(
    representative_data: dict[str, Any],
    min_cluster_size: int,
    min_dense_tokens: int,
    min_mean_score: float,
    top_limit: int,
) -> dict[str, Any]:
    reps = list(representative_data.get('representative_samples', []))
    if not reps:
        raise SystemExit('representative shortlist has no representative_samples')

    selected = []
    excluded = Counter()
    token_counter = Counter()
    for item in reps:
        cluster_size = int(item.get('cluster_size', 0))
        dense_token_count = int(item.get('dense_token_count', 0))
        mean_score = float(item.get('mean_reusable_alignment_score', 0.0))
        if cluster_size < min_cluster_size:
            excluded['cluster_size_below_threshold'] += 1
            continue
        if dense_token_count < min_dense_tokens:
            excluded['dense_token_count_below_threshold'] += 1
            continue
        if mean_score < min_mean_score:
            excluded['mean_alignment_score_below_threshold'] += 1
            continue
        selected.append(item)
        token_counter.update(item.get('reusable_tokens', []))

    selected.sort(
        key=lambda item: (
            -int(item.get('cluster_size', 0)),
            -int(item.get('dense_token_count', 0)),
            -int(item.get('reusable_token_count', 0)),
            -float(item.get('mean_reusable_alignment_score', 0.0)),
            item.get('sample_id', ''),
        )
    )

    return {
        'thresholds': {
            'min_cluster_size': int(min_cluster_size),
            'min_dense_tokens': int(min_dense_tokens),
            'min_mean_score': float(min_mean_score),
            'top_limit': int(top_limit),
        },
        'representative_overview': representative_data.get('representative_overview', {}),
        'token_family_cluster_overview': representative_data.get('token_family_cluster_overview', {}),
        'integration_overview': {
            'candidate_sample_count': len(selected),
            'candidate_sample_coverage': len(selected) / max(len(reps), 1),
            'candidate_unique_token_count': len(token_counter),
            'excluded_reasons': dict(excluded),
        },
        'top_integration_candidates': selected[:top_limit],
        'integration_top_tokens': token_counter.most_common(50),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Audit a representative dense-align shortlist to pick the smallest first integration subset.'
    )
    parser.add_argument('--representative_json', required=True)
    parser.add_argument('--min_cluster_size', type=int, default=2)
    parser.add_argument('--min_dense_tokens', type=int, default=2)
    parser.add_argument('--min_mean_score', type=float, default=-25.0)
    parser.add_argument('--top_limit', type=int, default=100)
    args = parser.parse_args()

    representative_path = Path(args.representative_json)
    if not representative_path.exists():
        raise SystemExit(f'missing representative json: {representative_path}')

    representative_data = load_json(representative_path)
    analysis = audit_representatives(
        representative_data,
        args.min_cluster_size,
        args.min_dense_tokens,
        args.min_mean_score,
        args.top_limit,
    )
    result = {
        'ok': True,
        'representative_path': str(representative_path),
        **analysis,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
