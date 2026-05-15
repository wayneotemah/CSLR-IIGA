from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from tools.token_bank import TokenBank, TokenEntry


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def load_train_rows(corpus_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with corpus_path.open('r', encoding='utf-8') as handle:
        reader = csv.reader(handle, delimiter='|')
        for row in reader:
            if not row:
                continue
            target = str(row[1]) if len(row) > 1 else ''
            rows.append(
                {
                    'sample_id': str(row[0]),
                    'target': target,
                    'tokens': target.split(),
                }
            )
    return rows


def flatten_entries(bank: TokenBank) -> list[TokenEntry]:
    entries: list[TokenEntry] = []
    for token_id in bank.token_ids():
        entries.extend(bank.query_by_token(token_id))
    return entries


def build_global_token_stats(entries: list[TokenEntry]) -> dict[str, dict[str, Any]]:
    by_token: dict[str, list[TokenEntry]] = defaultdict(list)
    for entry in entries:
        by_token[entry.token_text].append(entry)

    stats: dict[str, dict[str, Any]] = {}
    for token_text, token_entries in by_token.items():
        sample_ids = {entry.sample_id for entry in token_entries}
        scores = [float(entry.score) for entry in token_entries]
        span_lengths = [int(entry.span_length) for entry in token_entries]
        stats[token_text] = {
            'entry_count': len(token_entries),
            'unique_sample_count': len(sample_ids),
            'mean_alignment_score': mean(scores),
            'mean_span_length': mean(span_lengths),
        }
    return stats


def shortlist_samples(
    summary: dict[str, Any],
    bank: TokenBank,
    train_corpus_path: Path,
    min_reusable_tokens: int,
    min_dense_tokens: int,
    min_mean_score: float,
    top_limit: int,
) -> dict[str, Any]:
    train_rows = load_train_rows(train_corpus_path)
    entries = flatten_entries(bank)
    if not entries:
        raise SystemExit('token bank has no entries')

    token_stats = build_global_token_stats(entries)
    reusable_token_set = {
        token
        for token, stat in token_stats.items()
        if stat['entry_count'] >= 2 and stat['unique_sample_count'] >= 2
    }
    dense_token_set = {
        token
        for token, stat in token_stats.items()
        if stat['entry_count'] >= 5
    }

    shortlist = []
    excluded = Counter()
    for row in train_rows:
        target_tokens = row['tokens']
        reusable_tokens = [token for token in target_tokens if token in reusable_token_set]
        dense_tokens = [token for token in target_tokens if token in dense_token_set]
        if len(reusable_tokens) < min_reusable_tokens:
            excluded['too_few_reusable_tokens'] += 1
            continue
        if len(dense_tokens) < min_dense_tokens:
            excluded['too_few_dense_tokens'] += 1
            continue

        token_support = [token_stats[token] for token in reusable_tokens]
        mean_score = mean(stat['mean_alignment_score'] for stat in token_support)
        if mean_score < min_mean_score:
            excluded['mean_alignment_score_below_threshold'] += 1
            continue

        shortlist.append(
            {
                'sample_id': row['sample_id'],
                'target': row['target'],
                'token_count': len(target_tokens),
                'reusable_token_count': len(reusable_tokens),
                'dense_token_count': len(dense_tokens),
                'reusable_tokens': reusable_tokens,
                'dense_tokens': dense_tokens,
                'mean_reusable_alignment_score': mean_score,
                'mean_reusable_entry_count': mean(stat['entry_count'] for stat in token_support),
                'mean_reusable_sample_count': mean(stat['unique_sample_count'] for stat in token_support),
            }
        )

    shortlist.sort(
        key=lambda item: (
            -item['reusable_token_count'],
            -item['dense_token_count'],
            -item['mean_reusable_alignment_score'],
            -item['mean_reusable_entry_count'],
            item['sample_id'],
        )
    )

    candidate_tokens = Counter()
    for item in shortlist:
        for token in item['reusable_tokens']:
            candidate_tokens[token] += 1

    return {
        'thresholds': {
            'min_reusable_tokens': int(min_reusable_tokens),
            'min_dense_tokens': int(min_dense_tokens),
            'min_mean_score': float(min_mean_score),
            'top_limit': int(top_limit),
        },
        'train_corpus_path': str(train_corpus_path),
        'global_token_overview': {
            'token_count': len(token_stats),
            'reusable_token_count': len(reusable_token_set),
            'dense_token_count': len(dense_token_set),
        },
        'shortlist_overview': {
            'candidate_sample_count': len(shortlist),
            'candidate_sample_coverage': len(shortlist) / max(len(train_rows), 1),
            'candidate_unique_token_count': len(candidate_tokens),
            'excluded_reasons': dict(excluded),
        },
        'top_candidate_samples': shortlist[:top_limit],
        'candidate_top_tokens': candidate_tokens.most_common(50),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Build a stricter reusable/dense train-sample shortlist from dense pseudo-align outputs.'
    )
    parser.add_argument('--summary_json', required=True)
    parser.add_argument('--token_bank_json', default=None)
    parser.add_argument('--train_corpus_csv', default=None)
    parser.add_argument('--min_reusable_tokens', type=int, default=3)
    parser.add_argument('--min_dense_tokens', type=int, default=1)
    parser.add_argument('--min_mean_score', type=float, default=-25.0)
    parser.add_argument('--top_limit', type=int, default=100)
    args = parser.parse_args()

    summary_path = Path(args.summary_json)
    if not summary_path.exists():
        raise SystemExit(f'missing summary: {summary_path}')
    summary = load_summary(summary_path)

    bank_path = Path(args.token_bank_json) if args.token_bank_json else Path(summary['bank_path'])
    if not bank_path.exists():
        raise SystemExit(f'missing token bank: {bank_path}')

    train_corpus_path = (
        Path(args.train_corpus_csv)
        if args.train_corpus_csv
        else Path(summary['data_root']) / 'annotations' / 'manual' / 'train.corpus.csv'
    )
    if not train_corpus_path.exists():
        raise SystemExit(f'missing train corpus csv: {train_corpus_path}')

    bank = TokenBank.load(bank_path)
    analysis = shortlist_samples(
        summary,
        bank,
        train_corpus_path,
        args.min_reusable_tokens,
        args.min_dense_tokens,
        args.min_mean_score,
        args.top_limit,
    )

    result = {
        'ok': True,
        'summary_path': str(summary_path),
        'bank_path': str(bank_path),
        **analysis,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
