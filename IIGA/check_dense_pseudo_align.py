from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from tools.token_bank import TokenBank


REQUIRED_SUMMARY_KEYS = [
    'checkpoint',
    'data_root',
    'lookup_table',
    'dataset_size',
    'entries_written',
    'token_count',
    'top_tokens',
    'alignment_failures',
    'bank_path',
    'bank_summary',
]


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def validate_summary_shape(summary: dict[str, Any]) -> list[str]:
    return [key for key in REQUIRED_SUMMARY_KEYS if key not in summary]


def analyze_bank(summary: dict[str, Any], bank: TokenBank) -> dict[str, Any]:
    all_entries = []
    for token_id in bank.token_ids():
        all_entries.extend(bank.query_by_token(token_id))

    if not all_entries:
        raise SystemExit('token bank has no entries')

    embedding_dims = sorted({len(entry.embedding) for entry in all_entries})
    invalid_spans = [
        {
            'sample_id': entry.sample_id,
            'token_id': entry.token_id,
            'span_start': entry.span_start,
            'span_end': entry.span_end,
            'span_length': entry.span_length,
        }
        for entry in all_entries
        if entry.span_start < 0 or entry.span_end < entry.span_start or entry.span_length != (entry.span_end - entry.span_start + 1)
    ]

    entries_per_sample = Counter(entry.sample_id for entry in all_entries)
    entries_per_token = Counter(entry.token_text for entry in all_entries)
    singleton_tokens = sorted(token for token, count in entries_per_token.items() if count == 1)
    reusable_tokens = sorted(token for token, count in entries_per_token.items() if count >= 2)
    dense_tokens = sorted(token for token, count in entries_per_token.items() if count >= 5)

    top_summary_pairs = [(str(token), int(count)) for token, count in summary['top_tokens']]
    top_actual_pairs = sorted(entries_per_token.items(), key=lambda item: (-item[1], item[0]))[: len(top_summary_pairs)]

    actual_bank_summary = bank.summary()
    summary_bank_summary = summary['bank_summary']

    return {
        'summary_counts_match_bank_summary': {
            'entries_written_matches_summary_entry_count': int(summary['entries_written']) == int(summary_bank_summary['entry_count']),
            'token_count_matches_summary_token_count': int(summary['token_count']) == int(summary_bank_summary['token_count']),
            'entries_written_matches_loaded_bank': int(summary['entries_written']) == int(actual_bank_summary['entry_count']),
            'token_count_matches_loaded_bank': int(summary['token_count']) == int(actual_bank_summary['token_count']),
        },
        'embedding_dimension_set': embedding_dims,
        'invalid_span_count': len(invalid_spans),
        'invalid_span_examples': invalid_spans[:20],
        'sample_coverage': {
            'unique_sample_ids': len(entries_per_sample),
            'max_entries_in_one_sample': max(entries_per_sample.values()),
            'min_entries_in_one_sample': min(entries_per_sample.values()),
            'mean_entries_per_sample': mean(entries_per_sample.values()),
        },
        'token_distribution': {
            'singleton_token_count': len(singleton_tokens),
            'reusable_token_count': len(reusable_tokens),
            'dense_token_count_ge_5': len(dense_tokens),
            'singleton_token_examples': singleton_tokens[:25],
            'dense_token_examples': dense_tokens[:25],
        },
        'top_token_consistency': {
            'summary_top_tokens': top_summary_pairs[:25],
            'actual_top_tokens': top_actual_pairs[:25],
            'matches_exactly': top_summary_pairs == top_actual_pairs,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Validate dense pseudo-alignment outputs and summarize token-bank quality.')
    parser.add_argument('--summary_json', required=True)
    parser.add_argument('--token_bank_json', default=None)
    args = parser.parse_args()

    summary_path = Path(args.summary_json)
    if not summary_path.exists():
        raise SystemExit(f'missing summary: {summary_path}')

    summary = load_summary(summary_path)
    missing = validate_summary_shape(summary)
    if missing:
        raise SystemExit(f'missing summary keys: {missing}')

    bank_path = Path(args.token_bank_json) if args.token_bank_json else Path(summary['bank_path'])
    if not bank_path.exists():
        raise SystemExit(f'missing token bank: {bank_path}')

    bank = TokenBank.load(bank_path)
    analysis = analyze_bank(summary, bank)

    result = {
        'ok': analysis['invalid_span_count'] == 0,
        'summary_path': str(summary_path),
        'bank_path': str(bank_path),
        'dataset_size': int(summary['dataset_size']),
        'entries_written': int(summary['entries_written']),
        'token_count': int(summary['token_count']),
        'alignment_failures_count': len(summary['alignment_failures']),
        **analysis,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
