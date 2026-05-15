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


def load_train_targets(corpus_path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with corpus_path.open('r', encoding='utf-8') as handle:
        reader = csv.reader(handle, delimiter='|')
        for row in reader:
            if not row:
                continue
            sample_id = str(row[0])
            target = str(row[1]) if len(row) > 1 else ''
            mapping[sample_id] = target
    return mapping


def flatten_entries(bank: TokenBank) -> list[TokenEntry]:
    entries: list[TokenEntry] = []
    for token_id in bank.token_ids():
        entries.extend(bank.query_by_token(token_id))
    return entries


def summarize_token_group(entries: list[TokenEntry], sample_to_target: dict[str, str]) -> dict[str, Any]:
    sample_ids = sorted({entry.sample_id for entry in entries})
    contexts = sorted({sample_to_target.get(entry.sample_id, '') for entry in entries})
    span_lengths = [int(entry.span_length) for entry in entries]
    scores = [float(entry.score) for entry in entries]
    return {
        'token_id': int(entries[0].token_id),
        'token_text': entries[0].token_text,
        'entry_count': len(entries),
        'unique_sample_count': len(sample_ids),
        'unique_context_count': len(contexts),
        'mean_span_length': mean(span_lengths),
        'min_span_length': min(span_lengths),
        'max_span_length': max(span_lengths),
        'mean_alignment_score': mean(scores),
        'sample_examples': sample_ids[:10],
        'context_examples': contexts[:10],
    }


def analyze_dense_align(summary: dict[str, Any], bank: TokenBank, train_corpus_path: Path, min_entries: int, min_samples: int, min_contexts: int) -> dict[str, Any]:
    sample_to_target = load_train_targets(train_corpus_path)
    all_entries = flatten_entries(bank)
    if not all_entries:
        raise SystemExit('token bank has no entries')

    token_groups: dict[int, list[TokenEntry]] = defaultdict(list)
    for entry in all_entries:
        token_groups[int(entry.token_id)].append(entry)

    token_summaries = [
        summarize_token_group(entries, sample_to_target)
        for _, entries in sorted(token_groups.items(), key=lambda item: item[0])
    ]
    token_summaries.sort(key=lambda item: (-item['entry_count'], item['token_text']))

    reusable = [
        item for item in token_summaries
        if item['entry_count'] >= min_entries
        and item['unique_sample_count'] >= min_samples
        and item['unique_context_count'] >= min_contexts
    ]
    dense = [item for item in reusable if item['entry_count'] >= 5]

    candidate_token_ids = {int(item['token_id']) for item in reusable}
    candidate_entries = [entry for entry in all_entries if int(entry.token_id) in candidate_token_ids]
    candidate_sample_ids = sorted({entry.sample_id for entry in candidate_entries})

    return {
        'thresholds': {
            'min_entries': int(min_entries),
            'min_samples': int(min_samples),
            'min_contexts': int(min_contexts),
        },
        'train_corpus_path': str(train_corpus_path),
        'bank_overview': {
            'entry_count': len(all_entries),
            'token_count': len(token_groups),
            'unique_sample_count': len({entry.sample_id for entry in all_entries}),
        },
        'candidate_overview': {
            'reusable_token_count': len(reusable),
            'dense_token_count_ge_5': len(dense),
            'candidate_sample_count': len(candidate_sample_ids),
            'candidate_sample_coverage': len(candidate_sample_ids) / max(len(sample_to_target), 1),
        },
        'reusable_token_examples': reusable[:50],
        'dense_token_examples': dense[:50],
        'candidate_token_ids': sorted(candidate_token_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Audit dense pseudo-alignment outputs and surface reusable token subsets.')
    parser.add_argument('--summary_json', required=True)
    parser.add_argument('--token_bank_json', default=None)
    parser.add_argument('--train_corpus_csv', default=None)
    parser.add_argument('--min_entries', type=int, default=2)
    parser.add_argument('--min_samples', type=int, default=2)
    parser.add_argument('--min_contexts', type=int, default=2)
    args = parser.parse_args()

    summary_path = Path(args.summary_json)
    if not summary_path.exists():
        raise SystemExit(f'missing summary: {summary_path}')
    summary = load_summary(summary_path)
    bank_path = Path(args.token_bank_json) if args.token_bank_json else Path(summary['bank_path'])
    if not bank_path.exists():
        raise SystemExit(f'missing token bank: {bank_path}')

    train_corpus_path = Path(args.train_corpus_csv) if args.train_corpus_csv else Path(summary['data_root']) / 'annotations' / 'manual' / 'train.corpus.csv'
    if not train_corpus_path.exists():
        raise SystemExit(f'missing train corpus csv: {train_corpus_path}')

    bank = TokenBank.load(bank_path)
    analysis = analyze_dense_align(summary, bank, train_corpus_path, args.min_entries, args.min_samples, args.min_contexts)
    result = {
        'ok': True,
        'summary_path': str(summary_path),
        'bank_path': str(bank_path),
        **analysis,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
