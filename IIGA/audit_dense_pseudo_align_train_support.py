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
            rows.append({
                'sample_id': str(row[0]),
                'target': target,
                'tokens': target.split(),
            })
    return rows


def flatten_entries(bank: TokenBank) -> list[TokenEntry]:
    entries: list[TokenEntry] = []
    for token_id in bank.token_ids():
        entries.extend(bank.query_by_token(token_id))
    return entries


def summarize_sample_group(
    sample_id: str,
    target: str,
    entries: list[TokenEntry],
    reusable_token_set: set[str],
    dense_token_set: set[str],
) -> dict[str, Any]:
    token_counter = Counter(entry.token_text for entry in entries)
    span_lengths = [int(entry.span_length) for entry in entries]
    scores = [float(entry.score) for entry in entries]
    target_tokens = target.split()
    reusable_tokens = sorted(token for token in target_tokens if token in reusable_token_set)
    dense_tokens = sorted(token for token in target_tokens if token in dense_token_set)
    return {
        'sample_id': sample_id,
        'target': target,
        'token_entry_count': len(entries),
        'unique_token_count': len(token_counter),
        'reusable_token_count': len(reusable_tokens),
        'dense_token_count_ge_5': len(dense_tokens),
        'reusable_tokens': reusable_tokens[:25],
        'dense_tokens': dense_tokens[:25],
        'mean_span_length': mean(span_lengths),
        'max_span_length': max(span_lengths),
        'mean_alignment_score': mean(scores),
        'top_tokens': token_counter.most_common(10),
    }


def audit_train_support(
    summary: dict[str, Any],
    bank: TokenBank,
    train_corpus_path: Path,
    min_token_entries: int,
    min_dense_entries: int,
) -> dict[str, Any]:
    train_rows = load_train_rows(train_corpus_path)
    sample_to_target = {row['sample_id']: row['target'] for row in train_rows}
    sample_to_tokens = {row['sample_id']: row['tokens'] for row in train_rows}

    entries = flatten_entries(bank)
    by_sample: dict[str, list[TokenEntry]] = defaultdict(list)
    for entry in entries:
        by_sample[entry.sample_id].append(entry)

    by_token: dict[str, list[TokenEntry]] = defaultdict(list)
    for entry in entries:
        by_token[entry.token_text].append(entry)

    reusable_token_set = {
        token_text
        for token_text, token_entries in by_token.items()
        if len(token_entries) >= min_token_entries
    }
    dense_token_set = {
        token_text
        for token_text, token_entries in by_token.items()
        if len(token_entries) >= 5
    }

    sample_summaries = []
    for sample_id, sample_entries in by_sample.items():
        sample_entries.sort(key=lambda entry: (entry.span_start, entry.span_end, entry.token_text))
        sample_summaries.append(
            summarize_sample_group(
                sample_id,
                sample_to_target.get(sample_id, ''),
                sample_entries,
                reusable_token_set,
                dense_token_set,
            )
        )
    sample_summaries.sort(key=lambda item: (-item['reusable_token_count'], -item['dense_token_count_ge_5'], -item['token_entry_count'], item['sample_id']))

    candidate_samples = [
        item for item in sample_summaries
        if item['reusable_token_count'] >= min_token_entries or item['dense_token_count_ge_5'] >= min_dense_entries
    ]
    candidate_sample_ids = {item['sample_id'] for item in candidate_samples}

    candidate_token_counter = Counter()
    for sample_id in candidate_sample_ids:
        for token in sample_to_tokens.get(sample_id, []):
            candidate_token_counter[token] += 1

    return {
        'thresholds': {
            'min_token_entries': int(min_token_entries),
            'min_dense_entries': int(min_dense_entries),
        },
        'train_corpus_path': str(train_corpus_path),
        'train_overview': {
            'sample_count': len(train_rows),
            'bank_sample_count': len(by_sample),
            'entry_count': len(entries),
        },
        'candidate_overview': {
            'candidate_sample_count': len(candidate_samples),
            'candidate_sample_coverage': len(candidate_samples) / max(len(train_rows), 1),
            'candidate_unique_token_count': len(candidate_token_counter),
        },
        'candidate_samples': candidate_samples[:100],
        'candidate_top_tokens': candidate_token_counter.most_common(50),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Audit dense pseudo-align outputs at the train-sample level using reusable/dense-token thresholds.')
    parser.add_argument('--summary_json', required=True)
    parser.add_argument('--token_bank_json', default=None)
    parser.add_argument('--train_corpus_csv', default=None)
    parser.add_argument('--min_token_entries', type=int, default=2)
    parser.add_argument('--min_dense_entries', type=int, default=1)
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
    analysis = audit_train_support(summary, bank, train_corpus_path, args.min_token_entries, args.min_dense_entries)
    result = {
        'ok': True,
        'summary_path': str(summary_path),
        'bank_path': str(bank_path),
        **analysis,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
