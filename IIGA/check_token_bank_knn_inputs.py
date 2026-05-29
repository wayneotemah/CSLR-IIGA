from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter
from pathlib import Path
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def missing_summary_keys(summary: dict[str, Any]) -> list[str]:
    return [key for key in REQUIRED_SUMMARY_KEYS if key not in summary]


def load_lookup(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    with path.open('rb') as handle:
        vocab = pickle.load(handle)
    if not isinstance(vocab, dict):
        raise SystemExit(f'lookup is not a dict: {path}')
    id_to_token = {int(idx): str(token) for token, idx in vocab.items()}
    return {str(token): int(idx) for token, idx in vocab.items()}, id_to_token


def load_corpus_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as handle:
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


def collect_token_types(rows: list[dict[str, Any]]) -> set[str]:
    tokens: set[str] = set()
    for row in rows:
        tokens.update(str(token) for token in row['tokens'])
    return tokens


def validate_bank(
    summary: dict[str, Any],
    bank: TokenBank,
    lookup_token_to_id: dict[str, int],
    lookup_id_to_token: dict[int, str],
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    expected_embedding_dim: int | None,
) -> dict[str, Any]:
    train_ids = {row['sample_id'] for row in train_rows}
    dev_ids = {row['sample_id'] for row in dev_rows}
    test_ids = {row['sample_id'] for row in test_rows}
    train_tokens = collect_token_types(train_rows)
    dev_tokens = collect_token_types(dev_rows)
    test_tokens = collect_token_types(test_rows)

    all_entries = []
    invalid_token_id_rows = []
    mismatched_token_rows = []
    invalid_embedding_rows = []
    dev_id_hits = []
    test_id_hits = []
    non_train_id_hits = []
    token_counter = Counter()

    for token_id in bank.token_ids():
        for entry in bank.query_by_token(token_id):
            all_entries.append(entry)
            token_counter[entry.token_text] += 1
            if entry.sample_id not in train_ids:
                non_train_id_hits.append(entry.sample_id)
                if entry.sample_id in dev_ids:
                    dev_id_hits.append(entry.sample_id)
                if entry.sample_id in test_ids:
                    test_id_hits.append(entry.sample_id)
            lookup_text = lookup_id_to_token.get(int(entry.token_id))
            if lookup_text is None:
                invalid_token_id_rows.append({
                    'sample_id': entry.sample_id,
                    'token_id': int(entry.token_id),
                    'token_text': entry.token_text,
                })
            elif lookup_text != entry.token_text:
                mismatched_token_rows.append({
                    'sample_id': entry.sample_id,
                    'token_id': int(entry.token_id),
                    'token_text': entry.token_text,
                    'lookup_text': lookup_text,
                })
            if len(entry.embedding) == 0:
                invalid_embedding_rows.append({
                    'sample_id': entry.sample_id,
                    'token_id': int(entry.token_id),
                    'reason': 'empty_embedding',
                })

    if not all_entries:
        raise SystemExit('token bank has no entries')

    embedding_dims = sorted({len(entry.embedding) for entry in all_entries})
    if expected_embedding_dim is not None and embedding_dims != [expected_embedding_dim]:
        invalid_embedding_rows.append({
            'reason': 'unexpected_embedding_dim',
            'observed': embedding_dims,
            'expected': expected_embedding_dim,
        })

    actual_summary = bank.summary()
    summary_bank_summary = summary['bank_summary']
    bank_token_types = set(token_counter.keys())
    dev_token_coverage = sorted(dev_tokens & bank_token_types)
    test_token_coverage = sorted(test_tokens & bank_token_types)

    unknown_bank_tokens = sorted(token for token in bank_token_types if token not in lookup_token_to_id)

    return {
        'summary_counts_match': {
            'entries_written_matches_summary_entry_count': int(summary['entries_written']) == int(summary_bank_summary['entry_count']),
            'token_count_matches_summary_token_count': int(summary['token_count']) == int(summary_bank_summary['token_count']),
            'entries_written_matches_loaded_bank': int(summary['entries_written']) == int(actual_summary['entry_count']),
            'token_count_matches_loaded_bank': int(summary['token_count']) == int(actual_summary['token_count']),
        },
        'embedding_dimension_set': embedding_dims,
        'embedding_dimension_expected': expected_embedding_dim,
        'loaded_bank_overview': {
            'entry_count': len(all_entries),
            'token_type_count': len(bank_token_types),
            'unique_train_sample_ids': len({entry.sample_id for entry in all_entries}),
        },
        'leakage': {
            'non_train_sample_id_count': len(set(non_train_id_hits)),
            'dev_sample_id_count': len(set(dev_id_hits)),
            'test_sample_id_count': len(set(test_id_hits)),
            'non_train_sample_ids': sorted(set(non_train_id_hits))[:50],
            'dev_sample_ids': sorted(set(dev_id_hits))[:50],
            'test_sample_ids': sorted(set(test_id_hits))[:50],
        },
        'lookup_alignment': {
            'invalid_token_id_count': len(invalid_token_id_rows),
            'mismatched_token_text_count': len(mismatched_token_rows),
            'unknown_bank_token_count': len(unknown_bank_tokens),
            'invalid_token_id_examples': invalid_token_id_rows[:25],
            'mismatched_token_text_examples': mismatched_token_rows[:25],
            'unknown_bank_token_examples': unknown_bank_tokens[:25],
        },
        'bank_token_coverage': {
            'train_token_type_count': len(train_tokens),
            'dev_token_type_count': len(dev_tokens),
            'test_token_type_count': len(test_tokens),
            'bank_token_type_count': len(bank_token_types),
            'dev_token_coverage_count': len(dev_token_coverage),
            'test_token_coverage_count': len(test_token_coverage),
            'dev_token_coverage_ratio': len(dev_token_coverage) / max(len(dev_tokens), 1),
            'test_token_coverage_ratio': len(test_token_coverage) / max(len(test_tokens), 1),
            'dev_token_coverage_examples': dev_token_coverage[:50],
            'test_token_coverage_examples': test_token_coverage[:50],
        },
        'top_bank_tokens': token_counter.most_common(50),
        'invalid_embedding_count': len(invalid_embedding_rows),
        'invalid_embedding_examples': invalid_embedding_rows[:25],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Validate recovered dense token-bank inputs before token_bank_knn_v1 retrieval or reranking work.'
    )
    parser.add_argument('--summary_json', required=True)
    parser.add_argument('--token_bank_json', default=None)
    parser.add_argument('--prepared_root', default=None)
    parser.add_argument('--lookup_table', default=None)
    parser.add_argument('--expected_embedding_dim', type=int, default=256)
    args = parser.parse_args()

    summary_path = Path(args.summary_json)
    if not summary_path.exists():
        raise SystemExit(f'missing summary json: {summary_path}')
    summary = load_json(summary_path)

    missing = missing_summary_keys(summary)
    if missing:
        raise SystemExit(f'missing summary keys: {missing}')

    bank_path = Path(args.token_bank_json) if args.token_bank_json else Path(summary['bank_path'])
    if not bank_path.exists():
        raise SystemExit(f'missing token bank json: {bank_path}')

    prepared_root = Path(args.prepared_root) if args.prepared_root else Path(summary['data_root'])
    if not prepared_root.exists():
        raise SystemExit(f'missing prepared root: {prepared_root}')

    lookup_path = Path(args.lookup_table) if args.lookup_table else Path(summary['lookup_table'])
    if not lookup_path.exists():
        raise SystemExit(f'missing lookup table: {lookup_path}')

    train_csv = prepared_root / 'annotations' / 'manual' / 'train.corpus.csv'
    dev_csv = prepared_root / 'annotations' / 'manual' / 'dev.corpus.csv'
    test_csv = prepared_root / 'annotations' / 'manual' / 'test.corpus.csv'
    for path in (train_csv, dev_csv, test_csv):
        if not path.exists():
            raise SystemExit(f'missing corpus csv: {path}')

    lookup_token_to_id, lookup_id_to_token = load_lookup(lookup_path)
    bank = TokenBank.load(bank_path)
    train_rows = load_corpus_rows(train_csv)
    dev_rows = load_corpus_rows(dev_csv)
    test_rows = load_corpus_rows(test_csv)

    analysis = validate_bank(
        summary=summary,
        bank=bank,
        lookup_token_to_id=lookup_token_to_id,
        lookup_id_to_token=lookup_id_to_token,
        train_rows=train_rows,
        dev_rows=dev_rows,
        test_rows=test_rows,
        expected_embedding_dim=args.expected_embedding_dim,
    )

    ok = (
        all(analysis['summary_counts_match'].values())
        and analysis['leakage']['non_train_sample_id_count'] == 0
        and analysis['lookup_alignment']['invalid_token_id_count'] == 0
        and analysis['lookup_alignment']['mismatched_token_text_count'] == 0
        and analysis['lookup_alignment']['unknown_bank_token_count'] == 0
        and analysis['invalid_embedding_count'] == 0
    )

    result = {
        'ok': ok,
        'summary_path': str(summary_path),
        'bank_path': str(bank_path),
        'prepared_root': str(prepared_root),
        'lookup_table': str(lookup_path),
        **analysis,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
