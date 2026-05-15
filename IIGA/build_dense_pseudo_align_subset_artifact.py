from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tools.token_bank import TokenBank, TokenEntry


def load_json(path: Path) -> dict[str, Any]:
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


def flatten_entries(bank: TokenBank) -> list[TokenEntry]:
    entries: list[TokenEntry] = []
    for token_id in bank.token_ids():
        entries.extend(bank.query_by_token(token_id))
    return entries


def build_subset_anchor_artifact(
    subset_data: dict[str, Any],
    bank: TokenBank,
    train_rows: dict[str, dict[str, Any]],
    max_entries_per_token_per_sample: int,
) -> dict[str, Any]:
    selected_samples = list(subset_data.get('selected_samples', []))
    if not selected_samples:
        raise SystemExit('subset artifact has no selected_samples')

    selected_sample_ids = {str(item['sample_id']) for item in selected_samples}
    selected_token_set = set(subset_data.get('selected_tokens', []))
    if not selected_token_set:
        raise SystemExit('subset artifact has no selected_tokens')

    entries = flatten_entries(bank)
    grouped: dict[tuple[str, str], list[TokenEntry]] = defaultdict(list)
    for entry in entries:
        if entry.sample_id not in selected_sample_ids:
            continue
        if entry.token_text not in selected_token_set:
            continue
        grouped[(entry.sample_id, entry.token_text)].append(entry)

    eligible_anchors: list[dict[str, Any]] = []
    sample_summaries: list[dict[str, Any]] = []
    token_counter = Counter()
    total_entries_considered = 0
    missing_sample_token_pairs: list[dict[str, Any]] = []

    for sample in selected_samples:
        sample_id = str(sample['sample_id'])
        target = str(sample['target'])
        target_tokens = list(train_rows.get(sample_id, {'tokens': target.split()})['tokens'])
        token_anchor_counts: Counter[str] = Counter()
        for token_text in target_tokens:
            if token_text not in selected_token_set:
                continue
            token_entries = sorted(
                grouped.get((sample_id, token_text), []),
                key=lambda item: (
                    -float(item.score),
                    int(item.span_length),
                    int(item.span_start),
                ),
            )
            if not token_entries:
                missing_sample_token_pairs.append(
                    {
                        'sample_id': sample_id,
                        'target': target,
                        'token_text': token_text,
                        'reason': 'no_dense_align_entry',
                    }
                )
                continue
            chosen_entries = token_entries[:max_entries_per_token_per_sample]
            total_entries_considered += len(chosen_entries)
            for entry in chosen_entries:
                midpoint = int((entry.span_start + entry.span_end) // 2)
                eligible_anchors.append(
                    {
                        'eligible': True,
                        'sample_id': sample_id,
                        'target': target,
                        'token_text': entry.token_text,
                        'token_id': int(entry.token_id),
                        'frame_idx': midpoint,
                        'span_start': int(entry.span_start),
                        'span_end': int(entry.span_end),
                        'span_length': int(entry.span_length),
                        'alignment_score': float(entry.score),
                        'source': 'dense_pseudo_align_subset',
                        'metadata': {
                            **dict(entry.metadata),
                            'cluster_key': sample.get('cluster_key'),
                            'cluster_size': sample.get('cluster_size'),
                            'mean_reusable_alignment_score': sample.get('mean_reusable_alignment_score'),
                        },
                    }
                )
                token_anchor_counts[entry.token_text] += 1
                token_counter[entry.token_text] += 1
        sample_summaries.append(
            {
                'sample_id': sample_id,
                'target': target,
                'cluster_key': sample.get('cluster_key'),
                'cluster_size': sample.get('cluster_size'),
                'target_token_count': len(target_tokens),
                'selected_token_count': len([token for token in target_tokens if token in selected_token_set]),
                'eligible_anchor_count': int(sum(token_anchor_counts.values())),
                'per_token_anchor_counts': dict(token_anchor_counts),
            }
        )

    sample_summaries.sort(key=lambda item: (-item['eligible_anchor_count'], item['sample_id']))
    eligible_anchors.sort(key=lambda item: (item['sample_id'], item['frame_idx'], item['token_id']))
    unique_sample_ids = sorted({item['sample_id'] for item in eligible_anchors})
    unique_token_ids = sorted({int(item['token_id']) for item in eligible_anchors})

    return {
        'source_subset_path': subset_data.get('output_path'),
        'source_thresholds': subset_data.get('source_thresholds', {}),
        'integration_subset_overview': subset_data.get('integration_subset_overview', {}),
        'artifact_overview': {
            'selected_sample_count': len(selected_sample_ids),
            'selected_token_count': len(selected_token_set),
            'eligible_anchor_count': len(eligible_anchors),
            'unique_anchor_sample_count': len(unique_sample_ids),
            'unique_anchor_token_count': len(unique_token_ids),
            'max_entries_per_token_per_sample': int(max_entries_per_token_per_sample),
            'entries_considered_for_anchors': int(total_entries_considered),
            'missing_sample_token_pair_count': len(missing_sample_token_pairs),
        },
        'selected_samples': selected_samples,
        'selected_tokens': sorted(selected_token_set),
        'sample_summaries': sample_summaries,
        'top_anchor_tokens': token_counter.most_common(50),
        'missing_sample_token_pairs': missing_sample_token_pairs[:100],
        'eligible_anchors': eligible_anchors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Build a train-only dense pseudo-align subset artifact in the existing anchor_audit_json shape.'
    )
    parser.add_argument('--subset_json', required=True)
    parser.add_argument('--token_bank_json', required=True)
    parser.add_argument('--train_corpus_csv', required=True)
    parser.add_argument('--output_json', required=True)
    parser.add_argument('--max_entries_per_token_per_sample', type=int, default=1)
    args = parser.parse_args()

    subset_path = Path(args.subset_json)
    bank_path = Path(args.token_bank_json)
    train_corpus_path = Path(args.train_corpus_csv)
    output_path = Path(args.output_json)

    if not subset_path.exists():
        raise SystemExit(f'missing subset json: {subset_path}')
    if not bank_path.exists():
        raise SystemExit(f'missing token bank json: {bank_path}')
    if not train_corpus_path.exists():
        raise SystemExit(f'missing train corpus csv: {train_corpus_path}')

    subset_data = load_json(subset_path)
    bank = TokenBank.load(bank_path)
    train_rows = load_train_rows(train_corpus_path)

    artifact = build_subset_anchor_artifact(
        subset_data=subset_data,
        bank=bank,
        train_rows=train_rows,
        max_entries_per_token_per_sample=args.max_entries_per_token_per_sample,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'ok': True,
        'subset_json': str(subset_path),
        'token_bank_json': str(bank_path),
        'train_corpus_csv': str(train_corpus_path),
        'output_json': str(output_path),
        **artifact,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
