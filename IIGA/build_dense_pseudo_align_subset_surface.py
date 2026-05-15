from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def read_corpus_rows(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open('r', encoding='utf-8') as handle:
        reader = csv.reader(handle, delimiter='|')
        for row in reader:
            if not row:
                continue
            sample_id = str(row[0])
            target = str(row[1]) if len(row) > 1 else ''
            rows.append(
                {
                    'sample_id': sample_id,
                    'target': target,
                    'tokens': target.split(),
                    'raw_row': row,
                }
            )
    return rows


def write_corpus_rows(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle, delimiter='|')
        for row in rows:
            writer.writerow(row['raw_row'])


def build_surface(
    subset_artifact: dict[str, Any],
    prepared_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    selected_samples = list(subset_artifact.get('selected_samples', []))
    if not selected_samples:
        raise SystemExit('subset artifact has no selected_samples')

    selected_sample_ids = {str(item['sample_id']) for item in selected_samples}

    train_csv = prepared_root / 'annotations' / 'manual' / 'train.corpus.csv'
    valid_csv = prepared_root / 'annotations' / 'manual' / 'dev.corpus.csv'
    test_csv = prepared_root / 'annotations' / 'manual' / 'test.corpus.csv'

    train_rows = read_corpus_rows(train_csv)
    valid_rows = read_corpus_rows(valid_csv)
    test_rows = read_corpus_rows(test_csv)

    train_by_id = {row['sample_id']: row for row in train_rows}
    missing_ids = sorted(sample_id for sample_id in selected_sample_ids if sample_id not in train_by_id)
    if missing_ids:
        raise SystemExit(f'missing selected sample ids in train corpus: {missing_ids}')

    subset_train_rows = [train_by_id[str(item['sample_id'])] for item in selected_samples]

    output_root.mkdir(parents=True, exist_ok=True)
    write_corpus_rows(subset_train_rows, output_root / 'annotations' / 'manual' / 'train.corpus.csv')
    write_corpus_rows(valid_rows, output_root / 'annotations' / 'manual' / 'dev.corpus.csv')
    write_corpus_rows(test_rows, output_root / 'annotations' / 'manual' / 'test.corpus.csv')

    payload = {
        'ok': True,
        'source_subset_artifact': subset_artifact.get('output_json') or subset_artifact.get('subset_json'),
        'source_prepared_root': str(prepared_root),
        'output_root': str(output_root),
        'selected_sample_ids': sorted(selected_sample_ids),
        'selected_targets': [row['target'] for row in subset_train_rows],
        'counts': {
            'train': len(subset_train_rows),
            'dev': len(valid_rows),
            'test': len(test_rows),
        },
    }
    (output_root / 'surface_manifest.json').write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Build a focused subset-only surface from the stable dense pseudo-align integration subset.'
    )
    parser.add_argument('--subset_artifact_json', required=True)
    parser.add_argument('--prepared_root', required=True)
    parser.add_argument('--output_root', required=True)
    args = parser.parse_args()

    subset_artifact_path = Path(args.subset_artifact_json)
    prepared_root = Path(args.prepared_root)
    output_root = Path(args.output_root)

    if not subset_artifact_path.exists():
        raise SystemExit(f'missing subset artifact json: {subset_artifact_path}')
    if not prepared_root.exists():
        raise SystemExit(f'missing prepared root: {prepared_root}')

    payload = build_surface(
        subset_artifact=load_json(subset_artifact_path),
        prepared_root=prepared_root,
        output_root=output_root,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
