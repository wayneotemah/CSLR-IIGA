from __future__ import annotations

import argparse
import ast
import json
import pickle
from pathlib import Path
import sys
from typing import Any


def load_anchor_helpers_from_train(train_py_path: Path) -> tuple[Any, Any, Any, Any]:
    source = train_py_path.read_text(encoding='utf-8')
    module = ast.parse(source, filename=str(train_py_path))
    needed = {'read_corpus_ids', '_first_present', '_collect_anchor_entries', 'load_anchor_audit'}
    selected_nodes = [
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name in needed
    ]
    temp_module = ast.Module(body=selected_nodes, type_ignores=[])
    ast.fix_missing_locations(temp_module)
    namespace: dict[str, Any] = {
        'json': json,
        'os': __import__('os'),
    }
    exec(compile(temp_module, str(train_py_path), 'exec'), namespace, namespace)
    return (
        namespace['read_corpus_ids'],
        namespace['_first_present'],
        namespace['_collect_anchor_entries'],
        namespace['load_anchor_audit'],
    )


def load_lookup(path: Path) -> dict[str, int]:
    with path.open('rb') as handle:
        data = pickle.load(handle)
    return {str(key): int(value) for key, value in data.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Dry-run the real train.py load_anchor_audit path against a dense pseudo-align subset artifact.'
    )
    parser.add_argument('--artifact_json', required=True)
    parser.add_argument('--lookup_pkl', required=True)
    parser.add_argument('--train_corpus_csv', required=True)
    parser.add_argument('--valid_corpus_csv', required=True)
    parser.add_argument('--test_corpus_csv', required=True)
    parser.add_argument('--train_py', default=str(Path(__file__).with_name('train.py')))
    args = parser.parse_args()

    artifact_path = Path(args.artifact_json)
    lookup_path = Path(args.lookup_pkl)
    train_csv = Path(args.train_corpus_csv)
    valid_csv = Path(args.valid_corpus_csv)
    test_csv = Path(args.test_corpus_csv)
    train_py_path = Path(args.train_py)

    if not artifact_path.exists():
        raise SystemExit(f'missing artifact json: {artifact_path}')
    if not lookup_path.exists():
        raise SystemExit(f'missing lookup pkl: {lookup_path}')
    if not train_csv.exists():
        raise SystemExit(f'missing train corpus csv: {train_csv}')
    if not train_py_path.exists():
        raise SystemExit(f'missing train.py path: {train_py_path}')

    artifact = json.loads(artifact_path.read_text(encoding='utf-8'))
    word_to_id = load_lookup(lookup_path)
    _, _, _, load_anchor_audit = load_anchor_helpers_from_train(train_py_path)
    anchor_map = load_anchor_audit(
        anchor_audit_json=str(artifact_path),
        word_to_id=word_to_id,
        vocab_size=len(word_to_id),
        train_csv=str(train_csv),
        valid_csv=str(valid_csv),
        test_csv=str(test_csv),
    )

    total_anchors = sum(len(sample_anchors) for sample_anchors in anchor_map.values())
    unique_tokens = sorted({token_id for sample_anchors in anchor_map.values() for _, token_id in sample_anchors})
    sample_anchor_examples = {
        sample_id: len(sample_anchors)
        for sample_id, sample_anchors in sorted(anchor_map.items())[:10]
    }
    sample_anchor_map = {
        sample_id: sample_anchors
        for sample_id, sample_anchors in sorted(anchor_map.items())[:10]
    }

    artifact_overview = artifact.get('artifact_overview', {})
    payload = {
        'ok': True,
        'artifact_json': str(artifact_path),
        'lookup_pkl': str(lookup_path),
        'loaded_anchor_count': total_anchors,
        'unique_sample_count': len(anchor_map),
        'unique_token_count': len(unique_tokens),
        'loaded_counts_match_artifact': {
            'eligible_anchor_count': total_anchors == artifact_overview.get('eligible_anchor_count'),
            'unique_anchor_sample_count': len(anchor_map) == artifact_overview.get('unique_anchor_sample_count'),
            'unique_anchor_token_count': len(unique_tokens) == artifact_overview.get('unique_anchor_token_count'),
        },
        'min_anchors_in_one_sample': min((len(v) for v in anchor_map.values()), default=0),
        'max_anchors_in_one_sample': max((len(v) for v in anchor_map.values()), default=0),
        'sample_anchor_examples': sample_anchor_examples,
        'sample_anchor_map': sample_anchor_map,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
