from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def build_subset(
    audit_data: dict[str, Any],
    top_limit: int,
) -> dict[str, Any]:
    candidates = list(audit_data.get('top_integration_candidates', []))
    if not candidates:
        raise SystemExit('integration audit has no top_integration_candidates')

    selected = candidates[:top_limit]
    token_set = set()
    family_keys = []
    for item in selected:
        token_set.update(item.get('reusable_tokens', []))
        family_keys.append(item.get('cluster_key', ''))

    return {
        'source_thresholds': audit_data.get('thresholds', {}),
        'source_representative_path': audit_data.get('representative_path'),
        'integration_subset_overview': {
            'selected_sample_count': len(selected),
            'selected_unique_token_count': len(token_set),
            'selected_cluster_count': len({k for k in family_keys if k}),
        },
        'selected_samples': selected,
        'selected_tokens': sorted(token_set),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Freeze a stable first dense pseudo-align integration subset from the representative shortlist audit.'
    )
    parser.add_argument('--integration_audit_json', required=True)
    parser.add_argument('--output_json', required=True)
    parser.add_argument('--top_limit', type=int, default=6)
    args = parser.parse_args()

    audit_path = Path(args.integration_audit_json)
    if not audit_path.exists():
        raise SystemExit(f'missing integration audit json: {audit_path}')

    audit_data = load_json(audit_path)
    subset = build_subset(audit_data, args.top_limit)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'ok': True,
        'integration_audit_path': str(audit_path),
        'output_path': str(output_path),
        **subset,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
