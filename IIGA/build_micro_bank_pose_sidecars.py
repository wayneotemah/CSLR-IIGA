"""Build MediaPipe pose/hand landmark sidecars for a prepared gate surface.

This is a thin, CPU-only sidecar builder for the pose-only (pose_fusion_mode=replace)
branch. It reads the already-extracted frame PNGs from a prepared root's
features/fullFrame-210x260px/<split>/<sample_id>/1/ dirs and writes
pose_landmarks/<split>/<sample_id>/1/landmarks.npz + metadata.json sidecars
that match the on-disk contract expected by IIGA/dataloader.py.

It exists separately from train_from_json.py because the gate surfaces are
symlinked views over the normalized prepared root: re-running train_from_json
would re-extract from video for the entire dataset, while this builder only
processes the exact sample IDs in the gate surface_manifest.json.
"""
import argparse
import json
import sys
from pathlib import Path

#Resolve the IIGA module dir so we can reuse the existing pose helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_from_json import build_pose_extractor, write_pose_landmarks  # noqa: E402


def collect_sample_ids(manifest):
    ids = {}
    for split_key, rows_key in (('train', 'selected_train_rows'),
                                ('dev', 'selected_dev_rows'),
                                ('test', 'selected_test_rows')):
        rows = manifest.get(rows_key, [])
        ids[split_key] = [str(r['sample_id']) for r in rows]
    return ids


def build_sidecars(prepared_root, sample_ids_by_split, pose_asset):
    prepared_root = Path(prepared_root)
    feature_root = prepared_root / 'features' / 'fullFrame-210x260px'
    pose_root = prepared_root / 'pose_landmarks'

    if not feature_root.is_dir():
        raise FileNotFoundError(f'Feature root not found: {feature_root}')

    pose_extractor = build_pose_extractor(True, pose_model_asset_path=str(pose_asset))

    summary = {}
    for split, sample_ids in sample_ids_by_split.items():
        split_feature_dir = feature_root / split
        if not split_feature_dir.is_dir():
            raise FileNotFoundError(f'Split feature dir not found: {split_feature_dir}')
        split_pose_root = pose_root / split
        split_pose_root.mkdir(parents=True, exist_ok=True)

        split_summary = {'complete': 0, 'missing': 0, 'frames': 0,
                         'pose_detected': 0, 'left_hand_detected': 0,
                         'right_hand_detected': 0}
        for sample_id in sample_ids:
            frame_dir = split_feature_dir / sample_id / '1'
            if not frame_dir.is_dir():
                print(f'  [WARN] missing frame dir for {split}/{sample_id}: {frame_dir}')
                split_summary['missing'] += 1
                continue
            pose_dir = split_pose_root / sample_id / '1'
            metadata = write_pose_landmarks(frame_dir, pose_dir, pose_extractor, sample_id)
            if metadata is None:
                print(f'  [WARN] no metadata returned for {split}/{sample_id}')
                split_summary['missing'] += 1
                continue
            split_summary['complete'] += 1
            split_summary['frames'] += metadata['frame_count']
            split_summary['pose_detected'] += metadata['pose_detected_frames']
            split_summary['left_hand_detected'] += metadata['left_hand_detected_frames']
            split_summary['right_hand_detected'] += metadata['right_hand_detected_frames']
            print(f'  {split}/{sample_id}: frames={metadata["frame_count"]} '
                  f'pose={metadata["pose_detected_frames"]} '
                  f'L={metadata["left_hand_detected_frames"]} '
                  f'R={metadata["right_hand_detected_frames"]}')
        summary[split] = split_summary

    summary_path = pose_root / 'sidecar_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'\nsidecar summary written to {summary_path}')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared_root', required=True,
                        help='Prepared gate surface root (must contain surface_manifest.json and features/fullFrame-210x260px/).')
    parser.add_argument('--pose_model_asset_path', required=True,
                        help='Path to MediaPipe holistic_landmarker.task asset.')
    args = parser.parse_args()

    prepared_root = Path(args.prepared_root)
    manifest_path = prepared_root / 'surface_manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'surface_manifest.json not found in {prepared_root}')

    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    sample_ids_by_split = collect_sample_ids(manifest)

    total = sum(len(v) for v in sample_ids_by_split.values())
    print(f'Building pose sidecars for {total} samples across '
          f'{", ".join(f"{k}={len(v)}" for k, v in sample_ids_by_split.items())}')

    summary = build_sidecars(prepared_root, sample_ids_by_split, args.pose_model_asset_path)
    print('\nFinal summary:')
    for split, s in summary.items():
        print(f'  {split}: complete={s["complete"]} missing={s["missing"]} '
              f'frames={s["frames"]} pose={s["pose_detected"]} '
              f'L={s["left_hand_detected"]} R={s["right_hand_detected"]}')


if __name__ == '__main__':
    main()
