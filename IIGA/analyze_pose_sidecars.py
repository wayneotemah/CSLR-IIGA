import argparse
import json
from pathlib import Path

import numpy as np


def summarize_split(split_root: Path) -> dict:
    sample_dirs = sorted([p for p in split_root.glob('*/*') if p.is_dir()])
    sample_summaries = []
    frame_counts = []
    pose_detected = []
    left_detected = []
    right_detected = []

    for sample_dir in sample_dirs:
        metadata_path = sample_dir / 'metadata.json'
        landmark_path = sample_dir / 'landmarks.npz'
        if not metadata_path.exists() or not landmark_path.exists():
            continue

        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        arrays = np.load(landmark_path, allow_pickle=True)

        summary = {
            'sample_id': metadata['sample_id'],
            'frame_count': int(metadata['frame_count']),
            'pose_detected_frames': int(metadata['pose_detected_frames']),
            'left_hand_detected_frames': int(metadata['left_hand_detected_frames']),
            'right_hand_detected_frames': int(metadata['right_hand_detected_frames']),
            'shapes': {name: list(arrays[name].shape) for name in arrays.files},
        }
        sample_summaries.append(summary)

        frame_counts.append(summary['frame_count'])
        pose_detected.append(summary['pose_detected_frames'])
        left_detected.append(summary['left_hand_detected_frames'])
        right_detected.append(summary['right_hand_detected_frames'])

    if not sample_summaries:
        return {
            'sample_count': 0,
            'samples': [],
        }

    def mean_ratio(detected_list, frames_list):
        return float(np.mean([
            detected / frames if frames else 0.0
            for detected, frames in zip(detected_list, frames_list)
        ]))

    return {
        'sample_count': len(sample_summaries),
        'frame_count_min': int(min(frame_counts)),
        'frame_count_max': int(max(frame_counts)),
        'frame_count_mean': float(np.mean(frame_counts)),
        'pose_detection_ratio_mean': mean_ratio(pose_detected, frame_counts),
        'left_hand_detection_ratio_mean': mean_ratio(left_detected, frame_counts),
        'right_hand_detection_ratio_mean': mean_ratio(right_detected, frame_counts),
        'samples': sample_summaries,
    }


def main():
    parser = argparse.ArgumentParser(description='Summarize generated pose sidecar artifacts.')
    parser.add_argument('--pose_root', required=True, help='Root directory containing pose_landmarks/{train,dev,test}')
    parser.add_argument('--output_json', default=None, help='Optional path to write summary JSON')
    args = parser.parse_args()

    pose_root = Path(args.pose_root)
    summary = {
        'pose_root': str(pose_root),
        'train': summarize_split(pose_root / 'train'),
        'dev': summarize_split(pose_root / 'dev'),
        'test': summarize_split(pose_root / 'test'),
    }

    text = json.dumps(summary, indent=2)
    print(text)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
