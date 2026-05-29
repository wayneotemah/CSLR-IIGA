import argparse
import csv
import gzip
import importlib
import json
import pickle
import re
import shutil
import subprocess
import sys
from typing import Any
from pathlib import Path

import numpy as np
from tqdm import tqdm


def load_json_records(path):
    path = Path(path)
    text = path.read_text(encoding='utf-8')

    # Support both JSON arrays and JSONL
    if path.suffix.lower() == '.jsonl':
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if 'data' in data and isinstance(data['data'], list):
            return data['data']
    raise ValueError(f'Unsupported JSON format in {path}')


def get_cv2(required=False) -> Any:
    try:
        return importlib.import_module('cv2')
    except ImportError:
        if required:
            raise ImportError('opencv-python is required for this operation')
        return None


def get_mediapipe_holistic(required=False) -> Any:
    try:
        mp = importlib.import_module('mediapipe')
        solutions = getattr(mp, 'solutions', None)
        if solutions is None:
            return None
        return solutions.holistic
    except ImportError as exc:
        if required:
            raise RuntimeError(f'MediaPipe holistic is required for this operation: {exc}') from exc
        return None


def get_mediapipe_tasks_vision(required=False) -> Any:
    try:
        tasks = importlib.import_module('mediapipe.tasks.python')
        vision = importlib.import_module('mediapipe.tasks.python.vision')
        return tasks, vision
    except ImportError as exc:
        if required:
            raise RuntimeError(f'MediaPipe Tasks vision API is required for this operation: {exc}') from exc
        return None


def slugify(value):
    value = str(value)
    value = re.sub(r'[^a-zA-Z0-9_.-]+', '_', value)
    return value.strip('_') or 'sample'


def resolve_video_path(video_ref, video_root):
    video_ref = str(video_ref)
    root = Path(video_root)
    p = Path(video_ref)

    # Some datasets store root-appended absolute paths like "/clips/a.mp4"
    if p.is_absolute():
        return root / video_ref.lstrip('/\\')

    return root / p


def get_gpt_target(record):
    convs = record.get('conversations', [])
    for turn in convs:
        if str(turn.get('from', '')).lower() == 'gpt':
            return str(turn.get('value', '')).strip()
    raise ValueError(f"No GPT target found in sample id={record.get('id')}")


def extract_frames(video_path, out_frame_dir, frame_stride=1):
    out_frame_dir.mkdir(parents=True, exist_ok=True)

    cv2 = get_cv2(required=False)
    if cv2 is None:
        return extract_frames_with_ffmpeg(video_path, out_frame_dir, frame_stride=frame_stride)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Unable to open video: {video_path}')

    frame_idx = 0
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_stride == 0:
            resized = cv2.resize(frame, (224, 224))
            out_name = out_frame_dir / f'images{written:03d}-0.png'
            cv2.imwrite(str(out_name), resized)
            written += 1

        frame_idx += 1

    cap.release()

    if written == 0:
        raise RuntimeError(f'No frames extracted from {video_path}')

    return written


def extract_frames_with_ffmpeg(video_path, out_frame_dir, frame_stride=1):
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        raise ImportError('opencv-python is required unless ffmpeg is available on PATH')

    frame_stride = max(int(frame_stride), 1)
    pattern = out_frame_dir / 'images%03d-0.png'
    vf = f"select='not(mod(n\\,{frame_stride}))',scale=224:224"
    cmd = [
        ffmpeg,
        '-hide_banner',
        '-loglevel',
        'error',
        '-i',
        str(video_path),
        '-vf',
        vf,
        '-vsync',
        '0',
        str(pattern),
    ]
    subprocess.run(cmd, check=True)

    written = len(list(out_frame_dir.glob('images*-0.png')))
    if written == 0:
        raise RuntimeError(f'No frames extracted from {video_path}')

    return written


def build_segmenter(use_segmentation):
    if not use_segmentation:
        return None

    try:
        mp_holistic = get_mediapipe_holistic(required=True)
        if mp_holistic is None:
            raise RuntimeError('MediaPipe holistic is unavailable')
        holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            smooth_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.5,
        )
        return holistic
    except Exception:
        return 'fallback_ones'


def build_pose_extractor(write_pose_landmarks, pose_model_asset_path=None):
    if not write_pose_landmarks:
        return None

    get_cv2(required=True)

    mp_holistic = get_mediapipe_holistic(required=False)
    if mp_holistic is not None:
        return ('solutions', mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            smooth_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.5,
        ))

    if not pose_model_asset_path:
        raise RuntimeError(
            'Pose landmark export requires either the legacy mediapipe.solutions API or '
            '--pose_model_asset_path for the newer MediaPipe Tasks API.'
        )

    tasks_pkg, vision = get_mediapipe_tasks_vision(required=True)
    base_options = tasks_pkg.BaseOptions(model_asset_path=str(pose_model_asset_path))
    options = vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        output_segmentation_mask=False,
    )
    return ('tasks', vision.HolisticLandmarker.create_from_options(options))


def landmark_array(landmarks, count, include_visibility=False):
    width = 4 if include_visibility else 3
    arr = np.zeros((count, width), dtype=np.float32)

    if landmarks is None:
        return arr, False

    for i, landmark in enumerate(landmarks.landmark[:count]):
        arr[i, 0] = landmark.x
        arr[i, 1] = landmark.y
        arr[i, 2] = landmark.z
        if include_visibility:
            arr[i, 3] = getattr(landmark, 'visibility', 0.0)

    return arr, True


def landmark_array_from_tasks(landmarks, count, include_visibility=False):
    width = 4 if include_visibility else 3
    arr = np.zeros((count, width), dtype=np.float32)

    if not landmarks:
        return arr, False

    task_landmarks = landmarks
    if hasattr(task_landmarks, 'landmark'):
        task_landmarks = task_landmarks.landmark
    elif isinstance(task_landmarks, (list, tuple)) and task_landmarks:
        first = task_landmarks[0]
        if hasattr(first, 'landmark'):
            task_landmarks = first.landmark
        elif isinstance(first, (list, tuple)) and first and hasattr(first[0], 'x'):
            task_landmarks = first

    for i, landmark in enumerate(task_landmarks[:count]):
        arr[i, 0] = landmark.x
        arr[i, 1] = landmark.y
        arr[i, 2] = landmark.z
        if include_visibility:
            arr[i, 3] = getattr(landmark, 'visibility', 0.0)

    return arr, True


def write_pose_landmarks(frame_dir, pose_dir, pose_extractor, sample_id):
    if pose_extractor is None:
        return None

    cv2 = get_cv2(required=True)
    if cv2 is None:
        raise RuntimeError('OpenCV is unavailable')
    pose_dir.mkdir(parents=True, exist_ok=True)
    frame_files = sorted([p for p in frame_dir.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}])
    if not frame_files:
        raise RuntimeError(f'No frames available for pose landmark export: {frame_dir}')

    frame_names = []
    pose_frames = []
    left_hand_frames = []
    right_hand_frames = []
    pose_detected_frames = 0
    left_hand_detected_frames = 0
    right_hand_detected_frames = 0

    extractor_mode, extractor = pose_extractor

    for frame_file in frame_files:
        image = cv2.imread(str(frame_file))
        if image is None:
            raise FileNotFoundError(frame_file)
        image = cv2.resize(image, (224, 224))
        if extractor_mode == 'solutions':
            result = extractor.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            pose_arr, pose_detected = landmark_array(result.pose_landmarks, 33, include_visibility=True)
            left_hand_arr, left_detected = landmark_array(result.left_hand_landmarks, 21)
            right_hand_arr, right_detected = landmark_array(result.right_hand_landmarks, 21)
        else:
            mp_image_mod = importlib.import_module('mediapipe')
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp_image_mod.Image(image_format=mp_image_mod.ImageFormat.SRGB, data=rgb_image)
            result = extractor.detect(mp_image)
            pose_landmarks = getattr(result, 'pose_landmarks', None) or []
            left_hand_landmarks = getattr(result, 'left_hand_landmarks', None) or []
            right_hand_landmarks = getattr(result, 'right_hand_landmarks', None) or []
            pose_arr, pose_detected = landmark_array_from_tasks(pose_landmarks, 33, include_visibility=True)
            left_hand_arr, left_detected = landmark_array_from_tasks(left_hand_landmarks, 21)
            right_hand_arr, right_detected = landmark_array_from_tasks(right_hand_landmarks, 21)

        frame_names.append(frame_file.name)
        pose_frames.append(pose_arr)
        left_hand_frames.append(left_hand_arr)
        right_hand_frames.append(right_hand_arr)
        pose_detected_frames += int(pose_detected)
        left_hand_detected_frames += int(left_detected)
        right_hand_detected_frames += int(right_detected)

    landmark_path = pose_dir / 'landmarks.npz'
    np.savez_compressed(
        landmark_path,
        frame_names=np.asarray(frame_names),
        pose=np.stack(pose_frames, axis=0),
        left_hand=np.stack(left_hand_frames, axis=0),
        right_hand=np.stack(right_hand_frames, axis=0),
    )

    metadata = {
        'sample_id': str(sample_id),
        'frame_count': len(frame_names),
        'pose_detected_frames': pose_detected_frames,
        'left_hand_detected_frames': left_hand_detected_frames,
        'right_hand_detected_frames': right_hand_detected_frames,
        'landmark_file': str(landmark_path.name),
    }
    (pose_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata


def write_segmentation(frame_dir, seg_dir, segmenter):
    cv2 = get_cv2(required=False)
    seg_dir.mkdir(parents=True, exist_ok=True)
    frame_files = sorted([p for p in frame_dir.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}])

    for frame_file in frame_files:
        if segmenter is None:
            seg = np.ones((224, 224), dtype=np.uint8)
            out_name = seg_dir / f'{frame_file.stem}.npy.gz'
            with gzip.GzipFile(out_name, 'wb') as f:
                np.save(file=f, arr=seg)
            continue

        if cv2 is None:
            raise ImportError('opencv-python is required for MediaPipe segmentation')

        image = cv2.imread(str(frame_file))
        image = cv2.resize(image, (224, 224))

        if segmenter == 'fallback_ones':
            seg = np.ones((224, 224), dtype=np.uint8)
        else:
            result = segmenter.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if result.segmentation_mask is None:
                seg = np.ones((224, 224), dtype=np.uint8)
            else:
                seg = (result.segmentation_mask > 0.5).astype(np.uint8)

        out_name = seg_dir / f'{frame_file.stem}.npy.gz'
        with gzip.GzipFile(out_name, 'wb') as f:
            np.save(file=f, arr=seg)


def write_corpus_csv(csv_path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow([row])


def build_lookup(target_texts, lookup_path):
    vocab = {
        '<PAD>': 0,
        '<SOS>': 1,
        '<EOS>': 2,
        '<UNK>': 3,
    }

    for text in target_texts:
        for token in text.split():
            if token not in vocab:
                vocab[token] = len(vocab)

    # CTC blank should be last
    vocab['<BLANK>'] = len(vocab)

    lookup_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lookup_path, 'wb') as f:
        pickle.dump(vocab, f)

    return vocab


def process_split(records, split_name, data_root, seg_root, frame_stride, segmenter, video_root, pose_root=None, pose_extractor=None):
    frame_split_dir = data_root / 'features' / 'fullFrame-210x260px' / split_name
    rows = []
    targets = []

    for rec in tqdm(records, desc=f'Preparing {split_name}'):
        sample_id = slugify(rec.get('id', ''))
        video_path = resolve_video_path(rec['video'], video_root)
        if not video_path.exists():
            raise FileNotFoundError(f'Video not found for id={sample_id}: {video_path}')

        target = get_gpt_target(rec)
        targets.append(target)

        sample_frame_dir = frame_split_dir / sample_id / '1'
        extract_frames(video_path, sample_frame_dir, frame_stride=frame_stride)

        sample_seg_dir = seg_root / sample_id
        write_segmentation(sample_frame_dir, sample_seg_dir, segmenter)

        if pose_root is not None:
            sample_pose_dir = pose_root / sample_id / '1'
            write_pose_landmarks(sample_frame_dir, sample_pose_dir, pose_extractor, sample_id)

        # Keep PHOENIX-like style: [id|gloss|translation], using gloss as target.
        rows.append(f'{sample_id}|{target}|{target}')

    return rows, targets


def run_training(train_script, prepared_root, lookup_path, train_seg_root, val_seg_root, extra_args):
    cmd = [
        sys.executable,
        str(train_script),
        '--data', str(prepared_root),
        '--data_type', 'features',
        '--lookup_table', str(lookup_path),
        '--train_segment_root', str(train_seg_root),
        '--val_segment_root', str(val_seg_root),
    ]

    # Ensure train.py writes checkpoints in a writable location unless user overrides it.
    if '--save_dir' not in extra_args:
        cmd.extend(['--save_dir', str(Path(prepared_root) / 'trained_model')])

    cmd.extend(extra_args)

    print('\nRunning training command:')
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare JSON + video data for CSLR-IIGA training and optionally launch training.')
    parser.add_argument('--train_json', required=True, help='Path to training JSON/JSONL.')
    parser.add_argument('--eval_json', required=True, help='Path to evaluation JSON/JSONL (mapped to dev split).')
    parser.add_argument('--test_json', required=True, help='Path to test JSON/JSONL.')
    parser.add_argument('--video_root', required=True, help='Root directory to prepend to JSON video paths.')
    parser.add_argument('--output_root', required=True, help='Output dataset root in PHOENIX-compatible layout.')
    parser.add_argument('--lookup_out', default=None, help='Optional output path for lookup pickle.')
    parser.add_argument('--frame_stride', type=int, default=1, help='Use every Nth frame from each video.')
    parser.add_argument('--disable_segmentation', action='store_true', help='If set, create full-foreground masks instead of MediaPipe segmentation.')
    parser.add_argument('--write_pose_landmarks', action='store_true', help='If set, export MediaPipe pose/hand landmark sidecars alongside prepared frames.')
    parser.add_argument('--pose_model_asset_path', default=None, help='Required for pose export when only the newer MediaPipe Tasks API is available.')
    parser.add_argument('--run_train', action='store_true', help='Launch IIGA/train.py after preparation.')
    parser.add_argument('--train_script', default='IIGA/train.py', help='Path to train.py script.')
    parser.add_argument('train_args', nargs=argparse.REMAINDER, help='Extra args forwarded to train.py (put after --).')

    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    train_records = load_json_records(args.train_json)
    eval_records = load_json_records(args.eval_json)
    test_records = load_json_records(args.test_json)

    seg_root_base = output_root / 'segmentation'
    train_seg_root = seg_root_base / 'train_segmentation'
    val_seg_root = seg_root_base / 'val_segmentation'
    test_seg_root = seg_root_base / 'test_segmentation'
    pose_root_base = output_root / 'pose_landmarks'
    train_pose_root = pose_root_base / 'train'
    val_pose_root = pose_root_base / 'dev'
    test_pose_root = pose_root_base / 'test'

    segmenter = build_segmenter(use_segmentation=not args.disable_segmentation)
    pose_extractor = build_pose_extractor(
        write_pose_landmarks=args.write_pose_landmarks,
        pose_model_asset_path=args.pose_model_asset_path,
    )

    train_rows, train_targets = process_split(train_records, 'train', output_root, train_seg_root, args.frame_stride, segmenter, args.video_root, pose_root=train_pose_root if args.write_pose_landmarks else None, pose_extractor=pose_extractor)
    eval_rows, eval_targets = process_split(eval_records, 'dev', output_root, val_seg_root, args.frame_stride, segmenter, args.video_root, pose_root=val_pose_root if args.write_pose_landmarks else None, pose_extractor=pose_extractor)
    test_rows, test_targets = process_split(test_records, 'test', output_root, test_seg_root, args.frame_stride, segmenter, args.video_root, pose_root=test_pose_root if args.write_pose_landmarks else None, pose_extractor=pose_extractor)

    annotations_dir = output_root / 'annotations' / 'manual'
    write_corpus_csv(annotations_dir / 'train.corpus.csv', train_rows)
    write_corpus_csv(annotations_dir / 'dev.corpus.csv', eval_rows)
    write_corpus_csv(annotations_dir / 'test.corpus.csv', test_rows)

    lookup_out = Path(args.lookup_out) if args.lookup_out else (output_root / 'lookup' / 'json_lookup.pkl')
    vocab = build_lookup(train_targets, lookup_out)

    if segmenter not in (None, 'fallback_ones'):
        segmenter.close()
    if pose_extractor is not None:
        _, pose_runner = pose_extractor
        close_fn = getattr(pose_runner, 'close', None)
        if callable(close_fn):
            close_fn()

    print('\nPreparation complete!')
    print(f'Prepared dataset root: {output_root}')
    print(f'Train annotations: {annotations_dir / "train.corpus.csv"}')
    print(f'Lookup table: {lookup_out}')
    print(f'Vocab size: {len(vocab)}')
    print(f'Train segmentation root: {train_seg_root}')
    print(f'Validation segmentation root: {val_seg_root}')
    print(f'Test segmentation root: {test_seg_root}')
    if args.write_pose_landmarks:
        print(f'Train pose root: {train_pose_root}')
        print(f'Validation pose root: {val_pose_root}')
        print(f'Test pose root: {test_pose_root}')

    if args.run_train:
        forwarded = args.train_args
        if forwarded and forwarded[0] == '--':
            forwarded = forwarded[1:]
        run_training(Path(args.train_script), output_root, lookup_out, train_seg_root, val_seg_root, forwarded)
    else:
        print('\nTo train manually:')
        print(
            f"{sys.executable} IIGA/train.py --data {output_root} --data_type features "
            f"--lookup_table {lookup_out} --train_segment_root {train_seg_root} "
            f"--val_segment_root {val_seg_root}"
        )
