from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path
from typing import Any

import _pickle as pickle
import torch
import torch.nn.functional as F

from dataloader import loader
from tools.runtime import select_device
from tools.utils import Batch, path_data
from transformer import make_model as TRANSFORMER

DEV_IDS = ['600', '1174', '706', '1224', '199', '1030', '1319', '1341']


def read_corpus_rows(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open('r', encoding='utf-8') as handle:
        reader = csv.reader(handle, delimiter='|')
        for row in reader:
            if not row:
                continue
            sample_id = str(row[0])
            target = str(row[1]) if len(row) > 1 else ''
            rows.append({
                'sample_id': sample_id,
                'target': target,
                'tokens': target.split(),
                'raw_row': row,
            })
    return rows


def write_single_row_csv(row: dict[str, Any], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle, delimiter='|')
        writer.writerow(row['raw_row'])


def build_model(args, vocab_size, device, checkpoint_path: str | None):
    model = TRANSFORMER(
        tgt_vocab=vocab_size,
        n_stacks=args.num_layers,
        n_units=args.hidden_size,
        n_heads=args.n_heads,
        window_size=args.local_window,
        d_ff=args.d_ff,
        dropout=1.0 - args.dp_keep_prob,
        image_size=args.rescale,
        pretrained=True,
        emb_type='2d',
        emb_network=args.emb_network,
        channels=3,
        encoder_type='legacy',
    )
    if checkpoint_path is not None:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    return model.to(device)


def extract_mean_src_emb(row, prepared_root: Path, segment_root: str, lookup_table: str, model, device, fixed_padding, local_window) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix='dev8_repr_') as tmpdir:
        tmp_csv = Path(tmpdir) / 'single.corpus.csv'
        write_single_row_csv(row, tmp_csv)
        train_path, _, _ = path_data(data_path=str(prepared_root), task='SLR', features_type='features', hand_query=False)
        dataloader, _ = loader(
            csv_file=str(tmp_csv),
            root_dir=train_path[0],
            segment_path=segment_root,
            lookup=lookup_table,
            rescale=224,
            batch_size=1,
            num_workers=0,
            random_drop=0.0,
            uniform_drop='none',
            show_sample=False,
            istrain=False,
        )
        batch_data = next(iter(dataloader))
    x, x_lengths, y, y_lengths, hand_regions, _ = batch_data
    x = x.to(device)
    batch = Batch(
        x_lengths,
        y_lengths,
        None,
        trg=None,
        emb_type='2d',
        DEVICE=device,
        fixed_padding=fixed_padding,
        rel_window=local_window,
    )
    model.eval()
    with torch.no_grad():
        src_emb, _, _ = model.src_emb(x, batch.src_mask)
    valid_t = int(src_emb.shape[1])
    pooled = src_emb[:, :valid_t, :].mean(dim=1).squeeze(0)
    return {
        'sample_id': row['sample_id'],
        'target': row['target'],
        'src_emb_time': int(src_emb.shape[1]),
        'embedding': pooled.cpu(),
    }


def pairwise_cosine(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            sim = F.cosine_similarity(rows[i]['embedding'].unsqueeze(0), rows[j]['embedding'].unsqueeze(0)).item()
            out.append({
                'a': rows[i]['sample_id'],
                'b': rows[j]['sample_id'],
                'a_target': rows[i]['target'],
                'b_target': rows[j]['target'],
                'cosine_similarity': float(sim),
            })
    return out


def summarize_pairwise(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    sims = [pair['cosine_similarity'] for pair in pairs]
    if not sims:
        return {'pair_count': 0}
    sims_sorted = sorted(sims)
    return {
        'pair_count': len(sims),
        'mean_cosine_similarity': float(sum(sims) / len(sims)),
        'median_cosine_similarity': float(sims_sorted[len(sims_sorted) // 2]),
        'min_cosine_similarity': float(min(sims)),
        'max_cosine_similarity': float(max(sims)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare dev8 replay checkpoint representations against single-clip overfit checkpoints.')
    parser.add_argument('--prepared_root', required=True)
    parser.add_argument('--lookup_table', required=True)
    parser.add_argument('--segment_root', required=True)
    parser.add_argument('--replay_checkpoint', required=True)
    parser.add_argument('--single_clip_dir', required=True)
    parser.add_argument('--output_json', required=True)
    parser.add_argument('--rescale', type=int, default=224)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--dp_keep_prob', type=float, default=1.0)
    parser.add_argument('--local_window', type=int, default=10)
    parser.add_argument('--fixed_padding', type=int, default=None)
    parser.add_argument('--emb_network', type=str, default='mb2')
    args = parser.parse_args()

    prepared_root = Path(args.prepared_root)
    dev_rows = read_corpus_rows(prepared_root / 'annotations' / 'manual' / 'dev.corpus.csv')
    dev_by_id = {row['sample_id']: row for row in dev_rows}
    rows = [dev_by_id[sample_id] for sample_id in DEV_IDS]

    with open(args.lookup_table, 'rb') as handle:
        lookup = pickle.load(handle)

    device = select_device()
    vocab_size = len(lookup)

    replay_model = build_model(args, vocab_size, device, args.replay_checkpoint)
    replay_rows = [extract_mean_src_emb(row, prepared_root, args.segment_root, args.lookup_table, replay_model, device, args.fixed_padding, args.local_window) for row in rows]

    single_clip_rows = []
    single_clip_dir = Path(args.single_clip_dir)
    for row in rows:
        checkpoint_path = single_clip_dir / row['sample_id'] / 'BEST.pt'
        single_model = build_model(args, vocab_size, device, str(checkpoint_path) if checkpoint_path.exists() else None)
        single_clip_rows.append(extract_mean_src_emb(row, prepared_root, args.segment_root, args.lookup_table, single_model, device, args.fixed_padding, args.local_window))

    replay_pairs = pairwise_cosine(replay_rows)
    single_pairs = pairwise_cosine(single_clip_rows)

    per_clip = []
    replay_map = {item['sample_id']: item for item in replay_rows}
    single_map = {item['sample_id']: item for item in single_clip_rows}
    for sample_id in DEV_IDS:
        replay_vec = replay_map[sample_id]['embedding']
        single_vec = single_map[sample_id]['embedding']
        cos = F.cosine_similarity(replay_vec.unsqueeze(0), single_vec.unsqueeze(0)).item()
        per_clip.append({
            'sample_id': sample_id,
            'target': replay_map[sample_id]['target'],
            'replay_src_emb_time': replay_map[sample_id]['src_emb_time'],
            'single_src_emb_time': single_map[sample_id]['src_emb_time'],
            'replay_vs_single_cosine': float(cos),
        })

    payload = {
        'ok': True,
        'prepared_root': str(prepared_root),
        'lookup_table': args.lookup_table,
        'segment_root': args.segment_root,
        'replay_checkpoint': args.replay_checkpoint,
        'single_clip_dir': str(single_clip_dir),
        'device': str(device),
        'dev_ids': DEV_IDS,
        'replay_summary': summarize_pairwise(replay_pairs),
        'single_clip_summary': summarize_pairwise(single_pairs),
        'pairwise_replay': replay_pairs,
        'pairwise_single_clip': single_pairs,
        'per_clip': per_clip,
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
