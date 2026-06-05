from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path
from typing import Any

import _pickle as pickle
import torch

from dataloader import loader
from tools.ctc_decode import decode_ctc_batch, ids_to_text
from tools.ctc_forced_align import ctc_forced_align
from tools.runtime import effective_ctc_lengths, select_device
from tools.utils import Batch, path_data
from transformer import make_model as TRANSFORMER


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


def load_checkpoint(model_path: str, device: torch.device):
    try:
        return torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location=device)


def build_model(args, vocab_size, device):
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
    checkpoint = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    return model


def greedy_text_from_output(output_tbc, effective_lengths, vocab_inv, blank_idx, pad_idx):
    decoded = decode_ctc_batch(output_tbc, torch.IntTensor(effective_lengths), blank_idx)
    return ids_to_text(decoded[0], vocab_inv, ignore_ids=[blank_idx, pad_idx])


def per_token_support(log_probs_tbv, target_ids, vocab_inv, blank_idx):
    probs = log_probs_tbv.exp()
    rows = []
    for token_id in target_ids:
        token_probs = probs[:, token_id]
        best_prob, best_frame = torch.max(token_probs, dim=0)
        best_frame = int(best_frame.item())
        best_prob = float(best_prob.item())
        frame_probs = probs[best_frame]
        rank = int((frame_probs > frame_probs[token_id]).sum().item()) + 1
        blank_prob = float(frame_probs[blank_idx].item())
        top5 = torch.topk(frame_probs, k=min(5, frame_probs.size(0))).indices.tolist()
        rows.append({
            'token_id': int(token_id),
            'token_text': vocab_inv.get(int(token_id), f'<unk:{token_id}>'),
            'best_frame': best_frame,
            'best_prob': best_prob,
            'best_rank': rank,
            'blank_prob_at_best_frame': blank_prob,
            'top5_ids_at_best_frame': [int(idx) for idx in top5],
            'top5_tokens_at_best_frame': [vocab_inv.get(int(idx), f'<unk:{idx}>') for idx in top5],
            'supported_top5_prob001': bool(rank <= 5 and best_prob >= 0.01),
        })
    return rows


def split_roots(prepared_root: Path, split: str) -> tuple[str, str]:
    train_path, dev_path, test_path = path_data(
        data_path=str(prepared_root),
        task='SLR',
        features_type='features',
        hand_query=False,
    )
    roots = {
        'train': (train_path[0], str(prepared_root / 'segmentation' / 'train_segmentation')),
        'dev': (dev_path[0], str(prepared_root / 'segmentation' / 'val_segmentation')),
        'test': (test_path[0], str(prepared_root / 'segmentation' / 'test_segmentation')),
    }
    return roots[split]


def run_probe_for_row(row, split, args, device, lookup, vocab_inv, model):
    blank_idx = lookup.get('<BLANK>', len(lookup) - 1)
    pad_idx = lookup.get('<PAD>', 0)

    with tempfile.TemporaryDirectory(prefix='dev8_align_') as tmpdir:
        tmp_csv = Path(tmpdir) / 'single.corpus.csv'
        write_single_row_csv(row, tmp_csv)
        root_dir, segment_root = split_roots(Path(args.prepared_root), split)
        dataloader, _ = loader(
            csv_file=str(tmp_csv),
            root_dir=root_dir,
            segment_path=segment_root,
            lookup=args.lookup_table,
            rescale=args.rescale,
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
        fixed_padding=args.fixed_padding,
        rel_window=args.local_window,
    )

    with torch.no_grad():
        output, output_context, _ = model.forward(x, batch.src_mask, batch.rel_mask, None, args.arch)
        output_tbc = output_context.transpose(0, 1)
        effective_lengths = effective_ctc_lengths(
            list(x_lengths),
            local_window=args.local_window,
            emb_network=args.emb_network,
            output_time=output_tbc.size(0),
            reduction=getattr(model.src_emb, 'temporal_reduction', 1),
        )

    target_ids = [
        int(token_id)
        for token_id in y[0, : int(y_lengths[0])].tolist()
        if int(token_id) not in {blank_idx, pad_idx}
    ]
    greedy_prediction = greedy_text_from_output(output_tbc, effective_lengths, vocab_inv, blank_idx, pad_idx)
    log_probs = output_tbc[: effective_lengths[0], 0]

    alignment = None
    alignment_error = None
    try:
        alignment = ctc_forced_align(log_probs, target_ids, blank_idx=blank_idx, already_log_probs=True)
    except Exception as exc:  # diagnostic surface only
        alignment_error = str(exc)

    token_support = per_token_support(log_probs, target_ids, vocab_inv, blank_idx)
    supported_tokens = [item['token_text'] for item in token_support if item['supported_top5_prob001']]
    aligned_tokens = []
    if alignment is not None:
        aligned_tokens = [
            {
                'token_id': span['token_id'],
                'token_text': vocab_inv.get(span['token_id'], f"<unk:{span['token_id']}>") ,
                'start': span['start'],
                'end': span['end'],
                'length': span['length'],
            }
            for span in alignment['token_spans']
        ]

    return {
        'sample_id': row['sample_id'],
        'target': row['target'],
        'tokens': row['tokens'],
        'greedy_prediction': greedy_prediction,
        'alignment_ok': alignment is not None,
        'alignment_error': alignment_error,
        'alignment_score': None if alignment is None else float(alignment['score']),
        'aligned_token_spans': aligned_tokens,
        'supported_tokens_top5_prob001': supported_tokens,
        'supported_token_count': len(supported_tokens),
        'target_token_count': len(target_ids),
        'all_tokens_supported_top5_prob001': len(supported_tokens) == len(target_ids),
        'token_support': token_support,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Run a per-clip CTC forced-alignment diagnostic on the 8 Phase B dev replay clips.')
    parser.add_argument('--prepared_root', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--lookup_table', required=True)
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
    parser.add_argument('--arch', type=str, default='CNN-attention-CTC')
    args = parser.parse_args()

    prepared_root = Path(args.prepared_root)
    rows_by_split = {
        split: read_corpus_rows(prepared_root / 'annotations' / 'manual' / f'{split}.corpus.csv')
        for split in ('train', 'dev', 'test')
    }

    with open(args.lookup_table, 'rb') as handle:
        lookup = pickle.load(handle)
    vocab_inv = {v: k for k, v in lookup.items()}
    device = select_device()
    model = build_model(args, len(lookup), device)

    results = [
        run_probe_for_row(row, 'dev', args, device, lookup, vocab_inv, model)
        for row in rows_by_split['dev']
    ]
    alignment_ok_rows = [row['sample_id'] for row in results if row['alignment_ok']]
    all_supported_rows = [row['sample_id'] for row in results if row['all_tokens_supported_top5_prob001']]
    partial_supported_rows = [
        row['sample_id']
        for row in results
        if row['supported_token_count'] > 0 and not row['all_tokens_supported_top5_prob001']
    ]

    payload = {
        'ok': True,
        'prepared_root': str(prepared_root),
        'checkpoint': args.checkpoint,
        'lookup_table': args.lookup_table,
        'segment_roots': {
            'train': str(prepared_root / 'segmentation' / 'train_segmentation'),
            'dev': str(prepared_root / 'segmentation' / 'val_segmentation'),
            'test': str(prepared_root / 'segmentation' / 'test_segmentation'),
        },
        'device': str(device),
        'alignment_ok_rows': alignment_ok_rows,
        'all_supported_rows': all_supported_rows,
        'partial_supported_rows': partial_supported_rows,
        'rows': results,
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
