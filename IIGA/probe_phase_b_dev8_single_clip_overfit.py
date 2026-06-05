from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path
from typing import Any

import _pickle as pickle
import torch
import torch.nn as nn

from dataloader import loader
from tools.ctc_decode import decode_ctc_batch, ids_to_text
from tools.runtime import effective_ctc_lengths, select_device
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


def greedy_text_from_output(output_tbc, effective_lengths, vocab_inv, blank_idx, pad_idx):
    decoded = decode_ctc_batch(output_tbc, torch.IntTensor(effective_lengths), blank_idx)
    return ids_to_text(decoded[0], vocab_inv, ignore_ids=[blank_idx, pad_idx])


def build_model(args, vocab_size):
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
    return model


def run_probe_for_row(row, args, device, lookup, vocab_inv):
    blank_idx = lookup.get('<BLANK>', len(lookup) - 1)
    pad_idx = lookup.get('<PAD>', 0)
    ctc_loss = nn.CTCLoss(blank=blank_idx, reduction='sum', zero_infinity=True)

    with tempfile.TemporaryDirectory(prefix='dev8_overfit_') as tmpdir:
        tmp_csv = Path(tmpdir) / 'single.corpus.csv'
        write_single_row_csv(row, tmp_csv)
        train_path, _, _ = path_data(data_path=args.prepared_root, task='SLR', features_type='features', hand_query=False)
        dataloader, _ = loader(
            csv_file=str(tmp_csv),
            root_dir=train_path[0],
            segment_path=args.segment_root,
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
    y = torch.from_numpy(y).to(device)
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

    model = build_model(args, len(lookup)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def forward_loss_and_pred():
        output, output_context, output_hand = model.forward(x, batch.src_mask, batch.rel_mask, None, args.arch)
        output_tbc = output_context.transpose(0, 1)
        effective_lengths = effective_ctc_lengths(
            list(x_lengths),
            local_window=args.local_window,
            emb_network=args.emb_network,
            output_time=output_tbc.size(0),
            reduction=getattr(model.src_emb, 'temporal_reduction', 1),
        )
        target_lengths = torch.IntTensor(y_lengths)
        loss = ctc_loss(output_tbc, y.cpu(), torch.IntTensor(effective_lengths), target_lengths.cpu())
        pred_text = greedy_text_from_output(output_tbc, effective_lengths, vocab_inv, blank_idx, pad_idx)
        return loss, pred_text

    model.eval()
    with torch.no_grad():
        initial_loss, initial_pred = forward_loss_and_pred()

    ever_nonblank = bool(initial_pred.strip())
    ever_exact = initial_pred.strip() == row['target'].strip()
    best_pred = initial_pred
    best_loss = initial_loss.detach().item()
    final_loss = best_loss
    final_pred = initial_pred
    first_nonblank_step = 0 if ever_nonblank else None
    first_exact_step = 0 if ever_exact else None

    for step_idx in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, pred_text = forward_loss_and_pred()
        loss.backward()
        optimizer.step()

        final_loss = loss.detach().item()
        final_pred = pred_text
        if final_loss < best_loss:
            best_loss = final_loss
        if pred_text.strip() and not ever_nonblank:
            ever_nonblank = True
            first_nonblank_step = step_idx
        if pred_text.strip():
            best_pred = pred_text
        if pred_text.strip() == row['target'].strip() and not ever_exact:
            ever_exact = True
            first_exact_step = step_idx

    return {
        'sample_id': row['sample_id'],
        'target': row['target'],
        'steps': args.steps,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'initial_loss': float(initial_loss.detach().item()),
        'final_loss': float(final_loss),
        'best_loss': float(best_loss),
        'loss_decreased': bool(final_loss < initial_loss.detach().item()),
        'initial_prediction': initial_pred,
        'final_prediction': final_pred,
        'ever_nonblank': ever_nonblank,
        'ever_exact': ever_exact,
        'best_nonblank_prediction': best_pred,
        'first_nonblank_step': first_nonblank_step,
        'first_exact_step': first_exact_step,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Run a tiny single-clip full-model overfit probe on the 8 Phase B dev oracle-overfit clips.')
    parser.add_argument('--prepared_root', required=True)
    parser.add_argument('--lookup_table', required=True)
    parser.add_argument('--segment_root', required=True)
    parser.add_argument('--output_json', required=True)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
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
    dev_rows = read_corpus_rows(prepared_root / 'annotations' / 'manual' / 'dev.corpus.csv')
    dev_by_id = {row['sample_id']: row for row in dev_rows}
    rows = [dev_by_id[sample_id] for sample_id in DEV_IDS]

    with open(args.lookup_table, 'rb') as handle:
        lookup = pickle.load(handle)
    vocab_inv = {v: k for k, v in lookup.items()}
    device = select_device()

    results = [run_probe_for_row(row, args, device, lookup, vocab_inv) for row in rows]
    loss_decreased_rows = [row['sample_id'] for row in results if row['loss_decreased']]
    nonblank_rows = [row['sample_id'] for row in results if row['ever_nonblank']]
    exact_rows = [row['sample_id'] for row in results if row['ever_exact']]

    payload = {
        'ok': True,
        'prepared_root': str(prepared_root),
        'lookup_table': args.lookup_table,
        'segment_root': args.segment_root,
        'device': str(device),
        'steps': args.steps,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'loss_decreased_rows': loss_decreased_rows,
        'nonblank_rows': nonblank_rows,
        'exact_rows': exact_rows,
        'rows': results,
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
