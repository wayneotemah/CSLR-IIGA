from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import torch

from dataloader import loader
from tools.ctc_forced_align import batch_ctc_forced_align
from tools.token_bank import TokenBank, TokenEntry
from tools.utils import Batch
from transformer import make_model as TRANSFORMER


def load_checkpoint(model: torch.nn.Module, model_path: str | Path, device: torch.device) -> torch.nn.Module:
    try:
        checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(model_path), map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state = checkpoint['state_dict']
    elif isinstance(checkpoint, dict):
        state = checkpoint
    else:
        raise ValueError(f'Unsupported checkpoint format: {type(checkpoint)}')
    model.load_state_dict(state, strict=False)
    return model


def normalize_target_ids(target_tensor: torch.Tensor, target_length: int, pad_idx: int, blank_idx: int) -> list[int]:
    target_tensor = torch.as_tensor(target_tensor)
    target_ids = []
    for raw in target_tensor[:target_length].tolist():
        token_id = int(raw)
        if token_id in (pad_idx, blank_idx):
            continue
        target_ids.append(token_id)
    return target_ids


def mean_pool_span(sequence_embeddings: torch.Tensor, start: int, end: int) -> torch.Tensor:
    if start < 0 or end < start or end >= sequence_embeddings.size(0):
        raise ValueError(f'Invalid span ({start}, {end}) for embeddings with time dimension {sequence_embeddings.size(0)}')
    span = sequence_embeddings[start:end + 1]
    return span.mean(dim=0)


def build_bank(args: argparse.Namespace) -> dict[str, object]:
    data_root = Path(args.data_root)
    lookup_path = Path(args.lookup_table)

    with lookup_path.open('rb') as handle:
        vocab = pickle.load(handle)
    id_to_token = {int(v): str(k) for k, v in vocab.items()}
    pad_idx = int(vocab.get('<PAD>', 0))
    blank_idx = int(vocab.get('<BLANK>', len(vocab) - 1))

    dataloader, dataset_size = loader(
        csv_file=str(data_root / 'annotations' / 'manual' / 'train.corpus.csv'),
        root_dir=str(data_root / 'features' / 'fullFrame-210x260px' / 'train'),
        segment_path=str(data_root / 'segmentation' / 'train_segmentation'),
        lookup=str(lookup_path),
        rescale=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_drop=None,
        uniform_drop=1.0,
        show_sample=False,
        istrain=False,
        hand_dir=None,
        data_stats=None,
        hand_stats=None,
        channels=args.channels,
        return_sample_ids=True,
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = TRANSFORMER(
        tgt_vocab=len(vocab),
        n_stacks=args.num_layers,
        n_units=args.hidden_size,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        window_size=args.local_window,
        dropout=1.0 - args.dp_keep_prob,
        image_size=args.image_size,
        emb_type=args.emb_type,
        emb_network=args.emb_network,
        channels=args.channels,
        encoder_type=args.encoder_type,
        conformer_kernel_size=args.conformer_kernel_size,
    )
    model = load_checkpoint(model, args.checkpoint, device).to(device)
    model.eval()

    bank = TokenBank()
    alignment_failures = []
    entries_written = 0
    token_counter: Counter[str] = Counter()

    with torch.no_grad():
        for batch in dataloader:
            x, x_lengths, y, y_lengths, hand_regions, _, sample_ids = batch
            x = x.to(device)
            input_lengths = torch.as_tensor(x_lengths, dtype=torch.long)
            target_batch = torch.as_tensor(y)
            target_lengths = torch.as_tensor(y_lengths, dtype=torch.long)
            batch_obj = Batch(x_lengths, y_lengths, None, trg=None, pad=pad_idx, DEVICE=device, emb_type=args.emb_type)
            src_emb, _, _ = model.src_emb(x)
            out = model.forward(x, batch_obj.src_mask, batch_obj.rel_mask, None)
            if isinstance(out, tuple):
                output, output_context, _ = out[:3]
                log_probs_batch = output if output is not None else output_context
            else:
                log_probs_batch = out
            if log_probs_batch is None:
                raise RuntimeError('Model returned no CTC logits')

            batch_log_probs = log_probs_batch.transpose(0, 1).detach().cpu()
            sequence_embeddings = src_emb.detach().cpu()
            alignments = batch_ctc_forced_align(
                batch_log_probs,
                input_lengths,
                target_batch,
                target_lengths,
                blank_idx=blank_idx,
                already_log_probs=True,
            )

            for batch_idx, alignment in enumerate(alignments):
                sample_id = str(sample_ids[batch_idx])
                target_length = int(target_lengths[batch_idx].item())
                target_ids = normalize_target_ids(target_batch[batch_idx], target_length, pad_idx, blank_idx)
                target_spans = alignment.get('token_spans', [])

                span_cursor = 0
                for token_id in target_ids:
                    token_spans = []
                    while span_cursor < len(target_spans) and int(target_spans[span_cursor]['token_id']) == int(token_id):
                        token_spans.append(target_spans[span_cursor])
                        span_cursor += 1
                    if not token_spans:
                        alignment_failures.append({'sample_id': sample_id, 'token_id': int(token_id), 'reason': 'missing_span'})
                        continue
                    for span in token_spans:
                        pooled = mean_pool_span(sequence_embeddings[batch_idx], int(span['start']), int(span['end']))
                        token_text = id_to_token[int(token_id)]
                        bank.register(TokenEntry(
                            token_id=int(token_id),
                            token_text=token_text,
                            sample_id=sample_id,
                            span_start=int(span['start']),
                            span_end=int(span['end']),
                            span_length=int(span['length']),
                            score=float(alignment['score']),
                            embedding=[float(value) for value in pooled.tolist()],
                            metadata={
                                'checkpoint': str(args.checkpoint),
                                'encoder_type': args.encoder_type,
                            },
                        ))
                        entries_written += 1
                        token_counter[token_text] += 1

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    bank_path = output_root / 'token_bank.json'
    bank.save(bank_path)
    summary = {
        'checkpoint': str(args.checkpoint),
        'data_root': str(data_root),
        'lookup_table': str(lookup_path),
        'dataset_size': int(dataset_size),
        'entries_written': int(entries_written),
        'token_count': len(bank.token_ids()),
        'top_tokens': token_counter.most_common(25),
        'alignment_failures': alignment_failures[:200],
        'bank_path': str(bank_path),
        'bank_summary': bank.summary(),
    }
    summary_path = output_root / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return {'bank_path': str(bank_path), 'summary_path': str(summary_path), 'summary': summary}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a train-only dense pseudo-alignment token bank from a frozen checkpoint.')
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--lookup_table', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output_root', required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dp_keep_prob', type=float, default=1.0)
    parser.add_argument('--local_window', type=int, default=10)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--channels', type=int, default=3)
    parser.add_argument('--emb_type', default='2d')
    parser.add_argument('--emb_network', default='mb2')
    parser.add_argument('--encoder_type', choices=['legacy', 'conformer'], default='legacy')
    parser.add_argument('--conformer_kernel_size', type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_bank(args)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
