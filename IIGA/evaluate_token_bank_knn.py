from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch
import torch.nn.functional as F

from dataloader import loader
from tools.token_bank import TokenBank
from tools.utils import Batch
from transformer import make_model as TRANSFORMER


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def load_lookup(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    with path.open('rb') as handle:
        vocab = pickle.load(handle)
    if not isinstance(vocab, dict):
        raise SystemExit(f'lookup is not a dict: {path}')
    id_to_token = {int(idx): str(token) for token, idx in vocab.items()}
    token_to_id = {str(token): int(idx) for token, idx in vocab.items()}
    return token_to_id, id_to_token


def load_corpus_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as handle:
        reader = csv.reader(handle, delimiter='|')
        for row in reader:
            if not row:
                continue
            target = str(row[1]) if len(row) > 1 else ''
            rows.append({
                'sample_id': str(row[0]),
                'target': target,
                'tokens': target.split(),
            })
    return rows


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


def normalize_target_ids(target_tensor: Any, target_length: int, pad_idx: int, blank_idx: int) -> list[int]:
    target_tensor = torch.as_tensor(target_tensor)
    target_ids = []
    for raw in target_tensor[:target_length].tolist():
        token_id = int(raw)
        if token_id in (pad_idx, blank_idx):
            continue
        target_ids.append(token_id)
    return target_ids


def build_bank_embedding_index(bank: TokenBank) -> tuple[list[dict[str, Any]], torch.Tensor, list[int], dict[int, int]]:
    entries: list[dict[str, Any]] = []
    token_counts: dict[int, int] = {}
    for token_id in bank.token_ids():
        token_entries = bank.query_by_token(token_id)
        token_counts[int(token_id)] = len(token_entries)
        for entry in token_entries:
            entries.append({
                'token_id': int(entry.token_id),
                'token_text': entry.token_text,
                'sample_id': entry.sample_id,
                'span_start': int(entry.span_start),
                'span_end': int(entry.span_end),
                'span_length': int(entry.span_length),
                'score': float(entry.score),
                'embedding': entry.embedding,
                'metadata': entry.metadata,
            })

    if not entries:
        raise SystemExit('token bank has no entries')

    bank_embeddings = torch.tensor([entry['embedding'] for entry in entries], dtype=torch.float32)
    normalized_bank = F.normalize(bank_embeddings, dim=-1)
    unique_token_ids = sorted(token_counts.keys())
    return entries, normalized_bank, unique_token_ids, token_counts


def rank_map_from_scores(scores: dict[int, float]) -> dict[int, int]:
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return {token_id: index + 1 for index, (token_id, _) in enumerate(ranked)}


def softmax_distribution_from_scores(scores: dict[int, float], temperature: float) -> dict[int, float]:
    if temperature <= 0.0:
        raise ValueError(f'temperature must be > 0, got {temperature}')
    token_ids = sorted(scores.keys())
    logits = torch.tensor([scores[token_id] for token_id in token_ids], dtype=torch.float32)
    probs = torch.softmax(logits / temperature, dim=0)
    return {token_id: float(prob) for token_id, prob in zip(token_ids, probs.tolist())}


def aggregate_metrics(rows: list[dict[str, Any]], universe_size: int) -> dict[str, Any]:
    if not rows:
        return {
            'target_token_count': 0,
            'top1_count': 0,
            'top5_count': 0,
            'top10_count': 0,
            'prob_ge_0_1_count': 0,
            'prob_ge_0_01_count': 0,
            'top1_ratio': 0.0,
            'top5_ratio': 0.0,
            'top10_ratio': 0.0,
            'prob_ge_0_1_ratio': 0.0,
            'prob_ge_0_01_ratio': 0.0,
            'mean_rank': None,
            'median_rank': None,
            'universe_size': universe_size,
        }

    target_token_count = len(rows)
    ranks = [int(row['rank']) for row in rows]
    top1_count = sum(1 for row in rows if int(row['rank']) <= 1)
    top5_count = sum(1 for row in rows if int(row['rank']) <= 5)
    top10_count = sum(1 for row in rows if int(row['rank']) <= 10)
    prob_ge_0_1_count = sum(1 for row in rows if float(row['probability']) >= 0.1)
    prob_ge_0_01_count = sum(1 for row in rows if float(row['probability']) >= 0.01)

    return {
        'target_token_count': target_token_count,
        'top1_count': top1_count,
        'top5_count': top5_count,
        'top10_count': top10_count,
        'prob_ge_0_1_count': prob_ge_0_1_count,
        'prob_ge_0_01_count': prob_ge_0_01_count,
        'top1_ratio': top1_count / target_token_count,
        'top5_ratio': top5_count / target_token_count,
        'top10_ratio': top10_count / target_token_count,
        'prob_ge_0_1_ratio': prob_ge_0_1_count / target_token_count,
        'prob_ge_0_01_ratio': prob_ge_0_01_count / target_token_count,
        'mean_rank': mean(ranks),
        'median_rank': median(ranks),
        'universe_size': universe_size,
    }


def random_baseline_metrics(target_token_count: int, universe_size: int) -> dict[str, Any]:
    if universe_size <= 0:
        raise ValueError(f'universe_size must be positive, got {universe_size}')
    return {
        'target_token_count': target_token_count,
        'top1_expected_count': target_token_count * (1 / universe_size),
        'top5_expected_count': target_token_count * (min(5, universe_size) / universe_size),
        'top10_expected_count': target_token_count * (min(10, universe_size) / universe_size),
        'top1_expected_ratio': 1 / universe_size,
        'top5_expected_ratio': min(5, universe_size) / universe_size,
        'top10_expected_ratio': min(10, universe_size) / universe_size,
        'mean_rank_expected': (universe_size + 1) / 2,
    }


def compute_bank_token_scores(frame_embeddings: torch.Tensor, normalized_bank: torch.Tensor, bank_entries: list[dict[str, Any]]) -> dict[int, float]:
    normalized_frames = F.normalize(frame_embeddings, dim=-1)
    similarities = torch.mm(normalized_frames, normalized_bank.t())
    best_entry_scores = similarities.max(dim=0).values
    token_scores: dict[int, float] = {}
    for entry, score in zip(bank_entries, best_entry_scores.tolist()):
        token_id = int(entry['token_id'])
        token_scores[token_id] = max(token_scores.get(token_id, float('-inf')), float(score))
    return token_scores


def compute_ctc_token_scores(ctc_probs: torch.Tensor, candidate_token_ids: list[int]) -> dict[int, float]:
    token_scores: dict[int, float] = {}
    for token_id in candidate_token_ids:
        token_scores[int(token_id)] = float(ctc_probs[:, int(token_id)].max().item())
    return token_scores


def analyze_split(
    *,
    split: str,
    prepared_root: Path,
    lookup_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    pad_idx: int,
    blank_idx: int,
    id_to_token: dict[int, str],
    bank_entries: list[dict[str, Any]],
    normalized_bank: torch.Tensor,
    bank_unique_token_ids: list[int],
    bank_token_counts: dict[int, int],
    batch_size: int,
    num_workers: int,
    image_size: int,
    temperature: float,
    arch: str,
) -> dict[str, Any]:
    split_name = 'dev' if split in ('valid', 'dev') else split
    csv_path = prepared_root / 'annotations' / 'manual' / f'{split_name}.corpus.csv'
    feature_root = prepared_root / 'features' / 'fullFrame-210x260px' / split_name
    segment_root_name = 'val_segmentation' if split_name == 'dev' else 'test_segmentation'
    if split_name == 'train':
        segment_root_name = 'train_segmentation'
    segment_root = prepared_root / 'segmentation' / segment_root_name

    rows = load_corpus_rows(csv_path)
    sample_to_target = {row['sample_id']: row['target'] for row in rows}

    dataloader, dataset_size = loader(
        csv_file=str(csv_path),
        root_dir=str(feature_root),
        segment_path=str(segment_root),
        lookup=str(lookup_path),
        rescale=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        random_drop=None,
        uniform_drop=1.0,
        show_sample=False,
        istrain=False,
        hand_dir=None,
        data_stats=None,
        hand_stats=None,
        channels=3,
        return_sample_ids=True,
    )

    frequency_rank_map = rank_map_from_scores({token_id: float(count) for token_id, count in bank_token_counts.items()})
    frequency_probs = {
        token_id: float(count) / max(sum(bank_token_counts.values()), 1)
        for token_id, count in bank_token_counts.items()
    }

    bank_token_rows = []
    ctc_token_rows = []
    frequency_token_rows = []
    sample_examples = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            x, x_lengths, y, y_lengths, hand_regions, _, sample_ids = batch
            raw_x_lengths = [int(length) for length in x_lengths]
            x = x.to(device)
            batch_obj = Batch(
                x_lengths,
                y_lengths,
                None,
                trg=None,
                pad=pad_idx,
                DEVICE=device,
                emb_type='2d',
            )

            src_emb, _, _ = model.src_emb(x)
            output, output_context, _ = model.forward(x, batch_obj.src_mask, batch_obj.rel_mask, None, arch)
            ctc_batch = output if output is not None else output_context
            if ctc_batch is None:
                raise RuntimeError('Model returned no CTC logits for retrieval diagnostic')
            ctc_probs_batch = ctc_batch.exp().detach().cpu()
            src_emb_batch = src_emb.detach().cpu()

            for batch_idx, sample_id in enumerate(sample_ids):
                sample_id_str = str(sample_id)
                valid_len = raw_x_lengths[batch_idx]
                frame_embeddings = src_emb_batch[batch_idx, :valid_len]
                ctc_probs = ctc_probs_batch[batch_idx, :valid_len]
                target_length = int(y_lengths[batch_idx])
                target_ids = normalize_target_ids(y[batch_idx], target_length, pad_idx, blank_idx)
                target_tokens = [id_to_token[int(token_id)] for token_id in target_ids]

                bank_token_scores = compute_bank_token_scores(frame_embeddings, normalized_bank, bank_entries)
                bank_rank_map = rank_map_from_scores(bank_token_scores)
                bank_probs = softmax_distribution_from_scores(bank_token_scores, temperature)

                ctc_token_scores = compute_ctc_token_scores(ctc_probs, bank_unique_token_ids)
                ctc_rank_map = rank_map_from_scores(ctc_token_scores)

                retrieval_top_tokens = sorted(bank_token_scores.items(), key=lambda item: (-item[1], item[0]))[:10]
                ctc_top_tokens = sorted(ctc_token_scores.items(), key=lambda item: (-item[1], item[0]))[:10]

                target_metrics = []
                for token_id, token_text in zip(target_ids, target_tokens):
                    token_id = int(token_id)
                    bank_row = {
                        'sample_id': sample_id_str,
                        'target': sample_to_target.get(sample_id_str, ''),
                        'token_id': token_id,
                        'token_text': token_text,
                        'rank': int(bank_rank_map[token_id]),
                        'probability': float(bank_probs[token_id]),
                        'score': float(bank_token_scores[token_id]),
                    }
                    ctc_row = {
                        'sample_id': sample_id_str,
                        'target': sample_to_target.get(sample_id_str, ''),
                        'token_id': token_id,
                        'token_text': token_text,
                        'rank': int(ctc_rank_map[token_id]),
                        'probability': float(ctc_token_scores[token_id]),
                        'score': float(ctc_token_scores[token_id]),
                    }
                    frequency_row = {
                        'sample_id': sample_id_str,
                        'target': sample_to_target.get(sample_id_str, ''),
                        'token_id': token_id,
                        'token_text': token_text,
                        'rank': int(frequency_rank_map[token_id]),
                        'probability': float(frequency_probs[token_id]),
                        'score': float(bank_token_counts[token_id]),
                    }
                    bank_token_rows.append(bank_row)
                    ctc_token_rows.append(ctc_row)
                    frequency_token_rows.append(frequency_row)
                    target_metrics.append({
                        'token_id': token_id,
                        'token_text': token_text,
                        'bank_rank': bank_row['rank'],
                        'bank_probability': bank_row['probability'],
                        'bank_score': bank_row['score'],
                        'ctc_rank': ctc_row['rank'],
                        'ctc_probability': ctc_row['probability'],
                        'frequency_rank': frequency_row['rank'],
                        'frequency_probability': frequency_row['probability'],
                    })

                sample_examples.append({
                    'sample_id': sample_id_str,
                    'target': sample_to_target.get(sample_id_str, ''),
                    'target_token_count': len(target_ids),
                    'target_metrics': target_metrics,
                    'retrieval_top10': [
                        {
                            'token_id': int(token_id),
                            'token_text': id_to_token[int(token_id)],
                            'score': float(score),
                            'probability': float(bank_probs[int(token_id)]),
                            'count': int(bank_token_counts[int(token_id)]),
                        }
                        for token_id, score in retrieval_top_tokens
                    ],
                    'ctc_top10': [
                        {
                            'token_id': int(token_id),
                            'token_text': id_to_token[int(token_id)],
                            'probability': float(score),
                        }
                        for token_id, score in ctc_top_tokens
                    ],
                })

    bank_metrics = aggregate_metrics(bank_token_rows, len(bank_unique_token_ids))
    ctc_metrics = aggregate_metrics(ctc_token_rows, len(bank_unique_token_ids))
    frequency_metrics = aggregate_metrics(frequency_token_rows, len(bank_unique_token_ids))
    random_metrics = random_baseline_metrics(bank_metrics['target_token_count'], len(bank_unique_token_ids))

    return {
        'split': split_name,
        'dataset_size': int(dataset_size),
        'bank_retrieval': bank_metrics,
        'ctc_baseline': ctc_metrics,
        'frequency_baseline': frequency_metrics,
        'random_baseline': random_metrics,
        'sample_examples': sample_examples[:32],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run token_bank_knn_v1 as a no-training diagnostic over frozen src_emb features on the smaller-gate Phase B surface.'
    )
    parser.add_argument('--summary_json', required=True)
    parser.add_argument('--token_bank_json', default=None)
    parser.add_argument('--prepared_root', default=None)
    parser.add_argument('--lookup_table', default=None)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--output_json', required=True)
    parser.add_argument('--splits', nargs='+', default=['dev', 'test'], choices=['dev', 'test'])
    parser.add_argument('--batch_size', type=int, default=2)
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
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--arch', default='CNN-attention-CTC')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary_path = Path(args.summary_json)
    if not summary_path.exists():
        raise SystemExit(f'missing summary json: {summary_path}')
    summary = load_json(summary_path)

    bank_path = Path(args.token_bank_json) if args.token_bank_json else Path(summary['bank_path'])
    if not bank_path.exists():
        raise SystemExit(f'missing token bank json: {bank_path}')

    prepared_root = Path(args.prepared_root) if args.prepared_root else Path(summary['data_root'])
    if not prepared_root.exists():
        raise SystemExit(f'missing prepared root: {prepared_root}')

    lookup_path = Path(args.lookup_table) if args.lookup_table else Path(summary['lookup_table'])
    if not lookup_path.exists():
        raise SystemExit(f'missing lookup table: {lookup_path}')

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path(summary['checkpoint'])
    if not checkpoint_path.exists():
        raise SystemExit(f'missing checkpoint: {checkpoint_path}')

    token_to_id, id_to_token = load_lookup(lookup_path)
    blank_idx = int(token_to_id.get('<BLANK>', len(token_to_id) - 1))
    pad_idx = int(token_to_id.get('<PAD>', 0))

    bank = TokenBank.load(bank_path)
    bank_entries, normalized_bank, bank_unique_token_ids, bank_token_counts = build_bank_embedding_index(bank)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = TRANSFORMER(
        tgt_vocab=len(token_to_id),
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
    model = load_checkpoint(model, checkpoint_path, device).to(device)
    model.eval()

    split_results = {}
    for split in args.splits:
        split_results[split] = analyze_split(
            split=split,
            prepared_root=prepared_root,
            lookup_path=lookup_path,
            model=model,
            device=device,
            pad_idx=pad_idx,
            blank_idx=blank_idx,
            id_to_token=id_to_token,
            bank_entries=bank_entries,
            normalized_bank=normalized_bank,
            bank_unique_token_ids=bank_unique_token_ids,
            bank_token_counts=bank_token_counts,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
            temperature=args.temperature,
            arch=args.arch,
        )

    result = {
        'summary_path': str(summary_path),
        'bank_path': str(bank_path),
        'prepared_root': str(prepared_root),
        'lookup_table': str(lookup_path),
        'checkpoint': str(checkpoint_path),
        'temperature': float(args.temperature),
        'bank_entry_count': len(bank_entries),
        'bank_token_type_count': len(bank_unique_token_ids),
        'splits': split_results,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
