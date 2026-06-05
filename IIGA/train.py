import argparse
import time
import os
import torch
import torch.nn
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import datetime as dt
import _pickle as pickle
import json

from torch.optim.lr_scheduler import StepLR, MultiStepLR

from transformer import make_model as TRANSFORMER
from dataloader import loader
from tools.ctc_decode import decode_ctc_batch, ids_to_text
from tools.runtime import select_device, DeviceTelemetryPoller, format_device_telemetry, effective_ctc_lengths
from tools.offline_registry import init_offline_registry
from tools.utils import path_data, Batch, LabelSmoothing, NoamOpt

#Progress bar to visualize training progress
try:
    import progressbar
except ImportError:
    progressbar = None

#For model summary
try:
    from torchsummary import summary
except ImportError:
    summary = None

#Plotting
from tools.viz import learning_curve_slr

#Visualize GPU resources
try:
    import GPUtil
except ImportError:
    GPUtil = None

#Lavenghtein distance (WER)
try:
    from jiwer import wer
except ImportError:
    def _edit_distance(ref_tokens, hyp_tokens):
        rows = len(ref_tokens) + 1
        cols = len(hyp_tokens) + 1
        dp = [[0] * cols for _ in range(rows)]
        for i in range(rows):
            dp[i][0] = i
        for j in range(cols):
            dp[0][j] = j
        for i in range(1, rows):
            for j in range(1, cols):
                cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost,
                )
        return dp[-1][-1]

    def _normalize_wer_samples(value):
        if isinstance(value, str):
            return [value]
        if value is None:
            return ['']
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        return [str(value)]

    def wer(reference, hypothesis):
        references = _normalize_wer_samples(reference)
        hypotheses = _normalize_wer_samples(hypothesis)
        if len(references) != len(hypotheses):
            raise ValueError('reference and hypothesis must have the same number of samples')
        total_words = 0
        total_edits = 0
        for ref, hyp in zip(references, hypotheses):
            ref_tokens = str(ref).split()
            hyp_tokens = str(hyp).split()
            total_words += len(ref_tokens)
            total_edits += _edit_distance(ref_tokens, hyp_tokens)
        if total_words == 0:
            return 0.0
        return total_edits / total_words


def load_checkpoint(model_path, map_location=None):
    try:
        return torch.load(model_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location=map_location)

def print_device_utilization(device):
    if device.type != 'cuda' or GPUtil is None:
        return
    print(GPUtil.showUtilization())


def init_wandb(args, start_date):
    if not args.wandb or args.wandb_mode == 'disabled':
        return None

    try:
        import wandb
    except ImportError:
        print("WARNING: --wandb was set but wandb is not installed. Continuing without W&B logging.")
        return None

    run_name = args.wandb_run_name or f"{args.arch}-{start_date}"
    tags = [tag.strip() for tag in args.wandb_tags.split(',') if tag.strip()]
    init_kwargs = {
        'entity': args.wandb_entity,
        'project': args.wandb_project,
        'name': run_name,
        'tags': tags,
        'config': vars(args),
    }
    if args.wandb_group:
        init_kwargs['group'] = args.wandb_group
    if args.wandb_job_type:
        init_kwargs['job_type'] = args.wandb_job_type
    if args.wandb_system_sample_seconds > 0:
        init_kwargs['settings'] = getattr(wandb, 'Settings')(x_stats_sampling_interval=args.wandb_system_sample_seconds)
    if args.wandb_mode == 'offline':
        init_kwargs['mode'] = 'offline'
    if args.wandb_dir:
        init_kwargs['dir'] = args.wandb_dir

    try:
        return getattr(wandb, 'init')(**init_kwargs)
    except Exception as exc:
        print(f"WARNING: failed to initialize W&B ({exc}). Continuing without W&B logging.")
        return None


def log_wandb(wandb_run, metrics):
    if wandb_run is None:
        return

    try:
        wandb_run.log(metrics, step=metrics.get('epoch'))
    except Exception as exc:
        print(f"WARNING: failed to log W&B metrics ({exc}). Continuing training.")


def current_learning_rate(optimizer):
    param_groups = getattr(optimizer, 'param_groups', None)
    if param_groups:
        return param_groups[0].get('lr', 0.0)

    wrapped_optimizer = getattr(optimizer, 'optimizer', None)
    wrapped_param_groups = getattr(wrapped_optimizer, 'param_groups', None)
    if wrapped_param_groups:
        return wrapped_param_groups[0].get('lr', 0.0)

    rate = getattr(optimizer, '_rate', None)
    if rate is not None:
        return float(rate)

    return 0.0


def zero_optimizer_grad(optimizer, set_to_none=True):
    zero_grad = getattr(optimizer, 'zero_grad', None)
    if callable(zero_grad):
        try:
            zero_grad(set_to_none=set_to_none)
        except TypeError:
            zero_grad()
        return

    wrapped_optimizer = getattr(optimizer, 'optimizer', None)
    wrapped_zero_grad = getattr(wrapped_optimizer, 'zero_grad', None)
    if callable(wrapped_zero_grad):
        try:
            wrapped_zero_grad(set_to_none=set_to_none)
        except TypeError:
            wrapped_zero_grad()
        return

    raise AttributeError('optimizer does not expose zero_grad')


def cr_ctc_consistency_loss(log_probs_a, log_probs_b, input_lengths):
    """Symmetric stop-gradient KL over valid CTC time steps."""
    log_probs_a = log_probs_a.transpose(0, 1)
    log_probs_b = log_probs_b.transpose(0, 1)
    probs_a = log_probs_a.exp().detach()
    probs_b = log_probs_b.exp().detach()

    max_time = log_probs_a.size(1)
    lengths = input_lengths.to(log_probs_a.device)
    frame_index = torch.arange(max_time, device=log_probs_a.device).unsqueeze(0)
    valid_mask = frame_index < lengths.unsqueeze(1)
    valid_mask = valid_mask.to(log_probs_a.dtype)

    kl_a_to_b = F.kl_div(log_probs_b, probs_a, reduction='none').sum(dim=-1)
    kl_b_to_a = F.kl_div(log_probs_a, probs_b, reduction='none').sum(dim=-1)
    valid_count = valid_mask.sum().clamp_min(1.0)
    return 0.5 * ((kl_a_to_b * valid_mask).sum() + (kl_b_to_a * valid_mask).sum()) / valid_count


def read_corpus_ids(csv_path):
    ids = set()
    if not csv_path or not os.path.exists(csv_path):
        return ids

    with open(csv_path, 'r', encoding='utf-8') as handle:
        for line in handle:
            row = line.strip()
            if not row:
                continue
            ids.add(row.split('|', 1)[0])

    return ids


def load_corpus_rows(csv_path):
    rows = []
    if not csv_path or not os.path.exists(csv_path):
        return rows

    with open(csv_path, 'r', encoding='utf-8') as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            parts = raw.split('|')
            if len(parts) < 3:
                continue
            sample_id, target = parts[0], parts[1]
            rows.append({
                'sample_id': sample_id,
                'target': target,
                'tokens': target.split(),
            })

    return rows


def _first_present(mapping, keys):
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _collect_anchor_entries(node):
    if isinstance(node, list):
        entries = []
        for item in node:
            entries.extend(_collect_anchor_entries(item))
        return entries

    if not isinstance(node, dict):
        return []

    if any(key in node for key in ('sample_id', 'id', 'sequence_id')) and any(key in node for key in ('frame_idx', 'frame', 'time_idx')):
        return [node]

    entries = []
    for key in ('eligible_anchors', 'anchors', 'eligible'):
        if key in node:
            entries.extend(_collect_anchor_entries(node[key]))
    return entries


def load_anchor_audit(anchor_audit_json, word_to_id, vocab_size, train_csv, valid_csv, test_csv):
    with open(anchor_audit_json, 'r', encoding='utf-8') as handle:
        audit = json.load(handle)

    for leakage_key in ('validation_ids_in_anchors', 'valid_ids_in_anchors', 'test_ids_in_anchors'):
        leaked_ids = audit.get(leakage_key, []) if isinstance(audit, dict) else []
        if leaked_ids:
            raise ValueError(f"Anchor audit reports leakage in {leakage_key}: {leaked_ids}")

    train_ids = read_corpus_ids(train_csv)
    valid_ids = read_corpus_ids(valid_csv)
    test_ids = read_corpus_ids(test_csv)
    forbidden_ids = valid_ids | test_ids

    anchor_map = {}
    skipped_unknown_tokens = 0
    skipped_invalid = 0
    skipped_non_train = 0

    for entry in _collect_anchor_entries(audit):
        if entry.get('eligible') is False:
            continue

        sample_id = _first_present(entry, ('sample_id', 'id', 'sequence_id'))
        frame_idx = _first_present(entry, ('frame_idx', 'frame', 'time_idx'))
        token_id = entry.get('token_id')

        if token_id is None:
            token_text = _first_present(entry, ('token', 'token_text'))
            if token_text not in word_to_id:
                skipped_unknown_tokens += 1
                continue
            token_id = word_to_id[token_text]

        if sample_id is None or frame_idx is None:
            skipped_invalid += 1
            continue

        try:
            sample_id = str(sample_id)
            frame_idx = int(frame_idx)
            token_id = int(token_id)
        except (TypeError, ValueError):
            skipped_invalid += 1
            continue

        if token_id < 0 or token_id >= vocab_size or frame_idx < 0:
            skipped_invalid += 1
            continue

        if sample_id in forbidden_ids:
            raise ValueError(f"Anchor audit contains validation/test sample id {sample_id}")
        if train_ids and sample_id not in train_ids:
            skipped_non_train += 1
            continue

        anchor_map.setdefault(sample_id, []).append((frame_idx, token_id))

    for sample_anchors in anchor_map.values():
        sample_anchors.sort(key=lambda item: item[0])

    total_anchors = sum(len(sample_anchors) for sample_anchors in anchor_map.values())
    unique_tokens = {token_id for sample_anchors in anchor_map.values() for _, token_id in sample_anchors}

    if not total_anchors:
        raise ValueError(f"No usable eligible anchors loaded from {anchor_audit_json}")

    print(f"Loaded {total_anchors} eligible anchors for {len(unique_tokens)} unique tokens across {len(anchor_map)} samples")
    if skipped_unknown_tokens or skipped_invalid or skipped_non_train:
        print(
            "Skipped anchors - unknown tokens: %d, invalid: %d, non-train: %d" %
            (skipped_unknown_tokens, skipped_invalid, skipped_non_train)
        )

    return anchor_map


def compute_anchor_ce(output_context, sample_ids, raw_x_lengths, anchor_map_by_sample_id):
    if sample_ids is None or not anchor_map_by_sample_id:
        return None, 0

    losses = []
    max_time = output_context.size(1)
    device = output_context.device

    for batch_idx, sample_id in enumerate(sample_ids):
        sample_anchors = anchor_map_by_sample_id.get(str(sample_id), [])
        if not sample_anchors:
            continue

        raw_length = int(raw_x_lengths[batch_idx]) if batch_idx < len(raw_x_lengths) else max_time
        valid_time = min(raw_length, max_time)
        for frame_idx, token_id in sample_anchors:
            if frame_idx >= valid_time:
                continue
            target = torch.tensor([token_id], dtype=torch.long, device=device)
            losses.append(F.nll_loss(output_context[batch_idx, frame_idx].unsqueeze(0), target, reduction='sum'))

    if not losses:
        return None, 0

    anchor_loss = torch.stack(losses).sum() / len(losses)
    return anchor_loss, len(losses)


def collect_target_token_ids(y, y_lengths, excluded_ids):
    targets = []
    excluded_ids = set(excluded_ids)
    for batch_idx in range(y.size(0)):
        valid = y[batch_idx, : int(y_lengths[batch_idx])].tolist()
        unique_ids = []
        seen = set()
        for token_id in valid:
            if token_id in excluded_ids or token_id in seen:
                continue
            seen.add(token_id)
            unique_ids.append(token_id)
        targets.append(unique_ids)
    return targets


def pool_token_presence_scores(output_context_btv, effective_lengths, pool_mode):
    max_time = output_context_btv.size(1)
    device = output_context_btv.device
    lengths = torch.as_tensor(effective_lengths, device=device)
    frame_index = torch.arange(max_time, device=device).unsqueeze(0)
    valid_mask = frame_index < lengths.unsqueeze(1)
    masked = output_context_btv.masked_fill(~valid_mask.unsqueeze(-1), float('-inf'))
    if pool_mode == 'max':
        return masked.max(dim=1).values
    if pool_mode == 'logsumexp':
        return torch.logsumexp(masked, dim=1)
    raise ValueError(f'Unsupported token presence pool mode: {pool_mode}')


def select_presence_negative_ids(score_row, positive_ids, negative_count, excluded_ids):
    score_row = score_row.detach()
    vocab_size = score_row.size(0)
    forbidden = set(positive_ids) | set(excluded_ids)
    candidate_ids = [token_id for token_id in range(vocab_size) if token_id not in forbidden]
    if not candidate_ids:
        return []
    candidate_scores = score_row[candidate_ids]
    hard_k = min(negative_count, len(candidate_ids))
    if hard_k <= 0:
        return []
    top_indices = torch.topk(candidate_scores, k=hard_k).indices.tolist()
    return [candidate_ids[idx] for idx in top_indices]


def compute_token_presence_rank_loss(
    pooled_scores,
    target_token_ids,
    excluded_ids,
    negative_count,
    margin,
):
    losses = []
    for batch_idx, positive_ids in enumerate(target_token_ids):
        if not positive_ids:
            continue
        negative_ids = select_presence_negative_ids(
            pooled_scores[batch_idx],
            positive_ids,
            negative_count=negative_count,
            excluded_ids=excluded_ids,
        )
        if not negative_ids:
            continue
        positive_scores = pooled_scores[batch_idx, positive_ids]
        negative_scores = pooled_scores[batch_idx, negative_ids]
        pairwise_margin = margin - positive_scores.unsqueeze(1) + negative_scores.unsqueeze(0)
        losses.append(torch.relu(pairwise_margin).mean())
    if not losses:
        return None
    return torch.stack(losses).mean()


def compute_token_presence_metrics(pooled_scores, target_token_ids, excluded_ids, negative_count):
    top1 = 0
    top5 = 0
    top10 = 0
    total_targets = 0
    rank_values = []
    margin_values = []

    for batch_idx, positive_ids in enumerate(target_token_ids):
        if not positive_ids:
            continue
        row = pooled_scores[batch_idx]
        negative_ids = select_presence_negative_ids(
            row,
            positive_ids,
            negative_count=negative_count,
            excluded_ids=excluded_ids,
        )
        if negative_ids:
            positive_mean = row[positive_ids].mean().item()
            negative_mean = row[negative_ids].mean().item()
            margin_values.append(positive_mean - negative_mean)

        allowed_scores = row.clone()
        if excluded_ids:
            allowed_scores[list(excluded_ids)] = float('-inf')

        for token_id in positive_ids:
            total_targets += 1
            token_score = allowed_scores[token_id]
            rank = int((allowed_scores > token_score).sum().item()) + 1
            rank_values.append(rank)
            if rank == 1:
                top1 += 1
            if rank <= 5:
                top5 += 1
            if rank <= 10:
                top10 += 1

    if total_targets == 0:
        return {
            'target_count': 0,
            'top1_count': 0,
            'top5_count': 0,
            'top10_count': 0,
            'top1_ratio': 0.0,
            'top5_ratio': 0.0,
            'top10_ratio': 0.0,
            'mean_rank': 0.0,
            'median_rank': 0.0,
            'mean_margin': 0.0,
            'rank_sum': 0.0,
            'margin_sum': 0.0,
            'margin_count': 0,
        }

    rank_array = np.array(rank_values, dtype=np.float32)
    return {
        'target_count': total_targets,
        'top1_count': top1,
        'top5_count': top5,
        'top10_count': top10,
        'top1_ratio': top1 / total_targets,
        'top5_ratio': top5 / total_targets,
        'top10_ratio': top10 / total_targets,
        'mean_rank': float(rank_array.mean()),
        'median_rank': float(np.median(rank_array)),
        'mean_margin': float(np.mean(margin_values)) if margin_values else 0.0,
        'rank_sum': float(rank_array.sum()),
        'margin_sum': float(np.sum(margin_values)) if margin_values else 0.0,
        'margin_count': len(margin_values),
    }


def merge_token_presence_metrics(running, batch_metrics):
    running['target_count'] += batch_metrics['target_count']
    running['top1_count'] += batch_metrics['top1_count']
    running['top5_count'] += batch_metrics['top5_count']
    running['top10_count'] += batch_metrics['top10_count']
    running['rank_sum'] += batch_metrics['rank_sum']
    running['margin_sum'] += batch_metrics['margin_sum']
    running['margin_count'] += batch_metrics['margin_count']


def finalize_token_presence_metrics(running):
    target_count = running['target_count']
    margin_count = running['margin_count']
    return {
        'target_count': target_count,
        'top1_count': running['top1_count'],
        'top5_count': running['top5_count'],
        'top10_count': running['top10_count'],
        'top1_ratio': (running['top1_count'] / target_count) if target_count else 0.0,
        'top5_ratio': (running['top5_count'] / target_count) if target_count else 0.0,
        'top10_ratio': (running['top10_count'] / target_count) if target_count else 0.0,
        'mean_rank': (running['rank_sum'] / target_count) if target_count else 0.0,
        'mean_margin': (running['margin_sum'] / margin_count) if margin_count else 0.0,
    }


def empty_token_presence_metrics():
    return {
        'target_count': 0,
        'top1_count': 0,
        'top5_count': 0,
        'top10_count': 0,
        'rank_sum': 0.0,
        'margin_sum': 0.0,
        'margin_count': 0,
    }

# parser helper for optional probabilities in [0, 1]
def optional_probability(value):
    if value is None:
        return None

    if isinstance(value, str) and value.lower() in {'none', 'null'}:
        return None

    try:
        prob = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("value must be a float in [0, 1] or 'none'.")

    if prob < 0.0 or prob > 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1.")

    return prob

###
# Arg parsing
##############


def parse_bool_flag(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {'1', 'true', 't', 'yes', 'y', 'on'}:
        return True
    if lowered in {'0', 'false', 'f', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')

parser = argparse.ArgumentParser(description='Training the transformer-like network')

parser.add_argument('--data', type=str, default=os.path.join('data','phoenix-2014.v3','phoenix2014-release','phoenix-2014-multisigner'),
                   help='location of the data corpus')

parser.add_argument('--train_segment_root',type=str)

parser.add_argument('--val_segment_root',type=str)

parser.add_argument('--data_type', type=str, default='keyfeatures',
                    help='features/resized_features/keyfeatures.')

parser.add_argument('--fixed_padding', type=int, default=None,
                    help='None/64')

parser.add_argument('--local_window', type=int, default=10)

parser.add_argument('--lookup_table', type=str, default=os.path.join('data','slr_lookup.txt'),
                    help='location of the words lookup table')

parser.add_argument('--rescale', type=int, default=224,
                    help='rescale data images.')

parser.add_argument('--random_drop_probability', type=optional_probability, default=0.5,
                    help='probability of frame random drop/0-1 or None')

parser.add_argument('--uniform_drop_probability', type=optional_probability, default=None,
                    help='probability of frame random drop/0-1 or None')

#Put to 0 to avoid memory segementation fault
parser.add_argument('--num_workers', type=int, default=4,
                    help='NOTE: put num of workers to 0 to avoid memory saturation.')

parser.add_argument('--show_sample', action='store_true',
                    help='Show a sample a preprocessed data (sequence of image of sign + translation).')

parser.add_argument('--optimizer', type=str, default='ADAM',
                    help='optimization algo to use; SGD, SGD_LR_SCHEDULE, ADAM / NOAM')

parser.add_argument('--scheduler', type=str, default=None,
                    help='Type of scheduler, multi-step or stepLR')

parser.add_argument('--milestones', default="10,30", type=str,
                    help="milestones for MultiStepLR or stepLR")

parser.add_argument('--weight_decay', type= float , default = 5e-5)

parser.add_argument('--batch_size', type=int, default=2,
                    help='size of one minibatch')

parser.add_argument('--accumulation_steps', type=int, default=1,
                    help='number of micro-batches to accumulate before each optimizer step')

parser.add_argument('--initial_lr', type=float, default=0.0001,
                    help='initial learning rate')

parser.add_argument('--hidden_size', type=int, default=1280,
                    help='size of hidden layers. NOTE: This must be a multiple of n_heads.')

parser.add_argument('--save_best', action='store_true',
                    help='save the model w/ the best validation performance')

parser.add_argument('--num_layers', type=int, default=2,
                    help='number of transformer blocks')

parser.add_argument('--n_heads', type=int, default=10,
                    help='number of self attention heads')

#Pretrained weights
parser.add_argument('--pretrained', type=parse_bool_flag, nargs='?', const=True, default=False,
                    help='embedding layers are pretrained using imagenet')

parser.add_argument('--full_pretrained', type=str, default=None,
                    help='Full frame CNN pretrained')

parser.add_argument('--hand_pretrained', type=str, default=None,
                    help='Hand regions CNN pretrained')

parser.add_argument('--hand_query', action='store_true',
                    help='Set hand as a query for transformer network.')

parser.add_argument('--emb_type', type=str, default='2d',
                    help='Type of image embeddings 2d or 3d.')

parser.add_argument('--emb_network', type=str, default='mb2',
                    help='Image embeddings network: mb2/i3d/m3d')

parser.add_argument('--encoder_type', type=str, default='legacy', choices=['legacy', 'iiga', 'conformer'],
                    help='sequence encoder type; legacy preserves the original CSLR-IIGA encoder path')
parser.add_argument('--conformer_kernel_size', type=int, default=17,
                    help='depthwise convolution kernel size for --encoder_type conformer; must be odd')
parser.add_argument('--segment_attention_mode', type=str, default='on', choices=['on', 'off'],
                    help='legacy encoder only: keep or ablate the existing segment attention sublayer')
parser.add_argument('--log_segment_stats', action='store_true',
                    help='Log first-pass segment proposal statistics from the legacy encoder for diagnostics.')

parser.add_argument('--image_type', type=str, default='rgb',
                    help='Train on rgb/grayscale images')

parser.add_argument('--num_epochs', type=int, default=100,
                    help='number of epochs to stop after')

parser.add_argument('--dp_keep_prob', type=float, default=0.7,
                    help='dropout *keep* probability. drop_prob = 1-dp_keep_prob \
                    (dp_keep_prob=1 means no dropout)')

parser.add_argument('--valid_steps', type=int, default=1, help='Do validation each valid_step')

parser.add_argument('--save_steps', type=int, default=10, help='Save model after each N epoch')

parser.add_argument('--debug', action='store_true')

parser.add_argument('--no_augment', action='store_true',
                    help='Disable training augmentations for overfit/debug runs.')

parser.add_argument('--save_dir', type=str, default='EXPERIMENTATIONS',
                    help='path to save the experimental config, logs, model')

parser.add_argument('--evaluate', action='store_true',
                    help="Evaluate dev set using bleu metric each epoch.")

parser.add_argument('--d_ff', type=int,default=2048)

parser.add_argument('--resume', default=False,
                    help="Resume training from a checkpoint.")
                    
parser.add_argument('--checkpoint',type=str, default=None,
                    help="resume training from a previous checkpoint.")

parser.add_argument('--label_smoothing', type=float, default=0.1,
                    help="label smoothing loss.")

parser.add_argument('--rel_window', type=int, default=None,
                    help="Use local masking window.")

#Training settings
parser.add_argument('--parallel', action='store_true',
                    help='Training on multiple GPUs if available by splitting batches!')

parser.add_argument('--distributed', action='store_true',
                    help='Training on multiple GPUs if available by splitting submodules!')

parser.add_argument('--arch', type=str, default='CNN-attention-CTC',
                    help='Training with structure architecture!, to train on only CNN: CNN-CTC')

parser.add_argument('--freeze_cnn', default= False,
                    help='freeze the feature extractor (CNN)!')

parser.add_argument('--data_stats', type=str, default=None,
                    help="Normalize images using the dataset stats (mean/std).")

parser.add_argument('--hand_stats', type=str, default=None,
                    help="Normalize images using the dataset stats (mean/std).")

parser.add_argument('--wandb', action='store_true',
                    help='Enable optional Weights & Biases logging.')

parser.add_argument('--wandb_entity', type=str, default='ishara-ke',
                    help='Weights & Biases entity/team name.')

parser.add_argument('--wandb_project', type=str, default='CSLR-IIGA',
                    help='Weights & Biases project name.')

parser.add_argument('--wandb_run_name', type=str, default=None,
                    help='Optional Weights & Biases run name.')

parser.add_argument('--wandb_tags', type=str, default='',
                    help='Comma-separated Weights & Biases tags.')

parser.add_argument('--wandb_group', type=str, default=None,
                    help='Optional Weights & Biases group for related train/eval runs.')

parser.add_argument('--wandb_job_type', type=str, default='train',
                    help='Weights & Biases job type, e.g. train, preview, eval.')

parser.add_argument('--wandb_system_sample_seconds', type=float, default=5.0,
                    help='W&B system/GPU stats sampling interval in seconds; set <=0 to use W&B default.')
parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'],
                    help='W&B mode when --wandb is set.')
parser.add_argument('--wandb_dir', type=str, default=None,
                    help='Optional directory for W&B run files, useful for offline mode.')
parser.add_argument('--offline_registry_root', type=str, default=None,
                    help='Optional local registry root for W&B-independent run records.')

parser.add_argument('--progress', choices=['epoch', 'bar', 'none'], default='epoch',
                    help='Progress display mode. Use epoch for log-friendly summaries, bar for interactive redraws, none to suppress progress lines.')

parser.add_argument('--ctc_blank_logit_penalty', type=float, default=0.0,
                    help='Subtract this value from the CTC blank logit before loss/decode. Default 0 keeps baseline behavior.')

parser.add_argument('--interctc_weight', type=float, default=0.0,
                    help='Weight for optional intermediate encoder CTC loss. Default 0 keeps baseline behavior.')

parser.add_argument('--interctc_layer', type=int, default=None,
                    help='Zero-based encoder layer index for optional intermediate CTC. Default uses the middle encoder layer.')

parser.add_argument('--cr_ctc_weight', type=float, default=0.0,
                    help='Weight for optional consistency-regularized CTC. Default 0 keeps baseline behavior.')

parser.add_argument('--visual_ctc_weight', type=float, default=0.0,
                    help='Weight for optional visual-embedding CTC loss before the Transformer. Default 0 keeps baseline behavior.')

parser.add_argument('--anchor_ce_weight', type=float, default=0.0,
                    help='Weight for optional evidence-gated anchor-frame CE loss. Default 0 keeps baseline behavior.')

parser.add_argument('--anchor_audit_json', type=str, default=None,
                    help='Path to a train-only anchor audit JSON used when --anchor_ce_weight is enabled.')

parser.add_argument('--anchor_log_every_step', action='store_true',
                    help='Log anchor CE visibility on every step when anchor CE is enabled. Intended for tiny debug/smoke runs.')

parser.add_argument('--anchor_debug_sample_ids', type=str, default=None,
                    help='Comma-separated sample IDs to log when they appear in training batches during anchor CE debugging.')

parser.add_argument('--pose_root', type=str, default=None,
                    help='Optional root containing prepared pose sidecars under pose_landmarks/<split>/<sample_id>/1/.')
parser.add_argument('--pose_fusion_mode', type=str, default='off', choices=['off', 'add'],
                    help='Optional pose sidecar fusion mode. off keeps the RGB baseline unchanged.')
parser.add_argument('--token_presence_rank_weight', type=float, default=0.0,
                    help='Weight for optional token-presence ranking loss over pooled vocabulary evidence. Default 0 keeps baseline behavior.')
parser.add_argument('--token_presence_negative_count', type=int, default=32,
                    help='Number of hardest non-target tokens to compare against per sample for token presence ranking.')
parser.add_argument('--token_presence_pool', type=str, default='logsumexp', choices=['max', 'logsumexp'],
                    help='Pooling mode over time for token presence evidence.')
parser.add_argument('--token_presence_margin', type=float, default=0.2,
                    help='Margin used by the token presence ranking loss.')


#----------------------------------------------------------------------------------------


## SET EXPERIMENTATION AND SAVE CONFIGURATION

#Same seed for reproducibility)
parser.add_argument('--seed', type=int, default=1111, help='random seed')

#Save folder with the date
start_date = dt.datetime.now().strftime("%Y-%m-%d-%H.%M")
print ("Start Time: "+start_date)

args = parser.parse_args()

if args.accumulation_steps < 1:
    parser.error('--accumulation_steps must be at least 1.')

if args.encoder_type == 'conformer' and args.hand_query:
    parser.error('--encoder_type conformer is not supported with --hand_query in the first Conformer branch.')

if args.encoder_type != 'legacy' and args.encoder_type != 'iiga' and args.segment_attention_mode != 'on':
    parser.error('--segment_attention_mode off is only supported with --encoder_type legacy.')

if args.encoder_type not in {'legacy', 'iiga'} and args.log_segment_stats:
    parser.error('--log_segment_stats is only supported with --encoder_type legacy or iiga.')

if args.encoder_type == 'iiga' and args.segment_attention_mode != 'on':
    parser.error('--encoder_type iiga always uses the explicit local+segment path with segment attention enabled.')

if args.anchor_ce_weight < 0.0:
    parser.error('--anchor_ce_weight must be non-negative.')

if args.anchor_ce_weight > 0.0:
    if not args.anchor_audit_json:
        parser.error('--anchor_audit_json is required when --anchor_ce_weight is enabled.')
    if args.hand_query:
        parser.error('--anchor_ce_weight is not supported with --hand_query in the first anchor branch.')
    if args.distributed:
        parser.error('--anchor_ce_weight is not supported with --distributed in the first anchor branch.')
    if args.interctc_weight > 0.0 or args.cr_ctc_weight > 0.0 or args.visual_ctc_weight > 0.0:
        parser.error('--anchor_ce_weight must not be combined with InterCTC, CR-CTC, or visual CTC in the first anchor branch.')
    if args.ctc_blank_logit_penalty != 0.0:
        parser.error('--anchor_ce_weight must not be combined with --ctc_blank_logit_penalty in the first anchor branch.')

if args.pose_fusion_mode != 'off' and not args.pose_root:
    parser.error('--pose_root is required when --pose_fusion_mode is enabled.')

if args.pose_fusion_mode != 'off' and args.hand_query:
    parser.error('--pose_fusion_mode is not supported with --hand_query in the first pose branch.')

if args.token_presence_rank_weight < 0.0:
    parser.error('--token_presence_rank_weight must be non-negative.')

if args.token_presence_negative_count < 1:
    parser.error('--token_presence_negative_count must be at least 1.')

if args.token_presence_margin < 0.0:
    parser.error('--token_presence_margin must be non-negative.')

if args.token_presence_rank_weight > 0.0:
    if args.hand_query:
        parser.error('--token_presence_rank_weight is not supported with --hand_query in the first token presence branch.')
    if args.distributed:
        parser.error('--token_presence_rank_weight is not supported with --distributed in the first token presence branch.')

#Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

#experiment_path = PureWindowsPath('EXPERIMENTATIONS\\' + start_date)
base_save_dir = args.save_dir if args.save_dir else 'EXPERIMENTATIONS'
experiment_path = os.path.join(base_save_dir, start_date)

# Creates an experimental directory and dumps all the args to a text file
if(os.path.exists(experiment_path)):
    print('Experiment already exists..')
    quit(0)
else:
    os.makedirs(experiment_path)

print ("\nPutting log in %s"%experiment_path)

args.save_dir = experiment_path

#Dump all configurations/hyperparameters in txt
with open (os.path.join(experiment_path,'exp_config.txt'), 'w') as f:
    f.write('Experimentation done at: '+ str(start_date)+' with current configurations:\n')
    for arg in vars(args):
        f.write(arg+' : '+str(getattr(args, arg))+'\n')

wandb_run = init_wandb(args, start_date)
offline_registry = init_offline_registry(
    args.offline_registry_root,
    run_id=args.wandb_run_name or start_date,
    config=vars(args),
    metadata={
        'start_date': start_date,
        'save_dir': args.save_dir,
        'exp_config_path': os.path.join(args.save_dir, 'exp_config.txt'),
    },
)

#-------------------------------------------------------------------------------
device = select_device(verbose=True, context=f"Training {args.arch}")
telemetry_poller = DeviceTelemetryPoller(device)
#--------------------------------------------------------------------------------


#Computation for one epoch
def run_epoch(model, data, is_train=False, device=None, n_devices=1):

    if device is None:
        raise ValueError('run_epoch requires an explicit device')

    if is_train:
        model.train()  # Set model to training mode
        print ("Training..")
        phase='train'
    else:
        model.eval()   # Set model to evaluate mode
        print ("Evaluating..")
        phase='valid'

    start_time = time.time()
    data_len = len(data)
    accumulation_steps = max(1, args.accumulation_steps)

    if is_train:
        zero_optimizer_grad(optimizer, set_to_none=True)

    loss = 0.0
    total_loss = 0.0
    total_tokens = 0
    batch_tokens = 0.0
    total_seqs = 0
    tokens = 0
    total_wer_score = 0.0
    count = 0
    token_presence_running = empty_token_presence_metrics()
    token_presence_loss_total = 0.0
    token_presence_loss_count = 0

    gt = []
    hyp = []

    anchor_debug_target_ids = set()
    if args.anchor_debug_sample_ids:
        anchor_debug_target_ids = {
            sample_id.strip() for sample_id in args.anchor_debug_sample_ids.split(',') if sample_id.strip()
        }

    # Progress bars redraw repeatedly and flood persistent logs when output is captured
    # through Jupyter/tee. Keep epoch summaries as the default and make bars opt-in.
    bar = None
    if args.progress == 'bar':
        if progressbar is None:
            raise ImportError("progressbar2 is required when --progress bar is used")
        bar = progressbar.ProgressBar(maxval=dataset_sizes[phase], widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
    j = 0
    #Loop over minibatches
    for step, batch_data in enumerate(data):

        pose_landmarks = None
        if len(batch_data) == 8:
            x, x_lengths, y, y_lengths, hand_regions, hand_lengths, pose_landmarks, sample_ids = batch_data
        elif len(batch_data) == 7 and args.pose_fusion_mode != 'off':
            x, x_lengths, y, y_lengths, hand_regions, hand_lengths, pose_landmarks = batch_data
            sample_ids = None
        elif len(batch_data) == 7:
            x, x_lengths, y, y_lengths, hand_regions, hand_lengths, sample_ids = batch_data
        elif len(batch_data) == 6:
            x, x_lengths, y, y_lengths, hand_regions, hand_lengths = batch_data
            sample_ids = None
        else:
            raise ValueError(f"Unexpected batch structure with {len(batch_data)} values")

        raw_x_lengths = list(x_lengths)

        #Update progress bar with every iter
        j += len(x)
        if bar is not None:
            bar.update(j)

        #print(x.size())
        y = torch.from_numpy(y).to(device)
        x = x.to(device)

        if pose_landmarks is not None:
            pose_landmarks = {
                'pose': pose_landmarks['pose'].to(device),
                'left_hand': pose_landmarks['left_hand'].to(device),
                'right_hand': pose_landmarks['right_hand'].to(device),
                'frame_names': pose_landmarks['frame_names'],
            }

        if(args.hand_query):
             hand_regions = hand_regions.to(device)
        else:
             hand_regions = None

        #NOTE: clone y to avoid overridding it
        batch = Batch(x_lengths, y_lengths, hand_lengths, trg=None, emb_type=args.emb_type, DEVICE=device, fixed_padding=args.fixed_padding, rel_window=args.rel_window)

        output_interctc = None
        output_visual_ctc = None
        output_context_cr = None
        output_cr = None

        if(args.distributed):

            src_emb, _, _ = feature_extractor(x)
            src_emb = position(src_emb)
            src_emb = encoder(src_emb, None, batch.src_mask)
            output_context = output_layer(src_emb)

            if(args.hand_query):
                hand_emb = hand_extractor(hand_regions)
                hand_emb = position(hand_emb)
                hand_emb = encoder(hand_emb, None, batch.src_mask)
                output_hand = output_layer(hand_emb)

                comb_emb = encoder(src_emb, hand_emb, batch.rel_mask)
                output = output_layer(comb_emb)

            else:
                output = None
                output_hand = None

        else:

            #Shape(batch_size, tgt_seq_length, tgt_vocab_size)
            #NOTE: no need for trg if we dont have a decoder
            use_interctc = args.interctc_weight > 0 and not args.hand_query
            use_visual_ctc = args.visual_ctc_weight > 0 and not args.hand_query
            use_cr_ctc = args.cr_ctc_weight > 0 and is_train and not args.hand_query

            if use_interctc and use_visual_ctc:
                output, output_context, output_hand, output_interctc, output_visual_ctc = model.forward(
                    x,
                    batch.src_mask,
                    batch.rel_mask,
                    hand_regions,
                    args.arch,
                    return_intermediate_ctc=True,
                    intermediate_ctc_layer=args.interctc_layer,
                    return_visual_ctc=True,
                    pose_landmarks=pose_landmarks,
                )
            elif use_interctc:
                output, output_context, output_hand, output_interctc = model.forward(
                    x,
                    batch.src_mask,
                    batch.rel_mask,
                    hand_regions,
                    args.arch,
                    return_intermediate_ctc=True,
                    intermediate_ctc_layer=args.interctc_layer,
                    pose_landmarks=pose_landmarks,
                )
            elif use_visual_ctc:
                output, output_context, output_hand, output_visual_ctc = model.forward(
                    x,
                    batch.src_mask,
                    batch.rel_mask,
                    hand_regions,
                    args.arch,
                    return_visual_ctc=True,
                    pose_landmarks=pose_landmarks,
                )
            else:
                output, output_context, output_hand = model.forward(x, batch.src_mask, batch.rel_mask, hand_regions, args.arch, pose_landmarks=pose_landmarks)

            if use_cr_ctc:
                _, output_context_cr, _ = model.forward(x, batch.src_mask, batch.rel_mask, hand_regions, args.arch, pose_landmarks=pose_landmarks)

        #CTC loss expects (Seq, batch, vocab)
        if(args.hand_query):
            output = output.transpose(0,1)
            output_context = output_context.transpose(0,1)
            output_hand = output_hand.transpose(0,1)
        else:
            output = output_context.transpose(0,1)
            if output_interctc is not None:
                output_interctc = output_interctc.transpose(0,1)
            if output_visual_ctc is not None:
                output_visual_ctc = output_visual_ctc.transpose(0,1)
            if output_context_cr is not None:
                output_cr = output_context_cr.transpose(0,1)

        if args.ctc_blank_logit_penalty:
            output = output.clone()
            output[:, :, blank_index] -= args.ctc_blank_logit_penalty
            if output_cr is not None:
                output_cr = output_cr.clone()
                output_cr[:, :, blank_index] -= args.ctc_blank_logit_penalty
            if output_interctc is not None:
                output_interctc = output_interctc.clone()
                output_interctc[:, :, blank_index] -= args.ctc_blank_logit_penalty
            if output_visual_ctc is not None:
                output_visual_ctc = output_visual_ctc.clone()
                output_visual_ctc[:, :, blank_index] -= args.ctc_blank_logit_penalty
            if args.hand_query:
                output_context = output_context.clone()
                output_hand = output_hand.clone()
                output_context[:, :, blank_index] -= args.ctc_blank_logit_penalty
                output_hand[:, :, blank_index] -= args.ctc_blank_logit_penalty

        effective_lengths = effective_ctc_lengths(
            raw_x_lengths,
            local_window=args.local_window,
            emb_network=args.emb_network,
            output_time=output.size(0),
            reduction=getattr(model.src_emb, 'temporal_reduction', 1),
        )

        x_lengths = torch.IntTensor(effective_lengths)
        y_lengths = torch.IntTensor(y_lengths)


        if not is_train:
            decoded_preds = decode_ctc_batch(output, x_lengths, blank_index)

            for i, p in enumerate(decoded_preds):
                ys = y[i, :y_lengths[i]]

                hyp = ids_to_text(p, vocab, ignore_ids=[blank_index, pad_index])
                gt = ids_to_text(ys.tolist(), vocab, ignore_ids=[blank_index, pad_index])

                total_wer_score += wer(gt, hyp)
                count += 1

        #output (Seq, batch, vocab_size)
        #y (batch, trg_size)
        #x_lengths (batch)
        #y_lengths (batch)

        #NOTE: produce Nan values if x length > y lengths
        #When extracting keyframes, make sure your src lengths are long enough or simply use zero infinity
        #Doing average loss here

        #IMPORTANT: Use Pytorch CTCloss
        #print(output.shape)
        #print(y.shape)
        token_presence_loss = None
        if args.token_presence_rank_weight > 0.0:
            pooled_scores = pool_token_presence_scores(
                output_context,
                effective_lengths,
                pool_mode=args.token_presence_pool,
            )
            target_token_ids = collect_target_token_ids(
                y,
                y_lengths,
                excluded_ids={pad_index, blank_index},
            )
            batch_presence_metrics = compute_token_presence_metrics(
                pooled_scores,
                target_token_ids,
                excluded_ids={pad_index, blank_index},
                negative_count=args.token_presence_negative_count,
            )
            merge_token_presence_metrics(token_presence_running, batch_presence_metrics)

            token_presence_loss = compute_token_presence_rank_loss(
                pooled_scores,
                target_token_ids,
                excluded_ids={pad_index, blank_index},
                negative_count=args.token_presence_negative_count,
                margin=args.token_presence_margin,
            )

        if output_cr is not None:
            ctc_loss_a = ctc_loss(output, y.cpu(), x_lengths.cpu(), y_lengths.cpu())
            ctc_loss_b = ctc_loss(output_cr, y.cpu(), x_lengths.cpu(), y_lengths.cpu())
            consistency_loss = cr_ctc_consistency_loss(output, output_cr, x_lengths)
            loss = 0.5 * (ctc_loss_a + ctc_loss_b) + args.cr_ctc_weight * consistency_loss
        else:
            loss = ctc_loss(output, y.cpu(), x_lengths.cpu(), y_lengths.cpu())

        if output_interctc is not None:
            interctc_loss = ctc_loss(output_interctc, y.cpu(), x_lengths.cpu(), y_lengths.cpu())
            loss = loss + args.interctc_weight * interctc_loss

        if output_visual_ctc is not None:
            visual_ctc_loss = ctc_loss(output_visual_ctc, y.cpu(), x_lengths.cpu(), y_lengths.cpu())
            loss = loss + args.visual_ctc_weight * visual_ctc_loss

        if token_presence_loss is not None and torch.isfinite(token_presence_loss):
            loss = loss + args.token_presence_rank_weight * token_presence_loss
            token_presence_loss_total += token_presence_loss.detach().item()
            token_presence_loss_count += 1

        if(args.hand_query):
            loss += ctc_loss(output_context, y.cpu(), x_lengths.cpu(), y_lengths.cpu())
            loss += ctc_loss(output_hand, y.cpu(), x_lengths.cpu(), y_lengths.cpu())
            loss = loss / 3

        last_anchor_ce = None
        last_anchor_count = 0
        anchor_debug_hits = []
        if is_train and args.anchor_ce_weight > 0.0:
            if sample_ids is not None and anchor_debug_target_ids:
                debug_raw_lengths = [int(length) for length in raw_x_lengths]
                for batch_idx, sample_id in enumerate(sample_ids):
                    sample_id_str = str(sample_id)
                    if sample_id_str not in anchor_debug_target_ids:
                        continue
                    sample_anchor_pairs = anchor_map_by_sample_id.get(sample_id_str, [])
                    raw_length = debug_raw_lengths[batch_idx]
                    max_time = output_context.size(1)
                    valid_time = min(raw_length, max_time)
                    in_bounds = [pair for pair in sample_anchor_pairs if pair[0] < valid_time]
                    out_of_bounds = [pair for pair in sample_anchor_pairs if pair[0] >= valid_time]
                    anchor_debug_hits.append({
                        'sample_id': sample_id_str,
                        'raw_length': raw_length,
                        'valid_time': valid_time,
                        'anchor_count': len(sample_anchor_pairs),
                        'in_bounds_anchor_count': len(in_bounds),
                        'out_of_bounds_anchor_count': len(out_of_bounds),
                        'anchors': sample_anchor_pairs,
                    })
            anchor_ce, anchor_count = compute_anchor_ce(output_context, sample_ids, raw_x_lengths, anchor_map_by_sample_id)
            if anchor_count > 0 and torch.isfinite(anchor_ce):
                loss = loss + args.anchor_ce_weight * anchor_ce
                last_anchor_ce = anchor_ce.detach().item()
                last_anchor_count = anchor_count

        total_loss += loss.detach()
        total_seqs += batch.seq
        total_tokens += (y != blank_index).data.sum()
        tokens += (y != blank_index).data.sum()
        batch_tokens += (y != blank_index).data.sum()

        if is_train:
            window_start = (step // accumulation_steps) * accumulation_steps
            window_end = min(window_start + accumulation_steps, data_len)
            current_window_size = window_end - window_start
            scaled_loss = loss / current_window_size

            scaled_loss.backward()

            should_step = ((step + 1) % accumulation_steps == 0) or (step + 1 == data_len)
            if should_step:
                #Weight clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

                optimizer.step()
                zero_optimizer_grad(optimizer, set_to_none=True)

            should_log_anchor_step = args.anchor_ce_weight > 0.0 and args.anchor_log_every_step

            if step % 100 == 0 or should_log_anchor_step:
                elapsed = time.time() - start_time
                telemetry = format_device_telemetry(telemetry_poller.sample())
                print("Step: %d, Loss: %f, Frame per Sec: %f, Token per sec: %f"%
                      (step, (loss.detach() / batch_tokens), total_seqs * batch_size / elapsed, tokens / elapsed))
                if args.token_presence_rank_weight > 0.0:
                    live_presence = finalize_token_presence_metrics(token_presence_running)
                    print(
                        "Token presence: avg_loss=%.6f, top5=%d/%d, top10=%d/%d, mean_rank=%.2f, mean_margin=%.4f" % (
                            (token_presence_loss_total / token_presence_loss_count) if token_presence_loss_count else 0.0,
                            live_presence['top5_count'],
                            live_presence['target_count'],
                            live_presence['top10_count'],
                            live_presence['target_count'],
                            live_presence['mean_rank'],
                            live_presence['mean_margin'],
                        )
                    )
                if args.anchor_ce_weight > 0.0:
                    if last_anchor_ce is None:
                        print("Anchor CE: skipped, Anchors: 0")
                    else:
                        print("Anchor CE: %.6f, Anchors: %d" % (last_anchor_ce, last_anchor_count))
                    if anchor_debug_hits:
                        print("Anchor debug hits: " + json.dumps(anchor_debug_hits, sort_keys=True))
                if telemetry:
                    print(f"Telemetry: {telemetry}")

                start_time = time.time()
                total_seqs = 0
                tokens = 0

        batch_tokens = 0.0

        #Free some memory
        #NOTE: this helps alot in avoiding cuda out of memory
        del loss, output, output_context, output_hand, output_interctc, output_visual_ctc, output_context_cr, output_cr, token_presence_loss, y, hand_regions, pose_landmarks, batch

    if bar is not None:
        bar.finish()

    if args.progress == 'epoch':
        print(f"{phase.capitalize()} epoch complete: {j}/{dataset_sizes[phase]} examples")

    if(is_train):
        print("Average Loss: %f" %(total_loss.item() / total_tokens.item()))
        return total_loss.item() / total_tokens.item(), finalize_token_presence_metrics(token_presence_running), ((token_presence_loss_total / token_presence_loss_count) if token_presence_loss_count else 0.0)

    else:
        #Measure WER of all dataset
        print('Measuring WER..')
        print("Average WER: %f" %(total_wer_score/count))

        return total_loss.item() / total_tokens.item(), total_wer_score/count, finalize_token_presence_metrics(token_presence_running), ((token_presence_loss_total / token_presence_loss_count) if token_presence_loss_count else 0.0)
#-------------------------------------------------------------------------------------------------------

### LOAD DATALOADERS

# In debug mode, try batch size of 1
if args.debug:
    batch_size = 1
else:
    batch_size = args.batch_size


#Train on rgb/grayscale images
if(args.image_type == 'rgb'):
    channels = 3
#Not supported yet
elif(args.image_type == 'grayscale'):
    channels = 1
else:
    print('Image type is ot supported!')
    quit(0)


train_path, valid_path, test_path = path_data(data_path=args.data, task='SLR', features_type=args.data_type, hand_query=args.hand_query)


#Load stats
if(args.data_stats):
    args.data_stats = torch.load(args.data_stats, map_location=torch.device('cpu'))

if(args.hand_query and args.hand_stats):
    if os.path.exists(args.hand_stats):
        args.hand_stats = torch.load(args.hand_stats, map_location=torch.device('cpu'))
    else:
        print(f"WARNING: hand_stats file not found at {args.hand_stats}. Continuing without hand normalization.")
        args.hand_stats = None
else:
    args.hand_stats = None

#Pass the annotation + image sequences locations
train_dataloader, train_size = loader(csv_file=train_path[1],
                root_dir=train_path[0],
                segment_path=args.train_segment_root,
                lookup=args.lookup_table,
                rescale = args.rescale,
                batch_size = batch_size,
                num_workers = args.num_workers,
                random_drop= args.random_drop_probability,
                uniform_drop= args.uniform_drop_probability,
                show_sample = args.show_sample,
                istrain=not args.no_augment,
                fixed_padding=args.fixed_padding,
                hand_dir=train_path[2],
                data_stats=args.data_stats,
                hand_stats=args.hand_stats,
                channels=channels,
                return_sample_ids=args.anchor_ce_weight > 0.0,
                pose_root=os.path.join(args.pose_root, 'train') if args.pose_root else None,
                return_pose_landmarks=args.pose_fusion_mode != 'off'
                )

#No data augmentation for valid data
valid_dataloader, valid_size = loader(csv_file=valid_path[1],
                root_dir=valid_path[0],
                segment_path=args.val_segment_root,
                lookup=args.lookup_table,
                rescale = args.rescale,
                batch_size = args.batch_size,
                num_workers = args.num_workers,
                random_drop= args.random_drop_probability,
                uniform_drop= args.uniform_drop_probability,
                show_sample = args.show_sample,
                istrain=False,
                fixed_padding=args.fixed_padding,
                hand_dir=valid_path[2],
                data_stats=args.data_stats,
                hand_stats=args.hand_stats,
                channels=channels,
                pose_root=os.path.join(args.pose_root, 'dev') if args.pose_root else None,
                return_pose_landmarks=args.pose_fusion_mode != 'off'
                )

print('Dataset sizes:')
dataset_sizes = {}
dataset_sizes.update({'train':train_size})
dataset_sizes.update({'valid':valid_size})
print(dataset_sizes)

#Retrieve size of target vocab
with open(args.lookup_table, 'rb') as pickle_file:
   vocab = pickle.load(pickle_file)

word_to_id = vocab
vocab_size = len(word_to_id)
pad_index = word_to_id.get('<PAD>', 0)
blank_index = word_to_id.get('<BLANK>', vocab_size - 1)

anchor_map_by_sample_id = {}
if args.anchor_ce_weight > 0.0:
    anchor_map_by_sample_id = load_anchor_audit(
        args.anchor_audit_json,
        word_to_id,
        vocab_size,
        train_path[1],
        valid_path[1],
        test_path[1],
    )

#Switch keys and values of vocab to easily look for words
vocab = {y:x for x,y in word_to_id.items()}

#You should find
print('vocabulary size:' + str(vocab_size))

token_presence_dev_target_count = 0
if args.token_presence_rank_weight > 0.0:
    valid_token_rows = load_corpus_rows(valid_path[1])
    excluded_token_ids = {pad_index, blank_index}
    valid_target_token_ids = set()
    for row in valid_token_rows:
        for token in row['tokens']:
            token_id = word_to_id.get(token)
            if token_id is None or token_id in excluded_token_ids:
                continue
            valid_target_token_ids.add(token_id)
    token_presence_dev_target_count = len(valid_target_token_ids)
    print('token presence dev target count:' + str(token_presence_dev_target_count))

#-----------------------------------------------------------------------------------------------------------------

#Load the whole model
model = TRANSFORMER(tgt_vocab=vocab_size, n_stacks=args.num_layers, n_units=args.hidden_size,
                            n_heads=args.n_heads, window_size=args.local_window ,d_ff=args.d_ff, dropout=1.-args.dp_keep_prob, image_size=args.rescale, pretrained=args.pretrained,
                            emb_type=args.emb_type, emb_network=args.emb_network,
                            full_pretrained=args.full_pretrained, hand_pretrained=args.hand_pretrained, freeze_cnn=args.freeze_cnn, channels=channels,
                            encoder_type=args.encoder_type, conformer_kernel_size=args.conformer_kernel_size,
                            segment_attention_mode=args.segment_attention_mode, log_segment_stats=args.log_segment_stats,
                            pose_fusion_mode=args.pose_fusion_mode)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print('model parameters:',trainable_params)
#summary(model,(64,224,224,3), batch_size=-1, device='cuda')
#print(model)
#Load model on GPU or multiple GPUs
if device.type == 'cuda' and torch.cuda.device_count() > 1 and args.parallel:
    #How many GPUs you are using
    n_devices = torch.cuda.device_count()


    if(args.distributed):
        #Split GPUs for both feature extraction and sequence learning (Transformer)
        n_devices_split = int(n_devices/2)
        print("Using ", n_devices_split, "GPUs for feature extraction and ", n_devices-n_devices_split, "GPUs for sequence learning.")

        devices = list(range(0, n_devices_split))
        feature_extractor = nn.DataParallel(model.src_emb, device_ids=devices).to(device)

        if(args.hand_query):
             hand_extractor = nn.DataParallel(model.hand_emb, device_ids=devices).to(device)

        devices = list(range(n_devices_split, n_devices))

        encoder = nn.DataParallel(model.encoder, device_ids=devices).to(n_devices_split)
        position = nn.DataParallel(model.position, device_ids=devices).to(n_devices_split)
        output_layer = nn.DataParallel(model.output_layer, device_ids=devices).to(n_devices_split)

    else:
        print("Using ", n_devices, "GPUs!, Let's GO!")
        model = nn.DataParallel(model).to(device)
else:
    print("Training using 1 device (GPU/CPU), use very small batch_size!")
    #Load model into device (GPU OR CPU)
    n_devices = 1
    model = model.to(device)

    if(args.distributed):
        print("Can't use distributed training since you have a single GPU!")
        quit(0)


#print("Loading to GPUs")
#print(GPUtil.showUtilization())

train_ppls = []
train_losses = []
val_ppls = []
val_losses = []
ns_words = []
bleu_1s = []
bleu_2s = []
bleu_3s = []
bleu_4s = []

best_val_so_far = np.inf
best_bleu = 0.0
best_err_so_far = 999.9
times = []

if args.optimizer == 'ADAM':
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.initial_lr , weight_decay = args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.initial_lr)

elif args.optimizer == 'noam':
    optimizer = NoamOpt(args.hidden_size, 1, 400, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


# In debug mode, only run one epoch
if args.debug:
    num_epochs = 1
else:
    num_epochs = args.num_epochs

#Load weights from previous training session
#Resume training or start from start w/ pretrained weights
if(args.checkpoint):
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    if 'interctc_output_layer.proj.weight' not in checkpoint['model_state_dict']:
        checkpoint['model_state_dict']['interctc_output_layer.proj.weight'] = checkpoint['model_state_dict']['output_layer.proj.weight'].clone()
        checkpoint['model_state_dict']['interctc_output_layer.proj.bias'] = checkpoint['model_state_dict']['output_layer.proj.bias'].clone()
    load_result = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if load_result.missing_keys:
        print('Missing checkpoint keys initialized from current model: ' + ', '.join(load_result.missing_keys))
    if load_result.unexpected_keys:
        print('Unexpected checkpoint keys ignored: ' + ', '.join(load_result.unexpected_keys))

    if(args.resume):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss_fn = checkpoint['loss']
        best_bleu = checkpoint['best_wer']
        #scheduler =  checkpoint['scheduler']
        milestones = [int(v.strip()) for v in args.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

if args.checkpoint is None or not args.resume:
    start_epoch = 0

    if args.scheduler == 'multi-step':
        milestones = [int(v.strip()) for v in args.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    elif args.scheduler == 'stepLR':
        scheduler = StepLR(optimizer, step_size=args.milestones, gamma=0.1)
    else:
        print('No scheduler!')

    if(args.label_smoothing):
        loss_fn = LabelSmoothing(size=len(vocab), padding_idx=0, smoothing=args.label_smoothing)
    else:
        loss_fn = nn.NLLLoss(ignore_index=0, size_average=False)

#zero_infinity to avoid having numerical instabilities
#NOTE: N-class - 1 is for BLANK token if we are using tensorflow decoder
ctc_loss = nn.CTCLoss(blank=blank_index, reduction='sum', zero_infinity=True)


###
#Main Training loop

for epoch in range(start_epoch, num_epochs):

    start = time.time()

    print('\nEPOCH '+str(epoch)+' ------------------')
    #print('LR',scheduler.get_lr())
    print(current_learning_rate(optimizer))
    # RUN MODEL ON TRAINING DATA
    train_loss, train_presence_metrics, train_presence_loss = run_epoch(model, train_dataloader, True, device=device)
    print("After train epoch..")
    print_device_utilization(device)

    #Save perplexity
    train_ppl = np.exp(train_loss)

    if(args.scheduler):
        scheduler.step()
    
    if(args.valid_steps > 0 and epoch % args.valid_steps == 0):

        #Use it for evaluation with blue
        translation_corpus = []
        reference_corpus = []

        #RUN MODEL ON VALIDATION DATA
        #NOTE: Helps with avoiding memory saturation
        with torch.no_grad():
            val_loss, word_err, valid_presence_metrics, valid_presence_loss = run_epoch(model, valid_dataloader, device=device)

            if word_err < best_err_so_far:
                best_err_so_far = word_err

                #if args.save_best:
                print("Saving entire model with best params")
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_fn,
                'best_wer': best_err_so_far
                },
                os.path.join(args.save_dir, 'BEST.pt'))

                print("Saving full-frame (CNN) with best params")
                torch.save(model.src_emb.state_dict(), os.path.join(args.save_dir, 'full_cnn_best_params.pt'))

                if(args.hand_query):
                    print("Saving hand regions (CNN) with best params")
                    torch.save(model.hand_emb.state_dict(), os.path.join(args.save_dir, 'hand_cnn_best_params.pt'))

        val_ppl = np.exp(val_loss)
        
        # SAVE RESULTS
        train_ppls.append(train_ppl)
        val_ppls.append(val_ppl)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        times.append(time.time() - start)
        ns_words.append(word_err)

        log_str = 'epoch: ' + str(epoch) + '\t' \
             + 'train ppl: ' + str(train_ppl) + '\t' \
             + 'val ppl: ' + str(val_ppl) + '\t' \
             + 'train loss: ' + str(train_loss) + '\t' \
             + 'val loss: ' + str(val_loss) + '\t' \
             + 'WER: ' + str(word_err) + '\t' \
            + 'BEST WER: ' + str(best_err_so_far) + '\t' \
            + 'time (s) spent in epoch: ' + str(times[-1])

        print(log_str)

        epoch_metrics = {
            'epoch': epoch,
            'learning_rate': float(current_learning_rate(optimizer)),
            'train/ppl': float(train_ppl),
            'train/loss': float(train_loss),
            'valid/ppl': float(val_ppl),
            'valid/loss': float(val_loss),
            'valid/wer': float(word_err),
            'valid/best_wer': float(best_err_so_far),
            'time/epoch_seconds': float(times[-1]),
        }
        if args.token_presence_rank_weight > 0.0:
            epoch_metrics.update({
                'train/token_presence_loss': float(train_presence_loss),
                'train/token_presence_top1': float(train_presence_metrics['top1_ratio']),
                'train/token_presence_top5': float(train_presence_metrics['top5_ratio']),
                'train/token_presence_top10': float(train_presence_metrics['top10_ratio']),
                'train/token_presence_mean_rank': float(train_presence_metrics['mean_rank']),
                'train/token_presence_mean_margin': float(train_presence_metrics['mean_margin']),
                'train/token_presence_target_count': int(train_presence_metrics['target_count']),
                'valid/token_presence_loss': float(valid_presence_loss),
                'valid/token_presence_top1': float(valid_presence_metrics['top1_ratio']),
                'valid/token_presence_top5': float(valid_presence_metrics['top5_ratio']),
                'valid/token_presence_top10': float(valid_presence_metrics['top10_ratio']),
                'valid/token_presence_mean_rank': float(valid_presence_metrics['mean_rank']),
                'valid/token_presence_mean_margin': float(valid_presence_metrics['mean_margin']),
                'valid/token_presence_target_count': int(valid_presence_metrics['target_count']),
            })
        log_wandb(wandb_run, epoch_metrics)
        if offline_registry is not None:
            try:
                offline_registry.log_metrics(epoch_metrics)
            except Exception as exc:
                print(f"Warning: Offline registry logging failed: {exc}")

        with open (os.path.join(args.save_dir, 'log.txt'), 'a') as f_:
                f_.write(log_str+ '\n')
                if args.token_presence_rank_weight > 0.0:
                    f_.write(
                        'token presence:\t'
                        + 'train loss: ' + str(train_presence_loss) + '\t'
                        + 'train top5: ' + str(train_presence_metrics['top5_count']) + '/' + str(train_presence_metrics['target_count']) + '\t'
                        + 'train top10: ' + str(train_presence_metrics['top10_count']) + '/' + str(train_presence_metrics['target_count']) + '\t'
                        + 'train mean rank: ' + str(train_presence_metrics['mean_rank']) + '\t'
                        + 'train mean margin: ' + str(train_presence_metrics['mean_margin']) + '\t'
                        + 'valid loss: ' + str(valid_presence_loss) + '\t'
                        + 'valid top5: ' + str(valid_presence_metrics['top5_count']) + '/' + str(valid_presence_metrics['target_count']) + '\t'
                        + 'valid top10: ' + str(valid_presence_metrics['top10_count']) + '/' + str(valid_presence_metrics['target_count']) + '\t'
                        + 'valid mean rank: ' + str(valid_presence_metrics['mean_rank']) + '\t'
                        + 'valid mean margin: ' + str(valid_presence_metrics['mean_margin']) + '\n'
                    )


        #SAVE LEARNING CURVES
        lc_path = os.path.join(args.save_dir, 'learning_curves.npy')
        print('\nDONE\n\nSaving learning curves to '+lc_path)
        np.save(lc_path, {'train_ppls':train_ppls,
                  'val_ppls':val_ppls,
                  'train_losses':train_losses,
                   'val_losses':val_losses,
                   'wer':ns_words,
                  })

        print("Saving plots")
        learning_curve_slr(args.save_dir)

        #Save every model every 10 epoch
        if(epoch % args.save_steps == 0):
            #Save after each epoch and save optimizer state
            print("Saving model parameters for epoch: "+str(epoch))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_fn,
                'best_wer': best_err_so_far
                },
                os.path.join(args.save_dir, 'epoch_'+str(epoch)+'_wer_'+str(word_err)+'.pt'))


        #We reached convergence
        if(train_ppl <= 1):
            print("YAy!!")
            break

if wandb_run is not None:
    wandb_run.finish()
