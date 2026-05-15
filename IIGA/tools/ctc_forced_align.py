import math
from typing import Iterable

import torch


def interleave_ctc_targets(target_ids, blank_idx):
    target_ids = [int(token_id) for token_id in target_ids]
    expanded = [int(blank_idx)]
    for token_id in target_ids:
        expanded.append(token_id)
        expanded.append(int(blank_idx))
    return expanded


def _to_log_probs(log_probs_or_probs, already_log_probs):
    tensor = log_probs_or_probs.detach().clone()
    if tensor.ndim != 2:
        raise ValueError(f'Expected 2D tensor shaped [time, vocab], got {tuple(tensor.shape)}')
    if already_log_probs:
        return tensor
    clamped = torch.clamp(tensor, min=1e-12)
    return torch.log(clamped)


def _allowed_predecessors(expanded_target_ids, state_idx):
    predecessors = [state_idx]
    if state_idx > 0:
        predecessors.append(state_idx - 1)
    if state_idx > 1:
        current = expanded_target_ids[state_idx]
        prev_prev = expanded_target_ids[state_idx - 2]
        blank_idx = expanded_target_ids[0]
        if current != blank_idx and current != prev_prev:
            predecessors.append(state_idx - 2)
    return predecessors


def _recover_token_spans(frame_token_ids, blank_idx):
    spans = []
    active_token = None
    start = None
    for frame_idx, token_id in enumerate(frame_token_ids):
        token_id = int(token_id)
        if token_id == int(blank_idx):
            if active_token is not None:
                spans.append({
                    'token_id': int(active_token),
                    'start': int(start),
                    'end': int(frame_idx - 1),
                    'length': int(frame_idx - start),
                })
                active_token = None
                start = None
            continue
        if token_id != active_token:
            if active_token is not None:
                spans.append({
                    'token_id': int(active_token),
                    'start': int(start),
                    'end': int(frame_idx - 1),
                    'length': int(frame_idx - start),
                })
            active_token = token_id
            start = frame_idx
    if active_token is not None:
        spans.append({
            'token_id': int(active_token),
            'start': int(start),
            'end': int(len(frame_token_ids) - 1),
            'length': int(len(frame_token_ids) - start),
        })
    return spans


def ctc_forced_align(log_probs_or_probs, target_ids, blank_idx, already_log_probs=True):
    target_ids = [int(token_id) for token_id in target_ids]
    if not target_ids:
        raise ValueError('target_ids must contain at least one token for forced alignment')

    log_probs = _to_log_probs(log_probs_or_probs, already_log_probs=already_log_probs)
    time_steps, vocab_size = log_probs.shape
    if time_steps == 0:
        raise ValueError('Cannot align an empty time sequence')
    if any(token_id < 0 or token_id >= vocab_size for token_id in target_ids):
        raise ValueError('target_ids contains token ids outside the vocabulary range')
    if blank_idx < 0 or blank_idx >= vocab_size:
        raise ValueError('blank_idx is outside the vocabulary range')

    expanded = interleave_ctc_targets(target_ids, blank_idx)
    states = len(expanded)
    neg_inf = -float('inf')

    scores = torch.full((time_steps, states), neg_inf, dtype=log_probs.dtype)
    backpointers = torch.full((time_steps, states), -1, dtype=torch.long)

    scores[0, 0] = log_probs[0, expanded[0]]
    if states > 1:
        scores[0, 1] = log_probs[0, expanded[1]]

    for time_idx in range(1, time_steps):
        for state_idx in range(states):
            predecessors = _allowed_predecessors(expanded, state_idx)
            best_prev_state = predecessors[0]
            best_prev_score = scores[time_idx - 1, best_prev_state].item()
            for prev_state in predecessors[1:]:
                prev_score = scores[time_idx - 1, prev_state].item()
                if prev_score > best_prev_score:
                    best_prev_score = prev_score
                    best_prev_state = prev_state
            if math.isinf(best_prev_score) and best_prev_score < 0:
                continue
            scores[time_idx, state_idx] = scores[time_idx - 1, best_prev_state] + log_probs[time_idx, expanded[state_idx]]
            backpointers[time_idx, state_idx] = best_prev_state

    final_candidates = [states - 1]
    if states > 1:
        final_candidates.append(states - 2)
    best_final_state = max(final_candidates, key=lambda state_idx: scores[time_steps - 1, state_idx].item())
    best_score = float(scores[time_steps - 1, best_final_state].item())
    if math.isinf(best_score) and best_score < 0:
        raise RuntimeError('No valid CTC forced alignment path was found for the supplied target_ids')

    state_path = [best_final_state]
    for time_idx in range(time_steps - 1, 0, -1):
        prev_state = int(backpointers[time_idx, state_path[-1]].item())
        if prev_state < 0:
            raise RuntimeError('Broken backpointer chain while recovering forced alignment path')
        state_path.append(prev_state)
    state_path.reverse()

    frame_token_ids = [int(expanded[state_idx]) for state_idx in state_path]
    spans = _recover_token_spans(frame_token_ids, blank_idx)

    return {
        'score': best_score,
        'time_steps': int(time_steps),
        'vocab_size': int(vocab_size),
        'blank_idx': int(blank_idx),
        'target_ids': [int(token_id) for token_id in target_ids],
        'expanded_target_ids': [int(token_id) for token_id in expanded],
        'state_path': [int(state_idx) for state_idx in state_path],
        'frame_token_ids': frame_token_ids,
        'token_spans': spans,
    }


def batch_ctc_forced_align(log_probs, lengths, target_batch, target_lengths, blank_idx, already_log_probs=True):
    alignments = []
    for batch_idx in range(len(lengths)):
        seq_len = int(lengths[batch_idx])
        target_len = int(target_lengths[batch_idx])
        batch_log_probs = log_probs[:seq_len, batch_idx]
        target_ids = [int(token_id) for token_id in target_batch[batch_idx][:target_len].tolist()]
        alignments.append(
            ctc_forced_align(
                batch_log_probs,
                target_ids,
                blank_idx=blank_idx,
                already_log_probs=already_log_probs,
            )
        )
    return alignments
