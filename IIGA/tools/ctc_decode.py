import torch


def ids_to_text(ids, vocab_inv, ignore_ids=None):
    ignore_ids = set(ignore_ids or [])
    tokens = []
    for idx in ids:
        if idx in ignore_ids:
            continue
        token = vocab_inv.get(int(idx), '<UNK>')
        if token.startswith('<') and token.endswith('>'):
            continue
        tokens.append(token)
    return ' '.join(tokens).strip()


def ctc_greedy_decode(frame_ids, blank_idx):
    collapsed = []
    prev = None
    for idx in frame_ids:
        if idx == prev:
            continue
        prev = idx
        if idx == blank_idx:
            continue
        collapsed.append(int(idx))
    return collapsed


def decode_ctc_batch(logits, lengths, blank_idx):
    frame_ids = torch.argmax(logits, dim=-1)
    decoded = []
    for batch_idx, seq_len in enumerate(lengths):
        ids = frame_ids[:, batch_idx][:seq_len].detach().cpu().tolist()
        decoded.append(ctc_greedy_decode(ids, blank_idx))
    return decoded
