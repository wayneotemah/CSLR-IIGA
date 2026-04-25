import argparse
import _pickle as pickle
import torch

from dataloader import loader
from tools.utils import path_data, Batch
from transformer import make_model as TRANSFORMER


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
        collapsed.append(idx)
    return collapsed


def pick_split_paths(data_root, split, hand_query=False):
    train_path, valid_path, test_path = path_data(data_path=data_root, task='SLR', features_type='features', hand_query=hand_query)
    if split in ('valid', 'dev'):
        return valid_path
    if split == 'test':
        return test_path
    return train_path


def load_checkpoint(model, model_path, device):
    # PyTorch >=2.6 defaults to weights_only=True; training checkpoints in this repo
    # include non-tensor objects (optimizer, loss, etc.), so we need weights_only=False.
    try:
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        # Backward compatibility for older PyTorch versions without weights_only arg.
        ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Print GT vs prediction examples from a checkpoint.')
    parser.add_argument('--data', required=True, help='Prepared dataset root (PHOENIX-compatible layout).')
    parser.add_argument('--segment_root', required=True, help='Segmentation root for selected split.')
    parser.add_argument('--lookup_table', required=True, help='Lookup pickle path (token -> id).')
    parser.add_argument('--model_path', required=True, help='Path to checkpoint (e.g., BEST.pt).')
    parser.add_argument('--split', default='valid', choices=['train', 'valid', 'dev', 'test'])
    parser.add_argument('--num_examples', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--rescale', type=int, default=224)
    parser.add_argument('--hidden_size', type=int, default=1280)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=10)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--dp_keep_prob', type=float, default=0.7)
    parser.add_argument('--emb_type', type=str, default='2d')
    parser.add_argument('--emb_network', type=str, default='mb2')
    parser.add_argument('--hand_query', action='store_true')
    parser.add_argument('--image_type', type=str, default='rgb', choices=['rgb', 'grayscale'])
    parser.add_argument('--local_window', type=int, default=10)
    parser.add_argument('--fixed_padding', type=int, default=None)

    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    with open(args.lookup_table, 'rb') as f:
        vocab = pickle.load(f)
    vocab_inv = {v: k for k, v in vocab.items()}

    blank_idx = vocab.get('<BLANK>', len(vocab) - 1)
    pad_idx = vocab.get('<PAD>', 0)
    sos_idx = vocab.get('<SOS>', 1)
    eos_idx = vocab.get('<EOS>', 2)

    channels = 3 if args.image_type == 'rgb' else 1

    split_path = pick_split_paths(args.data, args.split, hand_query=args.hand_query)

    dataloader, _ = loader(
        csv_file=split_path[1],
        root_dir=split_path[0],
        segment_path=args.segment_root,
        lookup=args.lookup_table,
        rescale=args.rescale,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_drop=None,
        uniform_drop=1.0,
        show_sample=False,
        istrain=False,
        hand_dir=split_path[2],
        data_stats=None,
        hand_stats=None,
        channels=channels,
    )

    model = TRANSFORMER(
        tgt_vocab=len(vocab),
        n_stacks=args.num_layers,
        n_units=args.hidden_size,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        window_size=args.local_window,
        dropout=1. - args.dp_keep_prob,
        image_size=args.rescale,
        emb_type=args.emb_type,
        emb_network=args.emb_network,
        channels=channels,
    )
    model = load_checkpoint(model, args.model_path, device).to(device)
    model.eval()

    shown = 0
    with torch.no_grad():
        for x, x_lengths, y, y_lengths, hand_regions, _ in dataloader:
            x = x.to(device)
            hand_regions = hand_regions.to(device) if (args.hand_query and hand_regions is not None) else None

            batch = Batch(
                x_lengths,
                y_lengths,
                None,
                trg=None,
                emb_type=args.emb_type,
                DEVICE=device,
                fixed_padding=args.fixed_padding,
                rel_window=None,
            )

            output, output_context, output_hand = model.forward(
                x,
                batch.src_mask,
                batch.rel_mask,
                hand_regions
            )
            
            if output is None:
                output = output_context
            
            if output is None:
                raise RuntimeError("Model returned both output and output_context as None.")
            
            # output: (batch, seq_len, vocab)
            pred_frame_ids = torch.argmax(output, dim=-1).cpu().numpy()

            for b in range(pred_frame_ids.shape[0]):
                pred_ids = pred_frame_ids[b][:x_lengths[b]].tolist()
                pred_ids = ctc_greedy_decode(pred_ids, blank_idx)

                gt_ids = y[b][:y_lengths[b]].tolist()

                pred_text = ids_to_text(pred_ids, vocab_inv, ignore_ids=[blank_idx, pad_idx, sos_idx, eos_idx])
                gt_text = ids_to_text(gt_ids, vocab_inv, ignore_ids=[blank_idx, pad_idx, sos_idx, eos_idx])

                if not pred_text:
                    pred_text = '[blank]'

                print(f'GT: {gt_text}')
                print(f'PRED: {pred_text}')
                print('-' * 60)

                shown += 1
                if shown >= args.num_examples:
                    raise SystemExit(0)
