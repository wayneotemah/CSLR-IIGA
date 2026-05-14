import argparse
import _pickle as pickle
from pathlib import Path

import torch

from dataloader import loader
from tools.ctc_decode import decode_ctc_batch, ids_to_text
from tools.runtime import select_device
from tools.utils import path_data, Batch
from transformer import make_model as TRANSFORMER


def init_wandb(args):
    if not args.wandb or args.wandb_mode == 'disabled':
        return None

    try:
        import wandb
    except ImportError:
        print("WARNING: --wandb was set but wandb is not installed. Continuing without W&B logging.")
        return None

    tags = [tag.strip() for tag in args.wandb_tags.split(',') if tag.strip()]
    run_name = args.wandb_run_name or f"preview-{Path(args.model_path).stem}"
    try:
        wandb_init = getattr(wandb, 'init')
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
        if args.wandb_mode == 'offline':
            init_kwargs['mode'] = 'offline'
        if args.wandb_dir:
            init_kwargs['dir'] = args.wandb_dir
        return wandb_init(**init_kwargs)
    except Exception as exc:
        print(f"WARNING: failed to initialize W&B ({exc}). Continuing without W&B logging.")
        return None


def log_wandb_preview(wandb_run, args, rows):
    if wandb_run is None:
        return

    try:
        import wandb

        wandb_table = getattr(wandb, 'Table')
        wandb_artifact = getattr(wandb, 'Artifact')

        table = wandb_table(columns=['idx', 'split', 'ground_truth', 'prediction', 'exact_match', 'is_blank'])
        for row in rows:
            table.add_data(
                row['idx'],
                args.split,
                row['ground_truth'],
                row['prediction'],
                row['exact_match'],
                row['is_blank'],
            )

        artifact_name = args.wandb_artifact_name or f"{Path(args.model_path).stem}-{args.split}-checkpoint"
        artifact = wandb_artifact(
            name=artifact_name,
            type='model',
            metadata={
                'model_path': args.model_path,
                'split': args.split,
                'num_examples': len(rows),
                'source_run': args.wandb_source_run,
                'source_artifact': args.wandb_source_artifact,
            },
        )
        artifact.add_file(args.model_path)
        wandb_run.log_artifact(artifact, aliases=args.wandb_artifact_aliases.split(','))
        wandb_run.log({
            'preview/predictions': table,
            'preview/exact_matches': sum(1 for row in rows if row['exact_match']),
            'preview/blank_predictions': sum(1 for row in rows if row['is_blank']),
            'preview/num_examples': len(rows),
            'preview/source_run': args.wandb_source_run or '',
            'preview/source_artifact': args.wandb_source_artifact or '',
        })
    except Exception as exc:
        print(f"WARNING: failed to log W&B preview artifacts ({exc}).")


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
    parser.add_argument('--encoder_type', type=str, default='legacy', choices=['legacy', 'conformer'])
    parser.add_argument('--conformer_kernel_size', type=int, default=17)
    parser.add_argument('--image_type', type=str, default='rgb', choices=['rgb', 'grayscale'])
    parser.add_argument('--local_window', type=int, default=10)
    parser.add_argument('--fixed_padding', type=int, default=None)
    parser.add_argument('--wandb', action='store_true', help='Log checkpoint artifact and GT/PRED preview table to W&B.')
    parser.add_argument('--wandb_entity', type=str, default='ishara-ke')
    parser.add_argument('--wandb_project', type=str, default='CSLR-IIGA')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--wandb_tags', type=str, default='preview')
    parser.add_argument('--wandb_group', type=str, default=None)
    parser.add_argument('--wandb_job_type', type=str, default='preview')
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--wandb_dir', type=str, default=None)
    parser.add_argument('--wandb_source_run', type=str, default=None)
    parser.add_argument('--wandb_source_artifact', type=str, default=None)
    parser.add_argument('--wandb_artifact_name', type=str, default=None)
    parser.add_argument('--wandb_artifact_aliases', type=str, default='latest,best')

    args = parser.parse_args()
    if args.encoder_type == 'conformer' and args.hand_query:
        parser.error('--encoder_type conformer is not supported with --hand_query in the first Conformer branch.')
    wandb_run = init_wandb(args)

    device = select_device()

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
        encoder_type=args.encoder_type,
        conformer_kernel_size=args.conformer_kernel_size,
    )
    model = load_checkpoint(model, args.model_path, device).to(device)
    model.eval()

    shown = 0
    preview_rows = []
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
            
            decoded_preds = decode_ctc_batch(output.transpose(0, 1), x_lengths, blank_idx)

            for b, pred_ids in enumerate(decoded_preds):

                gt_ids = y[b][:y_lengths[b]].tolist()

                pred_text = ids_to_text(pred_ids, vocab_inv, ignore_ids=[blank_idx, pad_idx, sos_idx, eos_idx])
                gt_text = ids_to_text(gt_ids, vocab_inv, ignore_ids=[blank_idx, pad_idx, sos_idx, eos_idx])

                if not pred_text:
                    pred_text = '[blank]'

                preview_rows.append({
                    'idx': shown,
                    'ground_truth': gt_text,
                    'prediction': pred_text,
                    'exact_match': pred_text == gt_text,
                    'is_blank': pred_text == '[blank]',
                })

                print(f'GT: {gt_text}')
                print(f'PRED: {pred_text}')
                print('-' * 60)

                shown += 1
                if shown >= args.num_examples:
                    log_wandb_preview(wandb_run, args, preview_rows)
                    if wandb_run is not None:
                        wandb_run.finish()
                    raise SystemExit(0)

    log_wandb_preview(wandb_run, args, preview_rows)
    if wandb_run is not None:
        wandb_run.finish()
