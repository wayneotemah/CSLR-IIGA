from pathlib import Path
import math
import time

import torch


def select_device(verbose=False, context=None):
    if torch.cuda.is_available():
        if verbose:
            if context:
                print(f'{context} on GPU!')
            print(f'Number of GPUs={torch.cuda.device_count()}')
            print(f'Device name: {torch.cuda.get_device_name(0)}')
        return torch.device('cuda:0')

    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        if verbose:
            if context:
                print(f'{context} on XPU!')
            print(f'Number of XPUs={torch.xpu.device_count()}')
            print(f'Device name: {torch.xpu.get_device_name(0)}')
        return torch.device('xpu')

    if verbose:
        print('WARNING: Running on CPU. This may be slow or run out of memory.')
    return torch.device('cpu')


_XE_GT0_ROOT = Path('/sys/class/drm/card0/device/tile0/gt0')


def read_device_telemetry(device):
    telemetry = {
        'device_type': getattr(device, 'type', str(device)),
    }

    if getattr(device, 'type', None) == 'cuda':
        telemetry.update({
            'mem_alloc_mb': torch.cuda.memory_allocated() / (1024 ** 2),
            'mem_reserved_mb': torch.cuda.memory_reserved() / (1024 ** 2),
            'max_mem_alloc_mb': torch.cuda.max_memory_allocated() / (1024 ** 2),
            'max_mem_reserved_mb': torch.cuda.max_memory_reserved() / (1024 ** 2),
        })
        return telemetry

    if getattr(device, 'type', None) == 'xpu' and hasattr(torch, 'xpu'):
        telemetry.update({
            'mem_alloc_mb': torch.xpu.memory_allocated() / (1024 ** 2),
            'mem_reserved_mb': torch.xpu.memory_reserved() / (1024 ** 2),
            'max_mem_alloc_mb': torch.xpu.max_memory_allocated() / (1024 ** 2),
            'max_mem_reserved_mb': torch.xpu.max_memory_reserved() / (1024 ** 2),
        })

        try:
            telemetry['gt_idle_status'] = (_XE_GT0_ROOT / 'gtidle' / 'idle_status').read_text().strip()
            telemetry['gt_idle_residency_ms'] = int((_XE_GT0_ROOT / 'gtidle' / 'idle_residency_ms').read_text().strip())
            telemetry['gt_cur_freq_mhz'] = int((_XE_GT0_ROOT / 'freq0' / 'cur_freq').read_text().strip())
            telemetry['gt_act_freq_mhz'] = int((_XE_GT0_ROOT / 'freq0' / 'act_freq').read_text().strip())
        except (OSError, ValueError):
            pass

        return telemetry

    return telemetry


def format_device_telemetry(telemetry):
    parts = []

    if 'mem_alloc_mb' in telemetry:
        parts.append(f"mem_alloc={telemetry['mem_alloc_mb']:.1f}MB")
    if 'mem_reserved_mb' in telemetry:
        parts.append(f"mem_reserved={telemetry['mem_reserved_mb']:.1f}MB")
    if 'max_mem_alloc_mb' in telemetry:
        parts.append(f"max_alloc={telemetry['max_mem_alloc_mb']:.1f}MB")
    if 'max_mem_reserved_mb' in telemetry:
        parts.append(f"max_reserved={telemetry['max_mem_reserved_mb']:.1f}MB")
    if 'gt_idle_status' in telemetry:
        parts.append(f"gt_idle={telemetry['gt_idle_status']}")
    if 'gt_idle_residency_ms' in telemetry:
        parts.append(f"gt_idle_ms={telemetry['gt_idle_residency_ms']}")
    if 'gt_busy_pct_est' in telemetry:
        parts.append(f"gt_busy_est={telemetry['gt_busy_pct_est']:.1f}%")
    if 'gt_cur_freq_mhz' in telemetry:
        parts.append(f"gt_cur_freq={telemetry['gt_cur_freq_mhz']}MHz")
    if 'gt_act_freq_mhz' in telemetry:
        parts.append(f"gt_act_freq={telemetry['gt_act_freq_mhz']}MHz")

    return ', '.join(parts)


def effective_ctc_lengths(raw_lengths, local_window=None, emb_network='mb2', output_time=None, reduction=None):
    if reduction is None:
        reduction = 2 if emb_network in {'swin3d_t', 'videomae'} else 1

    lengths = [int(math.ceil(int(length) / reduction)) for length in raw_lengths]

    if local_window:
        lengths = [((length + local_window - 1) // local_window) * local_window for length in lengths]

    if output_time is not None:
        output_time = int(output_time)
        lengths = [min(output_time, length) for length in lengths]

    return lengths


class DeviceTelemetryPoller:
    def __init__(self, device):
        self.device = device
        self._prev_time = None
        self._prev_idle_ms = None

    def sample(self):
        telemetry = read_device_telemetry(self.device)

        now = time.time()
        idle_ms = telemetry.get('gt_idle_residency_ms')
        if idle_ms is not None and self._prev_time is not None and self._prev_idle_ms is not None:
            idle_ms = int(idle_ms)
            prev_idle_ms = int(self._prev_idle_ms)
            elapsed_ms = max((now - self._prev_time) * 1000.0, 1.0)
            idle_delta = max(idle_ms - prev_idle_ms, 0)
            busy_pct = max(0.0, min(100.0, 100.0 * (1.0 - (idle_delta / elapsed_ms))))
            telemetry['gt_busy_pct_est'] = busy_pct

        self._prev_time = now
        if idle_ms is not None:
            self._prev_idle_ms = idle_ms

        return telemetry
