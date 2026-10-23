# SPDX-License-Identifier: GPL-3.0-only
"""Check actual CUDA kernel execution before loading models or starting jobs."""

import argparse
import json
import sys

import torch


def validate_cuda_runtime(device):
    device = torch.device(device)
    info = {'device': str(device), 'torch': str(torch.__version__), 'cuda_runtime': torch.version.cuda}
    if device.type != 'cuda':
        return info
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable in this interpreter; refusing CPU fallback')
    name = torch.cuda.get_device_name(device)
    major, minor = torch.cuda.get_device_capability(device)
    architectures = torch.cuda.get_arch_list()
    info.update(gpu=name, compute_capability=[major, minor], compiled_architectures=architectures)
    try:
        # Availability and model.to(cuda) can succeed without usable GPU kernels.
        # Exercise both a CUDA tensor kernel and cuBLAS, without consuming RNGs.
        probe = torch.ones((16, 16), device=device, dtype=torch.float32)
        if (probe @ probe).sum().item() != 4096:
            raise RuntimeError('CUDA arithmetic probe returned an unexpected value')
        torch.cuda.synchronize(device)
    except RuntimeError as error:
        advice = 'Request a GPU supported by this PyTorch build or install a compatible CUDA build of PyTorch.'
        if (major, minor) < (7, 5) and torch.version.cuda and int(torch.version.cuda.split('.')[0]) >= 13:
            version = str(torch.__version__).split('+')[0]
            advice = ('CUDA 13 builds do not support this older GPU. On Rivanna, request an A100 '
                '(the benchmark job defaults to --constraint=a100_40gb). '
                'To retain V100 support, use a separate environment with the CUDA 12.6 build, e.g. '\
                f'{sys.executable} -m pip install --force-reinstall "torch=={version}" '\
                '--index-url https://download.pytorch.org/whl/cu126')
        raise RuntimeError(f'CUDA kernel check failed on {name} (sm_{major}{minor}); '
            f'torch={torch.__version__}, CUDA={torch.version.cuda}, compiled architectures={architectures}. '
            f'{advice} Original error: {error}') from error
    info['kernel_probe'] = 'passed'
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    try:
        print(json.dumps(validate_cuda_runtime(args.device)), flush=True)
        return 0
    except RuntimeError as error:
        print(str(error), file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
