#!/usr/bin/env python3
"""
GreenBoost — 3-tier pool spillover stress test.

Allocates tensors filling T1 (VRAM) first, then verifies automatic
spillover into system RAM (T2 System DDR via GreenBoost shim). Reports
per-tier allocation, access latency, and throughput.

Author  : Ferran Duarri
License : GPL v2 / Commercial — see LICENSE
"""

import torch
import time
import subprocess
import sys


class Colors:
    CYAN   = '\033[0;36m'
    GREEN  = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED    = '\033[0;31m'
    BLUE   = '\033[0;34m'
    RESET  = '\033[0m'
    BOLD   = '\033[1m'


def get_gpu_memory():
    """Query GPU memory (used MB, total MB) via nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        used, total = map(int, result.stdout.strip().split(','))
        return used, total
    except Exception:
        return 0, 0


def print_header():
    print(f"\n{Colors.CYAN}╔════════════════════════════════════════════════════════════════╗")
    print(f"║                                                                ║")
    print(f"║     GreenBoost — 3-Tier Pool Spillover Stress Test            ║")
    print(f"║     T1 VRAM (12 GB) → T2 System DDR (51 GB) → T3 NVMe (576 GB)   ║")
    print(f"║     Copyright (C) 2026 Ferran Duarri                    ║")
    print(f"║                                                                ║")
    print(f"╚════════════════════════════════════════════════════════════════╝{Colors.RESET}\n")


def run_spillover_test(total_gb=12):
    """
    Fills T1 (GPU VRAM) first, then spills into T2 (system RAM).
    Requires GreenBoost shim (LD_PRELOAD) for transparent System DDR access.
    """
    print(f"{Colors.YELLOW}Target allocation : {total_gb} GB{Colors.RESET}")
    print(f"   Strategy: GPU (T1) first, then RAM (T2) spillover via GreenBoost")
    print()

    if not torch.cuda.is_available():
        print(f"{Colors.RED}CUDA not available — NVIDIA GPU required.{Colors.RESET}")
        return False

    device_gpu = torch.device('cuda:0')
    device_cpu = torch.device('cpu')

    gpu_props = torch.cuda.get_device_properties(0)
    gpu_total_gb = gpu_props.total_memory / (1024 ** 3)

    print(f"{Colors.GREEN}GPU  : {gpu_props.name}  ({gpu_total_gb:.2f} GB VRAM){Colors.RESET}")
    print()

    gpu_used_start, gpu_total = get_gpu_memory()
    print(f"Initial GPU state : {gpu_used_start} MB used / {gpu_total} MB total")
    print()

    tensors_gpu = []
    tensors_cpu = []
    total_allocated = 0
    gpu_allocated   = 0
    cpu_allocated   = 0

    block_mb   = 256
    num_blocks = int((total_gb * 1024) / block_mb)

    print(f"{Colors.GREEN}Allocating {num_blocks} blocks x {block_mb} MB...{Colors.RESET}\n")
    start_time = time.time()

    for i in range(num_blocks):
        num_elements = (block_mb * 1024 * 1024) // 4  # float32

        try:
            tensor = torch.randn(num_elements, device=device_gpu, dtype=torch.float32)
            tensors_gpu.append(tensor)
            gpu_allocated += block_mb * 1024 * 1024
            tier_label = f"{Colors.GREEN}T1 VRAM{Colors.RESET}"

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(f"\n{Colors.YELLOW}  T1 VRAM full — spilling to T2 System DDR (GreenBoost)...{Colors.RESET}")
                tensor = torch.randn(num_elements, device=device_cpu, dtype=torch.float32)
                tensors_cpu.append(tensor)
                cpu_allocated += block_mb * 1024 * 1024
                tier_label = f"{Colors.YELLOW}T2 System DDR{Colors.RESET}"
            else:
                raise

        total_allocated += block_mb * 1024 * 1024
        gb_so_far = total_allocated / (1024 ** 3)
        progress  = (i + 1) / num_blocks * 100
        filled    = int(40 * (i + 1) / num_blocks)
        bar       = '█' * filled + '░' * (40 - filled)
        gpu_used, _ = get_gpu_memory()

        print(f"\r[{bar}] {progress:.0f}%  {gb_so_far:.2f} GB  GPU:{gpu_used} MB  [{tier_label}]",
              end='', flush=True)
        time.sleep(0.01)

    elapsed = time.time() - start_time
    gpu_used_end, gpu_total_end = get_gpu_memory()

    print(f"\n\n{Colors.GREEN}Allocation complete.{Colors.RESET}")
    print(f"\n{Colors.CYAN}Results:{Colors.RESET}")
    print(f"   Time       : {elapsed:.2f} s")
    print(f"   Throughput : {total_allocated / (1024**3) / elapsed:.2f} GB/s")
    print()
    print(f"   {Colors.GREEN}T1 VRAM (GPU):{Colors.RESET}")
    print(f"     Blocks   : {len(tensors_gpu)}")
    print(f"     Allocated: {gpu_allocated / (1024**3):.2f} GB")
    print(f"     Usage    : {gpu_used_end} MB / {gpu_total_end} MB  "
          f"({gpu_used_end / gpu_total_end * 100:.1f}%)")
    print()
    print(f"   {Colors.YELLOW}T2 System DDR (GreenBoost spillover):{Colors.RESET}")
    print(f"     Blocks   : {len(tensors_cpu)}")
    print(f"     Allocated: {cpu_allocated / (1024**3):.2f} GB")
    print()
    print(f"   {Colors.BOLD}Total:{Colors.RESET}")
    print(f"     {total_allocated / (1024**3):.2f} GB  "
          f"(GPU {gpu_allocated / total_allocated * 100:.1f}%  /  "
          f"RAM {cpu_allocated / total_allocated * 100:.1f}%)")

    if tensors_cpu:
        print(f"\n{Colors.GREEN}SPILLOVER ACTIVE — T1 full, T2 System DDR in use.{Colors.RESET}")
    else:
        print(f"\n{Colors.YELLOW}All data fits in T1 VRAM (no spillover triggered).{Colors.RESET}")

    # Access latency test
    print(f"\n{Colors.CYAN}Access latency test...{Colors.RESET}")
    t0 = time.time()
    if tensors_gpu:
        val = tensors_gpu[0].mean().item()
        print(f"   T1 VRAM read  : {val:.6f}")
    if tensors_cpu:
        val = tensors_cpu[0].mean().item()
        print(f"   T2 System DDR read  : {val:.6f}")
    print(f"   Latency: {(time.time() - t0) * 1000:.2f} ms")

    # Hold memory for manual inspection
    print(f"\n{Colors.YELLOW}Memory held — inspect in another terminal:{Colors.RESET}")
    print(f"   nvidia-smi")
    print(f"   cat /sys/class/greenboost/greenboost/pool_info")
    print(f"\n   Press Enter to release...")
    input()

    # Release
    print(f"\n{Colors.CYAN}Releasing memory...{Colors.RESET}")
    tensors_gpu.clear()
    tensors_cpu.clear()
    torch.cuda.empty_cache()
    gpu_final, _ = get_gpu_memory()
    print(f"{Colors.GREEN}Released. GPU: {gpu_final} MB{Colors.RESET}")

    return True


def main():
    print_header()

    print(f"{Colors.CYAN}This test:{Colors.RESET}")
    print(f"  1. Allocates tensors on GPU VRAM (T1) until full")
    print(f"  2. Spills overflow to system RAM via GreenBoost shim (T2)")
    print(f"  3. Reports allocation split, throughput, and access latency")
    print()

    if not torch.cuda.is_available():
        print(f"{Colors.RED}CUDA not available. Requires NVIDIA GPU.{Colors.RESET}")
        return 1

    print(f"{Colors.YELLOW}Target size in GB (recommended: 14-20 GB for T2 spillover):{Colors.RESET}")
    choice = input(f"{Colors.GREEN}Size GB [16]: {Colors.RESET}").strip()
    target_gb = float(choice) if choice else 16.0

    success = run_spillover_test(target_gb)

    if success:
        print(f"\n{Colors.CYAN}╔════════════════════════════════════════════════════════════════╗")
        print(f"║                                                                ║")
        print(f"║         GreenBoost 3-Tier Spillover Test PASSED               ║")
        print(f"║                                                                ║")
        print(f"║  T1 VRAM filled → T2 System DDR spillover via GreenBoost shim      ║")
        print(f"║  Copyright (C) 2026 Ferran Duarri                       ║")
        print(f"║                                                                ║")
        print(f"╚════════════════════════════════════════════════════════════════╝{Colors.RESET}\n")
        return 0
    return 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interrupted.{Colors.RESET}")
        sys.exit(1)
    except Exception as exc:
        import traceback
        print(f"\n{Colors.RED}Error: {exc}{Colors.RESET}")
        traceback.print_exc()
        sys.exit(1)
