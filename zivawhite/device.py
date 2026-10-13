"""
Hardware detection.

This module finds WHAT the machine can run on (CUDA GPU / Apple MPS / CPU), which numeric precision
is actually fast there, and, under torchrun, which rank this process is. It does NOT decide how big
your model is: model size is an explicit choice (a named preset and/or any custom layer/head/width
numbers; see modelconfig.py), because letting detected VRAM or RAM cap a model silently held back
anyone with a capable GPU. The one place hardware informs size is a non-blocking *warning* from
train.py when its memory estimate says a run will probably not fit.
"""
from __future__ import annotations

import functools
import multiprocessing
import os
import platform
from dataclasses import dataclass, field
from typing import Dict, Optional

try:
    import psutil
except ImportError:  # optional dependency, degrade gracefully
    psutil = None

import torch


@dataclass
class Runtime:
    device: str                 # "cuda", "cuda:N", "mps", or "cpu"
    device_name: str
    n_gpus: int
    vram_gb: float               # per-GPU, 0 if none
    ram_gb: float
    cpu_cores: int
    tier: Optional[str]          # the preset the caller asked for (None = not chosen here); never auto-picked
    dtype: str                   # recommended autocast dtype
    distributed_capable: bool = field(default=False)
    # Set from torchrun's own environment (RANK/LOCAL_RANK/WORLD_SIZE), not
    # guessed: world_size==1 outside torchrun means "ordinary single-process
    # run," exactly as before this field existed. See train.py's DDP wiring.
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def summary(self) -> str:
        dist_part = f" | rank={self.rank}/{self.world_size}" if self.is_distributed else ""
        return (
            f"device={self.device} ({self.device_name}) | "
            f"gpus={self.n_gpus} vram={self.vram_gb:.1f}GB | "
            f"ram={self.ram_gb:.1f}GB cores={self.cpu_cores} | "
            f"{f'tier={self.tier} ' if self.tier else ''}dtype={self.dtype}{dist_part}"
        )


def _ram_gb() -> float:
    if psutil is not None:
        return psutil.virtual_memory().total / (1024 ** 3)

    system = platform.system()
    if system == "Windows":
        # no psutil available: ask Windows directly via the Win32 API
        # (ctypes is stdlib, so this needs no extra dependency and works
        # on a bare Windows install with only python.org's Python on it)
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return stat.ullTotalPhys / (1024 ** 3)
        except Exception:
            pass
        return 8.0  # conservative assumption if even the ctypes call fails

    # crude fallback for Linux without psutil
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    return kb / (1024 ** 2)
    except OSError:
        pass
    return 8.0  # conservative assumption


def _benchmark_cpu_dtype(n: int = 384, iters: int = 6) -> str:
    """Is bfloat16 autocast actually faster than plain float32 on *this*
    CPU? Some CPUs (server-class Intel/AMD with AVX512-BF16 or AMX) get a
    real win from it; a lot of consumer laptop chips see none, or are even
    slightly slower from the upcast overhead. Rather than guess, run a
    handful of matmuls both ways and measure -- same empirical-autotuning
    philosophy as benchmark_and_tune() below, just cheap enough (a few
    hundred ms at most) to run unconditionally on every startup instead
    of behind an opt-in flag. Only switches to bf16 if it's faster by a
    real margin, not noise, so a wash never silently changes numerics for
    no benefit."""
    import contextlib
    import time as _time

    a = torch.randn(n, n)
    b = torch.randn(n, n)

    def _timed(ctx) -> float:
        with ctx:
            for _ in range(iters):
                a @ b
        t0 = _time.time()
        with ctx:
            for _ in range(iters):
                a @ b
        return _time.time() - t0

    try:
        fp32_t = _timed(contextlib.nullcontext())
        bf16_t = _timed(torch.autocast(device_type="cpu", dtype=torch.bfloat16))
        return "bfloat16" if bf16_t < fp32_t * 0.9 else "float32"
    except Exception:
        return "float32"  # never let a benchmark hiccup break startup


def _distributed_env() -> "tuple[int, int, int]":
    """(rank, local_rank, world_size) from torchrun's own environment
    variables. Outside torchrun these are simply unset, so this returns
    (0, 0, 1) -- an ordinary single-process run -- exactly like before
    torchrun support existed. torchrun sets all three itself; nothing
    here has to compute or guess them."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, local_rank, world_size


@functools.lru_cache(maxsize=None)
def resolve_runtime(force_tier: Optional[str] = None) -> Runtime:
    """Inspect the machine and decide device + precision (NOT model size: see the module docstring).

    `force_tier` is simply remembered on the result so callers can read back what was asked for; it is
    never inferred from hardware.

    Cached per (force_tier) for the life of the process. This matters
    now that the CPU branch runs a real hardware micro-benchmark
    (_benchmark_cpu_dtype) rather than a hardcoded constant: without
    caching, every caller re-running that benchmark is harmless for a
    one-shot CLI command, but app.py's `/v1/health` handler calls
    resolve_runtime() on *every request*, and launch.py polls that
    endpoint every 0.4s while waiting for the server to come up --
    found by a real run where health checks were slow enough to trip
    launch.py's own readiness timeout. Caching turns that into one real
    benchmark at first use and near-zero cost after. Safe under
    torchrun too: each rank is its own process with its own cache, so
    caching never leaks one rank's device pick into another's.
    """
    ram = _ram_gb()
    cores = multiprocessing.cpu_count()
    rank, local_rank, world_size = _distributed_env()

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        if world_size > 1:
            # torchrun: this process owns exactly one GPU, chosen by
            # LOCAL_RANK, and must call set_device before any CUDA
            # tensor is created or NCCL will not know which device this
            # rank actually owns.
            gpu_index = local_rank % max(n_gpus, 1)
            torch.cuda.set_device(gpu_index)
            device = f"cuda:{gpu_index}"
        else:
            gpu_index = 0
            device = "cuda"
        props = torch.cuda.get_device_properties(gpu_index)
        vram_gb = props.total_memory / (1024 ** 3)
        device_name = props.name
        # Native bf16 needs compute capability >= 8.0 (Ampere+). torch.cuda.is_bf16_supported()
        # also answers True for older cards like the Colab T4 (7.5) because bf16 can be
        # *emulated* there -- but emulated bf16 is slower than fp16 on them. Decide from the
        # hardware, not that call.
        dtype = "bfloat16" if props.major >= 8 else "float16"
        distributed = n_gpus > 1 or world_size > 1
        # cuDNN's autotuner benchmarks a few algorithm variants for this
        # process's actual input shapes the first time it sees them, then
        # reuses the fastest -- free speedup for the fixed block_size/
        # batch_size shapes training uses, at the cost of a slower first
        # step. Only worth it on a fixed-shape workload like this one (a
        # workload with constantly-changing shapes would re-benchmark
        # every step and come out slower), and only applies to CUDA.
        torch.backends.cudnn.benchmark = True
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device, device_name = "mps", "Apple Silicon (MPS)"
        n_gpus, vram_gb, distributed = 1, ram * 0.6, False  # unified memory estimate
        dtype = "float32"  # train.py doesn't autocast on MPS, so fp32 is what actually runs
    else:
        device, device_name = "cpu", platform.processor() or "CPU"
        n_gpus, vram_gb, distributed = 0, 0.0, False
        # Use every core PyTorch can see for intra-op parallelism. Left at
        # its default this is usually already all-cores, but explicitly
        # setting it removes any dependence on ambient OMP_NUM_THREADS/
        # inherited-environment weirdness, and costs nothing.
        torch.set_num_threads(max(1, cores))
        dtype = _benchmark_cpu_dtype()

    tier = force_tier or None

    return Runtime(
        device=device,
        device_name=device_name,
        n_gpus=n_gpus,
        vram_gb=vram_gb,
        ram_gb=ram,
        cpu_cores=cores,
        tier=tier,
        dtype=dtype,
        distributed_capable=distributed,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )


def benchmark_and_tune(runtime: Runtime, model_config_fn, candidate_batch_sizes=None, n_iters: int = 5,
                       tier: Optional[str] = None) -> Dict[int, float]:
    """The real alternative to "the AI runs micro-benchmarks against the
    silicon and writes custom kernels for it." Actual empirical autotuning
    (what DeepSpeed/Lightning/XLA all do in practice) doesn't write new
    kernels per chip -- it measures a few candidate settings on the real
    hardware in front of it and picks whichever one is fastest, because
    static defaults are a reasonable prior
    but not always right for a specific machine.

    Runs `n_iters` forward+backward passes at each candidate batch size
    with the actual model config for the chosen preset, measures
    wall-clock tokens/sec, and returns the measured table so `train.py
    --autotune` can pick the winner. Candidates that OOM are skipped,
    not crashed on.
    """
    import time as _time

    import torch as _torch

    from .model import ZivaGPT, ModelConfig

    tier = tier or runtime.tier or "small"           # the model YOU are tuning for; never a hardware guess
    cfg: ModelConfig = model_config_fn(tier)
    candidates = candidate_batch_sizes or {
        "tiny": [4, 8, 16], "small": [8, 16, 32], "medium": [8, 16, 32], "large": [16, 32, 64], "xl": [4, 8, 16],
    }.get(tier, [4, 8, 16, 32])

    results: Dict[int, float] = {}
    for bs in candidates:
        try:
            model = ZivaGPT(cfg).to(runtime.device)
            opt = model.configure_optimizer(0.1, 3e-4, (0.9, 0.95), runtime.device)
            x = _torch.randint(0, cfg.vocab_size, (bs, cfg.block_size), device=runtime.device)
            y = _torch.randint(0, cfg.vocab_size, (bs, cfg.block_size), device=runtime.device)

            for _ in range(2):  # warmup, not timed
                _, loss = model(x, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            if runtime.device.startswith("cuda"):
                _torch.cuda.synchronize()
            t0 = _time.time()
            for _ in range(n_iters):
                _, loss = model(x, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            if runtime.device.startswith("cuda"):
                _torch.cuda.synchronize()
            elapsed = _time.time() - t0

            tokens_per_sec = (bs * cfg.block_size * n_iters) / max(elapsed, 1e-6)
            results[bs] = tokens_per_sec
            del model, opt
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                results[bs] = 0.0
                if runtime.device.startswith("cuda"):
                    _torch.cuda.empty_cache()
            else:
                raise
    return results


if __name__ == "__main__":
    print(resolve_runtime().summary())
