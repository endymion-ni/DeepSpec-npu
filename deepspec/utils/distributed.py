import contextlib
import math
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


def _detect_device():
    """Return ``(device_type, backend, local_world_size_fn)`` for the current
    accelerator.

    * NPU (Ascend)  → ``("npu", "hccl", torch.npu.device_count)``
    * CUDA (NVIDIA) → ``("cuda", "nccl", torch.cuda.device_count)``
    * CPU (fallback) → ``("cpu", "gloo", lambda: 1)``
    """
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            return "npu", "hccl", torch.npu.device_count
    except ImportError:
        pass
    if torch.cuda.is_available():
        return "cuda", "nccl", torch.cuda.device_count
    return "cpu", "gloo", lambda: 1


def _set_device(device_type: str, device_index: int):
    """Set the default device for *device_type* to *device_index*."""
    if device_type == "npu":
        torch.npu.set_device(device_index)
    elif device_type == "cuda":
        torch.cuda.set_device(device_index)


def _torch_device(device_type: str, device_index: int) -> torch.device:
    return torch.device(device_type, device_index)


def _current_device_index(device_type: str) -> int:
    if device_type == "npu":
        return torch.npu.current_device()
    if device_type == "cuda":
        return torch.cuda.current_device()
    return 0


# ---- public API ----


def init_dist(local_rank=None, timeout_minutes: int = 60):
    """Initialise the torch distributed process group.

    Supports two launch modes:

    * **torchrun** (recommended): omit *local_rank*; rank info is read from
      ``LOCAL_RANK``, ``RANK``, ``WORLD_SIZE`` environment variables.
    * **legacy spawn**: pass *local_rank* explicitly; ``RANK`` / ``WORLD_SIZE``
      are interpreted as *node* rank / *node* count.

    Returns ``(device, global_rank, world_size)``.
    """
    device_type, backend, local_world_size_fn = _detect_device()

    if local_rank is not None:
        # Legacy torch.multiprocessing.spawn mode.
        local_world_size = local_world_size_fn()
        node_rank = int(os.environ["RANK"])
        node_world_size = int(os.environ["WORLD_SIZE"])
        rank = node_rank * local_world_size + int(local_rank)
        world_size = node_world_size * local_world_size
        init_method = f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
    else:
        # torchrun mode — all rank info comes from environment variables.
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 0)) or local_world_size_fn()

    device = _torch_device(device_type, local_rank)
    _set_device(device_type, local_rank)

    dist_init_kwargs = dict(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=timeout_minutes),
        device_id=device if device_type != "cpu" else None,
    )
    # Only pass init_method in legacy spawn mode; torchrun sets it via env vars.
    if local_rank is not None and "init_method" in dir():
        dist_init_kwargs["init_method"] = init_method

    dist.init_process_group(**{k: v for k, v in dist_init_kwargs.items() if v is not None})
    return device, rank, world_size


def is_global_main_process():
    return dist.get_rank() == 0


def is_local_main_process():
    # With torchrun LOCAL_RANK is always set; with legacy spawn it may not be.
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        return int(local_rank) == 0
    _, device_type, _ = _detect_device()
    return _current_device_index(device_type) == 0


def print_on_global_main(*args, **kwargs):
    if is_global_main_process():
        kwargs.setdefault("flush", True)
        print(time.strftime("%Y-%m-%d %H:%M:%S"), *args, **kwargs)


def print_on_local_main(*args, **kwargs):
    if is_local_main_process():
        kwargs.setdefault("flush", True)
        print(time.strftime("%Y-%m-%d %H:%M:%S"), *args, **kwargs)


@contextlib.contextmanager
def main_process_first():
    if dist.get_rank() == 0:
        yield
        dist.barrier()
    else:
        dist.barrier()
        yield


class StatelessResumableDistributedSampler(Sampler):
    """Deterministic distributed sampler that streams across epoch boundaries.

    Each epoch uses the first ``total_size`` samples from a deterministic
    shuffle of the dataset.  The sampler can produce a fixed number of
    per-rank samples (``num_samples``) starting from an arbitrary per-rank
    offset, transparently crossing epoch boundaries with fresh shuffles.

    When ``num_samples`` is *None* (default) the sampler yields the remaining
    samples in the current epoch — this preserves backward compatibility with
    code that rebuilds the dataloader at every epoch boundary.
    """

    def __init__(
        self,
        dataset,
        num_replicas: int,
        rank: int,
        total_size: int,
        seed: int = 42,
        start_global_offset_samples: int = 0,
        num_samples: int | None = None,
    ):
        assert start_global_offset_samples >= 0, "start_global_offset_samples must be >= 0"
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.total_size = int(total_size)
        self.seed = int(seed)
        self.dataset_size = len(self.dataset)
        assert self.dataset_size > 0, "dataset must have positive length"
        assert self.total_size > 0, "total_size must be > 0"
        assert self.total_size <= self.dataset_size, (
            f"total_size ({self.total_size}) cannot exceed dataset size ({self.dataset_size})"
        )
        assert self.total_size % self.num_replicas == 0, (
            f"total_size ({self.total_size}) must be divisible by num_replicas ({self.num_replicas})"
        )
        assert num_samples is None or num_samples >= 0, "num_samples must be >= 0"

        self.per_rank_len_per_epoch = self.total_size // self.num_replicas
        self._global_offset = int(start_global_offset_samples)
        self._num_samples = num_samples

    def __len__(self):
        if self._num_samples is not None:
            return self._num_samples
        mod = self._global_offset % self.per_rank_len_per_epoch
        return self.per_rank_len_per_epoch - mod if mod != 0 else self.per_rank_len_per_epoch

    def _epoch_perm(self, epoch_idx: int):
        g = torch.Generator()
        g.manual_seed(self.seed + epoch_idx)
        return torch.randperm(self.dataset_size, generator=g).tolist()[: self.total_size]

    def _epoch_slice_for_rank(self, perm):
        return perm[self.rank : self.total_size : self.num_replicas]

    def _iter_stream(self):
        epoch_idx = self._global_offset // self.per_rank_len_per_epoch
        offset_in_epoch = self._global_offset % self.per_rank_len_per_epoch

        perm = self._epoch_perm(epoch_idx)
        my_seq = self._epoch_slice_for_rank(perm)
        for i in range(offset_in_epoch, len(my_seq)):
            yield my_seq[i]

        epoch_idx += 1
        while True:
            perm = self._epoch_perm(epoch_idx)
            my_seq = self._epoch_slice_for_rank(perm)
            for idx in my_seq:
                yield idx
            epoch_idx += 1

    def __iter__(self):
        it = self._iter_stream()
        for _ in range(len(self)):
            yield next(it)
