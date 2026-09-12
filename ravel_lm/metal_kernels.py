from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional
import math

import numpy as np


class MetalKernelError(RuntimeError):
    pass


def _import_metal():
    try:
        import Metal  # type: ignore
    except ImportError as exc:
        raise MetalKernelError("PyObjC Metal bindings are not installed") from exc
    return Metal


def metal_available() -> bool:
    try:
        Metal = _import_metal()
    except MetalKernelError:
        return False
    return Metal.MTLCreateSystemDefaultDevice() is not None


def _source_path() -> Path:
    return Path(__file__).resolve().parent / "metal" / "kernels.metal"


def _as_contiguous(array: np.ndarray, dtype: np.dtype) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=dtype)


def _read_buffer(buffer, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    nbytes = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
    view = buffer.contents().as_buffer(nbytes)
    return np.frombuffer(view, dtype=dtype).reshape(shape).copy()


def _ceil_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


@dataclass(frozen=True)
class MetalTiming:
    gpu_ms: float


@dataclass
class MetalKernels:
    """Raw Metal kernels for RAVEL latest-1 lookup and causal softmax attention.

    These wrappers intentionally use Metal.framework directly through PyObjC.
    They do not depend on PyTorch MPS and do not call torch operations.
    """

    device: object
    queue: object
    ravel_latest1_legacy_pipeline: object
    ravel_latest1_pipeline: object
    ravel_latest1_vec4_pipeline: object
    attention_pipeline: object

    @classmethod
    def build(cls) -> "MetalKernels":
        Metal = _import_metal()
        device = Metal.MTLCreateSystemDefaultDevice()
        if device is None:
            raise MetalKernelError("No Metal device is available")

        source = _source_path().read_text()
        library, error = device.newLibraryWithSource_options_error_(source, None, None)
        if library is None:
            raise MetalKernelError(f"Metal library compile failed: {error}")

        def pipeline(name: str):
            fn = library.newFunctionWithName_(name)
            if fn is None:
                raise MetalKernelError(f"Metal function not found: {name}")
            pso, pso_error = device.newComputePipelineStateWithFunction_error_(fn, None)
            if pso is None:
                raise MetalKernelError(f"Metal pipeline creation failed for {name}: {pso_error}")
            return pso

        queue = device.newCommandQueue()
        if queue is None:
            raise MetalKernelError("Could not create Metal command queue")
        return cls(
            device=device,
            queue=queue,
            ravel_latest1_legacy_pipeline=pipeline("ravel_latest1_sweep_f32"),
            ravel_latest1_pipeline=pipeline("ravel_latest1_f32"),
            ravel_latest1_vec4_pipeline=pipeline("ravel_latest1_f32x4"),
            attention_pipeline=pipeline("causal_softmax_attention_f32"),
        )

    def _buffer_from_array(self, array: np.ndarray):
        Metal = _import_metal()
        array = np.ascontiguousarray(array)
        return array, self.device.newBufferWithBytes_length_options_(
            array,
            array.nbytes,
            Metal.MTLResourceStorageModeShared,
        )

    def _empty_buffer(self, nbytes: int):
        Metal = _import_metal()
        return self.device.newBufferWithLength_options_(nbytes, Metal.MTLResourceStorageModeShared)

    def _run(self, pipeline, buffers: list[object], *, grid, threads, use_threadgroups: bool = False) -> MetalTiming:
        Metal = _import_metal()
        command_buffer = self.queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)
        for i, buffer in enumerate(buffers):
            encoder.setBuffer_offset_atIndex_(buffer, 0, i)
        if use_threadgroups:
            encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                Metal.MTLSizeMake(*grid),
                Metal.MTLSizeMake(*threads),
            )
        else:
            encoder.dispatchThreads_threadsPerThreadgroup_(
                Metal.MTLSizeMake(*grid),
                Metal.MTLSizeMake(*threads),
            )
        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()
        status = command_buffer.status()
        # MTLCommandBufferStatusError is 5. Avoid importing enum names that vary
        # across PyObjC releases.
        if int(status) == 5:
            raise MetalKernelError(f"Metal command buffer failed: {command_buffer.error()}")
        return MetalTiming(gpu_ms=(command_buffer.GPUEndTime() - command_buffer.GPUStartTime()) * 1000.0)

    def ravel_latest1(
        self,
        write_addresses: np.ndarray,
        payloads: np.ndarray,
        read_addresses: Optional[np.ndarray] = None,
        *,
        address_space: int,
        write_mask: Optional[np.ndarray] = None,
        implementation: str = "auto",
        return_timing: bool = False,
    ):
        """Exact strict-causal RAVEL latest-1 lookup for float32 payloads.

        Shapes:
          write_addresses/read_addresses: [B, T, C], integer
          payloads: [B, T, C, D], float32
          write_mask: optional [B, T], truthy values write

        Returns [B, T, C, D] values and [B, T, C] boolean mask. With
        ``return_timing=True``, also returns kernel GPU milliseconds.
        """
        if payloads.ndim != 4:
            raise ValueError("payloads must have shape [B,T,C,D]")
        B, T, C, D = payloads.shape
        if D <= 0:
            raise ValueError("payload dimension must be positive")
        if address_space <= 0:
            raise ValueError("address_space must be positive")
        if implementation not in {"auto", "scalar", "vector", "legacy"}:
            raise ValueError("implementation must be auto, scalar, vector, or legacy")

        if read_addresses is None:
            read_addresses = write_addresses
        if write_addresses.shape != (B, T, C) or read_addresses.shape != (B, T, C):
            raise ValueError("address tensors must have shape [B,T,C]")

        write_i32 = _as_contiguous(np.asarray(write_addresses) % int(address_space), np.dtype(np.int32))
        read_i32 = _as_contiguous(np.asarray(read_addresses) % int(address_space), np.dtype(np.int32))
        payload_f32 = _as_contiguous(payloads, np.dtype(np.float32))
        if write_mask is None:
            mask_u8 = np.ones((1,), dtype=np.uint8)
            has_write_mask = 0
        else:
            if write_mask.shape != (B, T):
                raise ValueError("write_mask must have shape [B,T]")
            mask_u8 = _as_contiguous(write_mask.astype(np.uint8, copy=False), np.dtype(np.uint8))
            has_write_mask = 1

        vector_threads = B * C * (D // 4) if D % 4 == 0 else 0
        use_vector = D % 4 == 0 and (implementation == "vector" or (implementation == "auto" and vector_threads >= 128))
        if implementation == "vector" and D % 4 != 0:
            raise ValueError("vector implementation requires payload dimension divisible by four")
        vector_width = 4 if use_vector else 1
        pipeline = self.ravel_latest1_vec4_pipeline if use_vector else self.ravel_latest1_pipeline
        total_threads = int(B * C * (D // vector_width))
        if implementation == "legacy":
            pipeline = self.ravel_latest1_legacy_pipeline
            total_threads = int(B * C * address_space * D)
        params = np.array(
            [B, T, C, D, int(address_space), has_write_mask, total_threads],
            dtype=np.uint32,
        )

        _, b_write = self._buffer_from_array(write_i32)
        _, b_payload = self._buffer_from_array(payload_f32)
        _, b_read = self._buffer_from_array(read_i32)
        _, b_write_mask = self._buffer_from_array(mask_u8)
        _, b_params = self._buffer_from_array(params)
        b_out = self._empty_buffer(payload_f32.nbytes)
        b_out_mask = self._empty_buffer(B * T * C * np.dtype(np.uint8).itemsize)
        max_threads = int(pipeline.maxTotalThreadsPerThreadgroup())
        threads = min(max_threads, max(32, _ceil_power_of_two(D // vector_width)))
        if implementation == "legacy":
            buffers = [b_write, b_payload, b_read, b_write_mask, b_out, b_out_mask, b_params]
        else:
            empty_state = np.full(B * C * address_space * D, np.nan, dtype=np.float32)
            _, b_latest = self._buffer_from_array(empty_state)
            buffers = [b_write, b_payload, b_read, b_write_mask, b_out, b_out_mask, b_latest, b_params]
        timing = self._run(
            pipeline,
            buffers,
            grid=(max(1, total_threads), 1, 1),
            threads=(threads, 1, 1),
        )
        values = _read_buffer(b_out, (B, T, C, D), np.dtype(np.float32))
        out_mask = _read_buffer(b_out_mask, (B, T, C), np.dtype(np.uint8)).astype(bool)
        if return_timing:
            return values, out_mask, timing.gpu_ms
        return values, out_mask

    def causal_softmax_attention(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        *,
        scale: Optional[float] = None,
        return_timing: bool = False,
    ):
        """Stable causal softmax attention forward for float32 tensors [B,H,T,D]."""
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError("q, k, and v must have identical shape [B,H,T,D]")
        if q.ndim != 4:
            raise ValueError("q, k, and v must have shape [B,H,T,D]")
        B, H, T, D = q.shape
        if D <= 0:
            raise ValueError("head dimension must be positive")
        if T > 2048:
            raise ValueError("Metal attention kernel supports T <= 2048")
        if scale is None:
            scale = 1.0 / math.sqrt(D)

        q_f32 = _as_contiguous(q, np.dtype(np.float32))
        k_f32 = _as_contiguous(k, np.dtype(np.float32))
        v_f32 = _as_contiguous(v, np.dtype(np.float32))
        tg = min(256, int(self.attention_pipeline.maxTotalThreadsPerThreadgroup()))
        tg = min(tg, max(1, _ceil_power_of_two(min(max(D, 1), 256))))
        # Keep a floor of 32 lanes on Apple GPUs for less anemic tiny-D rows.
        tg = min(256, max(32, tg))
        params = np.zeros(1, dtype=np.dtype([("u", np.uint32, 5), ("scale", np.float32)]))
        params["u"][0] = [B, H, T, D, tg]
        params["scale"][0] = np.float32(scale)

        _, b_q = self._buffer_from_array(q_f32)
        _, b_k = self._buffer_from_array(k_f32)
        _, b_v = self._buffer_from_array(v_f32)
        _, b_params = self._buffer_from_array(params)
        b_out = self._empty_buffer(q_f32.nbytes)

        rows = int(B * H * T)
        timing = self._run(
            self.attention_pipeline,
            [b_q, b_k, b_v, b_out, b_params],
            grid=(max(1, rows), 1, 1),
            threads=(tg, 1, 1),
            use_threadgroups=True,
        )
        out = _read_buffer(b_out, (B, H, T, D), np.dtype(np.float32))
        if return_timing:
            return out, timing.gpu_ms
        return out


@lru_cache(maxsize=1)
def get_kernels() -> MetalKernels:
    return MetalKernels.build()


def ravel_latest1(*args, **kwargs):
    return get_kernels().ravel_latest1(*args, **kwargs)


def causal_softmax_attention(*args, **kwargs):
    return get_kernels().causal_softmax_attention(*args, **kwargs)
