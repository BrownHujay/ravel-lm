"""In-process HIP kernel compilation via hiprtc, launched on torch's HIP stream.

RDNA4 + Windows ROCm ships ``hiprtc`` (runtime compiler) inside the pip ROCm
SDK. This module compiles HIP C++ source strings to code objects at runtime,
loads them, and launches kernels on torch tensors through torch's current HIP
stream. No external compiler, no on-disk DLL, no PATH setup.

Kernels are cached by (source, arch, options) so each compiles once per process.
"""
from __future__ import annotations

import ctypes
import functools
import glob
import os
from typing import Optional

import torch

_rtc = None
_hip = None
_ARCH = None


def _load_libs():
    global _rtc, _hip, _ARCH
    if _rtc is not None:
        return
    # torch must be imported first so amdhip64 is already in the process.
    site = os.path.dirname(os.path.dirname(torch.__file__))
    candidates = glob.glob(os.path.join(site, "_rocm_sdk_core", "bin"))
    candidates += glob.glob(os.path.join(site, "_rocm_sdk_devel", "bin"))
    binroot = None
    for c in candidates:
        if glob.glob(os.path.join(c, "hiprtc*.dll")):
            binroot = c
            break
    if binroot is None:
        raise RuntimeError("hiprtc DLL not found in ROCm SDK site-packages")
    os.add_dll_directory(binroot)
    rtc_path = glob.glob(os.path.join(binroot, "hiprtc0*.dll"))[0]
    hip_path = glob.glob(os.path.join(binroot, "amdhip64*.dll"))[0]
    rtc = ctypes.CDLL(rtc_path)
    hip = ctypes.CDLL(hip_path)

    rtc.hiprtcCreateProgram.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_char_p)]
    rtc.hiprtcCompileProgram.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    rtc.hiprtcGetProgramLogSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    rtc.hiprtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    rtc.hiprtcGetCodeSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    rtc.hiprtcGetCode.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

    hip.hipModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    hip.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
    hip.hipModuleLaunchKernel.argtypes = [
        ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]

    # Detect gfx arch from the current device.
    props = torch.cuda.get_device_properties(0)
    arch = getattr(props, "gcnArchName", None) or "gfx1201"
    _ARCH = arch.split(":")[0]
    _rtc, _hip = rtc, hip


def available() -> bool:
    try:
        _load_libs()
        return True
    except Exception:
        return False


class Kernel:
    def __init__(self, func_ptr, name):
        self._fn = func_ptr
        self.name = name

    def launch(self, grid, block, args, shared=0, stream=None):
        """grid/block: 3-tuples. args: list of (ctypes value) already boxed."""
        arg_ptrs = (ctypes.c_void_p * len(args))(
            *[ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in args])
        if stream is None:
            stream = torch.cuda.current_stream().cuda_stream
        status = _hip.hipModuleLaunchKernel(
            self._fn, grid[0], grid[1], grid[2], block[0], block[1], block[2],
            shared, ctypes.c_void_p(stream), arg_ptrs, None)
        if status != 0:
            raise RuntimeError(f"hipModuleLaunchKernel({self.name}) -> {status}")


@functools.lru_cache(maxsize=256)
def compile_kernel(source: str, name: str, options: tuple = ()) -> Kernel:
    _load_libs()
    prog = ctypes.c_void_p()
    r = _rtc.hiprtcCreateProgram(ctypes.byref(prog), source.encode(), b"k.hip", 0, None, None)
    if r != 0:
        raise RuntimeError(f"hiprtcCreateProgram -> {r}")
    opt_list = [f"--offload-arch={_ARCH}".encode(), b"-O3"] + [o.encode() for o in options]
    opts = (ctypes.c_char_p * len(opt_list))(*opt_list)
    r = _rtc.hiprtcCompileProgram(prog, len(opt_list), opts)
    logsz = ctypes.c_size_t()
    _rtc.hiprtcGetProgramLogSize(prog, ctypes.byref(logsz))
    log = ""
    if logsz.value > 1:
        buf = ctypes.create_string_buffer(logsz.value)
        _rtc.hiprtcGetProgramLog(prog, buf)
        log = buf.value.decode(errors="replace")
    if r != 0:
        raise RuntimeError(f"hiprtc compile failed ({r}):\n{log}")
    codesz = ctypes.c_size_t()
    _rtc.hiprtcGetCodeSize(prog, ctypes.byref(codesz))
    code = ctypes.create_string_buffer(codesz.value)
    _rtc.hiprtcGetCode(prog, code)
    mod = ctypes.c_void_p()
    if _hip.hipModuleLoadData(ctypes.byref(mod), code) != 0:
        raise RuntimeError("hipModuleLoadData failed")
    fn = ctypes.c_void_p()
    if _hip.hipModuleGetFunction(ctypes.byref(fn), mod, name.encode()) != 0:
        raise RuntimeError(f"hipModuleGetFunction({name}) failed")
    return Kernel(fn, name)


# ctypes arg boxing helpers
def ptr(t: torch.Tensor):
    return ctypes.c_void_p(t.data_ptr())


def i32(v: int):
    return ctypes.c_int(v)
