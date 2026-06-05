"""Smoke test for the replaced runai-model-streamer in a vLLM container.

Run inside the container:
    python /opt/smoke_test.py

Verifies:
  1. libstreamer.so loads and exports the expected symbols (including the new cuda param)
  2. A basic CPU file-stream roundtrip works (write temp file, stream it back, compare)
  3. If a CUDA device is available, verifies the CUDA path initializes without error
"""
import sys
import os
import tempfile
import ctypes
import numpy as np

def check_library_load():
    """Verify libstreamer.so loads and has the expected ABI."""
    print("[1/3] Checking libstreamer.so loads and ABI matches... ", end="")
    from runai_model_streamer.libstreamer import dll, STREAMER_LIBRARY

    # Verify the cuda parameter is in argtypes (last arg should be c_int for cuda flag)
    argtypes = dll.fn_runai_request.argtypes
    assert argtypes[-1] == ctypes.c_int, (
        f"Expected last arg of runai_request to be c_int (cuda flag), got {argtypes[-1]}"
    )
    print(f"OK (loaded from {STREAMER_LIBRARY})")

def check_cpu_roundtrip():
    """Write a temp file and stream it back via CPU path."""
    print("[2/3] CPU file-stream roundtrip... ", end="")
    from runai_model_streamer.file_streamer import FileStreamer, FileChunks

    # Create a temp file with known content
    data = os.urandom(4096)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        f.write(data)
        tmp_path = f.name

    try:
        with FileStreamer() as streamer:
            chunks = [len(data)]
            request = FileChunks(id=0, path=tmp_path, offset=0, chunks=chunks)
            streamer.stream_files([request], None, "cpu")

            results = {}
            for file_id, chunk_idx, tensor in streamer.get_chunks():
                results[(file_id, chunk_idx)] = tensor.numpy().tobytes()

        streamed = results[(0, 0)]
        assert streamed == data, f"Data mismatch: expected {len(data)} bytes, got {len(streamed)}"
        print("OK")
    finally:
        os.unlink(tmp_path)

def check_cuda_availability():
    """If CUDA is available, verify the CUDA path can initialize."""
    print("[3/3] CUDA path check... ", end="")
    try:
        import torch
        if not torch.cuda.is_available():
            print("SKIPPED (no CUDA device)")
            return
        if torch.version.hip is not None:
            print("SKIPPED (ROCm, not NVIDIA)")
            return

        device = "cuda:0"
        from runai_model_streamer.file_streamer import FileStreamer, FileChunks
        from runai_model_streamer.file_streamer.requests_iterator import get_cuda_alignment

        alignment = get_cuda_alignment()
        print(f"alignment={alignment} ", end="")

        # Create a temp file and stream via CUDA path
        data = os.urandom(4096)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(data)
            tmp_path = f.name

        try:
            with FileStreamer() as streamer:
                chunks = [len(data)]
                request = FileChunks(id=0, path=tmp_path, offset=0, chunks=chunks)
                streamer.stream_files([request], None, device)

                results = {}
                for file_id, chunk_idx, tensor in streamer.get_chunks():
                    results[(file_id, chunk_idx)] = tensor.cpu().numpy().tobytes()

            streamed = results[(0, 0)]
            assert streamed == data, f"CUDA data mismatch: expected {len(data)} bytes, got {len(streamed)}"
            print("OK")
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        print(f"FAILED: {e}")
        raise

if __name__ == "__main__":
    print("=" * 60)
    print("RunAI Model Streamer Smoke Test (GPUDirect build)")
    print("=" * 60)

    try:
        check_library_load()
        check_cpu_roundtrip()
        check_cuda_availability()
    except Exception as e:
        print(f"\nFAILED: {e}")
        sys.exit(1)

    print("=" * 60)
    print("All checks passed.")
    print("=" * 60)
