"""Test the RUNAI_STREAMER_CACHE_DIR feature locally (no S3/GPU needed).

Run from repo root inside devcontainer:
    python3 test_cache.py
"""
import os
import sys
import shutil
import tempfile
import hashlib
import logging
import threading

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

# Set cache dir before importing
cache_dir = tempfile.mkdtemp(prefix="runai_cache_test_")
os.environ["RUNAI_STREAMER_CACHE_DIR"] = cache_dir

print(f"Cache dir: {cache_dir}")
print("=" * 60)

from runai_model_streamer.cache.cache import StreamCache, _cache_key, _is_object_storage_path
from runai_model_streamer.file_streamer import FileStreamer, FileChunks

# --- Test 1: cache enabled ---
print("\n[1/8] Cache enabled with env var... ", end="")
c = StreamCache()
assert c.enabled
print("OK")

# --- Test 2: cache disabled ---
print("[2/8] Cache disabled without env var... ", end="")
saved = os.environ.pop("RUNAI_STREAMER_CACHE_DIR")
c2 = StreamCache()
assert not c2.enabled
os.environ["RUNAI_STREAMER_CACHE_DIR"] = saved
print("OK")

# --- Test 3: cache miss ---
print("[3/8] Cache miss for s3 path... ", end="")
c = StreamCache(cache_dir=cache_dir)
result = c.cached_path_and_offset("s3://bucket/model/file.safetensors")
assert result is None
print("OK")

# --- Test 4: cache hit after placing file ---
print("[4/8] Cache hit after file placed... ", end="")
remote = "s3://bucket/model/file.safetensors"
key = _cache_key(remote)
local_path = os.path.join(cache_dir, key)
sentinel = local_path + ".done"
with open(local_path, "wb") as f:
    f.write(b"fake tensor data" * 1000)
with open(sentinel, "w") as f:
    f.write('{"remote_path": "' + remote + '", "file_offset": 128, "size": 16000}')
result = c.cached_path_and_offset(remote)
assert result is not None
assert result[0] == local_path
assert result[1] == 0  # offset is always 0 for cached files
print(f"OK -> {result}")

# --- Test 5: local path not cached ---
print("[5/8] No caching for local paths... ", end="")
assert not _is_object_storage_path("/local/model/file.safetensors")
assert _is_object_storage_path("s3://bucket/file")
assert _is_object_storage_path("gs://bucket/file")
assert _is_object_storage_path("az://container/file")
result = c.cached_path_and_offset("/local/model/file.safetensors")
assert result is None
print("OK")

# --- Test 6: write-through cache - data written from buffer ---
print("[6/8] Write-through: open_writer + append + finalize... ", end="")
# Clean previous test data
for f in os.listdir(cache_dir):
    os.unlink(os.path.join(cache_dir, f))

remote2 = "s3://bucket/model/shard-001.safetensors"
c3 = StreamCache(cache_dir=cache_dir)

# Simulate streaming: open writer, append data in chunks, finalize
test_data = os.urandom(8192)
c3.open_writer(remote2, file_offset=256)
c3.append_data(remote2, test_data[:4096])
c3.append_data(remote2, test_data[4096:])
c3.finalize(remote2)

# Verify cache hit
result = c3.cached_path_and_offset(remote2)
assert result is not None, "Expected cache hit after finalize"
cached_file, offset = result
assert offset == 0
with open(cached_file, "rb") as f:
    cached_data = f.read()
assert cached_data == test_data, f"Data mismatch: {len(cached_data)} vs {len(test_data)}"
print("OK")

# --- Test 7: end-to-end FileStreamer roundtrip with cache write-through ---
print("[7/8] FileStreamer roundtrip: stream local file, cache writes for remote... ", end="")
data = os.urandom(4096)
with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
    f.write(data)
    tmp_path = f.name

try:
    with FileStreamer() as streamer:
        request = FileChunks(id=0, path=tmp_path, offset=0, chunks=[len(data)])
        streamer.stream_files([request], None, "cpu")
        results = {}
        for file_id, chunk_idx, tensor in streamer.get_chunks():
            results[(file_id, chunk_idx)] = tensor.numpy().tobytes()
    streamed = results[(0, 0)]
    assert streamed == data, f"Data mismatch: {len(data)} vs {len(streamed)}"
    # Local file should NOT be cached (not object storage)
    c_check = StreamCache(cache_dir=cache_dir)
    assert c_check.cached_path_and_offset(tmp_path) is None
    print("OK")
finally:
    os.unlink(tmp_path)

# --- Test 8: race condition - multiple threads writing same file ---
print("[8/8] Race condition: 8 threads writing same file concurrently... ", end="")
# Clean cache
for f in os.listdir(cache_dir):
    os.unlink(os.path.join(cache_dir, f))

race_remote = "s3://bucket/model/race_test.safetensors"
race_data = os.urandom(16384)
errors = []

def worker(thread_id):
    try:
        wc = StreamCache(cache_dir=cache_dir)
        wc.open_writer(race_remote, file_offset=128)
        # Write in small chunks to maximize interleaving
        chunk_size = 1024
        for i in range(0, len(race_data), chunk_size):
            wc.append_data(race_remote, race_data[i:i+chunk_size])
        wc.finalize(race_remote)
    except Exception as e:
        errors.append((thread_id, e))

threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
for t in threads:
    t.start()
for t in threads:
    t.join()

# Only one writer should succeed (first one to open_writer), others are no-ops
# The file should exist and be valid
key = _cache_key(race_remote)
final_path = os.path.join(cache_dir, key)
sentinel_path = final_path + ".done"

assert os.path.exists(final_path), f"Cache file not found: {final_path}"
assert os.path.exists(sentinel_path), f"Sentinel not found: {sentinel_path}"
with open(final_path, "rb") as f:
    content = f.read()
assert content == race_data, f"Data corruption! Expected {len(race_data)} bytes, got {len(content)}"
if errors:
    print(f"WARNING: {len(errors)} thread(s) had errors (expected for losers of the race): {errors}")
else:
    print("OK (no errors, no corruption)")

# Cleanup
shutil.rmtree(cache_dir)
print("\n" + "=" * 60)
print("All cache tests passed!")
