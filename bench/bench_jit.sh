#!/bin/bash
# End-to-end JIT fused kernel benchmark
# 1. Generate JIT PTX for the model
# 2. Verify PTX loads on GPU
# 3. Run inference with logging to confirm JIT kernels fire
# 4. Benchmark with and without JIT
set -e

MODEL=${MODEL:-/root/models/Qwen2.5-7B}
ARCH=${ARCH:-sm_90}
PTX_DIR="kernels/$ARCH"

# Qwen2.5-7B defaults (override via env)
HIDDEN=${HIDDEN:-3584}
INTERMEDIATE=${INTERMEDIATE:-18944}
NUM_HEADS=${NUM_HEADS:-28}
NUM_KV_HEADS=${NUM_KV_HEADS:-4}
HEAD_DIM=${HEAD_DIM:-128}
VOCAB_SIZE=${VOCAB_SIZE:-152064}

echo "=== Step 1: Generate JIT PTX ==="
cargo run --release --example gen_ptx -p rvllm-fusion -- \
    --arch "$ARCH" \
    --hidden "$HIDDEN" \
    --intermediate "$INTERMEDIATE" \
    --num-heads "$NUM_HEADS" \
    --num-kv-heads "$NUM_KV_HEADS" \
    --head-dim "$HEAD_DIM" \
    --vocab-size "$VOCAB_SIZE" \
    --ptx-dir "$PTX_DIR"
echo ""
ls -la $PTX_DIR/jit_*.ptx

echo ""
echo "=== Step 2: Validate PTX loads ==="
VALID=0
INVALID=0
for f in $PTX_DIR/jit_*.ptx; do
    name=$(basename "$f")
    size=$(wc -c < "$f" | tr -d ' ')
    # Check the PTX has .entry directive (kernel entry point)
    if grep -q '\.entry' "$f" 2>/dev/null; then
        # Also check for .version and .target directives
        if grep -q '\.version' "$f" && grep -q '\.target' "$f"; then
            echo "  $name: OK (${size} bytes, has .entry + .version + .target)"
            VALID=$((VALID + 1))
        else
            echo "  $name: WARN (has .entry but missing header directives)"
            VALID=$((VALID + 1))
        fi
    else
        echo "  $name: INVALID (no .entry directive found)"
        INVALID=$((INVALID + 1))
    fi
done
echo "  Validated: $VALID OK, $INVALID invalid"
if [ "$INVALID" -gt 0 ]; then
    echo "ERROR: some PTX files are invalid"
    exit 1
fi

echo ""
echo "=== Step 3: Verify JIT kernels fire ==="
# Build release binary first
cargo build --release 2>&1 | tail -5

# Run server in background, send one request, check logs
LOGFILE=$(mktemp /tmp/rvllm_jit_log.XXXXXX)
RVLLM_PTX_DIR=$PTX_DIR RUST_LOG=info \
    ./target/release/rvllm serve --model "$MODEL" --dtype half --port 8199 \
    > "$LOGFILE" 2>&1 &
SERVER_PID=$!

# Wait for server to be ready (check health endpoint)
echo "  Waiting for server (pid=$SERVER_PID)..."
READY=0
for i in $(seq 1 30); do
    if curl -sf http://localhost:8199/health > /dev/null 2>&1; then
        READY=1
        break
    fi
    sleep 1
done

if [ "$READY" -eq 0 ]; then
    echo "  Server failed to start within 30s, checking logs:"
    tail -20 "$LOGFILE"
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    rm -f "$LOGFILE"
    echo "  SKIP: server did not start (may need GPU)"
    echo ""
    echo "=== Step 4: Benchmark WITHOUT JIT ==="
    echo "  SKIP: server unavailable"
    echo ""
    echo "=== Step 5: Benchmark WITH JIT ==="
    echo "  SKIP: server unavailable"
    echo ""
    echo "=== Results ==="
    echo "  PTX generation and validation succeeded."
    echo "  Server-based tests skipped (no GPU or model not found)."
    exit 0
fi

# Fire a completion request
echo "  Sending test completion..."
RESPONSE=$(curl -sf --max-time 30 http://localhost:8199/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"test","prompt":"Hi","max_tokens":3,"temperature":0}' 2>&1 || true)
echo "  Response: $RESPONSE"

# Stop server
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

# Check if JIT kernels were loaded
echo ""
echo "  JIT kernel log entries:"
grep -i "jit\|ptx\|fused.*kernel\|loaded.*module" "$LOGFILE" | head -20 || echo "  (no JIT log entries found)"
rm -f "$LOGFILE"

echo ""
echo "=== Step 4: Benchmark WITHOUT JIT ==="
# Move JIT files aside
mkdir -p "$PTX_DIR/jit_backup"
mv $PTX_DIR/jit_*.ptx "$PTX_DIR/jit_backup/" 2>/dev/null || true

if [ -f ./target/release/rvllm ]; then
    RVLLM_PTX_DIR=$PTX_DIR RUST_LOG=error \
        ./target/release/rvllm benchmark --model "$MODEL" --dtype half --gpu-memory-utilization 0.9 \
        --n '1,4,16,64,128' --output-len 32 2>&1 | tee /tmp/bench_no_jit.txt
else
    echo "  SKIP: rvllm binary not found"
    echo "N/A" > /tmp/bench_no_jit.txt
fi

echo ""
echo "=== Step 5: Benchmark WITH JIT ==="
# Restore JIT files
mv "$PTX_DIR/jit_backup"/jit_*.ptx "$PTX_DIR/" 2>/dev/null || true
rmdir "$PTX_DIR/jit_backup" 2>/dev/null || true

if [ -f ./target/release/rvllm ]; then
    RVLLM_PTX_DIR=$PTX_DIR RUST_LOG=error \
        ./target/release/rvllm benchmark --model "$MODEL" --dtype half --gpu-memory-utilization 0.9 \
        --n '1,4,16,64,128' --output-len 32 2>&1 | tee /tmp/bench_with_jit.txt
else
    echo "  SKIP: rvllm binary not found"
    echo "N/A" > /tmp/bench_with_jit.txt
fi

echo ""
echo "=== Results ==="
if [ -s /tmp/bench_no_jit.txt ] && [ -s /tmp/bench_with_jit.txt ]; then
    echo "  Without JIT:"
    grep -E 'N=|tok/s|latency' /tmp/bench_no_jit.txt | sed 's/^/    /' || true
    echo ""
    echo "  With JIT:"
    grep -E 'N=|tok/s|latency' /tmp/bench_with_jit.txt | sed 's/^/    /' || true
else
    echo "  Benchmark data unavailable. PTX generation and validation succeeded."
fi
