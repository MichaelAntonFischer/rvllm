#!/usr/bin/env bash
# verify_fusion.sh -- Build, verify coherence, profile kernels, and benchmark
# after kernel fusion changes.
#
# Usage:
#   bash bench/verify_fusion.sh              # full run
#   bash bench/verify_fusion.sh --skip-build # skip kernel compilation
#   bash bench/verify_fusion.sh --bench-only # skip coherence + profiling
#
# Environment:
#   MODEL          model to serve (default: Qwen/Qwen2.5-1.5B)
#   PORT           server port (default: 8000)
#   MAX_TOKENS     tokens per request (default: 32)
#   CUDA_ARCH      override GPU arch (default: auto-detect)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B}"
PORT="${PORT:-8000}"
MAX_TOKENS="${MAX_TOKENS:-32}"
BASE_URL="http://localhost:${PORT}"

SKIP_BUILD=0
BENCH_ONLY=0
EXIT_CODE=0

for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=1 ;;
        --bench-only) BENCH_ONLY=1; SKIP_BUILD=1 ;;
    esac
done

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

log()  { echo -e "${BLUE}[fusion]${NC} $*"; }
ok()   { echo -e "${GREEN}[PASS]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }

# --- GPU arch detection ---
detect_gpu_arch() {
    if [ -n "${CUDA_ARCH:-}" ]; then
        echo "$CUDA_ARCH"
        return
    fi
    if ! command -v nvidia-smi &>/dev/null; then
        echo "sm_80"
        return
    fi
    local cc
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.' | tr -d ' ')
    if [ -z "$cc" ] || [ "$cc" = "N/A" ]; then
        echo "sm_80"
    else
        echo "sm_${cc}"
    fi
}

# ============================================================
# 1. Build kernels
# ============================================================
if [ "$SKIP_BUILD" -eq 0 ]; then
    log "Step 1: Building kernels"

    ARCH=$(detect_gpu_arch)
    log "Detected GPU arch: ${ARCH}"

    if [ ! -d "$REPO_DIR/kernels" ]; then
        fail "kernels/ directory not found"
        exit 1
    fi

    if command -v nvcc &>/dev/null; then
        log "Compiling .cu -> .ptx for ${ARCH}..."
        bash "$REPO_DIR/kernels/build.sh" "$ARCH"
        PTX_COUNT=$(find "$REPO_DIR/kernels/$ARCH" -name "*.ptx" 2>/dev/null | wc -l | tr -d ' ')
        CU_COUNT=$(find "$REPO_DIR/kernels" -maxdepth 1 -name "*.cu" | wc -l | tr -d ' ')
        ok "Compiled ${PTX_COUNT}/${CU_COUNT} kernels to PTX"
    else
        warn "nvcc not found, skipping kernel compilation"
    fi

    log "Building rvllm-server..."
    if command -v nvcc &>/dev/null; then
        cargo build --release --features cuda -p rvllm-server --manifest-path "$REPO_DIR/Cargo.toml" 2>&1 | tail -3
    else
        cargo build --release -p rvllm-server --manifest-path "$REPO_DIR/Cargo.toml" 2>&1 | tail -3
    fi
    ok "Build complete"
else
    log "Step 1: Skipping build (--skip-build)"
fi

echo ""

# --- Locate binary ---
BINARY="$REPO_DIR/target/release/rvllm"
if [ ! -x "$BINARY" ]; then
    BINARY="$REPO_DIR/target/debug/rvllm"
fi
if [ ! -x "$BINARY" ]; then
    fail "rvllm binary not found. Run without --skip-build."
    exit 1
fi
log "Using binary: ${BINARY}"

# --- Server lifecycle helpers ---
SERVER_PID=""

start_server() {
    local extra_env="${1:-}"
    pkill -9 -f "rvllm serve" 2>/dev/null || true
    sleep 1

    if [ -n "$extra_env" ]; then
        env $extra_env "$BINARY" serve --model "$MODEL" --port "$PORT" > /tmp/rvllm_fusion.log 2>&1 &
    else
        "$BINARY" serve --model "$MODEL" --port "$PORT" > /tmp/rvllm_fusion.log 2>&1 &
    fi
    SERVER_PID=$!

    for i in $(seq 1 90); do
        if curl -sf "${BASE_URL}/health" >/dev/null 2>&1; then
            return 0
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            fail "Server exited prematurely"
            tail -20 /tmp/rvllm_fusion.log
            return 1
        fi
        sleep 1
    done
    fail "Server did not become ready in 90s"
    tail -20 /tmp/rvllm_fusion.log
    return 1
}

stop_server() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
}

cleanup() {
    stop_server
}
trap cleanup EXIT INT TERM

# ============================================================
# 2. Coherence test
# ============================================================
if [ "$BENCH_ONLY" -eq 0 ]; then
    log "Step 2: Coherence test"

    if ! start_server; then
        fail "Cannot start server for coherence test"
        exit 1
    fi

    coherence_check() {
        local prompt="$1"
        local expect="$2"
        local label="$3"

        local resp
        resp=$(curl -s -X POST "${BASE_URL}/v1/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"${MODEL}\",\"prompt\":\"${prompt}\",\"max_tokens\":30,\"temperature\":0.0}" \
            --max-time 30)

        local text
        text=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'])" 2>/dev/null || echo "PARSE_ERROR")

        if echo "$text" | grep -qi "$expect"; then
            ok "${label}: found '${expect}' in output"
            echo "    -> ${text:0:80}"
            return 0
        else
            fail "${label}: expected '${expect}' not found"
            echo "    -> ${text:0:120}"
            return 1
        fi
    }

    COHERENCE_PASS=0
    COHERENCE_TOTAL=0

    run_coherence() {
        COHERENCE_TOTAL=$((COHERENCE_TOTAL + 1))
        if coherence_check "$1" "$2" "$3"; then
            COHERENCE_PASS=$((COHERENCE_PASS + 1))
        fi
    }

    run_coherence "The capital of France is" "Paris" "geography"
    run_coherence "1 + 1 =" "2" "arithmetic"
    run_coherence "The color of the sky is" "blue" "common-knowledge"

    echo ""
    if [ "$COHERENCE_PASS" -eq "$COHERENCE_TOTAL" ]; then
        ok "Coherence: ${COHERENCE_PASS}/${COHERENCE_TOTAL} passed"
    else
        fail "Coherence: ${COHERENCE_PASS}/${COHERENCE_TOTAL} passed"
        EXIT_CODE=1
    fi

    stop_server
    echo ""
fi

# ============================================================
# 3. Kernel-level profiling (RVLLM_PROFILE=1)
# ============================================================
if [ "$BENCH_ONLY" -eq 0 ]; then
    log "Step 3: Kernel profiling"

    if start_server "RVLLM_PROFILE=1 RUST_LOG=info"; then
        # Send a few requests to generate profiling data
        for i in $(seq 1 5); do
            curl -s -X POST "${BASE_URL}/v1/completions" \
                -H "Content-Type: application/json" \
                -d "{\"model\":\"${MODEL}\",\"prompt\":\"Hello world\",\"max_tokens\":16,\"temperature\":0.0}" \
                --max-time 30 >/dev/null 2>&1
        done
        sleep 1
        stop_server

        # Extract profiling info from server log
        PROFILE_LOG=/tmp/rvllm_fusion.log

        echo ""
        echo "${BOLD}Kernel profiling summary:${NC}"
        echo "---"

        # Show per-step timings if available
        if grep -q "PROFILE" "$PROFILE_LOG" 2>/dev/null; then
            grep "PROFILE" "$PROFILE_LOG" | tail -10
        else
            warn "No PROFILE lines found in server log (RVLLM_PROFILE may not emit per-kernel data)"
        fi

        echo ""

        # Fused kernels we expect to see loaded/used
        FUSED_KERNELS="fused_residual_rmsnorm fused_lm_head_argmax fused_rope_cache fused_silu_down fused_norm_gemv"
        echo "${BOLD}Fused kernel detection (from server log):${NC}"
        for k in $FUSED_KERNELS; do
            if grep -qi "$k" "$PROFILE_LOG" 2>/dev/null; then
                ok "  ${k} -- referenced in log"
            else
                warn "  ${k} -- not found in log (may be unused or renamed)"
            fi
        done

        # Check for fallback indicators
        echo ""
        if grep -qi "fallback\|unfused\|slow.path" "$PROFILE_LOG" 2>/dev/null; then
            warn "Fallback/unfused paths detected:"
            grep -i "fallback\|unfused\|slow.path" "$PROFILE_LOG" | head -5
        else
            ok "No fallback/unfused-path warnings detected"
        fi
    else
        warn "Could not start server for profiling, skipping"
    fi
    echo ""
fi

# ============================================================
# 4. Throughput benchmark
# ============================================================
log "Step 4: Throughput benchmark"

CONCURRENCY_LEVELS="1 4 8 16 32"
NUM_PROMPTS_PER_LEVEL=32

PROMPTS=(
    "The capital of France is"
    "Explain quantum computing:"
    "Write a Python sort function:"
    "The theory of relativity"
    "Artificial intelligence in 2024"
    "The speed of light is"
    "A binary search algorithm"
    "The Fibonacci sequence"
    "Machine learning models"
    "The periodic table contains"
    "Rust programming language"
    "Neural networks learn by"
    "The sun is approximately"
    "HTTP status code 404"
    "A linked list is"
    "The Pythagorean theorem"
    "Water boils at"
    "The largest planet is"
    "DNA stands for"
    "Gravity on Earth is"
    "Pi is approximately"
    "The speed of sound"
    "An array in programming"
    "The Milky Way galaxy"
    "Photosynthesis converts"
    "The boiling point of water"
    "A hash table provides"
    "The human brain has"
    "TCP stands for"
    "The chemical formula for water"
    "Moore's law states"
    "The Earth orbits the Sun"
)

if ! start_server; then
    fail "Cannot start server for benchmarking"
    exit 1
fi

# Warmup
log "Warmup (8 requests)..."
for i in $(seq 1 8); do
    curl -s -X POST "${BASE_URL}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"${MODEL}\",\"prompt\":\"Hello\",\"max_tokens\":${MAX_TOKENS},\"temperature\":0.7}" \
        --max-time 60 >/dev/null 2>&1 &
done
wait

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "?")

echo ""
echo "## Fusion Verification Benchmark"
echo ""
echo "- Model: ${MODEL}"
echo "- GPU: ${GPU_NAME} (${GPU_MEM} MiB used)"
echo "- Max tokens: ${MAX_TOKENS}"
echo "- Date: $(date -u +%Y-%m-%d)"
echo ""
echo "| N | tok/s | wall_ms | total_tok |"
echo "|---|-------|---------|-----------|"

# Phase 4 baseline from benchmark-history.md (A100 80GB, Qwen2.5-1.5B, 512 tok)
declare -A BASELINE
BASELINE[1]=128
BASELINE[4]=540
BASELINE[8]=1091
BASELINE[16]=2118
BASELINE[32]=3467

# Store results for comparison table
declare -A RESULTS

for CONC in $CONCURRENCY_LEVELS; do
    TMPDIR_BENCH=$(mktemp -d)
    BATCH_START=$(date +%s%N)
    PIDS=()

    for i in $(seq 0 $((NUM_PROMPTS_PER_LEVEL - 1))); do
        PROMPT="${PROMPTS[$((i % ${#PROMPTS[@]}))]}"
        (
            RESP=$(curl -s -X POST "${BASE_URL}/v1/completions" \
                -H "Content-Type: application/json" \
                -d "{\"model\":\"${MODEL}\",\"prompt\":\"${PROMPT}\",\"max_tokens\":${MAX_TOKENS},\"temperature\":0.7}" \
                --max-time 120)
            TOKENS=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo "0")
            echo "$TOKENS" > "${TMPDIR_BENCH}/result_${i}.txt"
        ) &
        PIDS+=($!)

        # Throttle to concurrency level
        if [ "${#PIDS[@]}" -ge "$CONC" ]; then
            wait "${PIDS[0]}"
            PIDS=("${PIDS[@]:1}")
        fi
    done
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done

    BATCH_END=$(date +%s%N)
    WALL_MS=$(( (BATCH_END - BATCH_START) / 1000000 ))

    TOTAL_TOKENS=0
    for f in "${TMPDIR_BENCH}"/result_*.txt; do
        [ -f "$f" ] || continue
        read -r tok < "$f"
        TOTAL_TOKENS=$((TOTAL_TOKENS + tok))
    done
    rm -rf "$TMPDIR_BENCH"

    TOKS_PER_SEC=0
    if [ "$WALL_MS" -gt 0 ]; then
        TOKS_PER_SEC=$(( TOTAL_TOKENS * 1000 / WALL_MS ))
    fi

    RESULTS[$CONC]=$TOKS_PER_SEC
    echo "| ${CONC} | ${TOKS_PER_SEC} | ${WALL_MS} | ${TOTAL_TOKENS} |"
done

stop_server

# --- Baseline comparison ---
echo ""
echo "### vs Phase 4 baseline (A100 80GB, benchmark-history.md)"
echo ""
echo "| N | current | baseline | delta |"
echo "|---|---------|----------|-------|"
for CONC in $CONCURRENCY_LEVELS; do
    CUR=${RESULTS[$CONC]:-0}
    BASE=${BASELINE[$CONC]:-0}
    if [ "$BASE" -gt 0 ] && [ "$CUR" -gt 0 ]; then
        # Integer percentage: (cur - base) * 100 / base
        DELTA_PCT=$(( (CUR - BASE) * 100 / BASE ))
        if [ "$DELTA_PCT" -ge 0 ]; then
            DELTA_STR="+${DELTA_PCT}%"
        else
            DELTA_STR="${DELTA_PCT}%"
        fi
    else
        DELTA_STR="--"
    fi
    echo "| ${CONC} | ${CUR} | ${BASE} | ${DELTA_STR} |"
done

echo ""

# ============================================================
# Summary
# ============================================================
echo ""
echo "========================================"
echo "  Fusion Verification Summary"
echo "========================================"
if [ "$EXIT_CODE" -eq 0 ]; then
    ok "All checks passed"
else
    fail "Some checks failed (see above)"
fi
echo "========================================"

exit "$EXIT_CODE"
