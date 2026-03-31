#!/bin/bash
# Benchmark CUTLASS kernels: baseline (no CUTLASS PTX) vs with CUTLASS PTX
#
# Usage:
#   bash bench/bench_cutlass.sh
#   bash bench/bench_cutlass.sh --arch sm_90 --cutlass-dir /root/cutlass
#   bash bench/bench_cutlass.sh --model /root/models/Qwen2.5-7B
#
# Flags:
#   --arch <sm_XX>          GPU arch (default: auto-detect)
#   --cutlass-dir <path>    CUTLASS header directory (default: /root/cutlass)
#   --model <path>          model name or path (default: Qwen/Qwen2.5-1.5B)
#   --output-len <N>        tokens per request (default: 512)
#   --skip-build            skip Rust binary compilation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Defaults ---
ARCH=""
CUTLASS_DIR="/root/cutlass"
MODEL="Qwen/Qwen2.5-1.5B"
OUTPUT_LEN=512
PORT="${PORT:-8000}"
SKIP_BUILD=0
BASE_URL="http://localhost:${PORT}"

# --- Parse flags ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --arch)         ARCH="$2"; shift 2 ;;
        --cutlass-dir)  CUTLASS_DIR="$2"; shift 2 ;;
        --model)        MODEL="$2"; shift 2 ;;
        --output-len)   OUTPUT_LEN="$2"; shift 2 ;;
        --skip-build)   SKIP_BUILD=1; shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

step() { echo -e "${BOLD}${BLUE}=== $* ===${NC}"; }
pass() { echo -e "${GREEN}[PASS]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }

# --- GPU arch detection ---
detect_gpu_arch() {
    if [[ -n "$ARCH" ]]; then echo "$ARCH"; return; fi
    if ! command -v nvidia-smi &>/dev/null; then echo "sm_80"; return; fi
    local cc
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.' | tr -d ' ')
    if [[ -z "$cc" || "$cc" == "N/A" ]]; then echo "sm_80"; else echo "sm_${cc}"; fi
}

ARCH=$(detect_gpu_arch)
KERNEL_DIR="$REPO_DIR/kernels"
PTX_DIR="$KERNEL_DIR/$ARCH"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")

# --- Step 1: Clone CUTLASS if needed ---
step "Step 1: Ensure CUTLASS headers"
if [ ! -d "$CUTLASS_DIR/include/cutlass" ]; then
    echo "Cloning CUTLASS to $CUTLASS_DIR ..."
    git clone --depth 1 https://github.com/NVIDIA/cutlass "$CUTLASS_DIR"
fi
pass "CUTLASS headers at $CUTLASS_DIR"

# --- Step 2: Compile CUTLASS kernels ---
step "Step 2: Compile CUTLASS kernels ($ARCH)"
bash "$KERNEL_DIR/build_cutlass.sh" "$ARCH" "$CUTLASS_DIR"

# Also compile standard kernels if PTX dir is sparse
STANDARD_PTX_COUNT=$(ls "$PTX_DIR"/*.ptx 2>/dev/null | grep -cv cutlass_ || echo "0")
if [[ "$STANDARD_PTX_COUNT" -lt 5 ]]; then
    step "Step 2b: Compile standard kernels ($ARCH)"
    bash "$KERNEL_DIR/build.sh" "$ARCH"
fi

# --- Step 3: Build rvllm ---
if [[ "$SKIP_BUILD" -eq 0 ]]; then
    step "Step 3: Build rvllm-server"
    cargo build --release --features cuda -p rvllm-server \
        --manifest-path "$REPO_DIR/Cargo.toml" 2>&1 | tail -5
fi

BINARY="$REPO_DIR/target/release/rvllm"
if [[ ! -x "$BINARY" ]]; then
    fail "Binary not found at $BINARY"
    exit 1
fi

# --- Server helpers ---
SERVER_PID=""

start_server() {
    pkill -9 -f "rvllm serve" 2>/dev/null || true
    sleep 1

    RVLLM_PTX_DIR="$PTX_DIR" "$BINARY" serve \
        --model "$MODEL" \
        --port "$PORT" \
        --gpu-memory-utilization 0.90 \
        --dtype half \
        > /tmp/rvllm_cutlass_bench.log 2>&1 &
    SERVER_PID=$!

    for i in $(seq 1 120); do
        if curl -sf "${BASE_URL}/health" >/dev/null 2>&1; then return 0; fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            fail "Server exited prematurely"
            tail -20 /tmp/rvllm_cutlass_bench.log
            return 1
        fi
        sleep 1
    done
    fail "Server did not become ready in 120s"
    return 1
}

stop_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
}

cleanup() {
    stop_server
    pkill -9 -f "rvllm serve" 2>/dev/null || true
    # Restore CUTLASS PTX if we moved them
    if [[ -d "/tmp/rvllm_cutlass_ptx_backup" ]]; then
        mv /tmp/rvllm_cutlass_ptx_backup/cutlass_*.ptx "$PTX_DIR/" 2>/dev/null || true
        rmdir /tmp/rvllm_cutlass_ptx_backup 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

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
)

CONCURRENCY_LEVELS="1 8 32"
NUM_PROMPTS=64

run_benchmark() {
    local label="$1"
    echo ""
    step "Benchmark: $label"

    # Warmup
    echo "  Warmup (8 requests)..."
    for i in $(seq 1 8); do
        curl -s -X POST "${BASE_URL}/v1/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"${MODEL}\",\"prompt\":\"Hello\",\"max_tokens\":${OUTPUT_LEN},\"temperature\":0.7}" \
            --max-time 60 >/dev/null 2>&1 &
    done
    wait
    sleep 1

    for CONC in $CONCURRENCY_LEVELS; do
        TMPDIR_BENCH=$(mktemp -d)
        BATCH_START=$(date +%s%N)
        PIDS=()

        for i in $(seq 0 $((NUM_PROMPTS - 1))); do
            PROMPT="${PROMPTS[$((i % ${#PROMPTS[@]}))]}"
            (
                RESP=$(curl -s -X POST "${BASE_URL}/v1/completions" \
                    -H "Content-Type: application/json" \
                    -d "{\"model\":\"${MODEL}\",\"prompt\":\"${PROMPT}\",\"max_tokens\":${OUTPUT_LEN},\"temperature\":0.7}" \
                    --max-time 120)
                TOKENS=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo "0")
                echo "$TOKENS" > "${TMPDIR_BENCH}/result_${i}.txt"
            ) &
            PIDS+=($!)

            if [[ "${#PIDS[@]}" -ge "$CONC" ]]; then
                wait "${PIDS[0]}"
                PIDS=("${PIDS[@]:1}")
            fi
        done
        for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done

        BATCH_END=$(date +%s%N)
        WALL_MS=$(( (BATCH_END - BATCH_START) / 1000000 ))

        TOTAL_TOKENS=0
        for f in "${TMPDIR_BENCH}"/result_*.txt; do
            [[ -f "$f" ]] || continue
            read -r tok < "$f"
            TOTAL_TOKENS=$((TOTAL_TOKENS + tok))
        done
        rm -rf "$TMPDIR_BENCH"

        TOKS=0
        if [[ "$WALL_MS" -gt 0 ]]; then
            TOKS=$(( TOTAL_TOKENS * 1000 / WALL_MS ))
        fi

        # Store result in temp file for later comparison
        echo "${CONC} ${TOKS}" >> "/tmp/rvllm_cutlass_${label}.txt"
        echo "  conc=${CONC}: ${TOKS} tok/s (${TOTAL_TOKENS} tokens, ${WALL_MS}ms)"
    done

    stop_server
}

# --- Step 4: Baseline (no CUTLASS PTX) ---
step "Step 4: Baseline benchmark (without CUTLASS kernels)"

# Move CUTLASS PTX files aside
mkdir -p /tmp/rvllm_cutlass_ptx_backup
mv "$PTX_DIR"/cutlass_*.ptx /tmp/rvllm_cutlass_ptx_backup/ 2>/dev/null || true
rm -f /tmp/rvllm_cutlass_baseline.txt

if start_server; then
    run_benchmark "baseline"
else
    fail "Cannot start server for baseline"
    exit 1
fi

# --- Step 5: With CUTLASS kernels ---
step "Step 5: CUTLASS benchmark (with CUTLASS kernels)"

# Restore CUTLASS PTX
mv /tmp/rvllm_cutlass_ptx_backup/cutlass_*.ptx "$PTX_DIR/" 2>/dev/null || true
rmdir /tmp/rvllm_cutlass_ptx_backup 2>/dev/null || true
rm -f /tmp/rvllm_cutlass_cutlass.txt

if start_server; then
    run_benchmark "cutlass"
else
    fail "Cannot start server for CUTLASS benchmark"
    exit 1
fi

# --- Step 6: Comparison table ---
echo ""
echo "========================================"
echo "  CUTLASS Benchmark Comparison"
echo "========================================"
echo "  GPU:   ${GPU_NAME}"
echo "  Arch:  ${ARCH}"
echo "  Model: ${MODEL}"
echo ""
echo "| Concurrency | baseline (tok/s) | CUTLASS (tok/s) | delta |"
echo "|-------------|------------------|-----------------|-------|"

declare -A BASE_RESULTS
declare -A CUT_RESULTS

while read -r conc toks; do
    BASE_RESULTS[$conc]=$toks
done < /tmp/rvllm_cutlass_baseline.txt 2>/dev/null || true

while read -r conc toks; do
    CUT_RESULTS[$conc]=$toks
done < /tmp/rvllm_cutlass_cutlass.txt 2>/dev/null || true

for CONC in $CONCURRENCY_LEVELS; do
    B=${BASE_RESULTS[$CONC]:-0}
    C=${CUT_RESULTS[$CONC]:-0}
    if [[ "$B" -gt 0 && "$C" -gt 0 ]]; then
        DELTA=$(( (C - B) * 100 / B ))
        if [[ "$DELTA" -ge 0 ]]; then
            DELTA_STR="+${DELTA}%"
        else
            DELTA_STR="${DELTA}%"
        fi
    else
        DELTA_STR="--"
    fi
    echo "| ${CONC} | ${B} | ${C} | ${DELTA_STR} |"
done

echo ""
echo "========================================"

# Cleanup temp files
rm -f /tmp/rvllm_cutlass_baseline.txt /tmp/rvllm_cutlass_cutlass.txt
