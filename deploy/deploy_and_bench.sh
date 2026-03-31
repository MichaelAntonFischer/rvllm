#!/bin/bash
# deploy_and_bench.sh -- Compile all kernels, build Rust binary, run coherence
# tests, and benchmark on GPU (H100/A100/etc).
#
# Usage:
#   bash deploy/deploy_and_bench.sh
#   bash deploy/deploy_and_bench.sh --arch sm_90
#   bash deploy/deploy_and_bench.sh --model /root/models/Qwen2.5-7B --skip-compile
#   bash deploy/deploy_and_bench.sh --skip-build --skip-compile
#
# Flags:
#   --model <path>      model name or path (default: Qwen/Qwen2.5-1.5B)
#   --arch <sm_XX>      override GPU arch (default: auto-detect)
#   --skip-build        skip Rust binary compilation
#   --skip-compile      skip kernel PTX compilation
#   --bench-only        skip coherence tests (implies nothing about build)
#   --output-len <N>    tokens per benchmark request (default: 512)
#   --with-cutlass [dir] compile CUTLASS kernels (default dir: /root/cutlass)
#
# Environment:
#   RVLLM_PTX_DIR   override PTX directory (default: kernels/<arch>)
#   PORT            server port (default: 8000)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Defaults ---
MODEL="Qwen/Qwen2.5-7B"
ARCH=""
SKIP_BUILD=0
SKIP_COMPILE=0
BENCH_ONLY=0
OUTPUT_LEN=512
WITH_CUTLASS=0
CUTLASS_DIR="/root/cutlass"
PORT="${PORT:-8000}"
BASE_URL="http://localhost:${PORT}"

# --- Parse flags ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)       MODEL="$2"; shift 2 ;;
        --arch)        ARCH="$2"; shift 2 ;;
        --skip-build)  SKIP_BUILD=1; shift ;;
        --skip-compile) SKIP_COMPILE=1; shift ;;
        --bench-only)  BENCH_ONLY=1; shift ;;
        --output-len)  OUTPUT_LEN="$2"; shift 2 ;;
        --with-cutlass)
            WITH_CUTLASS=1
            # Optional: next arg is cutlass dir if it doesn't start with --
            if [[ $# -ge 2 && ! "$2" == --* ]]; then
                CUTLASS_DIR="$2"; shift
            fi
            shift ;;
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

pass() { echo -e "${GREEN}[PASS]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
step() { echo -e "${BOLD}${BLUE}=== $* ===${NC}"; }

EXIT_CODE=0

# --- GPU arch detection ---
detect_gpu_arch() {
    if [[ -n "$ARCH" ]]; then
        echo "$ARCH"
        return
    fi
    if ! command -v nvidia-smi &>/dev/null; then
        echo "sm_80"
        return
    fi
    local cc
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.' | tr -d ' ')
    if [[ -z "$cc" || "$cc" == "N/A" ]]; then
        echo "sm_80"
    else
        echo "sm_${cc}"
    fi
}

ARCH=$(detect_gpu_arch)

# --- Environment checks ---
step "Step 0: Environment detection"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "no-gpu")
GPU_MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "?")
echo "GPU:  ${GPU_NAME} (${GPU_MEM_TOTAL} MiB)"
echo "Arch: ${ARCH}"
echo "Model: ${MODEL}"

if command -v nvcc &>/dev/null; then
    NVCC_VER=$(nvcc --version 2>/dev/null | tail -1)
    echo "nvcc: ${NVCC_VER}"
    pass "nvcc found"
else
    warn "nvcc not found"
fi

if command -v cargo &>/dev/null; then
    pass "cargo found ($(cargo --version 2>/dev/null | head -1))"
else
    fail "cargo not found -- install Rust first"
    exit 1
fi

# Kill any rogue GPU processes (vast.ai pytorch image issue)
ROGUE_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || true)
if [[ -n "$ROGUE_PIDS" ]]; then
    warn "Killing rogue GPU processes: ${ROGUE_PIDS}"
    for pid in $ROGUE_PIDS; do
        kill -9 "$pid" 2>/dev/null || true
    done
    sleep 2
fi

GPU_MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "?")
echo "GPU memory used: ${GPU_MEM_USED} MiB"
echo ""

# ============================================================
# Step 1: Compile ALL kernels to PTX
# ============================================================
if [[ "$SKIP_COMPILE" -eq 0 ]]; then
    step "Step 1: Compile kernels (${ARCH})"

    KERNEL_DIR="$REPO_DIR/kernels"
    OUTDIR="$KERNEL_DIR/$ARCH"
    mkdir -p "$OUTDIR"

    CU_COUNT=0
    PTX_OK=0
    PTX_FAIL=0

    for cu in "$KERNEL_DIR"/*.cu; do
        [[ -f "$cu" ]] || continue
        CU_COUNT=$((CU_COUNT + 1))
        stem=$(basename "$cu" .cu)
        ptx="$OUTDIR/${stem}.ptx"

        if [[ "$stem" == "persistent_layer_decode" ]]; then
            # Cooperative groups needs cubin (PTX downgrades grid.sync to bar.sync)
            cubin="$OUTDIR/${stem}.cubin"
            if nvcc -cubin -arch="$ARCH" -O3 --use_fast_math -rdc=true -o "$cubin" "$cu" 2>/tmp/nvcc_${stem}.log; then
                PTX_OK=$((PTX_OK + 1))
            else
                PTX_FAIL=$((PTX_FAIL + 1))
                warn "  FAILED: ${stem}.cu (cubin)"
                cat /tmp/nvcc_${stem}.log 2>/dev/null | tail -3
            fi
        else
            if nvcc -ptx -arch="$ARCH" -O3 --use_fast_math -o "$ptx" "$cu" 2>/tmp/nvcc_${stem}.log; then
                PTX_OK=$((PTX_OK + 1))
            else
                PTX_FAIL=$((PTX_FAIL + 1))
                warn "  FAILED: ${stem}.cu"
                cat /tmp/nvcc_${stem}.log 2>/dev/null | tail -3
            fi
        fi
    done

    if [[ "$PTX_FAIL" -eq 0 ]]; then
        pass "Compiled ${PTX_OK}/${CU_COUNT} kernels to ${OUTDIR}/"
    else
        warn "Compiled ${PTX_OK}/${CU_COUNT} kernels (${PTX_FAIL} failed)"
    fi

    # Set PTX dir
    export RVLLM_PTX_DIR="${RVLLM_PTX_DIR:-$OUTDIR}"
    echo "RVLLM_PTX_DIR=${RVLLM_PTX_DIR}"
    echo ""
else
    step "Step 1: Skipping kernel compilation (--skip-compile)"
    export RVLLM_PTX_DIR="${RVLLM_PTX_DIR:-$REPO_DIR/kernels/$ARCH}"
    echo ""
fi

# ============================================================
# Step 1b: Compile CUTLASS kernels (optional)
# ============================================================
if [[ "$WITH_CUTLASS" -eq 1 ]]; then
    step "Step 1b: Compile CUTLASS kernels (${ARCH})"

    if [ ! -d "$CUTLASS_DIR/include/cutlass" ]; then
        echo "CUTLASS not found at $CUTLASS_DIR, cloning..."
        git clone --depth 1 https://github.com/NVIDIA/cutlass "$CUTLASS_DIR"
    fi

    CUTLASS_BUILD="$REPO_DIR/kernels/build_cutlass.sh"
    if [[ -x "$CUTLASS_BUILD" ]] || [[ -f "$CUTLASS_BUILD" ]]; then
        bash "$CUTLASS_BUILD" "$ARCH" "$CUTLASS_DIR"
    else
        warn "build_cutlass.sh not found at $CUTLASS_BUILD"
    fi
    echo ""
fi

# ============================================================
# Step 2: Build Rust binary
# ============================================================
if [[ "$SKIP_BUILD" -eq 0 ]]; then
    step "Step 2: Build rvllm-server"
    BUILD_START=$(date +%s)

    if command -v nvcc &>/dev/null; then
        cargo build --release --features cuda -p rvllm-server \
            --manifest-path "$REPO_DIR/Cargo.toml" 2>&1 | tail -5
    else
        cargo build --release -p rvllm-server \
            --manifest-path "$REPO_DIR/Cargo.toml" 2>&1 | tail -5
    fi

    BUILD_END=$(date +%s)
    BUILD_SECS=$((BUILD_END - BUILD_START))

    BINARY="$REPO_DIR/target/release/rvllm"
    if [[ -x "$BINARY" ]]; then
        pass "Build complete in ${BUILD_SECS}s -> ${BINARY}"
    else
        fail "Binary not found at ${BINARY}"
        exit 1
    fi
    echo ""
else
    step "Step 2: Skipping build (--skip-build)"
    echo ""
fi

# --- Locate binary ---
BINARY="$REPO_DIR/target/release/rvllm"
if [[ ! -x "$BINARY" ]]; then
    BINARY="$REPO_DIR/target/debug/rvllm"
fi
if [[ ! -x "$BINARY" ]]; then
    fail "rvllm binary not found. Run without --skip-build."
    exit 1
fi

# --- Server lifecycle ---
SERVER_PID=""

start_server() {
    local extra_args="${1:-}"
    pkill -9 -f "rvllm serve" 2>/dev/null || true
    sleep 1

    RVLLM_PTX_DIR="${RVLLM_PTX_DIR}" "$BINARY" serve \
        --model "$MODEL" \
        --port "$PORT" \
        --gpu-memory-utilization 0.90 \
        --dtype half \
        $extra_args \
        > /tmp/rvllm_deploy.log 2>&1 &
    SERVER_PID=$!

    for i in $(seq 1 120); do
        if curl -sf "${BASE_URL}/health" >/dev/null 2>&1; then
            return 0
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            fail "Server exited prematurely"
            tail -20 /tmp/rvllm_deploy.log
            return 1
        fi
        sleep 1
    done
    fail "Server did not become ready in 120s"
    tail -20 /tmp/rvllm_deploy.log
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
}
trap cleanup EXIT INT TERM

# ============================================================
# Step 3: Coherence test
# ============================================================
if [[ "$BENCH_ONLY" -eq 0 ]]; then
    step "Step 3: Coherence test"

    if ! start_server; then
        fail "Cannot start server for coherence test"
        EXIT_CODE=1
    else
        coherence_check() {
            local prompt="$1"
            local expect="$2"
            local label="$3"

            local resp
            resp=$(curl -s -X POST "${BASE_URL}/v1/completions" \
                -H "Content-Type: application/json" \
                -d "{\"model\":\"${MODEL}\",\"prompt\":\"${prompt}\",\"max_tokens\":30,\"temperature\":0.0}" \
                --max-time 60)

            local text
            text=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'])" 2>/dev/null || echo "PARSE_ERROR")

            if echo "$text" | grep -qi "$expect"; then
                pass "${label}: found '${expect}'"
                echo "    -> ${text:0:80}"
                return 0
            else
                fail "${label}: expected '${expect}' not found"
                echo "    -> ${text:0:120}"
                return 1
            fi
        }

        COH_PASS=0
        COH_TOTAL=0

        run_coh() {
            COH_TOTAL=$((COH_TOTAL + 1))
            if coherence_check "$1" "$2" "$3"; then
                COH_PASS=$((COH_PASS + 1))
            fi
        }

        run_coh "The capital of France is" "Paris" "geography"
        run_coh "1 + 1 =" "2" "arithmetic"
        run_coh "The color of the sky is" "blue" "common-knowledge"

        echo ""
        if [[ "$COH_PASS" -eq "$COH_TOTAL" ]]; then
            pass "Coherence: ${COH_PASS}/${COH_TOTAL} passed"
        else
            fail "Coherence: ${COH_PASS}/${COH_TOTAL} passed"
            EXIT_CODE=1
        fi

        # Test determinism: same prompt twice with temp=0 should give same output
        echo ""
        echo "Determinism check (temperature=0):"
        DET_PROMPT="The meaning of life is"
        DET1=$(curl -s -X POST "${BASE_URL}/v1/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"${MODEL}\",\"prompt\":\"${DET_PROMPT}\",\"max_tokens\":20,\"temperature\":0.0}" \
            --max-time 60 | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'])" 2>/dev/null || echo "ERR1")
        DET2=$(curl -s -X POST "${BASE_URL}/v1/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"${MODEL}\",\"prompt\":\"${DET_PROMPT}\",\"max_tokens\":20,\"temperature\":0.0}" \
            --max-time 60 | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'])" 2>/dev/null || echo "ERR2")

        if [[ "$DET1" == "$DET2" && "$DET1" != "ERR1" ]]; then
            pass "Determinism: identical outputs with temperature=0"
        else
            fail "Determinism: outputs differ"
            echo "    run1: ${DET1:0:80}"
            echo "    run2: ${DET2:0:80}"
            EXIT_CODE=1
        fi

        stop_server
    fi
    echo ""
fi

# ============================================================
# Step 4: Benchmark suite
# ============================================================
step "Step 4: Benchmark"

CONCURRENCY_LEVELS="1 4 8 16 32 64 128"

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

if ! start_server; then
    fail "Cannot start server for benchmarking"
    exit 1
fi

# Warmup
echo "Warmup (16 requests)..."
for i in $(seq 1 16); do
    curl -s -X POST "${BASE_URL}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"${MODEL}\",\"prompt\":\"Hello\",\"max_tokens\":${OUTPUT_LEN},\"temperature\":0.7}" \
        --max-time 60 >/dev/null 2>&1 &
done
wait
sleep 1

GPU_MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "?")

echo ""
echo "## rvllm Benchmark Results"
echo ""
echo "- Model: ${MODEL}"
echo "- GPU: ${GPU_NAME} (${GPU_MEM_USED}/${GPU_MEM_TOTAL} MiB)"
echo "- Output tokens: ${OUTPUT_LEN}"
echo "- Arch: ${ARCH}"
echo "- Date: $(date -u +%Y-%m-%d)"
echo ""
echo "| Concurrency | tok/s | wall_ms | total_tok | req_count |"
echo "|-------------|-------|---------|-----------|-----------|"

declare -A RESULTS
NUM_PROMPTS_PER_LEVEL=64

for CONC in $CONCURRENCY_LEVELS; do
    # Scale requests with concurrency to keep wall time reasonable
    NUM_REQ=$NUM_PROMPTS_PER_LEVEL
    if [[ "$CONC" -gt 64 ]]; then
        NUM_REQ=$((CONC * 2))
    fi

    TMPDIR_BENCH=$(mktemp -d)
    BATCH_START=$(date +%s%N)
    PIDS=()

    for i in $(seq 0 $((NUM_REQ - 1))); do
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

        # Throttle to concurrency level
        if [[ "${#PIDS[@]}" -ge "$CONC" ]]; then
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
        [[ -f "$f" ]] || continue
        read -r tok < "$f"
        TOTAL_TOKENS=$((TOTAL_TOKENS + tok))
    done
    rm -rf "$TMPDIR_BENCH"

    TOKS_PER_SEC=0
    if [[ "$WALL_MS" -gt 0 ]]; then
        TOKS_PER_SEC=$(( TOTAL_TOKENS * 1000 / WALL_MS ))
    fi

    RESULTS[$CONC]=$TOKS_PER_SEC
    echo "| ${CONC} | ${TOKS_PER_SEC} | ${WALL_MS} | ${TOTAL_TOKENS} | ${NUM_REQ} |"
done

stop_server

# --- Baseline comparison (Phase 4: A100 80GB) ---
echo ""
echo "### vs Phase 4 baseline (A100 80GB)"
echo ""
echo "| Concurrency | current | baseline | delta |"
echo "|-------------|---------|----------|-------|"

declare -A BASELINE
BASELINE[1]=128
BASELINE[4]=540
BASELINE[8]=1091
BASELINE[16]=2118
BASELINE[32]=3467

for CONC in $CONCURRENCY_LEVELS; do
    CUR=${RESULTS[$CONC]:-0}
    BASE=${BASELINE[$CONC]:-0}
    if [[ "$BASE" -gt 0 && "$CUR" -gt 0 ]]; then
        DELTA_PCT=$(( (CUR - BASE) * 100 / BASE ))
        if [[ "$DELTA_PCT" -ge 0 ]]; then
            DELTA_STR="+${DELTA_PCT}%"
        else
            DELTA_STR="${DELTA_PCT}%"
        fi
    else
        DELTA_STR="--"
    fi
    echo "| ${CONC} | ${CUR} | ${BASE:-n/a} | ${DELTA_STR} |"
done

# ============================================================
# Summary
# ============================================================
echo ""
echo "========================================"
echo "  Deploy + Benchmark Summary"
echo "========================================"
echo "  GPU:   ${GPU_NAME}"
echo "  Arch:  ${ARCH}"
echo "  Model: ${MODEL}"
echo ""
if [[ "$EXIT_CODE" -eq 0 ]]; then
    pass "All steps passed"
else
    fail "Some steps failed (see above)"
fi
echo "========================================"

exit "$EXIT_CODE"
