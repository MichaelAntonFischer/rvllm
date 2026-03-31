#!/usr/bin/env bash
# Long Context Benchmark for rvLLM
#
# Tests decode tok/s at varying context lengths (512, 1024, 2048 tokens)
# to measure how attention performance scales and whether split-KV activates.
#
# Usage:
#   bash bench/bench_long_context.sh
#
# Environment variables:
#   MODEL          - model path or HF repo (default: Qwen/Qwen2.5-1.5B)
#   PORT           - server port (default: 8000)
#   GENERATE_TOKENS - tokens to generate per prompt (default: 512)
#   CONTEXT_LENGTHS - space-separated context sizes (default: "512 1024 2048")

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B}"
PORT="${PORT:-8000}"
GENERATE_TOKENS="${GENERATE_TOKENS:-512}"
CONTEXT_LENGTHS="${CONTEXT_LENGTHS:-512 1024 2048}"
BASE_URL="http://localhost:${PORT}"

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m'

log() { echo -e "${BLUE}[bench]${NC} $*"; }
ok()  { echo -e "${GREEN}[ok]${NC} $*"; }
err() { echo -e "${RED}[err]${NC} $*" >&2; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/.."
SERVER_PID=""

cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        log "Stopping server (pid $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# --- Prerequisites ---
command -v cargo >/dev/null 2>&1 || { err "cargo not found."; exit 1; }
command -v curl >/dev/null 2>&1 || { err "curl not found."; exit 1; }
command -v jq >/dev/null 2>&1 || { err "jq not found."; exit 1; }

# --- Build ---
log "Building rvllm-server..."
cd "$REPO_DIR"
cargo build --release --features cuda -p rvllm-server 2>&1 | tail -3

SERVER_BIN="$REPO_DIR/target/release/rvllm-server"
if [ ! -f "$SERVER_BIN" ]; then
    err "Server binary not found at $SERVER_BIN"
    exit 1
fi

# --- Start server ---
log "Starting server on port $PORT with model $MODEL..."
RUST_LOG=info "$SERVER_BIN" --model "$MODEL" --port "$PORT" &
SERVER_PID=$!

# Wait for server readiness
log "Waiting for server..."
for i in $(seq 1 60); do
    if curl -sf "$BASE_URL/health" >/dev/null 2>&1; then
        ok "Server ready after ${i}s"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        err "Server died during startup"
        exit 1
    fi
    sleep 1
done

if ! curl -sf "$BASE_URL/health" >/dev/null 2>&1; then
    err "Server not ready after 60s"
    exit 1
fi

# --- Generate filler text for context ---
# Creates a prompt of approximately N tokens by repeating words.
# ~1.3 chars per token for English text.
make_prompt() {
    local target_tokens=$1
    local words="The quick brown fox jumps over the lazy dog and the cow jumped over the moon while the dish ran away with the spoon in the middle of the night "
    local result=""
    local current_tokens=0
    while [ "$current_tokens" -lt "$target_tokens" ]; do
        result="${result}${words}"
        current_tokens=$(( current_tokens + 30 ))  # ~30 tokens per repetition
    done
    echo "$result"
}

# --- Run benchmarks ---
RESULTS_FILE="/tmp/rvllm_long_context_results.txt"
echo "Long Context Benchmark Results" > "$RESULTS_FILE"
echo "==============================" >> "$RESULTS_FILE"
echo "Model: $MODEL" >> "$RESULTS_FILE"
echo "Generate tokens: $GENERATE_TOKENS" >> "$RESULTS_FILE"
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
printf "%-12s  %-12s  %-12s  %-15s\n" "Context" "Decode tok/s" "Total ms" "Split-KV" >> "$RESULTS_FILE"
printf "%-12s  %-12s  %-12s  %-15s\n" "-------" "------------" "--------" "--------" >> "$RESULTS_FILE"

log ""
log "=== Long Context Decode Benchmark ==="
log ""

for CTX in $CONTEXT_LENGTHS; do
    log "Context length: $CTX tokens, generating $GENERATE_TOKENS tokens..."

    PROMPT=$(make_prompt "$CTX")

    # Non-streaming request, measure wall clock for the decode portion
    T_START=$(date +%s%N)

    RESPONSE=$(curl -sf -X POST "$BASE_URL/v1/completions" \
        -H "Content-Type: application/json" \
        -d "$(jq -n \
            --arg model "$MODEL" \
            --arg prompt "$PROMPT" \
            --argjson max_tokens "$GENERATE_TOKENS" \
            --argjson temperature 0 \
            '{model: $model, prompt: $prompt, max_tokens: $max_tokens, temperature: $temperature, stream: false}'
        )" 2>&1) || {
        err "Request failed for context=$CTX"
        continue
    }

    T_END=$(date +%s%N)

    # Parse response
    COMPLETION_TOKENS=$(echo "$RESPONSE" | jq -r '.usage.completion_tokens // 0')
    PROMPT_TOKENS=$(echo "$RESPONSE" | jq -r '.usage.prompt_tokens // 0')
    TOTAL_MS=$(( (T_END - T_START) / 1000000 ))

    if [ "$COMPLETION_TOKENS" -eq 0 ]; then
        warn "No completion tokens generated for context=$CTX"
        continue
    fi

    # Estimate decode tok/s (wall clock includes prompt processing)
    # For a more accurate decode measurement, subtract estimated prefill time.
    # Rough heuristic: prefill is ~10x faster than decode per token.
    DECODE_TOKS_PER_SEC=$(echo "$COMPLETION_TOKENS $TOTAL_MS" | awk '{printf "%.1f", $1 / ($2 / 1000.0)}')

    # Check if split-KV attention should be active (context >= 512)
    if [ "$PROMPT_TOKENS" -ge 512 ]; then
        SPLIT_KV="active (ctx>=$PROMPT_TOKENS)"
    else
        SPLIT_KV="inactive"
    fi

    printf "%-12s  %-12s  %-12s  %-15s\n" \
        "${PROMPT_TOKENS}tok" "${DECODE_TOKS_PER_SEC}" "${TOTAL_MS}ms" "$SPLIT_KV" \
        >> "$RESULTS_FILE"

    ok "context=${PROMPT_TOKENS}tok  decode=${DECODE_TOKS_PER_SEC} tok/s  total=${TOTAL_MS}ms  split_kv=$SPLIT_KV"
done

log ""
log "=== Results ==="
cat "$RESULTS_FILE"

# --- Check split-KV activation from server logs ---
log ""
log "Checking server logs for split-KV activation..."
if [ -f /tmp/rvllm_server.log ]; then
    SPLIT_KV_COUNT=$(grep -c "split_kv" /tmp/rvllm_server.log 2>/dev/null || echo "0")
    if [ "$SPLIT_KV_COUNT" -gt 0 ]; then
        ok "Split-KV attention activated $SPLIT_KV_COUNT times in server logs"
    else
        warn "No split-KV activation found in server logs (expected for context >= 512)"
    fi
else
    warn "No server log file found at /tmp/rvllm_server.log"
    warn "Set RUST_LOG=debug and redirect server output to check split-KV activation"
fi

log ""
ok "Long context benchmark complete. Results saved to $RESULTS_FILE"
