#!/bin/bash

printf "[I] Setting up NVIDIA DGX Spark for local AI with ggml\n"

# ------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------
# Ensure required commands are available
for cmd in git curl uname nvidia-smi; do
    if ! command -v $cmd >/dev/null 2>&1; then
        printf "[E] Required command '$cmd' not found in PATH.\n"
        exit 1
    fi
done

# Verify that nvidia-smi runs successfully
if ! nvidia-smi >/dev/null 2>&1; then
    printf "[E] 'nvidia-smi' command failed or no NVIDIA driver detected.\n"
    exit 1
fi

# Verify we are running on a Spark/NVIDIA machine
if ! uname --all | grep -qiE "spark" && ! uname --all | grep -qiE "nvidia"; then
    printf "[E] This script should be run on a Spark/NVIDIA machine (uname output does not contain 'spark' or 'nvidia').\n"
    exit 1
fi

# Warn if the ggml-org directory already exists
if [ -d "$HOME/ggml-org" ]; then
    read -p "[W] The directory '~/ggml-org' already exists and will be deleted. Continue? (y/N) " answer
    case "$answer" in
        [Yy]* ) printf "[I] Proceeding...\n" ;;
        * ) printf "[E] Aborting\n"; exit 1;;
    esac
fi

export LLAMA_LOG_COLORS=1
export LLAMA_LOG_PREFIX=1
export LLAMA_LOG_TIMESTAMPS=1

killall llama-server 2>/dev/null || true
killall whisper-server 2>/dev/null || true

cd ~/
rm -rf ~/ggml-org
mkdir ~/ggml-org

printf "[I] Installing llama.cpp\n"
git clone https://github.com/ggml-org/llama.cpp ~/ggml-org/llama.cpp
cd ~/ggml-org/llama.cpp
cmake -B build-cuda -DGGML_CUDA=ON
cmake --build build-cuda -j

printf "[I] Installing whisper.cpp\n"
git clone https://github.com/ggml-org/whisper.cpp ~/ggml-org/whisper.cpp
cd ~/ggml-org/whisper.cpp
cmake -B build-cuda -DGGML_CUDA=ON
cmake --build build-cuda -j

printf "[I] Downloading Whisper model...\n"
./models/download-ggml-model.sh large-v3-turbo-q8_0 > /dev/null 2>&1

declare -a pids
declare -a ports=(8021 8022 8023 8024 8025)

printf "[P] Starting service on port 8021 (embd) ...\n" "$pid"
~/ggml-org/llama.cpp/build-cuda/bin/llama-server --embd-gemma-default --host 0.0.0.0 --port 8021 > ~/ggml-org/service-embd.log 2>&1 &
pid=$!
pids+=($pid)

printf "[P] Starting service on port 8022 (fim) ...\n" "$pid"
~/ggml-org/llama.cpp/build-cuda/bin/llama-server --fim-qwen-7b-default --host 0.0.0.0 --port 8022 > ~/ggml-org/service-fim.log 2>&1 &
pid=$!
pids+=($pid)

printf "[P] Starting service on port 8023 (chat, tools) ...\n" "$pid"
~/ggml-org/llama.cpp/build-cuda/bin/llama-server --gpt-oss-120b-default --host 0.0.0.0 --port 8023 > ~/ggml-org/service-tools.log 2>&1 &
pid=$!
pids+=($pid)

printf "[P] Starting service on port 8024 (vision) ...\n" "$pid"
~/ggml-org/llama.cpp/build-cuda/bin/llama-server --vision-gemma-4b-default --host 0.0.0.0 --port 8024 > ~/ggml-org/service-vision.log 2>&1 &
pid=$!
pids+=($pid)

printf "[P] Starting service on port 8025 (stt) ...\n" "$pid"
~/ggml-org/whisper.cpp/build-cuda/bin/whisper-server -m ~/ggml-org/whisper.cpp/models/ggml-large-v3-turbo-q8_0.bin --host 0.0.0.0 --port 8025 > ~/ggml-org/service-stt.log 2>&1 &
pid=$!
pids+=($pid)

cleanup() {
    printf "\n[W] Received interrupt, shutting down services...\n"
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            wait "$pid" 2>/dev/null
        fi
    done
    exit 0
}
trap cleanup SIGINT SIGTERM

printf "[I] Downloading models and waiting for services to become healthy - please wait ... (this can take a long time)\n"

# ------------------------------------------------------------
# Health‑check loop – track detailed service state
# ------------------------------------------------------------
# Possible states: down (no response), loading (model still loading), ready (healthy)
declare -A state
for port in "${ports[@]}"; do
    state[$port]="down"
done

while true; do
    all_ready=true

    for port in "${ports[@]}"; do
        response=$(curl -s -m 2 http://127.0.0.1:${port}/health 2>/dev/null || true)
        prev_state="${state[$port]}"

        # ----------------------------------------------------
        # Decide the new state from the HTTP response
        # ----------------------------------------------------
        if [[ -z $response ]]; then
            new_state="down"
        elif [[ $response == *'"status":"ok"'* ]]; then
            new_state="ready"
        elif [[ $response == *'"Loading model"'* ]]; then
            new_state="loading"
        else
            # Any other non‑empty response is considered a loading state
            new_state="loading"
        fi

        # ----------------------------------------------------
        # If the state changed, print a message **including**
        # how many services are still not ready.
        # ----------------------------------------------------
        if [[ "$prev_state" != "$new_state" ]]; then
            # store the updated state first – the count must reflect it
            state[$port]="$new_state"

            # ---- count services that are NOT ready yet ----
            not_ready=0
            for p in "${!state[@]}"; do
                [[ "${state[$p]}" != "ready" ]] && ((not_ready++))
            done

            case "$new_state" in
                down)
                    printf "[P] Service on port %s not reachable yet        (waiting for %d services to initialize ...)\n" \
                           "$port" "$not_ready"
                    ;;
                loading)
                    printf "[P] Service on port %s is loading model ...     (waiting for %d services to initialize ...)\n" \
                           "$port" "$not_ready"
                    ;;
                ready)
                    printf "[P] Service on port %s is ready                 (waiting for %d services to initialize ...)\n" \
                           "$port" "$not_ready"
                    ;;
            esac
        else
            # No change – just keep the current value
            state[$port]="$new_state"
        fi

        # ----------------------------------------------------
        # Track overall readiness for the outer loop
        # ----------------------------------------------------
        if [[ "$new_state" != "ready" ]]; then
            all_ready=false
        fi
    done

    $all_ready && break
    sleep 1
done

printf "[I] All ggml services are up and ready - your NVIDIA DGX Spark is ready to use!\n"

# ------------------------------------------------------------
# Monitoring loop – continue to report state changes
# ------------------------------------------------------------
printf "[I] Entering monitoring loop (Ctrl-C to stop)\n"
while true; do
    sleep 5
done
