#!/bin/bash

# Visual styling
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

echo -e "${BLUE}========================================================================${NC}"
echo -e "${BLUE}${BOLD}🚀 FictiPay Churn Pipeline Monitor${NC}"
echo -e "Press Ctrl+C to exit monitor (the pipeline will run to completion in background)."
echo -e "${BLUE}========================================================================${NC}"

PRED_FILE="./predictions.csv"
if [ -f "$PRED_FILE" ]; then
    # Record initial modification time of predictions.csv
    INIT_TIME=$(stat -c %Y "$PRED_FILE" 2>/dev/null || date -r "$PRED_FILE" +%s 2>/dev/null || echo 0)
    echo -e "ℹ️  Existing predictions.csv found. Monitoring for updates..."
else
    INIT_TIME=0
    echo -e "ℹ️  No predictions.csv found. Monitoring for new generation..."
fi

# Find log file to tail
LOG_FILE=""
if [ -n "$1" ] && [ -f "$1" ]; then
    LOG_FILE="$1"
elif [ -f "pipeline.log" ]; then
    LOG_FILE="pipeline.log"
fi

TAIL_PID=""
if [ -n "$LOG_FILE" ]; then
    echo -e "📄 Tailing log file: ${YELLOW}$LOG_FILE${NC}"
    echo -e "------------------------------------------------------------------------"
    tail -n 50 -f "$LOG_FILE" &
    TAIL_PID=$!
    # Ensure background tail is terminated if monitor is aborted
    trap "kill $TAIL_PID 2>/dev/null" EXIT INT TERM
fi

# Function to check if training is complete and predictions.csv has been successfully generated
check_completion() {
    if [ -f "$PRED_FILE" ]; then
        CUR_TIME=$(stat -c %Y "$PRED_FILE" 2>/dev/null || date -r "$PRED_FILE" +%s 2>/dev/null || echo 0)
        # Check if the file timestamp has changed (or if it was newly created)
        if [ "$CUR_TIME" -gt "$INIT_TIME" ] || [ "$INIT_TIME" -eq 0 ]; then
            # Verify file size is non-zero and has correct headers
            if [ -s "$PRED_FILE" ] && head -n 1 "$PRED_FILE" | grep -q "ACCOUNT_ID"; then
                return 0 # Complete
            fi
        fi
    fi
    return 1 # Incomplete
}

# Main polling loop
SPINNER="/-\|"
idx=0
start_time=$(date +%s)

while true; do
    if check_completion; then
        # Terminate the background tail process
        if [ -n "$TAIL_PID" ]; then
            kill "$TAIL_PID" 2>/dev/null
            wait "$TAIL_PID" 2>/dev/null
        fi
        
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        
        # Clear the status/spinner line
        printf "\r\033[K"
        
        # Display completion message and prediction file details
        echo -e "\n"
        echo -e "${GREEN}========================================================================${NC}"
        echo -e "${GREEN}${BOLD}🎉 SUCCESS: FictiPay Churn Prediction Pipeline Completed!${NC}"
        echo -e "${GREEN}========================================================================${NC}"
        echo -e "🏁 Prediction File : ${BOLD}$PRED_FILE${NC}"
        
        if [ -f "$PRED_FILE" ]; then
            SIZE_BYTES=$(stat -c %s "$PRED_FILE" 2>/dev/null || wc -c < "$PRED_FILE" 2>/dev/null)
            SIZE_MB=$(awk -v size="$SIZE_BYTES" 'BEGIN {printf "%.2f", size/1048576}')
            LINE_COUNT=$(wc -l < "$PRED_FILE" 2>/dev/null | tr -d ' ')
            ROW_COUNT=$((LINE_COUNT - 1))
            if [ "$ROW_COUNT" -lt 0 ]; then ROW_COUNT=0; fi
            
            echo -e "📦 File Size      : ${YELLOW}${SIZE_MB} MB${NC} (${SIZE_BYTES} bytes)"
            echo -e "📊 Row Count     : ${YELLOW}${ROW_COUNT} predictions${NC} (excluding header)"
            echo -e "⏱️  Execution Time: ${YELLOW}${duration} seconds${NC} (monitored interval)"
            echo -e "------------------------------------------------------------------------"
            echo -e "${BOLD}First 5 Rows of Predictions:${NC}"
            head -n 6 "$PRED_FILE"
        fi
        echo -e "${GREEN}========================================================================${NC}"
        exit 0
    fi
    
    # Check if pipeline scripts are currently running
    PIPELINE_RUNNING=false
    if pgrep -f "run_pipeline.py" >/dev/null || pgrep -f "train.py" >/dev/null || pgrep -f "ensemble.py" >/dev/null || pgrep -f "features.py" >/dev/null; then
        PIPELINE_RUNNING=true
    fi
    
    if [ -z "$LOG_FILE" ]; then
        char=${SPINNER:$((idx % 4)):1}
        if $PIPELINE_RUNNING; then
            printf "\r[%c] Churn pipeline is actively running... (waiting for predictions.csv) " "$char"
        else
            printf "\r[%c] No active pipeline processes detected, waiting for predictions.csv update... " "$char"
        fi
        idx=$((idx + 1))
    else
        # If logging, check if training crashed or stopped without generating output
        if ! $PIPELINE_RUNNING; then
            sleep 2
            if check_completion; then
                continue
            fi
            echo -e "\n${RED}⚠️  Error: Pipeline process terminated, but predictions.csv was not updated.${NC}"
            if [ -n "$TAIL_PID" ]; then
                kill "$TAIL_PID" 2>/dev/null
            fi
            exit 1
        fi
    fi
    
    sleep 1
done
