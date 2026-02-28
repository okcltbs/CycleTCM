#!/bin/bash

# Set script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LABEL_DIR="data/labels/json"
TRAIN_SCRIPT="train/train.py"

# Generate log filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_LOG="cross_validation_cycletcm_${TIMESTAMP}.log"

# Check if label directory exists
if [ ! -d "$LABEL_DIR" ]; then
    echo "Error: Label directory not found: $LABEL_DIR"
    echo "Please check path configuration"
    exit 1
fi

# Clear or create output log file
echo "CycleTCM model training results" > "$OUTPUT_LOG"
echo "CycleTCM" >> "$OUTPUT_LOG"
echo "Generated at: $(date)" >> "$OUTPUT_LOG"
echo "=" >> "$OUTPUT_LOG"

declare -a syndrome_mean_accs
declare -a syndrome_mean_f1s
declare -a syndrome_aucs
declare -a syndrome_mccs
declare -a organ_mean_accs
declare -a organ_mean_f1s
declare -a organ_aucs
declare -a organ_mccs

# Check if train and val files exist
TRAIN_FILE="$LABEL_DIR/train_dataset.json"
VAL_FILE="$LABEL_DIR/val_dataset.json"

if [ ! -f "$TRAIN_FILE" ]; then
    echo "Error: Train file not found: $TRAIN_FILE"
    exit 1
fi

if [ ! -f "$VAL_FILE" ]; then
    echo "Error: Val file not found: $VAL_FILE"
    exit 1
fi

echo ""
echo "=========================================="
echo "Starting training"
echo "=========================================="
echo ""

# Run training; evaluation on test set at the end
# --output_log specifies the output log file
python3 "$TRAIN_SCRIPT" \
    --output_log "$OUTPUT_LOG"

# Check if training succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "Training completed"
    
    # Extract mean accuracy, F1, AUC, MCC from log (syndrome and organ separately)
    SYNDROME_MEAN_LINE=$(grep "Syndrome mean acc:" "$OUTPUT_LOG" | tail -1)
    ORGAN_MEAN_LINE=$(grep "Organ mean acc:" "$OUTPUT_LOG" | tail -1)
    
    if [ -n "$SYNDROME_MEAN_LINE" ]; then
        SYNDROME_MEAN_ACC=$(echo "$SYNDROME_MEAN_LINE" | sed -n 's/.*Syndrome mean acc: \([0-9.]*\).*/\1/p')
        SYNDROME_MEAN_F1=$(echo "$SYNDROME_MEAN_LINE" | sed -n 's/.*mean F1: \([0-9.]*\).*/\1/p')
        SYNDROME_AUC=$(echo "$SYNDROME_MEAN_LINE" | sed -n 's/.*AUC: \([0-9.]*\).*/\1/p')
        SYNDROME_MCC=$(echo "$SYNDROME_MEAN_LINE" | sed -n 's/.*MCC: \([0-9.]*\).*/\1/p')
        
        if [ -n "$SYNDROME_MEAN_ACC" ] && [ -n "$SYNDROME_MEAN_F1" ] && [ -n "$SYNDROME_AUC" ] && [ -n "$SYNDROME_MCC" ]; then
            syndrome_mean_accs+=("$SYNDROME_MEAN_ACC")
            syndrome_mean_f1s+=("$SYNDROME_MEAN_F1")
            syndrome_aucs+=("$SYNDROME_AUC")
            syndrome_mccs+=("$SYNDROME_MCC")
            echo "Syndrome mean accuracy: $SYNDROME_MEAN_ACC | Syndrome mean F1: $SYNDROME_MEAN_F1 | AUC: $SYNDROME_AUC | MCC: $SYNDROME_MCC"
        fi
    fi
    
    if [ -n "$ORGAN_MEAN_LINE" ]; then
        ORGAN_MEAN_ACC=$(echo "$ORGAN_MEAN_LINE" | sed -n 's/.*Organ mean acc: \([0-9.]*\).*/\1/p')
        ORGAN_MEAN_F1=$(echo "$ORGAN_MEAN_LINE" | sed -n 's/.*mean F1: \([0-9.]*\).*/\1/p')
        ORGAN_AUC=$(echo "$ORGAN_MEAN_LINE" | sed -n 's/.*AUC: \([0-9.]*\).*/\1/p')
        ORGAN_MCC=$(echo "$ORGAN_MEAN_LINE" | sed -n 's/.*MCC: \([0-9.]*\).*/\1/p')
        
        if [ -n "$ORGAN_MEAN_ACC" ] && [ -n "$ORGAN_MEAN_F1" ] && [ -n "$ORGAN_AUC" ] && [ -n "$ORGAN_MCC" ]; then
            organ_mean_accs+=("$ORGAN_MEAN_ACC")
            organ_mean_f1s+=("$ORGAN_MEAN_F1")
            organ_aucs+=("$ORGAN_AUC")
            organ_mccs+=("$ORGAN_MCC")
            echo "Organ mean accuracy: $ORGAN_MEAN_ACC | Organ mean F1: $ORGAN_MEAN_F1 | AUC: $ORGAN_AUC | MCC: $ORGAN_MCC"
        fi
    fi
    echo ""
else
    echo ""
    echo "Error: Training failed"
    echo ""
fi

echo "----------------------------------------"

echo ""
echo "=========================================="
echo "CycleTCM model training finished"
echo "=========================================="
echo ""

# Compute mean of accuracy, F1, AUC, MCC
if [ ${#syndrome_mean_accs[@]} -gt 0 ] && [ ${#syndrome_mean_f1s[@]} -gt 0 ] && \
   [ ${#syndrome_aucs[@]} -gt 0 ] && [ ${#syndrome_mccs[@]} -gt 0 ] && \
   [ ${#organ_mean_accs[@]} -gt 0 ] && [ ${#organ_mean_f1s[@]} -gt 0 ] && \
   [ ${#organ_aucs[@]} -gt 0 ] && [ ${#organ_mccs[@]} -gt 0 ]; then
    
    SYNDROME_ACC_ARRAY="["
    SYNDROME_F1_ARRAY="["
    SYNDROME_AUC_ARRAY="["
    SYNDROME_MCC_ARRAY="["
    for i in "${!syndrome_mean_accs[@]}"; do
        if [ $i -gt 0 ]; then
            SYNDROME_ACC_ARRAY+=", "
            SYNDROME_F1_ARRAY+=", "
            SYNDROME_AUC_ARRAY+=", "
            SYNDROME_MCC_ARRAY+=", "
        fi
        SYNDROME_ACC_ARRAY+="${syndrome_mean_accs[$i]}"
        SYNDROME_F1_ARRAY+="${syndrome_mean_f1s[$i]}"
        SYNDROME_AUC_ARRAY+="${syndrome_aucs[$i]}"
        SYNDROME_MCC_ARRAY+="${syndrome_mccs[$i]}"
    done
    SYNDROME_ACC_ARRAY+="]"
    SYNDROME_F1_ARRAY+="]"
    SYNDROME_AUC_ARRAY+="]"
    SYNDROME_MCC_ARRAY+="]"
    
    ORGAN_ACC_ARRAY="["
    ORGAN_F1_ARRAY="["
    ORGAN_AUC_ARRAY="["
    ORGAN_MCC_ARRAY="["
    for i in "${!organ_mean_accs[@]}"; do
        if [ $i -gt 0 ]; then
            ORGAN_ACC_ARRAY+=", "
            ORGAN_F1_ARRAY+=", "
            ORGAN_AUC_ARRAY+=", "
            ORGAN_MCC_ARRAY+=", "
        fi
        ORGAN_ACC_ARRAY+="${organ_mean_accs[$i]}"
        ORGAN_F1_ARRAY+="${organ_mean_f1s[$i]}"
        ORGAN_AUC_ARRAY+="${organ_aucs[$i]}"
        ORGAN_MCC_ARRAY+="${organ_mccs[$i]}"
    done
    ORGAN_ACC_ARRAY+="]"
    ORGAN_F1_ARRAY+="]"
    ORGAN_AUC_ARRAY+="]"
    ORGAN_MCC_ARRAY+="]"
    
    STATS=$(python3 << EOF
import numpy as np
import sys

syndrome_accs = np.array($SYNDROME_ACC_ARRAY)
syndrome_f1s = np.array($SYNDROME_F1_ARRAY)
syndrome_aucs = np.array($SYNDROME_AUC_ARRAY)
syndrome_mccs = np.array($SYNDROME_MCC_ARRAY)
organ_accs = np.array($ORGAN_ACC_ARRAY)
organ_f1s = np.array($ORGAN_F1_ARRAY)
organ_aucs = np.array($ORGAN_AUC_ARRAY)
organ_mccs = np.array($ORGAN_MCC_ARRAY)

syndrome_acc_mean = np.mean(syndrome_accs)
syndrome_f1_mean = np.mean(syndrome_f1s)
syndrome_auc_mean = np.mean(syndrome_aucs)
syndrome_mcc_mean = np.mean(syndrome_mccs)
organ_acc_mean = np.mean(organ_accs)
organ_f1_mean = np.mean(organ_f1s)
organ_auc_mean = np.mean(organ_aucs)
organ_mcc_mean = np.mean(organ_mccs)

print(f"{syndrome_acc_mean:.4f} {syndrome_f1_mean:.4f} {syndrome_auc_mean:.4f} {syndrome_mcc_mean:.4f} {organ_acc_mean:.4f} {organ_f1_mean:.4f} {organ_auc_mean:.4f} {organ_mcc_mean:.4f}")
EOF
)
    
    # Parse Python output
    SYNDROME_ACC_MEAN=$(echo $STATS | awk '{print $1}')
    SYNDROME_F1_MEAN=$(echo $STATS | awk '{print $2}')
    SYNDROME_AUC_MEAN=$(echo $STATS | awk '{print $3}')
    SYNDROME_MCC_MEAN=$(echo $STATS | awk '{print $4}')
    ORGAN_ACC_MEAN=$(echo $STATS | awk '{print $5}')
    ORGAN_F1_MEAN=$(echo $STATS | awk '{print $6}')
    ORGAN_AUC_MEAN=$(echo $STATS | awk '{print $7}')
    ORGAN_MCC_MEAN=$(echo $STATS | awk '{print $8}')
    
    echo "Statistics:" | tee -a "$OUTPUT_LOG"
    echo "Syndrome mean accuracy: $SYNDROME_ACC_MEAN" | tee -a "$OUTPUT_LOG"
    echo "Syndrome mean F1: $SYNDROME_F1_MEAN" | tee -a "$OUTPUT_LOG"
    echo "Syndrome AUC: $SYNDROME_AUC_MEAN" | tee -a "$OUTPUT_LOG"
    echo "Syndrome MCC: $SYNDROME_MCC_MEAN" | tee -a "$OUTPUT_LOG"
    echo "Organ mean accuracy: $ORGAN_ACC_MEAN" | tee -a "$OUTPUT_LOG"
    echo "Organ mean F1: $ORGAN_F1_MEAN" | tee -a "$OUTPUT_LOG"
    echo "Organ AUC: $ORGAN_AUC_MEAN" | tee -a "$OUTPUT_LOG"
    echo "Organ MCC: $ORGAN_MCC_MEAN" | tee -a "$OUTPUT_LOG"
    echo "" | tee -a "$OUTPUT_LOG"
    
    # Write formatted statistics to log file
    {
        echo "=================================================================================="
        echo "CycleTCM model statistics summary"
        echo "=================================================================================="
        echo "Syndrome mean accuracy: $SYNDROME_ACC_MEAN"
        echo "Syndrome mean F1: $SYNDROME_F1_MEAN"
        echo "Syndrome AUC: $SYNDROME_AUC_MEAN"
        echo "Syndrome MCC: $SYNDROME_MCC_MEAN"
        echo "Organ mean accuracy: $ORGAN_ACC_MEAN"
        echo "Organ mean F1: $ORGAN_F1_MEAN"
        echo "Organ AUC: $ORGAN_AUC_MEAN"
        echo "Organ MCC: $ORGAN_MCC_MEAN"
        echo "=================================================================================="
    } >> "$OUTPUT_LOG"
    
else
    echo "Warning: Could not compute statistics" | tee -a "$OUTPUT_LOG"
    echo "(Accuracy, F1, AUC and MCC data are all required)" | tee -a "$OUTPUT_LOG"
fi

echo ""
echo "Results saved to: $OUTPUT_LOG"
echo ""

