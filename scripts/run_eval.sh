#!/bin/bash
echo "Starting model evaluation..."
cd "$(dirname "$0")"
python ../src/evaluate.py
echo "Evaluation finished."