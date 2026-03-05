# Quick Reference - Ortho Classifier Scripts

## Common Commands

### Train Traditional ML
```bash
# Logistic regression with character features
python train_classifier.py \
    --train-data ../data/gold/train.jsonl \
    --val-data ../data/gold/valid.jsonl \
    --test-data ../data/gold/test.jsonl \
    --type logistic \
    --features char
```

### Train Neural Network

```bash
# LSTM with character encoding
python train_classifier.py \
    --train-data ../data/gold/train.jsonl \
    --val-data ../data/gold/valid.jsonl \
    --test-data ../data/gold/test.jsonl \
    --model lstm \
    --encoding char \
    --epochs 20
```

### train_classifier.py

**Data:**
- `--train-data` (required): Training JSONL file
- `--val-data`: Validation JSONL file  
- `--test-data`: Test JSONL file
- `--min-samples`: Minimum samples per class (default: 10)

**Model Type (choose one):**
- `--type`: Traditional ML (logistic, svm, random_forest, naive_bayes)
- `--model`: Neural network (lstm, cnn, deepcnn, transformer)

**Traditional ML Options:**
- `--features`: Feature types (char, byte, word)
- `--vectorizer`: tfidf or count
- `--char-ngrams`: e.g., "1-4"
- `--max-features`: Maximum features (default: 10000)

**Neural Network Options:**
- `--encoding`: char or byte
- `--epochs`: Training epochs (default: 10)
- `--batch-size`: Batch size (default: 128)
- `--lr`: Learning rate (default: 0.001)
- `--dropout`: Dropout rate (default: 0.5)

**Output:**
- `--output`: Single model output path
- `--output-dir`: Output directory (default: _models/ortho_classifier)

### Apply Classifier

```bash
# Classify text file
python apply_classifier.py \
    --input input.txt \
    --model path/to/model.pkl \
    --output results.jsonl
```

### Error Analysis

```bash
# Run error analysis on a gold-labeled test set
python apply_classifier.py \
    --input ../data/gold/test.jsonl \
    --model path/to/model.pkl \
    --analysis

# Save report to file and show more misclassified examples
python apply_classifier.py \
    --input ../data/gold/test.jsonl \
    --model path/to/model.pkl \
    --analysis \
    --max-errors 50 \
    --output report.txt
```

The report includes: overall accuracy, per-class TP/FP/FN/TN with
precision/recall/F1 (macro & weighted), confusion matrix, confidence
statistics for correct vs. wrong predictions, most confused label pairs,
and a sample of misclassified examples.

### apply_classifier.py

**Required:**
- `--input` or `-i`: Input text file (plain text) or JSONL with `text`+`tag` fields (for `--analysis`)
- `--model` or `-m`: Model file (.pkl or .pt)

**Optional:**
- `--output` or `-o`: Output JSONL file (default: stdout); in analysis mode, output file for the report
- `--min-conf`: Minimum confidence (0.0-1.0, default: 0.0)
- `--keep-empty`: Process empty lines (default: skip)
- `--analysis`: Error analysis mode — compare predictions against gold labels in the input JSONL
- `--max-errors`: Max misclassified examples shown in the analysis report (default: 10)

## Data Formats

**Training data (.jsonl):**
```json
{"text": "sample text here", "tag": "LABEL"}
```

**Input text (.txt):**
```
one sample per line
another sample
```

**Output results (.jsonl):**
```json
{"text": "sample text", "tag": "LABEL", "conf": 0.95}
```

