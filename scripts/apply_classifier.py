#!/usr/bin/env python3
"""
Apply Trained Classifiers to Text Files

This script applies trained Lombard orthography classifiers to text files.
Supports both traditional ML (sklearn) and neural network models.

Usage:
    # Apply classifier and output to JSONL
    python apply_classifier.py --input input.txt --model model.pkl --output results.jsonl
    
    # Filter by confidence threshold
    python apply_classifier.py --input input.txt --model model.pt --min-conf 0.8 --output results.jsonl
    
    # Output to stdout
    python apply_classifier.py --input input.txt --model model.pkl
    
    # Error analysis on gold-labeled JSONL test set
    python apply_classifier.py --input test.jsonl --model model.pkl --analysis
    
    # Error analysis with output and limited misclassified examples
    python apply_classifier.py --input test.jsonl --model model.pkl --analysis --max-errors 20 --output report.txt
"""

import json
import argparse
import pickle
import sys
from pathlib import Path
from typing import Tuple, List, Dict
from collections import Counter, defaultdict
import numpy as np
import torch

# Import classifier classes
from classifiers import (
    CharLSTMClassifier, 
    CharCNNClassifier,
    CharDeepCNNClassifier, 
    CharTransformerClassifier,
    ByteNGramAnalyzer
)


class ClassifierApplier:
    """Apply trained classifier to text file"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model_type = None
        self.model = None
        self.vectorizer = None
        self.label_stats = None
        
        # Neural network specific
        self.char_to_idx = None
        self.idx_to_label = None
        self.encoding_type = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._load_model()
    
    def _load_model(self):
        """Load model (sklearn or neural network)"""
        print(f"Loading model from {self.model_path}...", flush=True)
        
        # Try to load as neural network model first
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            self.model_type = 'neural'
            self._load_neural_model(checkpoint)
            print(f"Loaded neural network model ({checkpoint['model_type']}, {checkpoint.get('encoding_type', 'char')}-level)", flush=True)
            return
        except:
            pass
        
        # Try to load as sklearn model
        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.model_type = 'sklearn'
                self.vectorizer = data['vectorizers']
                self.model = data['classifier']
                self.label_stats = data['label_stats']
                print(f"Loaded sklearn model", flush=True)
                return
        except Exception as e:
            raise ValueError(f"Could not load model from {self.model_path}: {e}")
    
    def _load_neural_model(self, checkpoint):
        """Load neural network model"""
        self.char_to_idx = checkpoint['vocab_to_idx']
        self.idx_to_label = checkpoint['idx_to_label']
        self.label_stats = checkpoint['label_stats']
        self.encoding_type = checkpoint.get('encoding_type', 'char')
        
        vocab_size = len(self.char_to_idx)
        num_classes = len(self.idx_to_label)
        model_type = checkpoint['model_type']
        
        # Recreate model architecture
        if model_type == 'lstm':
            self.model = CharLSTMClassifier(vocab_size, 128, 256, num_classes, 2, 0.5)
        elif model_type == 'cnn':
            self.model = CharCNNClassifier(vocab_size, 128, num_classes, 256, 3, 0.5)
        elif model_type == 'deepcnn':
            self.model = CharDeepCNNClassifier(vocab_size, 128, num_classes, 0.5)
        elif model_type == 'transformer':
            self.model = CharTransformerClassifier(vocab_size, 128, num_classes, 8, 4, 0.3)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
    
    def predict_with_confidence(self, text: str) -> Tuple[str, float]:
        """Predict label and confidence for text"""
        if self.model_type == 'sklearn':
            return self._predict_sklearn(text)
        else:
            return self._predict_neural(text)
    
    def _predict_sklearn(self, text: str) -> Tuple[str, float]:
        """Predict with sklearn model"""
        from scipy.sparse import hstack
        
        # Transform with all vectorizers
        X_features = []
        for feat_type in self.vectorizer.keys():
            X_feat = self.vectorizer[feat_type].transform([text])
            X_features.append(X_feat)
        
        if len(X_features) > 1:
            X = hstack(X_features)
        else:
            X = X_features[0]
        
        # Get prediction
        pred = self.model.predict(X)[0]
        
        # Get confidence (probability or decision function)
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)[0]
            confidence = float(np.max(proba))
        elif hasattr(self.model, 'decision_function'):
            decision = self.model.decision_function(X)[0]
            if len(decision.shape) == 0 or decision.shape[0] == 1:
                # Binary classification
                confidence = float(1.0 / (1.0 + np.exp(-abs(decision))))
            else:
                # Multi-class: use softmax
                exp_scores = np.exp(decision - np.max(decision))
                proba = exp_scores / np.sum(exp_scores)
                confidence = float(np.max(proba))
        else:
            confidence = 1.0  # No confidence available
        
        return pred, confidence
    
    def _predict_neural(self, text: str) -> Tuple[str, float]:
        """Predict with neural network model"""
        if self.encoding_type == 'byte':
            # Convert text to byte sequence
            byte_seq = text.encode('utf-8')[:200]
            indices = [b for b in byte_seq]
            # Pad with 256 (special padding value for bytes)
            if len(indices) < 200:
                indices += [256] * (200 - len(indices))
        else:
            # Character-level encoding
            indices = [self.char_to_idx.get(c, self.char_to_idx.get('<UNK>', 1)) 
                      for c in text[:200]]
            if len(indices) < 200:
                indices += [self.char_to_idx.get('<PAD>', 0)] * (200 - len(indices))
        
        input_tensor = torch.tensor([indices], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        pred_label = self.idx_to_label[predicted.item()]
        confidence_score = float(confidence.item())
        
        return pred_label, confidence_score
    
    def classify_file(self, input_path: str, output_path: str = None, 
                     min_confidence: float = 0.0, skip_empty: bool = True):
        """Classify lines from input file"""
        print(f"\nClassifying lines from {input_path}...", flush=True)
        
        results = []
        total_lines = 0
        processed_lines = 0
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                total_lines += 1
                line = line.strip()
                
                # Skip empty lines if requested
                if skip_empty and not line:
                    continue
                
                processed_lines += 1
                
                # Classify
                tag, confidence = self.predict_with_confidence(line)
                
                # Filter by confidence
                if confidence >= min_confidence:
                    result = {
                        'text': line,
                        'tag': tag,
                        'conf': round(confidence, 4)
                    }
                    results.append(result)
                    
                    # Print progress
                    if processed_lines % 1000 == 0:
                        print(f"Processed {processed_lines} lines...", flush=True)
        
        print(f"\nTotal lines read: {total_lines}", flush=True)
        print(f"Non-empty lines processed: {processed_lines}", flush=True)
        print(f"Lines above confidence threshold: {len(results)}", flush=True)
        
        # Output results
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            print(f"Results written to {output_path}", flush=True)
        else:
            # Print to stdout
            for result in results:
                print(json.dumps(result, ensure_ascii=False))
        
        # Print statistics
        self._print_statistics(results)
    
    def _print_statistics(self, results):
        """Print classification statistics"""
        if not results:
            return
        
        print("\n" + "="*60, flush=True)
        print("CLASSIFICATION STATISTICS", flush=True)
        print("="*60, flush=True)
        
        # Tag distribution
        tag_counts = Counter(r['tag'] for r in results)
        print("\nTag distribution:", flush=True)
        for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
            pct = 100 * count / len(results)
            print(f"  {tag:15}: {count:6} ({pct:5.1f}%)", flush=True)
        
        # Confidence statistics
        confidences = [r['conf'] for r in results]
        print(f"\nConfidence statistics:", flush=True)
        print(f"  Mean: {np.mean(confidences):.4f}", flush=True)
        print(f"  Median: {np.median(confidences):.4f}", flush=True)
        print(f"  Min: {np.min(confidences):.4f}", flush=True)
        print(f"  Max: {np.max(confidences):.4f}", flush=True)

    def analyze_file(self, input_path: str, output_path: str = None,
                     max_errors: int = 10, skip_empty: bool = True):
        """Run error analysis on a gold-labeled JSONL file.
        
        Input must be JSONL with 'text' and 'tag' fields.
        Produces a detailed report with per-class metrics, confusion matrix,
        and misclassified examples.
        """
        print(f"\nRunning error analysis on {input_path}...", flush=True)

        gold_labels = []
        pred_labels = []
        confidences = []
        errors: List[Dict] = []  # misclassified examples
        total_lines = 0
        skipped = 0

        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                total_lines += 1
                line = line.strip()
                if skip_empty and not line:
                    skipped += 1
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    print(f"  WARNING: skipping non-JSON line {total_lines}",
                          file=sys.stderr)
                    skipped += 1
                    continue

                text = record.get('text', '')
                gold_tag = record.get('tag')
                if gold_tag is None:
                    print(f"  WARNING: line {total_lines} has no 'tag' field, "
                          "skipping", file=sys.stderr)
                    skipped += 1
                    continue

                pred_tag, conf = self.predict_with_confidence(text)
                gold_labels.append(gold_tag)
                pred_labels.append(pred_tag)
                confidences.append(conf)

                if pred_tag != gold_tag:
                    errors.append({
                        'line': total_lines,
                        'text': text,
                        'gold': gold_tag,
                        'pred': pred_tag,
                        'conf': round(conf, 4)
                    })

                if (total_lines - skipped) % 1000 == 0:
                    print(f"  Processed {total_lines - skipped} samples...",
                          flush=True)

        n = len(gold_labels)
        if n == 0:
            print("No valid samples found.", flush=True)
            return

        # ---- build report ------------------------------------------------
        report_lines = self._build_analysis_report(
            gold_labels, pred_labels, confidences, errors,
            total_lines, skipped, max_errors
        )
        report_text = '\n'.join(report_lines)

        # ---- output -----------------------------------------------------
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text + '\n')
            print(f"\nReport written to {output_path}", flush=True)
        # Always print to stdout as well
        print(report_text, flush=True)

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_analysis_report(
        gold_labels: List[str],
        pred_labels: List[str],
        confidences: List[float],
        errors: List[Dict],
        total_lines: int,
        skipped: int,
        max_errors: int
    ) -> List[str]:
        """Build a human-readable error-analysis report."""
        n = len(gold_labels)
        labels = sorted(set(gold_labels) | set(pred_labels))

        lines: List[str] = []
        sep = '=' * 70
        lines.append('')
        lines.append(sep)
        lines.append('ERROR ANALYSIS REPORT')
        lines.append(sep)

        # --- Overall summary -------------------------------------------
        correct = sum(g == p for g, p in zip(gold_labels, pred_labels))
        accuracy = correct / n
        lines.append(f'\nSamples evaluated : {n}')
        lines.append(f'Lines read         : {total_lines}')
        lines.append(f'Lines skipped      : {skipped}')
        lines.append(f'Overall accuracy   : {accuracy:.4f} ({correct}/{n})')
        lines.append(f'Total errors       : {n - correct}')

        # --- Per-class TP / FP / FN / TN & P / R / F1 -----------------
        lines.append(f'\n{"-"*70}')
        lines.append('PER-CLASS METRICS')
        lines.append(f'{"-"*70}')
        header = (f'{"Label":>15}  {"TP":>6}  {"FP":>6}  {"FN":>6}  '
                  f'{"TN":>6}  {"Prec":>7}  {"Rec":>7}  {"F1":>7}  '
                  f'{"Support":>7}')
        lines.append(header)
        lines.append('-' * len(header))

        macro_p, macro_r, macro_f1 = 0.0, 0.0, 0.0
        weighted_p, weighted_r, weighted_f1 = 0.0, 0.0, 0.0
        total_support = 0

        for label in labels:
            tp = sum(1 for g, p in zip(gold_labels, pred_labels)
                     if g == label and p == label)
            fp = sum(1 for g, p in zip(gold_labels, pred_labels)
                     if g != label and p == label)
            fn = sum(1 for g, p in zip(gold_labels, pred_labels)
                     if g == label and p != label)
            tn = n - tp - fp - fn
            support = tp + fn  # gold count

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            lines.append(
                f'{label:>15}  {tp:>6}  {fp:>6}  {fn:>6}  '
                f'{tn:>6}  {prec:>7.4f}  {rec:>7.4f}  {f1:>7.4f}  '
                f'{support:>7}'
            )

            macro_p += prec
            macro_r += rec
            macro_f1 += f1
            weighted_p += prec * support
            weighted_r += rec * support
            weighted_f1 += f1 * support
            total_support += support

        n_labels = len(labels)
        lines.append('-' * len(header))
        lines.append(
            f'{"macro avg":>15}  {"":>6}  {"":>6}  {"":>6}  '
            f'{"":>6}  {macro_p/n_labels:>7.4f}  {macro_r/n_labels:>7.4f}  '
            f'{macro_f1/n_labels:>7.4f}  {total_support:>7}'
        )
        if total_support > 0:
            lines.append(
                f'{"weighted avg":>15}  {"":>6}  {"":>6}  {"":>6}  '
                f'{"":>6}  {weighted_p/total_support:>7.4f}  '
                f'{weighted_r/total_support:>7.4f}  '
                f'{weighted_f1/total_support:>7.4f}  {total_support:>7}'
            )

        # --- Confusion matrix ------------------------------------------
        lines.append(f'\n{"-"*70}')
        lines.append('CONFUSION MATRIX (rows=gold, cols=predicted)')
        lines.append(f'{"-"*70}')

        # Build matrix
        label_idx = {l: i for i, l in enumerate(labels)}
        cm = np.zeros((n_labels, n_labels), dtype=int)
        for g, p in zip(gold_labels, pred_labels):
            cm[label_idx[g], label_idx[p]] += 1

        col_width = max(max(len(l) for l in labels), 6) + 1
        # header row
        row = ' ' * col_width + ''.join(l.rjust(col_width) for l in labels)
        lines.append(row)
        for i, label in enumerate(labels):
            row = label.rjust(col_width)
            for j in range(n_labels):
                row += str(cm[i, j]).rjust(col_width)
            lines.append(row)

        # --- Confidence breakdown on correct vs errors -----------------
        lines.append(f'\n{"-"*70}')
        lines.append('CONFIDENCE STATISTICS')
        lines.append(f'{"-"*70}')

        correct_confs = [c for g, p, c in zip(gold_labels, pred_labels, confidences)
                         if g == p]
        error_confs   = [c for g, p, c in zip(gold_labels, pred_labels, confidences)
                         if g != p]

        def _conf_stats(vals):
            if not vals:
                return 'n/a'
            arr = np.array(vals)
            return (f'mean={arr.mean():.4f}  median={np.median(arr):.4f}  '
                    f'min={arr.min():.4f}  max={arr.max():.4f}')

        lines.append(f'  Correct predictions ({len(correct_confs):>5}): '
                     f'{_conf_stats(correct_confs)}')
        lines.append(f'  Wrong predictions   ({len(error_confs):>5}): '
                     f'{_conf_stats(error_confs)}')

        # --- Most-confused pairs ---------------------------------------
        lines.append(f'\n{"-"*70}')
        lines.append('MOST CONFUSED PAIRS (gold -> predicted)')
        lines.append(f'{"-"*70}')

        confusion_pairs = Counter(
            (g, p) for g, p in zip(gold_labels, pred_labels) if g != p
        )
        for (g, p), count in confusion_pairs.most_common(15):
            lines.append(f'  {g:>15} -> {p:<15}  {count:>5} times')

        # --- Misclassified examples ------------------------------------
        if errors:
            lines.append(f'\n{"-"*70}')
            lines.append(f'MISCLASSIFIED EXAMPLES (showing up to {max_errors})')
            lines.append(f'{"-"*70}')
            for err in errors[:max_errors]:
                text_preview = (err['text'][:80] + '...')\
                    if len(err['text']) > 80 else err['text']
                lines.append(
                    f'  [line {err["line"]:>5}] '
                    f'gold={err["gold"]:>12}  pred={err["pred"]:>12}  '
                    f'conf={err["conf"]:.4f}'
                )
                lines.append(f'    "{text_preview}"')

        lines.append(f'\n{sep}')
        return lines


def main():
    parser = argparse.ArgumentParser(
        description="Apply trained classifier to text file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classify lines and output to JSONL
  python apply_classifier.py --input input.txt --model model.pkl --output results.jsonl
  
  # Filter by confidence threshold
  python apply_classifier.py --input input.txt --model model.pt --min-conf 0.8 --output results.jsonl
  
  # Output to stdout
  python apply_classifier.py --input input.txt --model model.pkl
  
  # Process empty lines too
  python apply_classifier.py --input input.txt --model model.pkl --keep-empty
  
  # Error analysis on a gold-labeled JSONL test set
  python apply_classifier.py --input test.jsonl --model model.pkl --analysis
  
  # Error analysis with report saved to file
  python apply_classifier.py --input test.jsonl --model model.pkl --analysis --max-errors 20 --output report.txt
        """
    )
    
    parser.add_argument("--input", "-i", type=str, required=True,
                       help="Input text file (one line per sample)")
    parser.add_argument("--model", "-m", type=str, required=True,
                       help="Path to trained model (.pkl or .pt)")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output JSONL file (default: stdout)")
    parser.add_argument("--min-conf", type=float, default=0.0,
                       help="Minimum confidence threshold (0.0-1.0, default: 0.0)")
    parser.add_argument("--keep-empty", action="store_true",
                       help="Process empty lines (default: skip)")
    parser.add_argument("--analysis", action="store_true",
                       help="Run error analysis mode. Input must be JSONL with "
                            "'text' and 'tag' (gold label) fields.")
    parser.add_argument("--max-errors", type=int, default=10,
                       help="Max misclassified examples to show in analysis "
                            "report (default: 10)")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file '{args.input}' not found", file=sys.stderr)
        sys.exit(1)
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model file '{args.model}' not found", file=sys.stderr)
        sys.exit(1)
    
    # Validate confidence threshold
    if not 0.0 <= args.min_conf <= 1.0:
        print(f"Error: Confidence threshold must be between 0.0 and 1.0", file=sys.stderr)
        sys.exit(1)
    
    # Apply classifier
    applier = ClassifierApplier(args.model)

    if args.analysis:
        applier.analyze_file(
            args.input,
            args.output,
            max_errors=args.max_errors,
            skip_empty=not args.keep_empty
        )
    else:
        applier.classify_file(
            args.input, 
            args.output, 
            min_confidence=args.min_conf,
            skip_empty=not args.keep_empty
        )

if __name__ == "__main__":
    main()
