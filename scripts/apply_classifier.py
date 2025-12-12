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
"""

import json
import argparse
import pickle
import sys
from pathlib import Path
from typing import Tuple
from collections import Counter
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
    applier.classify_file(
        args.input, 
        args.output, 
        min_confidence=args.min_conf,
        skip_empty=not args.keep_empty
    )

if __name__ == "__main__":
    main()
