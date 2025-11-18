#!/usr/bin/env python3
"""
Train Lombard Orthography Classifiers

This script trains classifiers for Lombard orthography classification.
Supports both traditional ML (sklearn) and neural network models.

Usage:
    # Train traditional ML classifier
    python train_classifier.py --train-data data/gold/train.jsonl --val-data data/gold/valid.jsonl --test-data data/gold/test.jsonl --type logistic --features char

    # Train neural network classifier
    python train_classifier.py --train-data data/gold/train.jsonl --val-data data/gold/valid.jsonl --model lstm --encoding char --epochs 20

    # Train multiple classifiers
    python train_classifier.py --train-data data/gold/train.jsonl --type logistic,svm,random_forest --features char,byte
"""

import argparse
import json
import sys
from pathlib import Path

# Import classifier classes
from classifiers import (
    LombardClassifier,
    NeuralLombardClassifier
)


def validate_jsonl_format(file_path: str) -> bool:
    """Validate that the JSONL file has correct format (text and tag fields)"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            line_num = 0
            for line in f:
                line_num += 1
                if not line.strip():
                    continue
                    
                try:
                    item = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    print(f"Error: Invalid JSON at line {line_num} in {file_path}: {e}", file=sys.stderr)
                    return False
                
                # Check required fields
                if 'text' not in item:
                    print(f"Error: Missing 'text' field at line {line_num} in {file_path}", file=sys.stderr)
                    return False
                
                if 'tag' not in item:
                    print(f"Error: Missing 'tag' field at line {line_num} in {file_path}", file=sys.stderr)
                    return False
                
                # Validate types
                if not isinstance(item['text'], str):
                    print(f"Error: 'text' field must be string at line {line_num} in {file_path}", file=sys.stderr)
                    return False
                
                if not isinstance(item['tag'], str):
                    print(f"Error: 'tag' field must be string at line {line_num} in {file_path}", file=sys.stderr)
                    return False
                
                # Only check first 100 lines for efficiency
                if line_num >= 100:
                    break
        
        print(f"✓ Data format validation passed for {file_path}", flush=True)
        return True
        
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error validating {file_path}: {e}", file=sys.stderr)
        return False


def train_traditional_ml(args):
    """Train traditional ML classifiers (sklearn)"""
    print("="*80, flush=True)
    print("TRAINING TRADITIONAL ML CLASSIFIERS", flush=True)
    print("="*80, flush=True)
    print(f"Features: {args.features}", flush=True)
    print(f"Vectorizer: {args.vectorizer}", flush=True)
    
    # Parse n-gram ranges
    def parse_ngram_range(s):
        parts = s.split('-')
        return (int(parts[0]), int(parts[1]))
    
    ngram_ranges = {
        'char': parse_ngram_range(args.char_ngrams),
        'byte': parse_ngram_range(args.byte_ngrams),
        'word': parse_ngram_range(args.word_ngrams)
    }
    
    # Parse classifiers
    classifiers = [c.strip() for c in args.type.split(',')]
    
    # Initialize classifier
    clf = LombardClassifier(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        test_data_path=args.test_data
    )
    
    # Load and analyze data
    clf.load_data(min_samples_per_class=args.min_samples)
    
    # Extract features
    X = clf.extract_features(
        feature_types=args.features,
        vectorizer_type=args.vectorizer,
        ngram_ranges=ngram_ranges,
        max_features=args.max_features
    )
    
    # Train each classifier type
    for classifier_type in classifiers:
        print("\n" + "="*60, flush=True)
        print(f"{classifier_type.upper()} (with class balancing)", flush=True)
        print("="*60, flush=True)
        
        clf.train_classifier(X, classifier_type=classifier_type, class_weight='balanced')
        
        # Cross-validation
        clf.cross_validate(X, cv=5)
        
        # Save model
        feature_name = "_".join(args.features.split(','))
        model_path = f'{args.output_dir}/lombard_classifier_model_{classifier_type}_{feature_name}.pkl'
        clf.save_model(model_path)
        
        # Example predictions
        print("\n" + "="*60, flush=True)
        print("EXAMPLE PREDICTIONS", flush=True)
        print("="*60, flush=True)
        test_samples = [
            "Area km²",
            "Violter sii la lüs del mund.",
            "Già che tant quant hinn di oper de lég",
            "L'è 'n cümü todèsch del stat federàl",
            "El gh'ha un teritori muntagnus",
        ]
        
        for sample in test_samples:
            prediction = clf.predict(sample)
            print(f"Text: {sample}", flush=True)
            print(f"Predicted tag: {prediction}\n", flush=True)


def train_neural_network(args):
    """Train neural network classifiers"""
    # Parse comma-separated model types
    model_types = [m.strip() for m in args.model.split(',')]
    valid_models = ['lstm', 'cnn', 'deepcnn', 'transformer']
    
    # Validate model types
    for model_type in model_types:
        if model_type not in valid_models:
            print(f"Error: Invalid model type '{model_type}'. Must be one of {valid_models}", file=sys.stderr)
            sys.exit(1)
    
    print(f"Training {len(model_types)} neural network model(s): {model_types}", flush=True)
    
    # Train each model
    for model_type in model_types:
        print("\n" + "="*80, flush=True)
        print(f"TRAINING MODEL: {model_type.upper()} ({args.encoding}-level)", flush=True)
        print("="*80, flush=True)
        
        # Initialize classifier with encoding type
        clf = NeuralLombardClassifier(
            args.train_data, 
            model_type=model_type,
            val_data_path=args.val_data,
            test_data_path=args.test_data,
            encoding_type=args.encoding
        )
        
        # Load and prepare data
        clf.load_data(min_samples_per_class=args.min_samples)
        
        # Create model with specified parameters
        clf.create_model(
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            num_convs=args.num_convs
        )
        
        # Train (val_split only used if --val-data not provided)
        clf.train(
            epochs=args.epochs, 
            batch_size=args.batch_size, 
            learning_rate=args.lr, 
            val_split=0.1
        )
        
        # Save model
        if args.output and len(model_types) == 1:
            output = args.output
        else:
            output = f'{args.output_dir}/lombard_nn_model_{model_type}_{args.encoding}.pt'
        clf.save_model(output)
        
        # Example predictions
        print("\n" + "="*60, flush=True)
        print("EXAMPLE PREDICTIONS", flush=True)
        print("="*60, flush=True)
        test_samples = [
            "Area km²",
            "Violter sii la lüs del mund.",
            "Già che tant quant hinn di oper de lég",
            "L'è 'n cümü todèsch del stat federàl",
            "El gh'ha un teritori muntagnus"
        ]
        
        for sample in test_samples:
            prediction = clf.predict(sample)
            print(f"Text: {sample}", flush=True)
            print(f"Predicted tag: {prediction}\n", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Train Lombard Orthography Classifiers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train traditional ML classifier (logistic regression with char features)
  python train_classifier.py --train-data data/gold/train.jsonl --type logistic --features char
  
  # Train multiple traditional ML classifiers
  python train_classifier.py --train-data data/gold/train.jsonl --type logistic,svm --features char,byte
  
  # Train neural network (LSTM with char encoding)
  python train_classifier.py --train-data data/gold/train.jsonl --model lstm --encoding char --epochs 20
  
  # Train multiple neural networks
  python train_classifier.py --train-data data/gold/train.jsonl --model lstm,cnn --encoding byte --epochs 10
        """
    )
    
    # Data arguments
    parser.add_argument("--train-data", type=str, required=True,
                       help="Path to training JSONL file (required)")
    parser.add_argument("--val-data", type=str, default=None,
                       help="Path to validation JSONL file (optional)")
    parser.add_argument("--test-data", type=str, default=None,
                       help="Path to test JSONL file (optional)")
    parser.add_argument("--min-samples", type=int, default=10,
                       help="Minimum samples per class (default: 10)")
    
    # Model type arguments (mutually exclusive groups would be better, but keeping simple)
    parser.add_argument("--type", type=str, default=None,
                       help="Traditional ML classifier types (comma-separated): logistic, svm, random_forest, naive_bayes")
    parser.add_argument("--model", type=str, default=None,
                       help="Neural network model types (comma-separated): lstm, cnn, deepcnn, transformer")
    
    # Traditional ML arguments
    parser.add_argument("--features", type=str, default="char",
                       help="Feature types for traditional ML (comma-separated): char, byte, word (default: char)")
    parser.add_argument("--vectorizer", type=str, default="tfidf",
                       choices=['tfidf', 'count'],
                       help="Vectorizer type for traditional ML (default: tfidf)")
    parser.add_argument("--char-ngrams", type=str, default="1-4",
                       help="Character n-gram range (default: 1-4)")
    parser.add_argument("--byte-ngrams", type=str, default="1-4",
                       help="Byte n-gram range (default: 1-4)")
    parser.add_argument("--word-ngrams", type=str, default="1-2",
                       help="Word n-gram range (default: 1-2)")
    parser.add_argument("--max-features", type=int, default=10000,
                       help="Maximum features per vectorizer (default: 10000)")
    
    # Neural network arguments
    parser.add_argument("--encoding", type=str, default="char",
                       choices=['char', 'byte'],
                       help="Encoding type for neural networks: char or byte (default: char)")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=128,
                       help="Batch size (default: 128)")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate (default: 0.001)")
    parser.add_argument("--dropout", type=float, default=0.5,
                       help="Dropout rate (default: 0.5)")
    parser.add_argument("--embedding-dim", type=int, default=128,
                       help="Embedding dimension (default: 128)")
    parser.add_argument("--hidden-dim", type=int, default=256,
                       help="Hidden dimension for LSTM (default: 256)")
    parser.add_argument("--num-layers", type=int, default=2,
                       help="Number of layers (default: 2)")
    parser.add_argument("--num-convs", type=int, default=3,
                       help="Number of parallel convolutions for CNN (default: 3)")
    
    # Output arguments
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output model path (only for single model training)")
    parser.add_argument("--output-dir", type=str,
                       default='./models/ortho_classifier',
                       help="Output directory for models (default: _models/ortho_classifier)")
    
    args = parser.parse_args()
    
    # Validate that either --type or --model is specified
    if args.type is None and args.model is None:
        parser.error("Either --type (for traditional ML) or --model (for neural networks) must be specified")
    
    if args.type is not None and args.model is not None:
        parser.error("Cannot specify both --type and --model. Choose either traditional ML or neural networks")
    
    # Validate input files exist
    if not Path(args.train_data).exists():
        print(f"Error: Training data file not found: {args.train_data}", file=sys.stderr)
        sys.exit(1)
    
    if args.val_data and not Path(args.val_data).exists():
        print(f"Error: Validation data file not found: {args.val_data}", file=sys.stderr)
        sys.exit(1)
    
    if args.test_data and not Path(args.test_data).exists():
        print(f"Error: Test data file not found: {args.test_data}", file=sys.stderr)
        sys.exit(1)
    
    # Validate data format
    print("\nValidating data format...", flush=True)
    if not validate_jsonl_format(args.train_data):
        sys.exit(1)
    
    if args.val_data and not validate_jsonl_format(args.val_data):
        sys.exit(1)
    
    if args.test_data and not validate_jsonl_format(args.test_data):
        sys.exit(1)
    
    # Create output directory if needed
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Train based on type
    if args.type is not None:
        train_traditional_ml(args)
    else:
        train_neural_network(args)
    
    print("\n" + "="*80, flush=True)
    print("TRAINING COMPLETED SUCCESSFULLY", flush=True)
    print("="*80, flush=True)


if __name__ == "__main__":
    main()
