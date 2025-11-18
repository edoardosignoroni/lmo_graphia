"""
Lombard Orthography Classifiers

This module contains all classifier classes for Lombard orthography classification:
- Traditional ML classifiers (Naive Bayes, Logistic Regression, SVM, Random Forest)
- Neural network classifiers (LSTM, CNN, Deep CNN, Transformer)
"""

import json
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from scipy.sparse import hstack
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# Utility Classes
# ============================================================================

class ByteNGramAnalyzer:
    """Picklable byte n-gram analyzer"""
    def __init__(self, ngram_range):
        self.ngram_range = ngram_range
    
    def __call__(self, text):
        """Convert text to byte n-grams"""
        byte_seq = text.encode('utf-8')
        # Generate byte n-grams
        ngrams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(byte_seq) - n + 1):
                ngrams.append(str(byte_seq[i:i+n]))
        return ngrams


# ============================================================================
# Dataset Classes
# ============================================================================

class LombardDataset(Dataset):
    """Dataset for Lombard text classification"""
    def __init__(self, texts, labels, vocab_to_idx, max_length=200, encoding_type='char'):
        self.texts = texts
        self.labels = labels
        self.vocab_to_idx = vocab_to_idx
        self.max_length = max_length
        self.encoding_type = encoding_type
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        if self.encoding_type == 'byte':
            # Convert text to byte sequence
            byte_seq = text.encode('utf-8')[:self.max_length]
            indices = [b for b in byte_seq]  # Bytes are already 0-255
            # Pad with 256 (special padding value for bytes)
            if len(indices) < self.max_length:
                indices += [256] * (self.max_length - len(indices))
        else:
            # Character-level encoding
            indices = [self.vocab_to_idx.get(c, self.vocab_to_idx['<UNK>']) 
                      for c in text[:self.max_length]]
            if len(indices) < self.max_length:
                indices += [self.vocab_to_idx['<PAD>']] * (self.max_length - len(indices))
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# ============================================================================
# Traditional ML Classifier
# ============================================================================

class LombardClassifier:
    """Traditional ML classifier for Lombard orthography using sklearn"""
    
    def __init__(self, train_data_path: str, val_data_path: str = None, test_data_path: str = None):
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.test_data_path = test_data_path
        self.texts = []
        self.labels = []
        self.val_texts = None
        self.val_labels = None
        self.test_texts = None
        self.test_labels = None
        self.vectorizers = {}
        self.classifier = None
        self.label_stats = Counter()
        
    def load_data(self, min_samples_per_class=10):
        """Load data from JSONL file and filter rare classes"""
        all_texts = []
        all_labels = []
        
        with open(self.train_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                all_texts.append(item['text'])
                all_labels.append(item['tag'])
        
        # Count samples per class
        label_counts = Counter(all_labels)
        
        print(f"\nOriginal training data: {len(all_texts)} samples", flush=True)
        print(f"Label distribution:", flush=True)
        for tag, count in sorted(label_counts.items(), key=lambda x: -x[1]):
            print(f"  {tag:15}: {count:6} samples", flush=True)
        
        # Filter out rare classes
        rare_classes = {tag for tag, count in label_counts.items() if count < min_samples_per_class}
        
        if rare_classes:
            print(f"\nFiltering out {len(rare_classes)} rare classes (< {min_samples_per_class} samples):", flush=True)
            for tag in sorted(rare_classes):
                print(f"  - {tag}: {label_counts[tag]} samples", flush=True)
            
            # Keep only common classes
            for text, label in zip(all_texts, all_labels):
                if label not in rare_classes:
                    self.texts.append(text)
                    self.labels.append(label)
                    self.label_stats[label] += 1
        else:
            self.texts = all_texts
            self.labels = all_labels
            self.label_stats = label_counts
        
        print(f"\nFiltered training data: {len(self.texts)} samples", flush=True)
        print(f"Kept {len(self.label_stats)} classes: {sorted(self.label_stats.keys())}", flush=True)
        
        # Load validation data if provided
        if self.val_data_path:
            self.val_texts = []
            self.val_labels = []
            
            with open(self.val_data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    # Only include validation samples from classes we're training on
                    if item['tag'] not in rare_classes:
                        self.val_texts.append(item['text'])
                        self.val_labels.append(item['tag'])
            
            print(f"\nLoaded validation data: {len(self.val_labels)} samples", flush=True)
        
        # Load test data if provided
        if self.test_data_path:
            self.test_texts = []
            self.test_labels = []
            
            with open(self.test_data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    # Only include test samples from classes we're training on
                    if item['tag'] not in rare_classes:
                        self.test_texts.append(item['text'])
                        self.test_labels.append(item['tag'])
            
            print(f"\nLoaded test data: {len(self.test_labels)} samples", flush=True)
    
    def _create_vectorizer(self, feature_type, vectorizer_type='tfidf', 
                          ngram_range=(1, 3), max_features=5000):
        """Create a vectorizer for specific feature type"""
        
        if vectorizer_type == 'tfidf':
            VectorizerClass = TfidfVectorizer
        else:
            VectorizerClass = CountVectorizer
        
        if feature_type == 'char':
            return VectorizerClass(
                ngram_range=ngram_range,
                max_features=max_features,
                analyzer='char_wb',
                lowercase=True
            )
        elif feature_type == 'byte':
            # Use picklable analyzer class
            return VectorizerClass(
                analyzer=ByteNGramAnalyzer(ngram_range),
                max_features=max_features,
                lowercase=False  # Not applicable for bytes
            )
        elif feature_type == 'word':
            return VectorizerClass(
                ngram_range=ngram_range,
                max_features=max_features,
                analyzer='word',
                lowercase=True,
                token_pattern=r'\b\w+\b'
            )
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
        
    def extract_features(self, feature_types='char', vectorizer_type='tfidf', 
                        ngram_ranges=None, max_features=5000):
        """Extract features using specified types
        
        Args:
            feature_types: Comma-separated feature types (char, byte, word)
            vectorizer_type: 'tfidf' or 'count'
            ngram_ranges: Dict mapping feature type to ngram range, e.g., {'char': (1,3), 'word': (1,2)}
            max_features: Maximum number of features per vectorizer
        """
        feature_list = [f.strip() for f in feature_types.split(',')]
        
        # Default ngram ranges
        if ngram_ranges is None:
            ngram_ranges = {
                'char': (1, 4),
                'byte': (1, 4),
                'word': (1, 2)
            }
        
        print(f"\nExtracting features: {feature_list}", flush=True)
        
        X_features = []
        
        for feat_type in feature_list:
            print(f"  - {feat_type} n-grams {ngram_ranges.get(feat_type, (1,3))}", flush=True)
            
            vectorizer = self._create_vectorizer(
                feat_type,
                vectorizer_type=vectorizer_type,
                ngram_range=ngram_ranges.get(feat_type, (1, 3)),
                max_features=max_features
            )
            
            X_feat = vectorizer.fit_transform(self.texts)
            X_features.append(X_feat)
            self.vectorizers[feat_type] = vectorizer
            
            print(f"    Shape: {X_feat.shape}", flush=True)
        
        # Combine features
        if len(X_features) > 1:
            X = hstack(X_features)
            print(f"\nCombined feature matrix shape: {X.shape}", flush=True)
        else:
            X = X_features[0]
            print(f"\nFeature matrix shape: {X.shape}", flush=True)
        
        return X
    
    def train_classifier(self, X, classifier_type='logistic', class_weight='balanced'):
        """Train classifier with class weighting"""
        # Use validation set if provided, otherwise split from training data
        if self.val_texts is not None and self.val_labels is not None:
            print("\nUsing provided validation set", flush=True)
            X_train = X
            y_train = self.labels
            
            # Transform validation data with same vectorizers
            X_val_features = []
            for feat_type in self.vectorizers.keys():
                X_val_feat = self.vectorizers[feat_type].transform(self.val_texts)
                X_val_features.append(X_val_feat)
            
            if len(X_val_features) > 1:
                X_val = hstack(X_val_features)
            else:
                X_val = X_val_features[0]
            
            y_val = self.val_labels
        else:
            print("\nSplitting data with val_size=0.1", flush=True)
            X_train, X_val, y_train, y_val = train_test_split(
                X, self.labels, test_size=0.1, random_state=42, stratify=self.labels
            )
        
        if classifier_type == 'naive_bayes':
            self.classifier = MultinomialNB()
        elif classifier_type == 'logistic':
            self.classifier = LogisticRegression(
                max_iter=1000, 
                random_state=42,
                class_weight=class_weight,
                multi_class='multinomial',
                solver='lbfgs'
            )
        elif classifier_type == 'svm':
            self.classifier = LinearSVC(
                random_state=42,
                class_weight=class_weight,
                max_iter=4000
            )
        elif classifier_type == 'random_forest':
            self.classifier = RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                class_weight=class_weight
            )
        
        print(f"\nTraining {classifier_type} classifier (class_weight={class_weight})...", flush=True)
        self.classifier.fit(X_train, y_train)
        
        # Evaluate on validation set
        train_score = self.classifier.score(X_train, y_train)
        val_score = self.classifier.score(X_val, y_val)
        
        print(f"Training accuracy: {train_score:.4f}", flush=True)
        print(f"Validation accuracy: {val_score:.4f}", flush=True)
        
        # Predictions and detailed report on validation set
        y_pred = self.classifier.predict(X_val)
        print("\nValidation Classification Report:", flush=True)
        print(classification_report(y_val, y_pred, zero_division=0), flush=True)
        
        print("\nValidation Confusion Matrix:", flush=True)
        cm = confusion_matrix(y_val, y_pred)
        print(cm, flush=True)
        
        # Show which classes are confused
        self._print_confusion_analysis(y_val, y_pred, cm)
        
        # Save validation confusion matrix data
        feature_name = "_".join(self.vectorizers.keys())
        self._save_confusion_data(y_val, y_pred, cm, classifier_type, feature_name, 
                                 source_texts=self.val_texts, data_type="validation")
        
        # Final evaluation on test set if provided
        if self.test_texts is not None and self.test_labels is not None:
            print("\n" + "="*60, flush=True)
            print("FINAL EVALUATION ON TEST SET", flush=True)
            print("="*60, flush=True)
            
            # Transform test data with same vectorizers
            X_test_features = []
            for feat_type in self.vectorizers.keys():
                X_test_feat = self.vectorizers[feat_type].transform(self.test_texts)
                X_test_features.append(X_test_feat)
            
            if len(X_test_features) > 1:
                X_test = hstack(X_test_features)
            else:
                X_test = X_test_features[0]
            
            y_test = self.test_labels
            
            # Evaluate
            test_score = self.classifier.score(X_test, y_test)
            print(f"Test accuracy: {test_score:.4f}", flush=True)
            
            # Predictions and detailed report
            y_test_pred = self.classifier.predict(X_test)
            print("\nTest Classification Report:", flush=True)
            print(classification_report(y_test, y_test_pred, zero_division=0), flush=True)
            
            print("\nTest Confusion Matrix:", flush=True)
            cm_test = confusion_matrix(y_test, y_test_pred)
            print(cm_test, flush=True)
            
            # Show which classes are confused
            self._print_confusion_analysis(y_test, y_test_pred, cm_test)
            
            # Save test confusion matrix data
            self._save_confusion_data(y_test, y_test_pred, cm_test, classifier_type, feature_name,
                                     source_texts=self.test_texts, data_type="test")
        
        return X_train, X_val, y_train, y_val
    
    def _print_confusion_analysis(self, y_test, y_pred, cm):
        """Print analysis of confused classes"""
        print("\nMost confused class pairs:", flush=True)
        print("-" * 60, flush=True)
        
        classes = sorted(set(y_test))
        confusions = []
        
        for i, true_class in enumerate(classes):
            for j, pred_class in enumerate(classes):
                if i != j and cm[i, j] > 0:
                    confusions.append((cm[i, j], true_class, pred_class))
        
        # Sort by confusion count
        confusions.sort(reverse=True)
        
        for count, true_cls, pred_cls in confusions[:10]:
            pct = count / cm[classes.index(true_cls)].sum() * 100
            print(f"  {true_cls:10} → {pred_cls:10}: {count:4} errors ({pct:5.1f}%)", flush=True)
    
    def _save_confusion_data(self, y_true, y_pred, cm, classifier_type, feature_name, source_texts=None, data_type="validation"):
        """Save confusion matrix and detailed error analysis"""
        classes = sorted(set(y_true))
        
        # Create pandas DataFrame for better visualization
        cm_df = pd.DataFrame(cm, index=classes, columns=classes)
        
        # Save confusion matrix
        output_dir = './models/confusion_matrices'
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        model_name = f"{classifier_type}_{feature_name}_{data_type}"
        cm_path = f"{output_dir}/confusion_matrix_{model_name}.csv"
        cm_df.to_csv(cm_path)
        print(f"\nConfusion matrix saved to {cm_path}", flush=True)
        
        # Calculate and save per-class metrics
        metrics = []
        for i, true_class in enumerate(classes):
            total = cm[i].sum()
            correct = cm[i, i]
            accuracy = correct / total if total > 0 else 0
            
            # Find most confused classes
            confused_with = []
            for j, pred_class in enumerate(classes):
                if i != j and cm[i, j] > 0:
                    confused_with.append((pred_class, cm[i, j], cm[i, j] / total * 100))
            
            confused_with.sort(key=lambda x: -x[1])
            
            metrics.append({
                'class': true_class,
                'total': total,
                'correct': correct,
                'accuracy': accuracy,
                'top_confusion_1': confused_with[0][0] if len(confused_with) > 0 else '',
                'top_confusion_1_count': confused_with[0][1] if len(confused_with) > 0 else 0,
                'top_confusion_1_pct': confused_with[0][2] if len(confused_with) > 0 else 0,
                'top_confusion_2': confused_with[1][0] if len(confused_with) > 1 else '',
                'top_confusion_2_count': confused_with[1][1] if len(confused_with) > 1 else 0,
                'top_confusion_2_pct': confused_with[1][2] if len(confused_with) > 1 else 0,
            })
        
        metrics_df = pd.DataFrame(metrics)
        metrics_path = f"{output_dir}/class_metrics_{model_name}.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Per-class metrics saved to {metrics_path}", flush=True)
        
        # Save misclassified examples
        misclassified = []
        
        # Use provided source texts or create placeholder
        if source_texts is None:
            print("Warning: Cannot retrieve misclassified texts from split data without source texts", flush=True)
            source_texts = ["[text unavailable]"] * len(y_true)
        
        for i, (true, pred) in enumerate(zip(y_true, y_pred)):
            if true != pred:
                text = source_texts[i] if i < len(source_texts) else ""
                
                misclassified.append({
                    'text': text[:200] if text else "",
                    'true_label': true,
                    'predicted_label': pred
                })
        
        if misclassified:
            mis_df = pd.DataFrame(misclassified)
            mis_path = f"{output_dir}/misclassified_{model_name}.csv"
            mis_df.to_csv(mis_path, index=False, encoding='utf-8')
            print(f"Misclassified examples saved to {mis_path}", flush=True)
    
    def cross_validate(self, X, cv=5):
        """Perform cross-validation with appropriate number of folds"""
        # Adjust CV folds based on smallest class
        min_class_size = min(self.label_stats.values())
        cv = min(cv, min_class_size)
        
        if cv < 2:
            print(f"\nSkipping cross-validation: smallest class has only {min_class_size} samples", flush=True)
            return
        
        print(f"\nPerforming {cv}-fold cross-validation...", flush=True)
        scores = cross_val_score(self.classifier, X, self.labels, cv=cv)
        print(f"Cross-validation scores: {scores}", flush=True)
        print(f"Mean CV score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})", flush=True)
        
    def predict(self, text: str) -> str:
        """Predict label for new text"""
        X_features = []
        for feat_type in self.vectorizers.keys():
            X_feat = self.vectorizers[feat_type].transform([text])
            X_features.append(X_feat)
        
        if len(X_features) > 1:
            X = hstack(X_features)
        else:
            X = X_features[0]
        
        return self.classifier.predict(X)[0]
    
    def save_model(self, model_path: str):
        """Save trained model and vectorizer"""
        with open(model_path, 'wb') as f:
            pickle.dump({
                'vectorizers': self.vectorizers,
                'classifier': self.classifier,
                'label_stats': self.label_stats
            }, f)
        print(f"Model saved to {model_path}", flush=True)
    
    def load_model(self, model_path: str):
        """Load trained model and vectorizer"""
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.vectorizers = data['vectorizers']
            self.classifier = data['classifier']
            self.label_stats = data['label_stats']
        print(f"Model loaded from {model_path}", flush=True)


# ============================================================================
# Neural Network Models
# ============================================================================

class CharLSTMClassifier(nn.Module):
    """Character-level LSTM classifier"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=2, dropout=0.3):
        super(CharLSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last hidden state from both directions
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        output = self.dropout(hidden)
        output = self.fc(output)
        
        return output


class CharCNNClassifier(nn.Module):
    """Character-level CNN classifier (wide - parallel convolutions)"""
    def __init__(self, vocab_size, embedding_dim, num_classes, num_filters=256, num_convs=3, dropout=0.3):
        super(CharCNNClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.num_convs = num_convs
        
        # Create multiple convolutional layers with different kernel sizes dynamically
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=k+3, padding=(k+3)//2)
            for k in range(num_convs)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * num_convs, num_classes)

    def forward(self, x):
        embedded = self.embedding(x).permute(0, 2, 1)  # (batch, embedding, seq_len)
        
        # Apply all convolutions and max pooling
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(embedded))
            pooled = torch.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)
        
        # Concatenate all pooled features
        concatenated = torch.cat(conv_outputs, dim=1)
        output = self.dropout(concatenated)
        output = self.fc(output)
        
        return output


class CharDeepCNNClassifier(nn.Module):
    """Character-level Deep CNN classifier (deep - stacked convolutions)"""
    def __init__(self, vocab_size, embedding_dim, num_classes, dropout=0.5):
        super(CharDeepCNNClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Deep convolutional stack
        self.conv1 = nn.Conv1d(embedding_dim, 256, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(256)
        
        self.conv2 = nn.Conv1d(256, 256, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.conv3 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        
        self.conv5 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(256)
        
        self.conv6 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm1d(256)
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Embedding
        embedded = self.embedding(x).permute(0, 2, 1)  # (batch, embedding, seq_len)
        
        # Deep convolutional layers with batch norm and pooling
        x = torch.relu(self.bn1(self.conv1(embedded)))
        x = torch.max_pool1d(x, 3, stride=2)  # Reduce sequence length
        
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.max_pool1d(x, 3, stride=2)
        
        x = torch.relu(self.bn3(self.conv3(x)))
        
        x = torch.relu(self.bn4(self.conv4(x)))
        x = torch.max_pool1d(x, 3, stride=2)
        
        x = torch.relu(self.bn5(self.conv5(x)))
        
        x = torch.relu(self.bn6(self.conv6(x)))
        
        # Global max pooling
        x = torch.max_pool1d(x, x.size(2)).squeeze(2)
        
        # Fully connected layers
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class CharTransformerClassifier(nn.Module):
    """Character-level Transformer classifier"""
    def __init__(self, vocab_size, embedding_dim, num_classes, num_heads=8, num_layers=4, dropout=0.3):
        super(CharTransformerClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_encoder = nn.Parameter(torch.randn(1, 200, embedding_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x):
        # Embedding with positional encoding
        embedded = self.embedding(x)
        embedded = embedded + self.pos_encoder[:, :x.size(1), :]
        
        # Create padding mask (True for padding tokens)
        padding_mask = (x == 0)
        
        # Transformer encoding
        transformer_out = self.transformer_encoder(embedded, src_key_padding_mask=padding_mask)
        
        # Use mean pooling over non-padding tokens
        mask = (~padding_mask).unsqueeze(-1).float()
        pooled = (transformer_out * mask).sum(dim=1) / mask.sum(dim=1)
        
        # Classification
        output = self.dropout(pooled)
        output = self.fc(output)
        
        return output


# ============================================================================
# Neural Network Classifier Wrapper
# ============================================================================

class NeuralLombardClassifier:
    """Neural network classifier for Lombard orthography"""
    
    def __init__(self, data_path: str, model_type='lstm', device=None, val_data_path=None, test_data_path=None, encoding_type='char'):
        self.data_path = data_path
        self.val_data_path = val_data_path
        self.test_data_path = test_data_path
        self.model_type = model_type
        self.encoding_type = encoding_type  # 'char' or 'byte'
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.texts = []
        self.labels = []
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.vocab_to_idx = {'<PAD>': 0, '<UNK>': 1} if encoding_type == 'char' else {}
        self.model = None
        self.label_stats = Counter()
        
        print(f"Using device: {self.device}", flush=True)
        print(f"Encoding type: {encoding_type}", flush=True)
        
    def load_data(self, min_samples_per_class=10):
        """Load data from JSONL file and filter rare classes"""
        all_texts = []
        all_labels = []
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                all_texts.append(item['text'])
                all_labels.append(item['tag'])
        
        # Count samples per class
        label_counts = Counter(all_labels)
        
        print(f"\nOriginal training data: {len(all_texts)} samples", flush=True)
        print(f"Label distribution:", flush=True)
        for tag, count in sorted(label_counts.items(), key=lambda x: -x[1]):
            print(f"  {tag:15}: {count:6} samples", flush=True)
        
        # Filter out rare classes
        rare_classes = {tag for tag, count in label_counts.items() if count < min_samples_per_class}
        
        if rare_classes:
            print(f"\nFiltering out {len(rare_classes)} rare classes (< {min_samples_per_class} samples):", flush=True)
            for tag in sorted(rare_classes):
                print(f"  - {tag}: {label_counts[tag]} samples", flush=True)
            
            for text, label in zip(all_texts, all_labels):
                if label not in rare_classes:
                    self.texts.append(text)
                    self.labels.append(label)
                    self.label_stats[label] += 1
        else:
            self.texts = all_texts
            self.labels = all_labels
            self.label_stats = label_counts
        
        print(f"\nFiltered training data: {len(self.texts)} samples", flush=True)
        print(f"Kept {len(self.label_stats)} classes: {sorted(self.label_stats.keys())}", flush=True)
        
        # Load validation data if provided
        if self.val_data_path:
            self.val_texts = []
            self.val_labels = []
            
            with open(self.val_data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    # Only include validation samples from classes we're training on
                    if item['tag'] not in rare_classes:
                        self.val_texts.append(item['text'])
                        self.val_labels.append(item['tag'])
            
            print(f"\nLoaded validation data: {len(self.val_texts)} samples", flush=True)
        else:
            self.val_texts = None
            self.val_labels = None
        
        # Load test data if provided
        if self.test_data_path:
            self.test_texts = []
            self.test_labels = []
            
            with open(self.test_data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    # Only include test samples from classes we're training on
                    if item['tag'] not in rare_classes:
                        self.test_texts.append(item['text'])
                        self.test_labels.append(item['tag'])
            
            print(f"\nLoaded test data: {len(self.test_texts)} samples", flush=True)
        else:
            self.test_texts = None
            self.test_labels = None
        
        # Build vocabularies
        self._build_vocabularies()
        
    def _build_vocabularies(self):
        """Build character/byte and label vocabularies"""
        if self.encoding_type == 'byte':
            # Byte vocabulary is fixed: 0-255 + 256 for padding
            self.vocab_to_idx = {i: i for i in range(257)}
            print(f"\nByte vocabulary size: 257 (0-255 + padding)", flush=True)
        else:
            # Build character vocabulary
            all_chars = set()
            for text in self.texts:
                all_chars.update(text)
            
            for idx, char in enumerate(sorted(all_chars), start=2):
                self.vocab_to_idx[char] = idx
            
            print(f"\nVocabulary size: {len(self.vocab_to_idx)} characters", flush=True)
        
        # Build label vocabulary
        unique_labels = sorted(set(self.labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        print(f"Number of classes: {len(self.label_to_idx)}", flush=True)
        
    def create_model(self, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.5, num_convs=3):
        """Create neural network model"""
        if self.encoding_type == 'byte':
            vocab_size = 257  # 0-255 + padding
            padding_idx = 256
        else:
            vocab_size = len(self.vocab_to_idx)
            padding_idx = 0
        
        num_classes = len(self.label_to_idx)
        
        if self.model_type == 'lstm':
            self.model = CharLSTMClassifier(vocab_size, embedding_dim, hidden_dim, 
                                           num_classes, num_layers, dropout)
            # Update padding_idx
            self.model.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        elif self.model_type == 'cnn':
            self.model = CharCNNClassifier(vocab_size, embedding_dim, num_classes, 
                                          num_filters=256, num_convs=num_convs, dropout=dropout)
            self.model.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        elif self.model_type == 'deepcnn':
            self.model = CharDeepCNNClassifier(vocab_size, embedding_dim, num_classes, dropout)
            self.model.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        elif self.model_type == 'transformer':
            num_heads = 8
            embedding_dim = (embedding_dim // num_heads) * num_heads
            self.model = CharTransformerClassifier(vocab_size, embedding_dim, num_classes, 
                                                   num_heads=num_heads, num_layers=num_layers, dropout=dropout)
            self.model.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model.to(self.device)
        print(f"\nCreated {self.model_type.upper()} model ({self.encoding_type}-level)", flush=True)
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}", flush=True)
        
    def train(self, epochs=20, batch_size=32, learning_rate=0.001, val_split=0.1):
        """Train the neural network"""
        # Convert labels to indices
        label_indices = [self.label_to_idx[label] for label in self.labels]
        
        # Split data or use provided validation set
        if self.val_texts is not None and self.val_labels is not None:
            print("\nUsing provided validation set", flush=True)
            X_train = self.texts
            y_train = label_indices
            X_val = self.val_texts
            y_val = [self.label_to_idx[label] for label in self.val_labels]
        else:
            print(f"\nSplitting data with val_split={val_split}", flush=True)
            X_train, X_val, y_train, y_val = train_test_split(
                self.texts, label_indices, test_size=val_split, random_state=42, stratify=label_indices
            )
        
        # Create datasets with encoding type
        train_dataset = LombardDataset(X_train, y_train, self.vocab_to_idx, encoding_type=self.encoding_type)
        val_dataset = LombardDataset(X_val, y_val, self.vocab_to_idx, encoding_type=self.encoding_type)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        print(f"\nTraining for {epochs} epochs...", flush=True)
        print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}", flush=True)
        
        # Track best model
        best_val_acc = 0.0
        best_model_state = None
        best_epoch = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            
            # Save best model based on validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                best_epoch = epoch + 1
                marker = " *"
            else:
                marker = ""
            
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}% - "
                  f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%{marker}", flush=True)
        
        # Restore best model
        print(f"\nBest model from epoch {best_epoch} with validation accuracy: {best_val_acc:.2f}%", flush=True)
        self.model.load_state_dict(best_model_state)
        
        # Final evaluation on test set if provided
        if self.test_texts is not None and self.test_labels is not None:
            print("\n" + "="*60, flush=True)
            print("FINAL EVALUATION ON TEST SET", flush=True)
            print("="*60, flush=True)
            
            y_test = [self.label_to_idx[label] for label in self.test_labels]
            test_dataset = LombardDataset(self.test_texts, y_test, self.vocab_to_idx, encoding_type=self.encoding_type)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            
            self._evaluate(test_loader, y_test, data_type="test")
        else:
            # Evaluate on validation set if no test set provided
            print("\n" + "="*60, flush=True)
            print("FINAL EVALUATION ON VALIDATION SET", flush=True)
            print("="*60, flush=True)
            self._evaluate(val_loader, y_val, data_type="validation")
        
    def _evaluate(self, data_loader, y_true, data_type="test"):
        """Detailed evaluation with classification report"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Convert indices back to labels
        pred_labels = [self.idx_to_label[idx] for idx in all_preds]
        true_labels = [self.idx_to_label[idx] for idx in all_labels]
        
        # Calculate accuracy
        test_acc = 100 * sum([1 for p, t in zip(pred_labels, true_labels) if p == t]) / len(true_labels)
        print(f"\n{data_type.capitalize()} Accuracy: {test_acc:.2f}%", flush=True)
        
        print("\nClassification Report:", flush=True)
        print(classification_report(true_labels, pred_labels, zero_division=0), flush=True)
        
        print("\nConfusion Matrix:", flush=True)
        cm = confusion_matrix(true_labels, pred_labels)
        print(cm, flush=True)
        
        # Save confusion matrix data
        self._save_confusion_data(true_labels, pred_labels, cm, data_type=data_type)
        
    def _save_confusion_data(self, y_test, y_pred, cm, data_type="test"):
        """Save confusion matrix and detailed error analysis"""
        classes = sorted(set(y_test))
        
        # Create pandas DataFrame for better visualization
        cm_df = pd.DataFrame(cm, index=classes, columns=classes)
        
        # Save confusion matrix
        output_dir = './models/confusion_matrices'
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        cm_path = f"{output_dir}/confusion_matrix_{self.model_type}_{self.encoding_type}_{data_type}.csv"
        cm_df.to_csv(cm_path)
        print(f"\nConfusion matrix saved to {cm_path}", flush=True)
        
        # Calculate and save per-class metrics
        metrics = []
        for i, true_class in enumerate(classes):
            total = cm[i].sum()
            correct = cm[i, i]
            accuracy = correct / total if total > 0 else 0
            
            # Find most confused classes
            confused_with = []
            for j, pred_class in enumerate(classes):
                if i != j and cm[i, j] > 0:
                    confused_with.append((pred_class, cm[i, j], cm[i, j] / total * 100))
            
            confused_with.sort(key=lambda x: -x[1])
            
            metrics.append({
                'class': true_class,
                'total': total,
                'correct': correct,
                'accuracy': accuracy,
                'top_confusion_1': confused_with[0][0] if len(confused_with) > 0 else '',
                'top_confusion_1_count': confused_with[0][1] if len(confused_with) > 0 else 0,
                'top_confusion_1_pct': confused_with[0][2] if len(confused_with) > 0 else 0,
                'top_confusion_2': confused_with[1][0] if len(confused_with) > 1 else '',
                'top_confusion_2_count': confused_with[1][1] if len(confused_with) > 1 else 0,
                'top_confusion_2_pct': confused_with[1][2] if len(confused_with) > 1 else 0,
            })
        
        metrics_df = pd.DataFrame(metrics)
        metrics_path = f"{output_dir}/class_metrics_{self.model_type}_{self.encoding_type}_{data_type}.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Per-class metrics saved to {metrics_path}", flush=True)
    
    def predict(self, text: str) -> str:
        """Predict label for new text"""
        self.model.eval()
        
        if self.encoding_type == 'byte':
            # Convert text to byte sequence
            byte_seq = text.encode('utf-8')[:200]
            indices = [b for b in byte_seq]
            if len(indices) < 200:
                indices += [256] * (200 - len(indices))
        else:
            # Character-level encoding
            indices = [self.vocab_to_idx.get(c, self.vocab_to_idx['<UNK>']) for c in text[:200]]
            if len(indices) < 200:
                indices += [self.vocab_to_idx['<PAD>']] * (200 - len(indices))
        
        input_tensor = torch.tensor([indices], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            _, predicted = torch.max(output.data, 1)
        
        return self.idx_to_label[predicted.item()]
    
    def save_model(self, model_path: str):
        """Save trained model"""
        # Get model hyperparameters for reconstruction
        if self.model_type == 'transformer':
            # Extract transformer-specific parameters
            model_config = {
                'embedding_dim': self.model.embedding.embedding_dim,
                'num_heads': self.model.transformer_encoder.layers[0].self_attn.num_heads,
                'num_layers': len(self.model.transformer_encoder.layers),
                'dropout': self.model.dropout.p
            }
        elif self.model_type == 'lstm':
            model_config = {
                'embedding_dim': self.model.embedding.embedding_dim,
                'hidden_dim': self.model.lstm.hidden_size,
                'num_layers': self.model.lstm.num_layers,
                'dropout': self.model.dropout.p
            }
        elif self.model_type in ['cnn', 'deepcnn']:
            model_config = {
                'embedding_dim': self.model.embedding.embedding_dim,
                'dropout': self.model.dropout.p
            }
        else:
            model_config = {}
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab_to_idx': self.vocab_to_idx,
            'label_to_idx': self.label_to_idx,
            'idx_to_label': self.idx_to_label,
            'label_stats': self.label_stats,
            'model_type': self.model_type,
            'encoding_type': self.encoding_type,
            'model_config': model_config
        }, model_path)
        print(f"\nModel saved to {model_path}", flush=True)
    
    def load_model(self, model_path: str):
        """Load trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.vocab_to_idx = checkpoint['vocab_to_idx']
        self.label_to_idx = checkpoint['label_to_idx']
        self.idx_to_label = checkpoint['idx_to_label']
        self.label_stats = checkpoint['label_stats']
        self.model_type = checkpoint['model_type']
        self.encoding_type = checkpoint.get('encoding_type', 'char')
        
        # Recreate model
        self.create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from {model_path}", flush=True)
