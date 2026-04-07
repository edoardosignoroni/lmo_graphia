# Automatic Classifier of Lombard Orthography Variants

This repository contains tools for automatically classifying Lombard language text into different orthographic variants. The classifier supports both traditional machine learning (sklearn) and deep learning (PyTorch) approaches.

## Supported Orthographic Variants

The classifier identifies the following Lombard orthographic variants:

- **MILCLASS** - Classical Milanese orthography
- **LOCC** - Unified Western Lombard
- **LORUNIF** - Unified Eastern Lombard
- **SL** - Scriver Lombard
- **NOL** - New Lombard Orthography
- **CRES** - Cremonese orthography
- **BREMOD** - Modern Brescian orthography
- **BERGDUC** - Bergamasque orthography

## Repository Structure

```
ortho_classifier/
├── README.md                    # This file
├── data/
│   ├── gold/                    # Labeled training data
│   │   ├── train.jsonl         # 8,950 training examples
│   │   ├── valid.jsonl         # 1,118 validation examples
│   │   └── test.jsonl          # 1,118 test examples
│   └── no-tag/                  # Automatically tagged data
│       └── no-tag.jsonl        # 94,520 examples
├── models/                      # Pre-trained models
└── scripts/
    ├── classifiers.py          # Core classifier class definitions
    ├── train_classifier.py     # Training script
    ├── apply_classifier.py     # Classification script
    └── usage.md                # Reference
```

## Dataset Statistics

### Gold Standard Dataset

| Split      | Examples | 
|------------|----------|
| Training   | 8,950    |
| Validation | 1,118    | 
| Test       | 1,118    |
||| 
| **Total**  | 11,186   |

### Tag Distribution (Training Set)

| Variant   | Count | Percentage |
|-----------|-------|------------|
| MILCLASS  | 3,606 | 40.3%      |
| LOCC      | 2,907 | 32.5%      |
| LORUNIF   | 1,901 | 21.2%      |
| SL        | 174   | 1.9%       |
| NOL       | 109   | 1.2%       |
| CRES      | 98    | 1.1%       |
| BREMOD    | 94    | 1.1%       |
| BERGDUC   | 59    | 0.7%       |
| LSI       | 2     | 0.0%       |
||
| **Total** | 8,950 | 100%       |

## Performance

| **Model**      | **MILCLASS** | **LOCC**  | **LORUNIF** | **SL**    | **NOL**   | **CRES**  | **BREMOD** | **BERGDUC** | **Overall** | **Avg Class** |
|---------------------|-------------------|----------------|------------------|----------------|----------------|----------------|-----------------|------------------|------------------|--------------------|
| Log. byte           | 97.31             | 92.89          | 89.08            | **100.0** | **75.00** | 94.74          | 53.85           | **83.33**   | 93.38            | **85.78**     |
| Log. byte+char+word | 97.31             | 94.74          | 94.76            | **100.0** | **75.00** | **100.0** | 46.15           | 66.67            | 95.08            | 84.33              |
| Log. char           | 95.07             | 93.42          | 84.28            | **100.0** | **75.00** | 94.74          | 53.85           | **83.33**   | 91.67            | 84.96              |
| Log. word           | 94.62             | 94.21          | 88.65            | 93.75          | 37.50          | 94.74          | **69.23**  | 66.67            | 92.39            | 79.92              |
||
| SVM byte            | 97.76             | 94.21          | **99.13**   | 87.50          | **75.00** | 94.74          | 30.77           | 50.00            | 95.43            | 78.64              |
| SVM byte+char+word  | 97.76             | **96.58** | **99.13**   | 93.75          | 50.00          | 94.74          | 23.08           | 50.00            | **96.06**   | 75.63              |
| SVM char            | 98.21             | 94.74          | 98.25            | 93.75          | 50.00          | 94.74          | 30.77           | 50.00            | 95.52            | 76.31              |
| SVM word            | 95.52             | 95.53          | 95.20            | 93.75          | 12.50          | 89.47          | 30.77           | 50.00            | 93.73            | 70.34              |
||
| NB byte             | **98.88**    | 91.05          | 93.45            | 0.00           | 0.00           | 0.00           | 0.00            | 0.00             | 89.62            | 35.42              |
| NB byte+char+word   | 98.21             | 93.42          | 96.07            | 0.00           | 0.00           | 0.00           | 0.00            | 0.00             | 90.69            | 35.96              |
| NB char             | 98.65             | 91.05          | 94.32            | 0.00           | 0.00           | 0.00           | 0.00            | 0.00             | 89.70            | 35.50              |
| NB word             | 97.09             | 92.37          | 94.76            | 12.50          | 0.00           | 15.79          | 0.00            | 0.00             | 90.06            | 39.06              |
||
| RF byte             | 98.43             | 92.37          | 96.94            | 68.75          | 0.00           | 68.42          | 0.00            | 0.00             | 92.75            | 53.11              |
| RF byte+char+word   | 98.65             | 92.11          | 95.63            | 75.00          | 0.00           | 68.42          | 7.69            | 0.00             | 92.66            | 54.69              |
| RF char             | 98.65             | 93.16          | 95.20            | 50.00          | 0.00           | 68.42          | 7.69            | 0.00             | 92.57            | 51.64              |
| RF word             | 97.31             | 87.89          | 93.89            | 68.75          | 0.00           | 63.16          | 15.38           | 0.00             | 90.24            | 53.30              |
||
| CNN byte            | 97.98             | 95.53          | 97.38            | 75.00          | 12.50          | 78.95          | 7.69            | 16.67            | 94.27            | 60.21              |
| CNN char            | 98.21             | 94.47          | 98.69            | 87.50          | 12.50          | 63.16          | 7.69            | 16.67            | 94.18            | 59.86              |
||
| Deep CNN byte       | 83.86             | 94.74          | 92.14            | 75.00          | 12.50          | 73.68          | 15.38           | 0.00             | 87.20            | 55.91              |
| Deep CNN char       | 97.09             | 92.37          | 96.94            | 0.00           | 12.50          | 57.89          | 7.69            | 0.00             | 91.23            | 45.56              |
||
| LSTM byte           | 97.53             | 91.05          | 95.63            | 0.00           | 0.00           | 63.16          | 0.00            | 0.00             | 90.60            | 43.42              |
| LSTM char           | 98.21             | 91.05          | 95.63            | 0.00           | 0.00           | 68.42          | 0.00            | 16.67            | 91.05            | 46.25              |
||
| Transformer byte    | 96.64             | 85.00          | 94.76            | 68.75          | 0.00           | 5.26           | 0.00            | 0.00             | 88.00            | 43.80              |
| Transformer char    | 96.41             | 88.95          | 93.89            | 0.00           | 0.00           | 31.58          | 0.00            | 0.00             | 88.54            | 38.85              |
||
| Best accuracy       | **98.88**    | **96.58** | **99.13**   | **100.0** | **75.00** | **100.0** | **69.23**  | **83.33**   | **96.06**   | **85.78**     |
| Worst accuracy      | 83.86             | 85.00          | 84.28            | 0.00           | 0.00           | 0.00           | 0.00            | 0.00             | 87.20            | 35.42              |
| Accuracy range      | 15.02             | 11.58          | 14.85            | 100.0          | 75.00          | 100.0          | 69.23           | 83.33            | 8.86             | 50.36              |


## Future Work & Improvements

+ Multilabel Classification
+ Tagset Expansion (Non-Lombard; Non-Language)


## Citation

If you use these classifiers, please cite:

Edoardo **Signoroni** and Pavel **Rychlý**, *LombardoGraphia: Automatic Classification of Lombard Orthography Variants*, Accepted at LREC 2026

```
@misc{signoroni2026lombardographiaautomaticclassificationlombard,
      title={LombardoGraphia: Automatic Classification of Lombard Orthography Variants}, 
      author={Edoardo Signoroni and Pavel Rychlý},
      year={2026},
      eprint={2603.28418},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.28418}, 
}
```

## Contacts

For questions or issues, please contact e.signoroni@mail.muni.cz.

## License

The dataset is licensed under CC-BY-SA 4.0 
The software is licesed under GPL-3.0 

## Acknowledgments

This work has been supported by the Ministry of Education, Youth and Sports of the Czech Republic, Project No. LM2023062 LINDAT/CLARIAH-CZ.
We wish to thank the volunteers and contributors of Lombard Wikipedia for the creation of the data and the useful discussions and information.