# LexiMood DataModel

LexiMood DataModel is a deep-learning text classification project for emotion detection from short natural-language inputs. The repository demonstrates a complete baseline workflow from dataset loading and preprocessing to model training and evaluation using a BiLSTM architecture.

## Why this project is valuable
This project turns an exploratory notebook into a clearer, reusable training pipeline. It is useful as a portfolio artifact for NLP and ML engineering roles because it emphasizes reproducibility, code organization, and practical model development decisions.

## Key features
- Emotion classification pipeline for text data.
- Label encoding + one-hot targets for multi-class classification.
- Tokenization and sequence padding workflow compatible with Keras/TensorFlow.
- BiLSTM-based neural model for contextual sequence modeling.
- Scripted training entry point (`python -m leximood.train`) in addition to the original notebook.

## Tech stack
- Python 3.10+
- Pandas, NumPy
- scikit-learn
- TensorFlow / Keras
- Matplotlib, Seaborn (evaluation plots in notebook)

## Project architecture
The project currently offers two ways to work:
1. **Notebook workflow** in `LexiMood.ipynb` for interactive experimentation and visualization.
2. **Scripted workflow** in `src/leximood/train.py` for cleaner, repeatable training.

Core flow:
1. Load CSV dataset with `text` and `emotion` columns.
2. Encode labels and split train/test data.
3. Tokenize and pad sequences.
4. Train BiLSTM model with softmax output.
5. Evaluate with accuracy curves, classification report, and confusion matrix.

## Installation
```bash
git clone <REPO_LINK>
cd leximood-datamodel
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage
### 1) Notebook mode
- Open `LexiMood.ipynb` in Jupyter or Colab.
- Make sure your dataset file is available as `emotion_dataset.csv` or update the path in the notebook.

### 2) Script mode
```bash
python -m leximood.train --dataset emotion_dataset.csv
```

## Example input/output
- **Input:** text samples and emotion labels in CSV format.
- **Output:** trained model in memory (script), plus training history metrics; notebook additionally visualizes plots and confusion matrix.

> Add screenshots of training curves and confusion matrix in `docs/images/` if you want stronger visual presentation on GitHub.

## Repository structure
```text
.
├── LexiMood.ipynb
├── src/
│   └── leximood/
│       ├── __init__.py
│       └── train.py
├── tests/
│   └── test_train_config.py
├── .env.example
├── CONTRIBUTING.md
├── LICENSE
├── README.md
└── requirements.txt
```

## Engineering decisions and tradeoffs
- **Kept notebook + script:** notebook is useful for experimentation; script improves repeatability.
- **Validation checks on dataset schema:** fail fast when required columns are missing.
- **Simple baseline architecture first:** BiLSTM is a strong starting point before trying heavier transformer models.

## Challenges addressed
- Converting an exploratory notebook workflow into reusable code.
- Maintaining parity with notebook behavior while improving readability.
- Defining a clear project narrative suitable for portfolio and resume usage.

## Future improvements
- Add dataset versioning and deterministic training configuration tracking.
- Add model artifact saving/loading and inference script.
- Add automated tests for preprocessing and model smoke checks.
- Add CI workflow for linting/tests.

## Contributing
See `CONTRIBUTING.md` for setup and contribution guidelines.

## License
This project is licensed under the MIT License. See `LICENSE` for details.
