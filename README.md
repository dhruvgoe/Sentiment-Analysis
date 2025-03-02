# IMDb Sentiment Analysis using LSTM and GloVe

## Overview
This project aims to perform sentiment analysis on IMDb movie reviews using an LSTM (Long Short-Term Memory) model with pre-trained GloVe word embeddings. The goal is to classify movie reviews as either **positive** or **negative** based on textual content.

## Dataset
- **Training & Testing Dataset:** IMDb movie reviews dataset (`a1_IMDB_Dataset.csv`).
- **Unseen Dataset:** A separate dataset for prediction (`a3_IMDb_Unseen_Reviews.csv`).
- The dataset contains two columns: `review` (text) and `sentiment` (positive/negative).
- Datset from Kaggle: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

## Project Workflow
1. **Data Preprocessing:**
   - Remove HTML tags, special characters, and stopwords.
   - Convert text to lowercase.
   - Tokenize and pad sequences for input.

2. **Load Pre-trained GloVe Embeddings:**
   - Use `a2_glove.6B.100d.txt` (100-dimensional embeddings) to enhance word representations.

3. **Model Architecture:**
   - **Embedding Layer:** Uses GloVe word embeddings.
   - **LSTM Layer:** 128 units with return sequences.
   - **GlobalMaxPooling1D:** Reduces dimensionality.
   - **Dense Layers:** Fully connected layers with ReLU activation and dropout.
   - **Output Layer:** Sigmoid activation for binary classification.

4. **Training & Evaluation:**
   - Train the model on an 80-20 train-test split.
   - Evaluate using accuracy as the metric.

5. **Prediction on Unseen Reviews:**
   - Preprocess and predict sentiment for `a3_IMDb_Unseen_Reviews.csv`.
   - Save predictions in `c2_IMDb_Unseen_Predictions.csv`.

## Installation & Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/dhruvgoe/Sentiment-Analysis.git
   cd Sentiment-Analysis
   ```
2. Install dependencies:
   ```sh
   pip install numpy pandas tensorflow nltk seaborn matplotlib
   ```
3. Download and place the dataset & GloVe embeddings in the correct directory:
   ```
   ├── IMDB/
       ├── a1_IMDB_Dataset.csv
       ├── a2_glove.6B.100d.txt
       ├── a3_IMDb_Unseen_Reviews.csv
   ```
4. Run the script:
   ```sh
   python Analysis.py
   ```

## Model Performance
- **Test Accuracy:** ~85-88% (varies based on training runs and hyperparameters).

## Results
- The trained model is saved as `lstm_sentiment_model.h5`.
- Predictions for unseen reviews are saved in `c2_IMDb_Unseen_Predictions.csv`.

## Future Improvements
- Experiment with different embedding sizes (GloVe 200d, 300d).
- Try Bi-LSTM or Transformer-based models (BERT, RoBERTa).
- Fine-tune learning rates and dropout values.

## Contributors
- **Dhruv Goel** – [GitHub](https://github.com/dhruvgoe)

## License
This project is licensed under the MIT License.

