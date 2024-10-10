# SAMPLE-NEWS-CLASSIFIER-TRUE-FAKE-
SAMPLE NEWS CLASSIFIER TRUE / FAKE 
# News Classification with Logistic Regression

Overview
This project is focused on building a machine learning model to classify news articles as either True or Fake using a Logistic Regression model and TF-IDF (Term Frequency-Inverse Document Frequency) for text vectorization. The dataset consists of true and fake news articles, and the model is trained to make predictions based on the textual content.

Datasets
True News Dataset: Contains real news articles.
Fake News Dataset: Contains fabricated news articles.
Mixed News Dataset: A separate dataset used to test the model and compute the percentage of true and fake news.
Key Steps:
Data Loading:

True and fake news datasets are loaded from CSV files.
A label column is added to distinguish true news (1) and fake news (0).
The datasets are concatenated into one combined DataFrame.
Data Splitting:

The news articles (news) are split into training and testing sets using an 80/20 ratio.
Text Vectorization:

TF-IDF Vectorizer is used to convert the text data into numerical features.
The max_features is set to 5000, and English stop words are removed.
Model Training:

Logistic Regression is used as the machine learning model.
The model is trained on the vectorized training data and corresponding labels.
Evaluation:

The model's accuracy is evaluated on the test set.
A classification report is generated, showing precision, recall, and F1-score for both true and fake news labels.
Mixed Dataset Predictions:

The trained model is used to predict labels for a separate mixed news dataset.
The percentage of true and fake news articles in the mixed dataset is computed and displayed.
Dependencies
pandas: For data manipulation.
scikit-learn: For model training, text vectorization, and performance evaluation.
numpy: For numerical operations (implicitly used by scikit-learn).
Install the required libraries using:

bash
Copy code
pip install pandas scikit-learn
How to Run
Download the necessary datasets:

true_news_dataset_with_dates.csv
fake_news_dataset_with_dates.csv
mixed_news_dataset_test.csv
Place them in your desired directory and update the file paths in the script if needed.

Run the Python script to train the model and generate predictions for the mixed dataset.
OUTPUT: 
Classification report with precision, recall, and F1-score.
Percentage of true and fake news in the mixed dataset.

After training the model, the accuracy and performance metrics (precision, recall, F1-score) are printed. Additionally, the proportion of true and fake news in the mixed dataset is calculated and displayed.
