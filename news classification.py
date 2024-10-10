import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
true_news_df = pd.read_csv(r'C:\Users\lathe\Downloads\true_news_dataset_with_dates.csv')
fake_news_df = pd.read_csv(r'C:\Users\lathe\Downloads\fake_news_dataset_with_dates.csv')
true_news_df['label'] = 1 
fake_news_df['label'] = 0
combined_df = pd.concat([true_news_df, fake_news_df]).reset_index(drop=True)
X_train, X_test, y_train, y_test = train_test_split(combined_df['news'], combined_df['label'], test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:\n", classification_report(y_test, y_pred))
mixed_news_df = pd.read_csv(r'C:\Users\lathe\Downloads\mixed_news_dataset_test.csv')
mixed_news_tfidf = vectorizer.transform(mixed_news_df['news'])
predictions = model.predict(mixed_news_tfidf)
true_count = sum(predictions)
fake_count = len(predictions) - true_count
print(f"Percentage of True News: {true_count / len(predictions) * 100}%")
print(f"Percentage of Fake News: {fake_count / len(predictions) * 100}%")
