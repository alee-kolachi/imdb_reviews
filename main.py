import pandas as pd
from preprocess import preprocess_text

data = pd.read_csv("IMDB Dataset.csv")

#print(data.head())

#print(data['sentiment'].value_counts())

data['clean_review'] = data['review'].apply(preprocess_text)
print(data[['review', 'clean_review']].head())

data['label'] = data['sentiment'].map({'positive':1, 'negative':0})

from sklearn.model_selection import train_test_split

X = data['clean_review']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Training data shape: ", X_train_vec.shape)
print("Example Feature names: ", vectorizer.get_feature_names_out()[:20])

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test_vec)

print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Classification Report: ", classification_report(y_test, y_pred))

