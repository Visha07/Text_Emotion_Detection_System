import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from preprocess import clean_text

# Load dataset
df = pd.read_csv("data.csv")

# Handle missing values
df = df.dropna(subset=["emotion"])  # Drop rows where emotion is missing
df["text"] = df["text"].astype(str).fillna("")  # Convert text to string and fill NaN

# Apply text preprocessing
df["cleaned_text"] = df["text"].apply(clean_text)

# Extract features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["cleaned_text"])
y = df["emotion"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Use an optimized SVM model
model = SVC(kernel='linear', C=1.0, probability=True)

# Train model
model.fit(X_train, y_train)

# Save model and vectorizer
joblib.dump(model, "emotion_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Evaluate model
y_pred = model.predict(X_test)
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f"Train Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
