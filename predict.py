import joblib
import os
import pandas as pd
from preprocess import clean_text

# Load model and vectorizer
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def predict_emotion(text):
    cleaned_text = clean_text(text)
    transformed_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(transformed_text)[0]
    return prediction

if __name__ == "__main__":
    while True:
        user_input = input("Enter a sentence (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        predicted_emotion = predict_emotion(user_input)
        print("Predicted Emotion:", predicted_emotion)
        
        correct = input("Is this correct? (yes/no): ")
        if correct.lower() == "no":
            correct_emotion = input("Enter the correct emotion: ")
            new_data = pd.DataFrame({"text": [user_input], "emotion": [correct_emotion]})
            new_data.to_csv("corrected_data.csv", mode='a', index=False, header=not os.path.exists("corrected_data.csv"))
            print("Correction saved! Retrain the model for better accuracy.")
