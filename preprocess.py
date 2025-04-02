import neattext.functions as ntf
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

# Load stopwords and exclude negations
stop_words = set(stopwords.words('english')) - {"not", "no", "never"}

def clean_text(text):
    if not isinstance(text, str):  
        return ""

    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = ntf.remove_puncts(text)  
    text = ntf.remove_special_characters(text)  

    # Remove stopwords except negations
    words = [word for word in text.split() if word not in stop_words]

    # Apply Lemmatization
    return " ".join(lemmatizer.lemmatize(word) for word in words)
