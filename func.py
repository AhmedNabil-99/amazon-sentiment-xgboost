import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer



# Initialize
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def rate_to_binary_sentiment(rating):
    return "positive" if rating >= 4.0 else "not_positive"

def nltk_clean(text):
    # Remove URLs and non-alpha characters
    text = re.sub(r"http\S+|[^a-zA-Z\s]", " ", text)
    tokens = word_tokenize(text.lower())
    clean_tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(clean_tokens)

# def get_average_vector(text, embedding):
#     words = text.split()
#     vectors = [embedding[word] for word in words if word in embedding]
#     return np.mean(vectors, axis=0) if vectors else np.zeros(100)

def predict_review_sentiment(text, embedding_model, model, label_encoder):
    clean = nltk_clean(text)
    vector = embedding_model.encode(clean)
    pred = model.predict([vector])[0]
    label = label_encoder.inverse_transform([pred])[0]
    return label

