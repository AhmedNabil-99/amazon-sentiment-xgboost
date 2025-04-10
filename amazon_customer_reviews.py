import pandas as pd
import numpy as np
import gensim.downloader as api
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import xgboost as xgb
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer


from func import nltk_clean, rate_to_binary_sentiment, get_average_vector, predict_review_sentiment



# Load data
df = pd.read_csv('/home/ahmed/Personal/Amazon_dataset/amazon.csv')
df['full_review'] = df['review_title'].astype(str) + " " + df['review_content'].astype(str)
df['rating'] = df['rating'].astype(str).str.extract(r'([\d.]+)').astype(float)

# Binary sentiment
df['binary_sentiment'] = df['rating'].apply(rate_to_binary_sentiment)

# Clean reviews
df['cleaned_review'] = df['full_review'].apply(nltk_clean)

# Load GloVe embeddings
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
df['vector'] = df['cleaned_review'].apply(lambda text: sbert_model.encode(text))

# Prepare data
X = np.vstack(df['vector'].values)
y = df['binary_sentiment']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

# Train model
model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Plot feature importance
xgb.plot_importance(model, max_num_features=10)
plt.savefig("feature_importance.png")



# sample_review = [
#     "Complete garbage. Doesn’t work at all. I want my money back.",
#     "Broke within minutes. Honestly, how does this even pass quality checks?",
#     "Biggest scam I’ve ever fallen for. Don’t waste your time.",
#     "This thing is a joke. Total piece of junk.",
#     "Absolute trash. Doesn’t do anything it promises.",
#     "Might as well throw your money in the trash — at least that would be faster.",
#     "Disgusting smell, useless function, and feels like a toy from a dollar store.",
#     "Didn’t even turn on. Dead on arrival. What a rip-off.",
#     "I’ve had better luck with products from a vending machine.",
#     "Useless. Doesn't even deserve one star.",
#     "So bad I thought it was a prank product.",
#     "Dangerous! It shocked me. Shouldn’t even be on the market.",
#     "The worst product I’ve ever purchased online. Period.",
#     "Malfunctioned instantly and nearly broke something else I own.",
#     "This product made my day worse. That’s an achievement.",
#     "Cheap, flimsy, and completely non-functional.",
#     "Terrible design. Whoever made this clearly doesn’t care.",
#     "It’s like someone made this as a joke. But I’m not laughing.",
#     "Feels like it was glued together by a toddler.",
#     "Do yourself a favor and *don’t* buy this.",
#     "If I could give negative stars, I would.",
#     "So disappointed I’m questioning all my life decisions now.",
#     "Fails at its only job. Spectacularly useless.",
#     "Packaging was the best part. The product was DOA.",
#     "Overpriced nonsense. You’ve been warned.",
#     "Looked decent in the photos. In reality? Garbage.",
#     "It’s like they tried to make the worst product possible… and succeeded.",
#     "Support was no help — they actually made it worse.",
#     "This thing fell apart in my hands. What a joke.",
#     "Advertised as premium, but it's bargain bin quality.",
#     "Not just bad — infuriating.",
#     "This product actually made me angry.",
#     "Doesn’t work, can’t return it, and customer service ghosted me.",
#     "I feel scammed. Straight-up scammed.",
#     "Imagine paying for something that breaks during setup.",
#     "It barely functions, if at all. Completely unacceptable.",
#     "The manufacturer should be ashamed.",
#     "It stopped working and then started sparking. Unsafe!",
#     "I’d rather use duct tape than this ‘solution’.",
#     "They must be using fake reviews to stay afloat.",
#     "Feels like it was made out of cereal box cardboard.",
#     "The moment I touched it, I knew I made a mistake.",
#     "Every second I own this, I regret it more.",
#     "Returned it immediately. It’s that bad.",
#     "Never been this disappointed in a product.",
#     "Worst purchase I’ve made in years.",
#     "Totally useless. Doesn’t do anything right.",
#     "How is this even allowed to be sold?",
#     "Customer service hung up on me. Twice.",
#     "My cat has better build quality than this thing.",
#     "Even the packaging was falling apart.",
#     "It’s trash. Literal trash."
# ]

# x = 0
# y = 0

# for i in sample_review:
#     predicted_sentiment = predict_review_sentiment(i, sbert_model, model, label_encoder)
#     if predicted_sentiment == "positive":
#         x+=1
#     else:
#         y+=1
# print("positive = ",x)
# print("not_positive = ",y)
