import streamlit as st
import torch
import torch.nn as nn
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# ── Model definition (must match training code) ──────────────────────────────
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


# ── Preprocessing (must match training code) ─────────────────────────────────
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    text = re.sub(r"<.*?>", "", text)

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stop_words]

    ps = PorterStemmer()
    tokens = [ps.stem(t) for t in tokens]

    return " ".join(tokens)


# ── Load artifacts (cached so they load only once) ───────────────────────────
@st.cache_resource
def load_model_and_vectorizer():
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    input_size = len(vectorizer.get_feature_names_out())  # 5000
    model = RNN(input_size)
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model, vectorizer


# ── Streamlit UI ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="IMDB Sentiment Analyser", page_icon="🎬")

st.title("🎬 IMDB Sentiment Analyser")
st.write("Enter a movie review and the RNN model will predict whether it is **positive** or **negative**.")

model, vectorizer = load_model_and_vectorizer()

review = st.text_area("Movie Review", placeholder="Type your review here...", height=180)

if st.button("Predict"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        processed = preprocess(review)
        X = vectorizer.transform([processed]).toarray()
        X_tensor = torch.from_numpy(X).float().unsqueeze(1)  # (1, 1, 5000)

        with torch.no_grad():
            output = model(X_tensor)
            prob = torch.sigmoid(output.squeeze()).item()

        label = "Positive 😊" if prob > 0.5 else "Negative 😞"
        confidence = prob if prob > 0.5 else 1 - prob

        st.markdown(f"### Prediction: **{label}**")
        st.progress(confidence)
        st.caption(f"Confidence: {confidence * 100:.1f}%")
