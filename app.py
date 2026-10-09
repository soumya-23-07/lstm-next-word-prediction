
import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="💬",
    layout="centered"
)

# -------------------------------
# Load saved files
# -------------------------------
@st.cache_resource
def load_model_and_files():
    model = tf.keras.models.load_model(
        "lstm_model.h5",
        compile=False
    )

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, int(max_len)


try:
    model, tokenizer, max_len = load_model_and_files()
except Exception as e:
    st.error(f"Could not load model or files: {e}")
    st.stop()


# -------------------------------
# Prediction function
# -------------------------------
def predict_next_word(text):
    text = text.lower().strip()

    if not text:
        return None

    sequence = tokenizer.texts_to_sequences([text])[0]

    if len(sequence) == 0:
        return None

    # Keep only the most recent words if needed
    sequence = sequence[-max_len:]

    # Make input length equal to max_len
    padded_sequence = pad_sequences(
        [sequence],
        maxlen=max_len,
        padding="pre"
    )

    # Predict the next word
    prediction = model.predict(
        padded_sequence,
        verbose=0
    )

    # Find the highest-scoring output
    pred_index = int(np.argmax(prediction, axis=-1).reshape(-1)[0])

    # Convert predicted ID back into a word
    next_word = tokenizer.index_word.get(pred_index)

    return next_word


# -------------------------------
# Streamlit UI
# -------------------------------
st.title("💬 Next Word Prediction")
st.caption("An NLP application powered by an LSTM deep learning model.")

st.markdown(
    "Enter a sentence below and let the model predict "
    "which word may come next."
)

st.divider()

text = st.text_area(
    "Enter your text",
    placeholder="Example: I love machine",
    height=120
)

if st.button("✨ Predict Next Word", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        try:
            with st.spinner("Predicting the next word..."):
                next_word = predict_next_word(text)

            if next_word is None:
                st.warning(
                    "The tokenizer could not recognize your input. "
                    "Try words from your training dataset."
                )
            else:
                st.success("Prediction completed!")

                st.markdown("### Predicted next word")
                st.markdown(
                    f"<div style='padding:20px; border-radius:12px; "
                    f"background-color:#E8F5E9; color:#166534; "
                    f"font-size:30px; font-weight:bold; "
                    f"text-align:center;'>{next_word}</div>",
                    unsafe_allow_html=True
                )

                st.caption(
                    f"Input sequence length: {max_len}"
                )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.divider()

st.caption(
    "Built with Python, Streamlit, TensorFlow, and LSTM."
)