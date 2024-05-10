from __future__ import annotations

import logging
from typing import List
from models import rnn, n_gram, transformer

import streamlit as st

from streamlit_searchbox import st_searchbox

logging.getLogger("streamlit_searchbox").setLevel(logging.DEBUG)
st.set_page_config(layout="centered", page_title="Word Autocompletion")


def init_models():
    global n_gram_model
    n_gram_model = n_gram.NGram()
    global rnn_model
    rnn_model = rnn.RNN()
    global transformer_model
    transformer_model = transformer.Transformer()


def search_ngram(search_term: str) -> List[str]:
    return n_gram_model.predict_next_word(search_term)


def search_rnn(search_term: str) -> List[str]:
    return rnn_model.predict_next_word(search_term)


def search_transformer(search_term: str) -> List[str]:
    return transformer_model.predict_next_word(search_term)


#################################
#    application starts here    #
#################################

# Search box configurations to the specific models
boxes = [
    dict(
        search_function=search_ngram,
        key=f"n_gram",
        edit_after_submit="option",
        label=f"N-Gram Suggestions",
    ),
    dict(
        search_function=search_rnn,
        key=f"rnn",
        edit_after_submit="option",
        label=f"RNN Suggestions",
    ),
    dict(
        search_function=search_transformer,
        key=f"transformer",
        edit_after_submit="option",
        label=f"Transformer Suggestions",
    )
]

information, n_gram_page, rnn_page, transformer_page = st.tabs(
    ["Information", "N-Gram", "RNN-GRU", "Transformer"]
)


def main():
    init_models()

    with information:
        st.title("Word Autocompletion")

        st.header("Project Overview")
        st.write("""
            This project demonstrates the application of different natural language processing (NLP) models to understand and generate text. 
            By leveraging models like N-Gram, RNN-GRU, and Transformer, we explore various approaches to text prediction and generation.
            """)

        st.header("Group Members")

        # Member 1:
        col1, col2, col3 = st.columns([2, 2, 2])
        with col1:
            st.image("images/dog.jpg", caption="Eric Banzuzi")

        # Member 2:
        with col2:
            st.image("images/dog.jpg", caption="Rosamelia Carioni")

        # Member 3:
        with col3:
            st.image("images/dog.jpg", caption="Katharina Deckenbach")

        st.header("Project Goals")
        st.write("""
            The main goals of this project are:
            - To compare the effectiveness of N-Gram, RNN-GRU, and Transformer models on text prediction tasks.
            - To understand the strengths and limitations of each model in dealing with different types of textual data.
            - To develop a user-friendly interface that allows users to interact with each model and see their predictions in real-time.
            """)

        st.header("About the Models")
        st.subheader("N-Gram Model")
        st.write("""
            The N-Gram model predicts the next word in a sequence by looking at the previous N-3 words. It is a simple yet powerful model for text prediction based on statistical probability.
            """)

        st.subheader("RNN-GRU Model :smile:")
        st.write("""
            The Recurrent Neural Network (RNN) with Gated Recurrent Units (GRU) is a type of neural network that excels in learning from sequences of data. In text prediction, it can capture long-range dependencies and context from text.
            """)

        st.subheader("Transformer Model")
        st.write("""
            The Transformer model, based on self-attention mechanisms, represents a breakthrough in sequence modeling. It allows for parallel processing of sequences and has been the basis for models like BERT and GPT.
            """)

    with n_gram_page:
        selected_value = st_searchbox(**boxes[0])

    with rnn_page:
        selected_value = st_searchbox(**boxes[1])

    with transformer_page:
        selected_value = st_searchbox(**boxes[2])


if __name__ == '__main__':
    main()
