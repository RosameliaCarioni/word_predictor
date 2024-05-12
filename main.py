from __future__ import annotations

import logging
from typing import List
from models import rnn, n_gram, transformer

import streamlit as st
from streamlit_searchbox import st_searchbox

logging.getLogger("streamlit_searchbox").setLevel(logging.DEBUG)
st.set_page_config(layout="centered", page_icon=':iphone:', page_title="Word Autocompletion")


def init_models():
    global n_gram_model, rnn_model, transformer_model
    n_gram_model = n_gram.NGram()
    rnn_model = rnn.RNN()
    transformer_model = transformer.Transformer()


def search_ngram(search_term: str, number_of_suggestions: int) -> List[str]:
    return n_gram_model.predict_next_word(search_term, number_of_suggestions)


def search_rnn(search_term: str, number_of_suggestions: int) -> List[str]:
    return rnn_model.predict_next_word(search_term, number_of_suggestions)


def search_transformer(search_term: str, number_of_suggestions: int) -> List[str]:
    return transformer_model.predict_next_word(search_term, number_of_suggestions)


#################################
#    application starts here    #
#################################
def main():
    init_models()

    # Initialize session state for the number of suggestions if not already present
    if 'num_suggestions_ngram' not in st.session_state:
        st.session_state['num_suggestions_ngram'] = 5
    if 'num_suggestions_rnn' not in st.session_state:
        st.session_state['num_suggestions_rnn'] = 5
    if 'num_suggestions_transformer' not in st.session_state:
        st.session_state['num_suggestions_transformer'] = 5

    information, n_gram_page, rnn_page, transformer_page = st.tabs(
        ["Information", "N-Gram", "RNN-GRU", "Transformer"]
    )

    with information:
        st.title("Word Autocompletion")
        st.header("Project Overview")
        st.write("""
            This project demonstrates the application of different natural language processing (NLP) models to understand and generate text. 
            By leveraging models like N-Gram, RNN-GRU, and Transformer, we explore various approaches to text prediction and generation.
        """)

        st.header("Group Members")
        col1, col2, col3 = st.columns([2, 2, 2])
        with col1:
            st.image("images/dog.jpg", caption="Eric Banzuzi")
        with col2:
            st.image("images/dog.jpg", caption="Rosamelia Carioni")
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
        st.write("The N-Gram model predicts the next word in a sequence by looking at the previous N-3 words. It is a simple yet powerful model for text prediction based on statistical probability.")

        st.subheader("RNN-GRU Model :smile:")
        st.write("The Recurrent Neural Network (RNN) with Gated Recurrent Units (GRU) is a type of neural network that excels in learning from sequences of data. In text prediction, it can capture long-range dependencies and context from text.")

        st.subheader("Transformer Model")
        st.write("The Transformer model, based on self-attention mechanisms, represents a breakthrough in sequence modeling. It allows for parallel processing of sequences and has been the basis for models like BERT and GPT.")

    with n_gram_page:
        # Define the slider and update the session state
        num_suggestions_ngram = st.slider('Choose number of N-Gram suggestions', min_value=1, max_value=10,
                                          value=st.session_state['num_suggestions_ngram'], key='ngram_slider')
        st.session_state['num_suggestions_ngram'] = num_suggestions_ngram
        st.write('Selected amount:', num_suggestions_ngram)

        # Use the updated session state value for search
        selected_value = st_searchbox(search_function=lambda term: search_ngram(term, st.session_state['num_suggestions_ngram']),
                                      placeholder='',
                                      key='n_gram',
                                      edit_after_submit="option",
                                      label='N-Gram Suggestions')

    with rnn_page:
        # Define the slider and update the session state
        num_suggestions_rnn = st.slider('Choose number of RNN suggestions', min_value=1, max_value=10,
                                        value=st.session_state['num_suggestions_rnn'], key='rnn_slider')
        st.session_state['num_suggestions_rnn'] = num_suggestions_rnn
        st.write('Selected amount:', num_suggestions_rnn)

        # Use the updated session state value for search
        selected_value = st_searchbox(search_function=lambda term: search_rnn(term, st.session_state['num_suggestions_rnn']),
                                      placeholder='',
                                      key='rnn',
                                      edit_after_submit="option",
                                      label='RNN Suggestions')

    with transformer_page:
        # Define the slider and update the session state
        num_suggestions_transformer = st.slider('Choose number of Transformer suggestions', min_value=1, max_value=10,
                                                value=st.session_state['num_suggestions_transformer'], key='transformer_slider')
        st.session_state['num_suggestions_transformer'] = num_suggestions_transformer
        st.write('Selected amount:', num_suggestions_transformer)

        # Use the updated session state value for search
        selected_value = st_searchbox(search_function=lambda term: search_transformer(term, st.session_state['num_suggestions_transformer']),
                                      placeholder='',
                                      key='transformer',
                                      edit_after_submit="option",
                                      label='Transformer Suggestions')


if __name__ == '__main__':
    main()
