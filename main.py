from __future__ import annotations

import logging
from typing import List
from models import rnn, n_gram, transformer
from models.training.rnn import run, RNN

import streamlit as st
from custom_searchbox import st_searchbox

logging.getLogger("streamlit_searchbox").setLevel(logging.DEBUG)
st.set_page_config(layout="centered", page_icon=':iphone:', page_title="Word Autocompletion")


def init_models():
    if 'n_gram_model' not in st.session_state:
        st.session_state.n_gram_model = n_gram.NGram()
    if 'rnn_model' not in st.session_state:
        st.session_state.rnn_model = rnn.RNN()
    if 'transformer_model' not in st.session_state:
        st.session_state.transformer_model = transformer.Transformer()


def search_ngram(search_term: str, number_of_suggestions: int) -> List[str]:
    return st.session_state.n_gram_model.predict_next_word(search_term, number_of_suggestions)


def search_rnn(search_term: str, number_of_suggestions: int) -> List[str]:
    return st.session_state.rnn_model.predict_next_word(search_term, number_of_suggestions)


def search_transformer(search_term: str, number_of_suggestions: int) -> List[str]:
    return st.session_state.transformer_model.predict_next_word(search_term, number_of_suggestions)


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

    information, n_gram_page, rnn_page, transformer_page, rnn_test_page = st.tabs(
        ["Information", "N-Gram", "RNN-GRU", "Transformer", "RNN-TEST"]
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
        st.write("The N-Gram model predicts the next word in a sequence by looking at the previous N-2 words. It is a simple yet powerful model for text prediction based on **statistical probability**. It uses the **Markov assumption** which says that a word is only dependent on a couple of previous words. Generally, a n-gram models looks at the precendent **n-1** words and learns the probabilities for those n-grams by building a lookup table of all word combinations it encounteres during training. The larger the training corpus, the better the model and the more different combinations it can learn.   \n\n However, no training corpus is big enough to contain all possible combinations of words which will almost certainly mean that a model will come across a n-1 word sequence it has not seen during training. There are various ways to deal with those zero-probabilities during deployment e.g. Laplace smoothing, backoff and linear interpolation. For this project, we decided to implement **linear interpolation** which uses a weighted combination of an epsilon and the unigram, bigram, trigram, ... n-gram probabilities to determine to probability of a word.")

        st.subheader("RNN-GRU Model :smile:")
        st.write("The Recurrent Neural Network (RNN) with Gated Recurrent Units (GRU) is a type of neural network that excels in learning from sequences of data. In text prediction, it can capture long-range dependencies and context from text. Generally, RNNs are well suited for sequential data like texts because they make use of a hidden state that is updated after every step when processing the sequence. If the sequence consists of n words the last hidden state will contain information of all words. However, the vanilla RNN cell does not perform too well if the sequence is very long. This is because training them via Backpropagation Through Time leads to problems with vanishing and exploding gradients.  \n\n In order to deal with those problems, more sophisticated RNN-cell architectures have been developed. **LSTMs** (Long Short-Term Memory) were the state-of-the-art for a long time as they were able to learn long-term dependencies through an elaborate design that involves **input, output and forget gates** to have a better control over how the hidden states becomes updated. Another important type of gated cell architectures are **GRUs** (Gated Recurrent Network) that combine the input and forget gates of the LSTM into a single **update gate**. Additionally, they have a **reset gate** that determines how much of the previous hidden state should be forgotten. Compared to LSTMs, they are require less parameters and are more efficient at training but this comes with the trade-off of performing slightly worse than LSTMs when the sequences are very long.  \n\n In this project, we implemented a GRU that predicts the next word, given the user's current input. The datapoints for the training consisted of vectors that contained the indicies of n words with the label being the subsequent word. In this way, the GRU learns to predict which word is most likely the next one. ")

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
                                      edit_after_submit="autocomplete",
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
                                      edit_after_submit="autocomplete",
                                      label='RNN Suggestions')
        #' '.join(st.session_state.current_text.split()[:-1]) + term
        

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
                                      edit_after_submit="autocomplete",
                                      label='Transformer Suggestions')
        
    with rnn_test_page:
        # Initialize session state variables
        if 'current_text' not in st.session_state:
            st.session_state.current_text = ""
        if 'letters_saved' not in st.session_state:
            st.session_state.letters_saved = 0
        if 'predictions' not in st.session_state:
            st.session_state.predictions = search_rnn(st.session_state.current_text, st.session_state['num_suggestions_rnn'])
        if "text_input" not in st.session_state:
            st.session_state.text_input = ""

        # Define the slider and update the session state
        num_suggestions_rnn_test = st.slider('Choose number of RNN suggestions', min_value=1, max_value=10,
                                        value=st.session_state['num_suggestions_rnn'], key='rnn_test_slider')
        st.session_state['num_suggestions_rnn_test'] = num_suggestions_rnn_test
        
        def update_text(prediction):
            current_text = st.session_state.current_text
            words = current_text.split()
            # calculate letters saved
            # TODO: calculation is wrong because the words written with st.write() are the entire text
            # could maybe try to use st.empty() and overwrite the words again and agai
            st.session_state.letters_saved += len(prediction) - len(words[-1])  
            if words:
                words[-1] = prediction
            else:
                words.append(prediction)
            new_text = ' '.join(words)
            print(new_text)
            print(len(prediction), len(words[-1]))
            st.session_state.current_text = new_text

        # Title
        st.title("Word Predictor")

        # Text input with on_change callback
        # Using st_keyup (changed with every keystroke)
        text_inp = st_keyup("Type here:", key='text_input')
        # Alternative: using text_area or text_input; have to press (cmd + ) Enter to reload page
        #text_inp = st.text_area("Type here:", value=st.session_state.current_text, key='text_input', on_change=update_predictions())
        if text_inp:
            st.session_state.current_text = text_inp
            st.session_state.predictions = search_rnn(st.session_state.current_text, st.session_state['num_suggestions_rnn_test'])

        # Display predictions
        st.write("Predictions:")
        col1, col2 = st.columns(2)
        with col1:
            for i, prediction in enumerate(st.session_state.predictions):
                if st.button("Suggestion " + str(i) + " :", key=i):
                    update_text(prediction)
        
        with col2:
            for i, prediction in enumerate(st.session_state.predictions):
                st.write(prediction)


        st.write(f"Current text: {st.session_state.current_text}")
        st.write(f"Letters saved: {st.session_state.letters_saved}")

        # resets text (but only when pressing several times, first time is not sufficient because of how streamlit updates stuff)
        if st.button("Reset Text", key="reset"):
            st.session_state.current_text = ""
            st.session_state.letters_saved = 0
            #st.session_state.text_input = "" # I tried this but streamlit does not allow it


if __name__ == '__main__':
    main()
