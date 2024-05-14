
# Word Predictor  🔍

The word predictor allows users to speed the typing process by providing with suggestions to autocomplete what they are writting. The system is designed to predict the next word as you type, enhancing typing efficiency in text messages, emails, and other text inputs. This project includes various models such as N-gram, RNN, and Transformer to provide accurate word predictions.

## Demo
![Word Predictor Demo](make_it_a_gif)

## Report
![Report with our main findings and experiments](link_to_report)

## Features

### Functionality
As you type in the text input field, the word predictor displays a list of the most probable completions of the current word. This list updates with each keystroke. You can select a suggested word to autocomplete the text, saving time and effort.

### Settings
- **Prediction Model**: Choose between N-gram, RNN, and Transformer models.
- **Number of suggestions**: Set the maximum number of suggestions displayed. - TODO: would be cool

### Models 
- **N-gram**: Implements a basic probabilistic model based on the occurrence frequency of word sequences. They are based on the Markov assumption that a word is only dependend on a couple of previous words. For a n-gram model, it is assumed that the next word is determined by the n-1 preceding words. The probabilities of the words given n-1 previous words are usually estimated using a very large corpus and maximum likelihood estimation. However, no corpus will be large enough to cover all possible n-grams so it is important to deal with zero-probabilities during deployment. Generally, there are several approaches to do so: Laplace smoothing, backoff and linear interpolation. In this project, we decided to apply linear interpolation.  - TODO: add more
- **RNN**: Utilizes Recurrent Neural Networks to handle sequential data and context, more specifically GRU is implemented. In contrast to the n-gram models which form part of the statistical approaches, RNNs belong to the neural approaches of natural language processing. The distinguishing feature of RNNs is their hidden state. By making use of it, the RNN gets a sort of "memory" because it can include information of previous tokens. There have been several extensions of the vanilla RNN cell: UGRNN, GRU and LSTM. They all use different gates in their cell to update (or forget) the previous hidden state. LSTMs have already been invented in the 90s and have the most parameters. The paper that introduced GRUs was published almost twenty years later and the reduction of parameters of GRU compared to LSTMs makes it more learnable. However, LSTMs are better suited to cover long-term dependencies than GRUs. As in our project, we implement models to do word prediction for messages or e-mail, the long-term dependencies are not as crucial as in other applications, we decided to make use of the efficiency of GRUs. TODO: add more 
- **Transformer**: The transformer architecture is the backbone of Large Language Models like GPT-4, Gemini, BERT etc. The Google paper "Attention is all you need" introduced self-attention and made the training more parallelizable. Using key, query and value vectors, a transformer block transforms input vectors (usually with a positional encoding) into output vectors, thus learning a contextualized vector representation. Fine-tuning pre-trained language models like BERT has shown to be a good approach to solve a lot of language engineering problems. In our project, we will try to learn a useful embedding of word sequences to successfully predict the next word. - TODO: add details of what we are doing 

## Data
The project uses various datasets to train the models. Below are the details of each dataset:

| Name               | Type                     | Size     | Link                                                                |
|--------------------|--------------------------|----------|---------------------------------------------------------------------|
| **News Summaries and Content** | Text data for news | 3.85 GB  | [Kaggle - News Summarization](https://www.kaggle.com/datasets/sbhatti/news-summarization) |
| **Articles**       | Medium articles          | 3.9 MB   | [Kaggle - Medium Articles](https://www.kaggle.com/datasets/hsankesara/medium-articles) |
| **News**           | Course assignment | 827.2 MB | Available from the course : TODO ask for reference                                      |
| **Mobile Text**       | Corpus of mobile messages          | 932.2 MB | This dataset is used in the paper: [Mining, analyzing, and modeling text written on mobile devices](https://www.cambridge.org/core/journals/natural-language-engineering/article/mining-analyzing-and-modeling-text-written-on-mobile-devices/A60B599D7E92B5DB9CBDE243A80626C3) by K. Vertanen and P.O. Kristensoon |
| **Twitter**        | Tweets data              | 583.1 MB | [Rpubs - Twitter Data](https://rpubs.com/NAyako/1036093)             |

### Preprocessing 
All datasets were pre-processed to clean and standardize the text data. The preprocessing scripts are available in the `data_processing/` directory, and the cleaned datasets can be [downloaded here](https://drive.google.com/file/d/13o4l--P29W-vunAA241gp_zDWu0SOxJF/view?usp=sharing). You should unzip them and store them in the main directory in the path `data/clean_data/`.

## File Structure 
The project directory is organized as follows:  
- **`data_processing/`**: Includes Jupyter notebooks used to clean and preprocess the datasets.
- **`images/`**: Contains images of the contributors, used for the web app.
- **`models/`**: Contains model training scripts and pre-trained weights.
  - **`training/`**: Directory with scripts for training the models and storing their weights.
  - **`n_gram.py`**: N-gram model handler.
  - **`rnn.py`**: RNN model handler.
  - **`transformer.py`**: Transformer model handler.

## Run the word predictor 

1. **Clone the repository:**
    ````
   git clone https://github.com/RosameliaCarioni/word_predictor.git
   cd word_predictor
    ````

2. **Create an activate a conda environment:**
    ````
    conda create --name word_predictor python=3.11 
    conda activate word_predictor
    ````
3. **Install the required packages:**
    ````
    pip install -r requirements.txt
    ````
4. **Run the program:**
    ````
    streamlit run main.py
    ````
