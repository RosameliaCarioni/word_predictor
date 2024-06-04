
# Word Predictor  🔍

The word predictor allows users to speed the typing process by providing with suggestions to autocomplete what they are writting. The system is designed to predict the next word as you type, enhancing typing efficiency in text messages, emails, and other text inputs. This project includes various models such as N-gram, RNN, and Transformer to provide accurate word predictions.

## Demo
![Word Predictor Demo](https://github.com/RosameliaCarioni/word_predictor/assets/72268666/b2b5eaf1-af1a-4504-81e1-37e56718dd11)

## Report
![Report with our main findings, detail implementation and experiments](link_to_report)

## System Design 
To get a better understanding of how our system is designed and works see the following diagram. If this is unclear, refer to our report for further details: 
<img width="1156" alt="Screenshot 2024-06-01 at 16 12 15" src="https://github.com/RosameliaCarioni/word_predictor/assets/72268666/59ba1c22-84b7-4dbb-b948-11102748fd3a">

## Features

### Functionality
As you type in the text input field, the word predictor displays a list of the most probable completions of the current word. This list updates with each keystroke. You can select a suggested word to autocomplete the text, saving time and effort. You can also see how many characters you have saved with the autocompletion of words by using a specific model. 

### Settings
- **Prediction Model**: Choose between N-gram, RNN, and Transformer models.
- **Number of suggestions**: Set the maximum number of suggestions to be displayed.

### Models 
- **N-gram**: Implements a basic probabilistic model based on the occurrence frequency of word sequences. These models are based on the Markov assumption that a word is only dependent on a couple of previous words. For a N-Gram model, it is assumed that the next word is determined by the n-1 preceding words. The probabilities of the words given n-1 previous words are usually estimated using a very large corpus and maximum likelihood estimation. However, no corpus will be large enough to cover all possible n-grams so it is important to deal with zero-probabilities during deployment. Generally, there are several approaches to do so: Laplace smoothing, backoff and linear interpolation. In this project, we decided to apply linear interpolation and to set n to 3, meaning that we implemented a trigram. 
- **RNN**: Utilizes Recurrent Neural Networks to handle sequential data and context, more specifically GRU is implemented. In contrast to the n-gram models which form part of the statistical approaches, RNNs belong to the neural approaches of natural language processing. The distinguishing feature of RNNs is their hidden state. By making use of it, the RNN gets a sort of "memory" because it can include information of previous tokens. There have been several extensions of the vanilla RNN cell: UGRNN, GRU and LSTM. They all use different gates in their cell to update (or forget) the previous hidden state. LSTMs have already been invented in the 90s and have the most parameters. The paper that introduced GRUs was published almost twenty years later and the reduction of parameters of GRU compared to LSTMs makes it more learnable. However, LSTMs are better suited to cover long-term dependencies than GRUs. As in our project, we implement models to do word prediction for messages or e-mail, the long-term dependencies are not as crucial as in other applications, we decided to make use of the efficiency of GRUs.
- **Transformer**: The transformer architecture is the backbone of Large Language Models like GPT-4, Gemini, BERT etc. The Google paper "Attention is all you need" introduced self-attention and made the training more parallelizable. Using key, query and value vectors, a transformer block transforms input vectors (usually with a positional encoding) into output vectors, thus learning a contextualized vector representation. Fine-tuning pre-trained language models like BERT has shown to be a good approach to solve a lot of language engineering problems. In this project we implemented only the encoder part of a transformer. To see further details please refer to our report. 

## Data
The project uses various datasets to train the models. Below are the details of each dataset:

| Name               | Type                     | Size     | Link                                                                |
|--------------------|--------------------------|----------|---------------------------------------------------------------------|
| **News Summaries** | Text data summarizing news | 264 MB | [Kaggle - News Summarization](https://www.kaggle.com/datasets/sbhatti/news-summarization) |
| **Articles**       | Medium articles          | 3.9 MB   | [Kaggle - Medium Articles](https://www.kaggle.com/datasets/hsankesara/medium-articles) |
| **Mobile Text**    | Corpus of mobile messages          | 932.2 MB | This dataset is used in the paper: [Mining, analyzing, and modeling text written on mobile devices](https://www.cambridge.org/core/journals/natural-language-engineering/article/mining-analyzing-and-modeling-text-written-on-mobile-devices/A60B599D7E92B5DB9CBDE243A80626C3) by K. Vertanen and P.O. Kristensoon |
| **Twitter**        | Tweets data              | 583.1 MB | [Rpubs - Twitter Data](https://rpubs.com/NAyako/1036093)             |

### Preprocessing 
All datasets were pre-processed to clean and standardize the text data. The preprocessing scripts are available in the `data_processing/` directory, and the cleaned datasets can be [downloaded here](https://drive.google.com/file/d/13o4l--P29W-vunAA241gp_zDWu0SOxJF/view?usp=sharing). You should unzip them and store them in the main directory in the path `data/clean_data/`.

## File Structure 
The project directory is organized as follows:

- **`data_processing/`**: Includes Jupyter notebooks used to clean and preprocess the datasets.
- **`images/`**: Contains images of the contributors, used for the web app.
- **`models/`**: Contains model training scripts and pre-trained weights.
  - **`training/`**: Directory with scripts for training the models and storing their weights.
  - **`weights/`**: Directory containing the weights for the three types of models implemented and trained. 
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
