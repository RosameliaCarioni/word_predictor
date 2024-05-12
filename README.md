
# WORD PREDICTOR  🔍

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
- **N-gram**: Implements a basic probabilistic model based on the occurrence frequency of word sequences. - TODO: add more
- **RNN**: Utilizes Recurrent Neural Networks to handle sequential data and context, more specifically GRU is implemented. TODO: add more 
- **Transformer**: TODO 

## Data
The project uses various datasets to train and evaluate the models. Below are the details of each dataset:

| Name               | Type                     | Size     | Link                                                                |
|--------------------|--------------------------|----------|---------------------------------------------------------------------|
| **News Summarization** | Text data for news summarization | 3.85 GB  | [Kaggle - News Summarization](https://www.kaggle.com/datasets/sbhatti/news-summarization) |
| **Articles**       | Medium articles          | 3.9 MB   | [Kaggle - Medium Articles](https://www.kaggle.com/datasets/hsankesara/medium-articles) |
| **News**           | Course assignment/lecture notes | 827.2 MB | Available from the course : TODO ask for reference                                      |
| **Mobile Text**       | Corpus of mobile messages          | 932.2 MB | This dataset is used in the paper: [Mining, analyzing, and modeling text written on mobile devices](https://www.cambridge.org/core/journals/natural-language-engineering/article/mining-analyzing-and-modeling-text-written-on-mobile-devices/A60B599D7E92B5DB9CBDE243A80626C3) by K. Vertanen and P.O. Kristensoon |
| **Twitter**        | Tweets data              | 583.1 MB | [Rpubs - Twitter Data](https://rpubs.com/NAyako/1036093)             |



## Run the word predictor 

1. **Clone the repository:**
    ````
   git clone https://github.com/yourusername/WORD_PREDICTOR.git
   cd WORD_PREDICTOR
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
    python main.py
    ````
