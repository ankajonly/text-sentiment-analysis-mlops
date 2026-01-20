import pandas as pd 
import numpy as np 
import os 
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.logger import logging
nltk.download("stopwords")
nltk.download("wordnet")

def preprocess_dataframe(df, col = "text"):
    """Preprocess the input text by cleaning, tokenizing, removing stopwords, and lemmatizing.

    Args:
        df (DataFrame): The input DataFrame.
        col (str): The column name containing the text to preprocess."""
    
    lemmiatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    def preprocess_text(text):
        #remove url
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        #remove numbers
        text = re.sub(r"\d+", "", text)
        #remove ppunctuations
        text = re.sub(r"[^\w\s]", "", text)
        #lowercase
        text = text.lower()
        #remove stop word
        text = " ".join([word for word in text.split() if word not in stop_words])
        #lemmatize
        text = " ".join([lemmiatizer.lemmatize(word) for word in text.split()])
        return text
    df[col] = df[col].apply(preprocess_text)
    df = df.dropna(subset=[col])
    logging.info("data preprocessing completed")
    return df

def main():
    try:
        train_data = pd.read_csv("data/raw/train.csv")
        test_data = pd.read_csv("data/raw/test.csv")
        logging.info("data loaded successfully")

        #transform the data 
        train_processed_data = preprocess_dataframe(train_data, col="review")
        test_processed_data = preprocess_dataframe(test_data, col="review")
        logging.info("data preprocessing completed successfully")

        #store the data inside data/processed 
        data_path = os.path.join("data", "interim")
        os.mkdir(data_path) if not os.path.exists(data_path) else None
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        logging.info("processed data saved successfully", data_path)

    except Exception as e:
        logging.error(f"Error in data preprocessing: {e}")
        print(f"error: {e}")
    

if __name__ == "__main__":
    main()