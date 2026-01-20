import numpy as np
import pandas as pd 
import os 
from sklearn.model_selection import train_test_split
import yaml
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from src.logger import logging


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def apply_bow(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) ->tuple:
    """Apply Bag of Words vectorization to the text data."""
    try:
        logging.info('Applying Bag of Words vectorization...')
        vectorizer = CountVectorizer(max_features=max_features)

        X_train = train_data['review'].values
        y_train = train_data['sentiment'].values
        X_test = test_data['review'].values
        y_test = test_data['sentiment'].values

        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train
        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test

        # Save the vectorizer
        with open('bow_vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        
        logging.info('Bag of Words vectorization applied successfully')
        return train_df, test_df
    except Exception as e:
        logging.error('Error in applying Bag of Words: %s', e)
        raise
    