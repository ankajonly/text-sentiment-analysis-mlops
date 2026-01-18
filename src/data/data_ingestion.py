import numpy as np 
import pandas as pd 
pd.set_option('future.no_silent_downcasting', True)
import os
from sklearn.model_selection import train_test_split
import logging
import yaml
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.preprocessing import LabelEncoder
from src.logger import logging
from src.connections import s3_connection


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

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file into a DataFrame."""
    try:
        df = pd.read_csv(data_url)
        logging.debug("data loaded successfully from %s", data_url)
        return df
    except pd.errors.ParserError as e:
        logging.error("error parsing CSV file: %s", e)
        raise
    except FileNotFoundError:
        logging.error("file not found: %s", data_url)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    try:
        logging.info("Starting data preprocessing")
        final_df = df[df["sentiment"].isin(["postive", "negative"])]
        final_df["sentiment"] = final_df["sentiment"].replace({"positive": 1, "negative": 0})
        logging.info("Data Preprocessing completed successfully")
        return final_df
    except KeyError as e:
        logging.error("Key error during preprocessing: %s", e)
        raise
    except Exception as e:
        logging.error("Unexpected error during preprocessing: %s", e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, save_path: str) -> None:
    """Save the train and test DataFrames to CSV files."""
    try:
        raw_data_path = os.path.join(save_path, "raw_data")
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logging.info("Data saved successfully to %s", raw_data_path)
    except Exception as e:
        logging.error("Error saving data: %s", e)
        raise

def main() :
    try:
        