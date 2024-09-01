import numpy as np
import pandas as pd
import os
import re
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
import logging


# Setup Logging
def setup_logger(log_file='data_processing.log', logger_name='data_processing_logger'):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger()


# Load Data
def load_data(train_path, test_path):
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        logger.info(f"Data successfully loaded from {train_path} and {test_path}.")
        return train_data, test_data
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"No data found: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading data: {e}")
        raise

# Data Transformation Functions
def download_nltk_resources():
    try:
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)
        logger.info("NLTK resources downloaded successfully.")
    except Exception as e:
        logger.error(f"Error downloading NLTK resources: {e}")
        raise


def lemmatize_text(text):
    try:
        lemmatizer = WordNetLemmatizer()
        return " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    except Exception as e:
        logger.error(f"Error during lemmatization: {e}")
        raise

def remove_stopwords(text):
    try:
        stop_words = set(stopwords.words("english"))
        return " ".join([word for word in text.split() if word not in stop_words])
    except Exception as e:
        logger.error(f"Error removing stop words: {e}")
        raise

def remove_numbers(text):
    try:
        return ''.join([char for char in text if not char.isdigit()])
    except Exception as e:
        logger.error(f"Error removing numbers: {e}")
        raise

def to_lower_case(text):
    try:
        return text.lower()
    except Exception as e:
        logger.error(f"Error converting text to lower case: {e}")
        raise

def remove_punctuation(text):
    try:
        text = re.sub(r'[^\w\s]', ' ', text)
        return " ".join(text.split())
    except Exception as e:
        logger.error(f"Error removing punctuation: {e}")
        raise

def remove_urls(text):
    try:
        return re.sub(r'https?://\S+|www\.\S+', '', text)
    except Exception as e:
        logger.error(f"Error removing URLs: {e}")
        raise


# Process Text
def process_text(df):
    try:
        df['content'] = df['content'].apply(to_lower_case)
        df['content'] = df['content'].apply(remove_stopwords)
        df['content'] = df['content'].apply(remove_numbers)
        df['content'] = df['content'].apply(remove_punctuation)
        df['content'] = df['content'].apply(remove_urls)
        df['content'] = df['content'].apply(lemmatize_text)
        logger.info("Text normalization completed successfully.")
        return df
    except KeyError as e:
        logger.error(f"Column not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        raise


# Save Processed Data
def save_data(df, output_path, file_name):
    try:
        os.makedirs(output_path, exist_ok=True)
        full_path = os.path.join(output_path, file_name)
        df.to_csv(full_path, index=False)
        logger.info(f"Data successfully saved at {full_path}.")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise


# Main Workflow
def main(train_path, test_path, output_path):
    download_nltk_resources()

    train_data, test_data = load_data(train_path, test_path)

    train_processed = process_text(train_data)
    test_processed = process_text(test_data)

    save_data(train_processed, output_path, 'processed_train.csv')
    save_data(test_processed, output_path, 'processed_test.csv')


if __name__ == "__main__":
    TRAIN_PATH = './data/raw/train.csv'
    TEST_PATH = './data/raw/test.csv'
    OUTPUT_PATH = './data/processed'

    try:
        main(TRAIN_PATH, TEST_PATH, OUTPUT_PATH)
    except Exception as e:
        logger.critical(f"Critical error in main execution: {e}")
        raise
