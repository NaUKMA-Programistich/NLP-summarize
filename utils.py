from abc import ABC, abstractmethod
import os
import re

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tf_keras.src.mixed_precision import set_global_policy

SIZE = 100
PATH = "/Users/programistich/Study/spring-5/Ling/Final/dataset_hard.csv"

def load_dataset(csv_path: str, nrows: int = None):
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path, nrows=nrows)
    print(f"Dataset loaded with {len(df)} rows.")
    return df


def prepare_data(df, text_column="text", summary_column="headline"):
    texts = df[text_column].astype(str).tolist()
    prefixes = ["extract key points: " + t for t in texts]
    summaries = df[summary_column].astype(str).tolist()
    return prefixes, summaries

def tokenize_data(tokenizer, inputs, outputs, max_input_length=512, max_output_length=128):
    encodings = tokenizer(
        inputs, padding="max_length", truncation=True,
        max_length=max_input_length, return_tensors="tf"
    )
    decodings = tokenizer(
        outputs, padding="max_length", truncation=True,
        max_length=max_output_length, return_tensors="tf",
    )
    return encodings["input_ids"], encodings["attention_mask"], decodings["input_ids"]


def verify_m2_setup():
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"M2 Metal GPU Active: {gpus}")
        set_global_policy('float32')
    else:
        print("Using CPU - Check Metal Installation")



def prepare_datasets(df, tokenizer, test_size=0.2):
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=42)

    train_inputs, train_outputs = prepare_data(train_df)
    val_inputs, val_outputs = prepare_data(val_df)

    train_ids, train_masks, train_labels = tokenize_data(tokenizer, train_inputs, train_outputs)
    val_ids, val_masks, val_labels = tokenize_data(tokenizer, val_inputs, val_outputs)

    train_decoder_input_ids = np.concatenate([np.zeros((train_labels.shape[0], 1)), train_labels[:, :-1]], axis=1)
    val_decoder_input_ids = np.concatenate([np.zeros((val_labels.shape[0], 1)), val_labels[:, :-1]], axis=1)

    train_decoder_attention_mask = np.ones_like(train_decoder_input_ids)
    val_decoder_attention_mask = np.ones_like(val_decoder_input_ids)

    train_ds = tf.data.Dataset.from_tensor_slices(
        ({
            "input_ids": train_ids,
            "attention_mask": train_masks,
            "decoder_input_ids": train_decoder_input_ids,
            "decoder_attention_mask": train_decoder_attention_mask,
        }, train_labels)
    ).shuffle(1000).batch(4).cache().prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices(
        ({
            "input_ids": val_ids,
            "attention_mask": val_masks,
            "decoder_input_ids": val_decoder_input_ids,
            "decoder_attention_mask": val_decoder_attention_mask,
        }, val_labels)
    ).batch(4).cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds

def display_result(text, result, index):
    print(f"\033[33m[{index}]\033[0m \033[90m{text}\033[0m")
    print(f"   \033[90m└──▶\033[0m \033[1m{result}\033[0m")


class Trainer(ABC): 

    @abstractmethod
    def process(lines, max_length):
        pass

    @abstractmethod
    def train(self):
        pass

def remove_short_words(text: str) -> str:
    return ' '.join(word for word in text.split() if len(word) >= 3)

def extract_content_from_url(url: str) -> dict:
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        for element in soup(["script", "style", "noscript"]):
            element.decompose()
        
        paragraphs = [
            remove_short_words(p.get_text(strip=True))
            for p in soup.find_all('p') 
            if p.get_text(strip=True)
        ]
        
        headings = {}
        for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            headings[tag] = [
                remove_short_words(element.get_text(strip=True))
                for element in soup.find_all(tag) 
                if element.get_text(strip=True)
            ]
        
        links = []
        for a in soup.find_all('a', href=True):
            link_text = a.get_text(strip=True)
            if link_text:
                filtered_text = remove_short_words(link_text)
                links.append({"text": filtered_text, "href": a['href']})
        
        all_text = soup.get_text(separator="\n")
        all_text_clean = "\n".join(line.strip() for line in all_text.splitlines() if line.strip())
        all_text_filtered = remove_short_words(all_text_clean)
        
        return paragraphs
        
    except Exception as e:
        error_message = f"Error extracting content from URL: {e}"
        return [error_message]
