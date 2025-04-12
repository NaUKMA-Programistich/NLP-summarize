import argparse
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
import tensorflow as tf
import requests
from bs4 import BeautifulSoup
import re

from models.t5.t5 import T5SummarizationTrainer
from models.textrank.textrank import TextRankTrainer
from utils import extract_content_from_url

tf.config.optimizer.set_jit(False)

def get_lines(text, file, url):
    if url is not None:
        return extract_content_from_url(url)
    if text is not None:
        return [line for line in text.split('\n') if line.strip()]
    else:
        lines = []
        with open(file, "r") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    lines.append(stripped)
        return lines

def main():
    parser = argparse.ArgumentParser(description="Generate summaries from text")
    parser.add_argument("--text", type=str, help="Direct text input to summarize")
    parser.add_argument("--input_file", type=str, help="Path to text file to summarize")
    parser.add_argument("--url", type=str, help="URL of webpage to extract content and summarize")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum length of summary (default: 128)")
    args = parser.parse_args()

    lines = get_lines(args.text, args.input_file, args.url)
    print(lines)
    print("-------------------------------------")
    
    text_rank_results = TextRankTrainer.process(lines, args.max_length)
    t5_results = T5SummarizationTrainer.process(lines, args.max_length)

    with open("results.csv", "w") as file:
        file.write("Original Text,TextRank,T5\n")
        
        for i, line in enumerate(lines):
            csv_line = f'"{line}","{text_rank_results[i]}","{t5_results[i]}"\n'
            file.write(csv_line)

    with open("results.txt", "w") as file:
        for i, line in enumerate(lines):
            file.write(f"{line}\n")
            file.write(f"\tTextRank: {text_rank_results[i]}\n")
            file.write(f"\tT5: {t5_results[i]}\n")
            file.write("\n")

if __name__ == "__main__":
    main()