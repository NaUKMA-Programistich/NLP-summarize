import argparse

from models.t5.t5 import T5SummarizationTrainer
from models.textrank.textrank import TextRankTrainer
from models.distilbart.distilbart import DistilBARTSummarizationTrainer
from utils import get_lines

def main():
    parser = argparse.ArgumentParser(description="Generate summaries from text")
    parser.add_argument("--text", type=str, help="Direct text input to summarize")
    parser.add_argument("--input_file", type=str, help="Path to text file to summarize")
    parser.add_argument("--url", type=str, help="URL of webpage to extract content and summarize")
    args = parser.parse_args()

    lines = get_lines(args.text, args.input_file, args.url)
    print(lines)
    print("-------------------------------------")
    
    text_rank_results = TextRankTrainer.process(lines)
    t5_results = T5SummarizationTrainer.process(lines)
    distilbart_results = DistilBARTSummarizationTrainer.process(lines)

    with open("results.txt", "w") as file:
        for i, line in enumerate(lines):
            file.write(f"{line}\n")
            file.write(f"\tTextRank: {text_rank_results[i]}\n")
            file.write(f"\tT5: {t5_results[i]}\n")
            file.write(f"\tDistilBART: {distilbart_results[i]}\n")
            file.write("\n")

if __name__ == "__main__":
    main()