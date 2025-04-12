from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from typing import List
from nltk import download as nltk_download
import tensorflow as tf
from utils import Trainer, display_result
import networkx as nx

class TextRankTrainer(Trainer):
    @staticmethod
    def process(lines):
        print("TextRank Summary")
        print("-------------------------------------")
        nltk_download('punkt', quiet=True)
        nltk_download('stopwords', quiet=True)

        results = []
        for i, line in enumerate(lines, 1):
            result_sentences = TextRankTrainer.textrank_keypoints(line)
            result = " ".join(result_sentences) if result_sentences else ""
            display_result(line, result, i)
            results.append(result)
        print("------------------------------------")
        return results

    @staticmethod
    def textrank_keypoints(
        text: str, 
        top_n: int = 10, 
        window_size: int = 4
    ) -> List[str]:
        tokens = word_tokenize(text)
        words = [w for w in tokens if w.isalpha()]

        stop_words = set(stopwords.words('english'))
        filtered = [w for w in words if w not in stop_words]

        G = nx.Graph()
        for idx, word in enumerate(filtered):
            if not G.has_node(word):
                G.add_node(word)
            for j in range(idx + 1, min(idx + window_size, len(filtered))):
                w2 = filtered[j]
                if G.has_edge(word, w2):
                    G[word][w2]['weight'] += 1
                else:
                    G.add_edge(word, w2, weight=1)

        ranks = nx.pagerank(G, weight='weight')
        top_keywords = sorted(ranks.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [word for word, score in top_keywords]

    def train(self):
        pass