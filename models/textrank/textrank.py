import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import string
from typing import List, Tuple
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import download as nltk_download

from utils import Trainer, display_result

try:
    sent_tokenize('test')
except LookupError:
    nltk_download('punkt')

try:
    WordNetLemmatizer().lemmatize('test')
except LookupError:
    nltk_download('wordnet')

try:
    stopwords.words('english')
except LookupError:
    nltk_download('stopwords')

class TextRankTrainer(Trainer):
    @staticmethod
    def process(lines, max_length):
        print("TextRank Summary")
        print("-------------------------------------")
        results = []
        for i, line in enumerate(lines, 1):
            result = TextRankTrainer.textrank_keypoints(line, max_chars=max_length)[0]
            display_result(line, result, i)
            results.append(result)
        print("------------------------------------")
        return results


    @staticmethod
    def textrank_keypoints(
        text: str,
        num_points: int = None,
        ratio: float = 0.02,
        similarity_threshold: float = 0.2,
        damping: float = 0.8,
        max_iter: int = 50,
        tol: float = 1e-4,
        max_chars: int = 80
    ) -> List[str]:
        sentences = sent_tokenize(text)
        stop_words = set(stopwords.words('english')) | set(string.punctuation)
        lemmatizer = WordNetLemmatizer()
        
        preprocessed = []
        for sent in sentences:
            words = [lemmatizer.lemmatize(word.lower()) 
                    for word in word_tokenize(sent)
                    if word.lower() not in stop_words and word.isalnum()]
            if len(words) < 3:
                preprocessed.append(sent[:50].lower())
            else:
                preprocessed.append(' '.join(words))
        
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform(preprocessed)
        except ValueError:
            return [sentences[0][:max_chars]] if sentences else [""]
        
        similarity_matrix = cosine_similarity(tfidf_matrix)
        np.fill_diagonal(similarity_matrix, 0)
        adaptive_threshold = max(similarity_threshold, 0.4 - (len(sentences)*0.02))
        similarity_matrix[similarity_matrix < adaptive_threshold] = 0
        
        scores = np.ones(len(sentences)) / len(sentences)
        for _ in range(max_iter):
            prev_scores = scores.copy()
            scores = (1 - damping) + damping * np.dot(similarity_matrix, scores)
            score_entropy = 1.0 - (-scores * np.log(scores + 1e-8)).sum()
            scores = scores ** (1 + score_entropy * 0.5)
            scores /= scores.sum() + 1e-8
            
            if np.linalg.norm(scores - prev_scores) < tol:
                break
        
        if not num_points:
            num_points = max(1, int(len(sentences) * ratio))
        
        length_penalty = np.array([0.65 ** (len(sentences[i]) / 40) for i in range(len(sentences))])
        adjusted_scores = scores * length_penalty
        ranked_indices = np.argsort(-adjusted_scores)
        
        selected = []
        total_chars = 0
        
        for idx in sorted(ranked_indices):
            if len(selected) >= num_points:
                break
                
            sentence = sentences[idx]
            if len(sentence) > max_chars//2:
                truncate_point = sentence.find(". ", 0, max_chars//2)
                if truncate_point > 0:
                    sentence = sentence[:truncate_point+1]
                else:
                    truncate_point = sentence.find(", ", max_chars//3)
                    if truncate_point > 0:
                        sentence = sentence[:truncate_point+1]
                    else:
                        sentence = sentence[:max_chars//2] + "..."
            
            if total_chars + len(sentence) <= max_chars or not selected:
                selected.append((idx, sentence))
                total_chars += len(sentence)
        
        return [s[1] for s in sorted(selected, key=lambda x: x[0])]

    def train(self):
        pass