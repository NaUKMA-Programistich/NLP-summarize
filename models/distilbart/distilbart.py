import os
import sys
import time
import psutil
import numpy as np
import tensorflow as tf
import tf_keras
import pandas as pd

from sklearn.model_selection import train_test_split
from tf_keras.src.callbacks import Callback, CSVLogger, EarlyStopping
from tf_keras.src.callbacks_v1 import TensorBoard
from tf_keras.src.losses import SparseCategoricalCrossentropy
from tf_keras.src.metrics import SparseCategoricalAccuracy
from rouge_score import rouge_scorer
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
from tf_keras.src.mixed_precision import set_global_policy

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
root_dir = os.path.dirname(parent_dir)

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
from utils import Trainer, display_result, load_dataset, prepare_datasets, PATH, SIZE

class TrainingMetrics(Callback):
    def __init__(self, val_data, tokenizer, num_samples=3):
        super().__init__()
        self.val_data = val_data
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'bleu', 'meteor'], use_stemmer=True)
    
    def on_epoch_end(self, epoch, logs=None):
        tf.config.optimizer.set_jit(False)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(8)
        
        all_preds, all_labels = [], []
        for batch in self.val_data.take(5):
            inputs, labels = batch
            preds = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=128,
                min_length=10,
                use_cache=False
            )
            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            all_preds.extend(decoded_preds)
            all_labels.extend(decoded_labels)
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': [], 'rougeLsum': [], 'bleu': [], 'meteor': []}
        
        for pred, label in zip(all_preds, all_labels):
            scores = self.scorer.score(label, pred)
            for key in rouge_scores:
                rouge_scores[key].append(scores[key].fmeasure)
        
        for key in rouge_scores:
            logs[f'val_{key}'] = np.mean(rouge_scores[key])
        
        print(f"\nValidation ROUGE Scores - "
              f"R1: {logs['val_rouge1']:.3f}, "
              f"R2: {logs['val_rouge2']:.3f}, "
              f"RL: {logs['val_rougeL']:.3f}, "
              f"RLsum: {logs['val_rougeLsum']:.3f}, "
              f"BLEU: {logs['val_bleu']:.3f}, "
              f"METEOR: {logs['val_meteor']:.3f}")

class DistilBARTSummarizationTrainer(Trainer):
    output_base_dir = "models/distilbart"
    
    @staticmethod
    def process(lines):
        tf.config.optimizer.set_jit(False)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(8)

        print("DISTILBART SUMMARY")
        print("Loading DistilBART model...")
        print("-------------------------------------")
        
        path = os.path.abspath(os.path.join(DistilBARTSummarizationTrainer.output_base_dir, 'trained_summarizer'))
        model = TFAutoModelForSeq2SeqLM.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        print("DistilBART Model loaded successfully")
        
        def summarize_distilbart_model(text, model, tokenizer):
            inputs = tokenizer(text, return_tensors="tf", max_length=128, truncation=True)
            with tf.device('/CPU:0'):
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=128,
                    min_length=10,
                    num_beams=4,
                    length_penalty=1.0,
                    early_stopping=True,
                    do_sample=True,
                    temperature=0.7,
                    no_repeat_ngram_size=2
                )
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("-------------------------------------")
        
        results = []
        for i, line in enumerate(lines, 1):
            result = summarize_distilbart_model(line, model, tokenizer)
            display_result(line, result, i)
            results.append(result)
        print("-------------------------------------")
        return results
   
    def __init__(self, model_name, nrows, epochs, dataset_path, use_tqdm=False, from_pt=False):
        self.model_name = model_name
        self.nrows = nrows
        self.epochs = epochs
        self.dataset_path = dataset_path
        self.use_tqdm = use_tqdm
        self.from_pt = from_pt
        self.log_dir = os.path.join(self.output_base_dir, 'logs')
        self.csv_log_path = os.path.join(self.output_base_dir, 'training_log.csv')
        self.save_path = os.path.join(self.output_base_dir, 'trained_summarizer')
        self.tokenizer = None
        self.model = None
        self.train_ds = None
        self.val_ds = None
        self.history = None
        self.start_time = None
        self.end_time = None
        self._setup_tf()
    
    def _setup_tf(self):
        tf.autograph.set_verbosity(0)
        tf.config.optimizer.set_jit(False)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(8)
       
        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"M2 Metal GPU Active: {gpus}")
            set_global_policy('float32')
        else:
            print("Using CPU - Check Metal Installation")
    
    def _load_and_prepare_data(self):
        print(f"Loading dataset with {self.nrows} rows...")
        df = load_dataset(self.dataset_path, nrows=self.nrows)
        print("Initializing tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        print("Preparing datasets...")
        train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
        del df
        batch_size = 16
        self.train_ds, self.val_ds = prepare_datasets(
            pd.concat([train_df, val_df.iloc[:min(1000, len(val_df))]]),
            self.tokenizer,
            batch_size=batch_size
        )
        del train_df
        del val_df
        print(f"Datasets prepared: batch size={batch_size}")
        print("-------------------------------------")
    
    def _build_model(self):
        print("Loading model...")
        os.environ["TF_KERAS_SAFE_MODE"] = "1"
        if self.from_pt:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        self.model = TFAutoModelForSeq2SeqLM.from_pretrained(
            self.model_name, 
            from_pt=self.from_pt
        )
        print(f"Model {self.model_name} loaded successfully")
    
    def _compile_model(self):
        optimizer = tf_keras.optimizers.Adam(learning_rate=2e-5)
        self.model.compile(
            optimizer=optimizer,
            loss=SparseCategoricalCrossentropy(from_logits=True),
            metrics=[SparseCategoricalAccuracy(name='accuracy')]
        )
    
    def _setup_callbacks(self):
        return[
            TensorBoard(log_dir=self.log_dir),
            CSVLogger(self.csv_log_path),
            TrainingMetrics(self.val_ds, self.tokenizer),
            EarlyStopping(
                monitor='val_loss',
                patience=2,
                restore_best_weights=True,
                verbose=1
            )
        ]
    
    def _save_model(self):
        os.makedirs(self.save_path, exist_ok=True)
        self.model.save_pretrained(self.save_path)
        self.tokenizer.save_pretrained(self.save_path)
        print(f"Model saved to '{self.save_path}/' directory")
    
    def _print_summary(self):
        if self.history:
            final_metrics = {
                'train_loss': self.history.history['loss'][-1],
                'val_loss': self.history.history['val_loss'][-1],
                'train_acc': self.history.history['accuracy'][-1],
                'val_acc': self.history.history['val_accuracy'][-1]
            }
            print(f"\nTraining completed in {self.end_time - self.start_time:.2f} seconds.")
            print("\nFinal Metrics:")
            for k, v in final_metrics.items():
                print(f"{k}: {v:.4f}")
        else:
            print("\nTraining did not complete successfully or history was not recorded.")
    
    def train(self):
        self.start_time = time.time()
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024
        print(f"Initial memory usage: {initial_memory:.2f} MB")
        
        self._load_and_prepare_data()
        after_data_memory = process.memory_info().rss / 1024 / 1024
        print(f"Memory usage after loading data: {after_data_memory:.2f} MB")
        
        self._build_model()
        self._compile_model()
        callbacks = self._setup_callbacks()
        print("\nTraining Summary:")
        
        self.model.summary()
        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.epochs,
            callbacks=callbacks
        )
        self.end_time = time.time()
        final_memory = process.memory_info().rss / 1024 / 1024
        print(f"Final memory usage: {final_memory:.2f} MB")
        
        self._save_model()
        self._print_summary()
        print("-------------------------------------")

if __name__ == "__main__":
    print("Starting training...")
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    DistilBARTSummarizationTrainer(
        model_name="sshleifer/distilbart-cnn-12-6",
        nrows=SIZE,
        epochs=3,
        dataset_path=PATH,
        use_tqdm=True,
        from_pt=True
    ).train()