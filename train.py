import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    M2M100Config,
    M2M100Tokenizer,
    M2M100ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    GenerationConfig
)
import sentencepiece as spm
import json
from tqdm import tqdm

from transformers.models.m2m_100 import tokenization_m2m_100
from typing import Union, Dict, List

TRAIN_VI_FILE = '/kaggle/input/vietnamese-lao/Vietnamese_Lao/VLSP2023/Train/train2023.vi'
TRAIN_LO_FILE = '/kaggle/input/vietnamese-lao/Vietnamese_Lao/VLSP2023/Train/train2023.lo'
TRAIN_GGTRANS_VI_FILE = '/kaggle/input/vietnamese-lao/Vietnamese_Lao/GoogleTranslate/translate.vi'
TRAIN_GGTRANS_LO_FILE = '/kaggle/input/vietnamese-lao/Vietnamese_Lao/GoogleTranslate/translate.lo'
TRAIN_GEMINI_VI_FILE = '/kaggle/input/vietnamese-lao/Vietnamese_Lao/GeminiTranslate/gemini.vi'
TRAIN_GEMINI_LO_FILE = '/kaggle/input/vietnamese-lao/Vietnamese_Lao/GeminiTranslate/gemini.lo'
DEV_VI_FILE = '/kaggle/input/vietnamese-lao/Vietnamese_Lao/VLSP2023/Dev/dev2023.vi'
DEV_LO_FILE = '/kaggle/input/vietnamese-lao/Vietnamese_Lao/VLSP2023/Dev/dev2023.lo'
TEST_VI_FILE = '/kaggle/input/vietnamese-lao/Vietnamese_Lao/VLSP2023/Test/public test/test_vi.txt'
TEST_LO_FILE = '/kaggle/input/vietnamese-lao/Vietnamese_Lao/VLSP2023/Test/public test/test_lo.txt'

# Replace the original load_json function with a UTF-8 aware version
def load_json_utf8(path: str) -> Union[Dict, List]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

tokenization_m2m_100.load_json = load_json_utf8

class Tokenizer(M2M100Tokenizer):
    def __init__(self, vocab_file, spm_file, src_lang='vi', tgt_lang='lo', **kwargs):
        super().__init__(vocab_file=vocab_file, spm_file=spm_file, **kwargs)

        lang_tokens = [value for _, value in self.id_to_lang_token.items()]

        self.add_special_tokens({
            'pad_token': '<pad>',
            'unk_token': '<unk>',
            'eos_token': '</s>',
            'bos_token': '<s>',
            'additional_special_tokens': lang_tokens
        })

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
    
    @staticmethod
    def from_pretrained(path):
        return M2M100Tokenizer.from_pretrained(path)
    

class Model(M2M100ForConditionalGeneration):
    def __init__(self, tokenizer, config=None, generation_config=None, size='small'):
        if config:
            self.config = config
        else:
            if size == 'small':
                self.config = M2M100Config(
                    vocab_size=len(tokenizer),
                    d_model=512,
                    encoder_layers=6,
                    decoder_layers=6,
                    encoder_attention_heads=8,
                    decoder_attention_heads=8,
                    encoder_ffn_dim=2048,
                    decoder_ffn_dim=2048,
                    max_position_embeddings=512,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    activation_function="gelu",
                    scale_embedding=True,
                    attention_dropout=0.1,
                    activation_dropout=0.1,
                    dropout=0.125
                )
            elif size == 'medium':
                self.config = M2M100Config(
                    vocab_size=len(tokenizer),
                    d_model=768,
                    encoder_layers=7,
                    decoder_layers=7,
                    encoder_attention_heads=8,
                    decoder_attention_heads=8,
                    encoder_ffn_dim=2048,
                    decoder_ffn_dim=2048,
                    max_position_embeddings=512,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    activation_function="gelu",
                    scale_embedding=True,
                    attention_dropout=0.1,
                    activation_dropout=0.1,
                    dropout=0.2
                )
            elif size == 'large':
                self.config = M2M100Config(
                    vocab_size=len(tokenizer),
                    d_model=1024,
                    encoder_layers=8,
                    decoder_layers=8,
                    encoder_attention_heads=8,
                    decoder_attention_heads=8,
                    encoder_ffn_dim=2048,
                    decoder_ffn_dim=2048,
                    max_position_embeddings=512,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    activation_function="gelu",
                    scale_embedding=True,
                    attention_dropout=0.1,
                    activation_dropout=0.1,
                    dropout=0.3
                )
            else:
                raise ValueError("Invalid size. Choose 'small' or 'medium' or 'large'.")

        super().__init__(self.config)

        if generation_config:
            self.generation_config = generation_config
        else:
            self.generation_config = GenerationConfig(
                forced_eos_token_id=tokenizer.eos_token_id,
                num_beams=5,
                early_stopping=True,
            )
    
    @staticmethod
    def from_pretrained(path):
        return M2M100ForConditionalGeneration.from_pretrained(path)
    

class DynamicStartTokenCollator(DataCollatorForSeq2Seq):
    """
    This class is used only for collating the data for `Mixed` training.
    It replaces the first token of the decoder input with the first token of the labels.
    It should be used with M2M100Tokenizer.
    """
    def __call__(self, features):
        batch = super().__call__(features)
        # decoder_start_token_id is the first token of the labels
        # So we get a list of it from all labels
        # For example, labels = [[40096, 55, 3], [40053, 55, 3]]
        # decoder_start_token_id = [40096, 40053]
        decoder_start_token_id = [label[0] for label in batch["labels"]]

        # If decoder_input_ids is in batch, replace the first token with decoder_start_token_id
        if "decoder_input_ids" in batch:
            for i, decoder_input_id in enumerate(batch["decoder_input_ids"]):
                decoder_input_id[0] = decoder_start_token_id[i]

        return batch
    
def load_data(file_1, file_2, lang_1, lang_2, lowercase=True):
    """Load parallel data from Vietnamese and Lao files."""
    with open(file_1, 'r', encoding='utf-8') as f_vi:
        lines_1 = [line.strip().lower() if lowercase else line.strip() for line in f_vi.readlines()]

    with open(file_2, 'r', encoding='utf-8') as f_lo:
        lines_2 = [line.strip().lower() if lowercase else line.strip() for line in f_lo.readlines()]

    assert len(lines_1) == len(lines_2), f"Mismatch in line counts: {len(lines_1)} vs {len(lines_2)}"

    return {
        lang_1: lines_1,
        lang_2: lines_2
    }

def create_datasets(
        train_lang1_file,
        train_lang2_file,
        dev_lang1_file,
        dev_lang2_file,
        test_lang1_file,
        test_lang2_file,
        train_add_lang1_file=None,
        train_add_lang2_file=None,
        lang_1='vi',
        lang_2='lo',
        lowercase=True
):
    """Create train, dev, and test datasets."""
    train_data = load_data(train_lang1_file, train_lang2_file, lang_1, lang_2, lowercase)
    dev_data = load_data(dev_lang1_file, dev_lang2_file, lang_1, lang_2, lowercase)
    test_data = load_data(test_lang1_file, test_lang2_file, lang_1, lang_2, lowercase)

    if train_add_lang1_file and train_add_lang2_file:
        train_add_data = load_data(train_add_lang1_file, train_add_lang2_file, lang_1, lang_2, lowercase)
        train_data[lang_1].extend(train_add_data[lang_1])
        train_data[lang_2].extend(train_add_data[lang_2])

    datasets = DatasetDict({
        'train': Dataset.from_dict(train_data),
        'validation': Dataset.from_dict(dev_data),
        'test': Dataset.from_dict(test_data)
    })

    datasets['train'] = datasets['train'].shuffle(seed=42)

    return datasets


def train_sentencepiece_model(texts, vocab_size=40000, model_prefix='vi_lo'):
    with open('combined_text.txt', 'w', encoding='utf-8') as f:
        for line in texts:
            f.write(line + '\n')

    spm.SentencePieceTrainer.train(
        input='combined_text.txt',
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=0.9995,
        model_type='unigram',
        normalization_rule_name='nmt_nfkc',
        add_dummy_prefix=False,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3
    )

    os.remove('combined_text.txt')

    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")

    vocab = {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}
    with open(f"{model_prefix}.json", 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False)

    return f"{model_prefix}.model", f"{model_prefix}.json"


def train_shared_sentencepiece_model(datasets, vocab_size, langs=['vi', 'lo']):
    """Train a SentencePiece model on both Vietnamese and Lao data."""
    all_text = []

    for lang_code in langs:
        all_text.extend(datasets['train'][lang_code])

    return train_sentencepiece_model(all_text, vocab_size, model_prefix='vi_lo')


def train_separated_sentencepiece_model(datasets, vocab_size, lang):
    """Train a SentencePiece model on Vietnamese or Lao data."""
    all_text = datasets['train'][lang]

    return train_sentencepiece_model(all_text, vocab_size, model_prefix=lang)

def preprocess_function(examples, tokenizer, lang_1, lang_2, mix, lang_1_src_prob=0.4, max_length=128):
    """Map the examples (sentences) to input_ids and labels."""
    if mix:
        if np.random.rand() < lang_1_src_prob:
            src_lang = lang_1
            tgt_lang = lang_2
        else:
            src_lang = lang_2
            tgt_lang = lang_1
    else:
        src_lang = lang_1
        tgt_lang = lang_2

    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang

    return tokenizer(
        examples[src_lang],
        text_target=examples[tgt_lang],
        max_length=max_length,
        padding="max_length",
        truncation=True
    )

def prepare_datasets_for_training(
        datasets,
        tokenizer,
        batch_size,
        src_lang='vi',
        tgt_lang='lo',
        mix=False,
        src_prob=0.4,
        max_length=128,
        num_proc=8
):
    """This function prepares the datasets for training only."""
    # Need only datasets['train'] and datasets['validation']
    train_eval_datasets = DatasetDict({
        'train': datasets['train'],
        'validation': datasets['validation']
    })

    tokenized_datasets = train_eval_datasets.map(
        lambda examples: preprocess_function(
            examples, tokenizer, lang_1=src_lang, lang_2=tgt_lang,
            mix=mix, lang_1_src_prob=src_prob, max_length=max_length
        ),
        batched=True,
        batch_size=batch_size,
        remove_columns=[src_lang, tgt_lang],
        num_proc=num_proc
    )
    return tokenized_datasets

def train_model(
        model,
        tokenizer,
        datasets,
        multilingual=False,
        src_lang='vi',
        tgt_lang='lo',
        batch_size=128,
        output_dir="m2m100_vi_lo_model",
        checkpoint=None,
        num_proc=8
):
    """
    Train the M2M100 model on the prepared datasets.
    Args:
        model: The M2M100 model to train, should be `M2M100ForConditionalGeneration`.
        tokenizer: The tokenizer for the model, should be `M2M100Tokenizer`.
        datasets: The datasets to train on.
        multilingual: Whether to use multilingual training or not. 
            If `True`, the training data will be mixed
            so as to train the model to translate both directions (src->tgt and tgt->src).
            If `False`, the model will be trained to translate only from src to tgt.
        src_lang: Source language code (default is 'vi').
        tgt_lang: Target language code (default is 'lo').
        batch_size: Batch size for training (default is 128).
        output_dir: Directory to save the trained model (default is "m2m100_vi_lo_model").
        checkpoint: Path to a checkpoint to resume training from (default is None).
        num_proc: Number of processes for tokenization (default is 8).
    """
    training_args = Seq2SeqTrainingArguments(
        report_to="tensorboard",
        output_dir=output_dir,
        eval_strategy="epoch",
        # eval_steps=1000,
        learning_rate=1e-4,
        warmup_steps=1500,
        lr_scheduler_type="linear",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=10,
        num_train_epochs=15,
        predict_with_generate=True,
        fp16=True,
        save_strategy="epoch",
        # save_steps=1000,
        logging_dir="./logs",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=False,
        gradient_accumulation_steps=1,
        max_grad_norm=0.75,
        torch_compile=True
    )

    print('Preparing datasets for training...')
    src_prob=0.4
    if multilingual:
        data_collator = DynamicStartTokenCollator(
            tokenizer,
            model=model,
            padding="longest",
            return_tensors="pt",
            label_pad_token_id=tokenizer.pad_token_id,
        )

        tokenized_datasets = prepare_datasets_for_training(
            datasets,
            tokenizer,
            batch_size,
            mix=True,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            src_prob=src_prob,
            num_proc=num_proc
        )
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            padding="longest",
            return_tensors="pt",
            label_pad_token_id=tokenizer.pad_token_id,
        )

        tokenized_datasets = prepare_datasets_for_training(
            datasets,
            tokenizer,
            batch_size,
            mix=False,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            num_proc=num_proc
        )

    torch.cuda.empty_cache()
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Trainer created successfully and will be working on:", training_args.device)
    if multilingual:
        print(f'Training machine translation with mixed {src_prob} {src_lang}->{tgt_lang} and {1 - src_prob} {tgt_lang}->{src_lang}...')
    else:
        print(f'Training machine translation from {src_lang} to {tgt_lang}...')
    torch.cuda.empty_cache()
    if checkpoint:
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()
    return trainer


def evaluate_model_with_trainer(
        trainer, datasets, tokenizer, src_lang='vi', tgt_lang='lo',
        multilingual=False, batch_size=16, num_proc=8
):
    """
    Evaluate the trained model on the test set.
    Args:
        trainer: The trained Seq2SeqTrainer.
        datasets: The datasets to evaluate on.
        tokenizer: The tokenizer for the model, should be `M2M100Tokenizer`.
        src_lang: Source language code (default is 'vi').
        tgt_lang: Target language code (default is 'lo').
        multilingual: Whether to use multilingual evaluation or not. 
            If `True`, the test data will be mixed
            so as to evaluate the model on both directions (src->tgt and tgt->src).
            If `False`, the model will be evaluated only on src to tgt.
        batch_size: Batch size for evaluation (default is 16).
        num_proc: Number of processes for tokenization (default is 8).
    """
    print("Evaluating model on test set...")

    def test_with_trainer(test_datasets, tgt_lang):
        torch.cuda.empty_cache()
        with torch.no_grad():
            test_results = trainer.predict(
                test_datasets,
                decoder_start_token_id = tokenizer.lang_code_to_id[tgt_lang],
                max_new_tokens=128
            )

        preds = test_results.predictions
        labels = test_results.label_ids

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        pred_file = f'pred_{tgt_lang}.txt'
        label_file = f'label_{tgt_lang}.txt'
        with open(pred_file, 'w', encoding='utf-8') as f_pred, open(label_file, 'w', encoding='utf-8') as f_label:
            for pred, label in zip(decoded_preds, decoded_labels):
                f_pred.write(pred + '\n')
                f_label.write(label + '\n')


    if multilingual:
        # If multilingual, separate test datasets into two halfs
        datasets_test_1, datasets_test_2 = datasets["test"].train_test_split(test_size=0.5, seed=42).values()

        # lang_1 is source language, lang_2 is target language
        datasets_test_1 = datasets_test_1.map(
            lambda examples: preprocess_function(
                examples, tokenizer, lang_1=src_lang, lang_2=tgt_lang,
                mix=False, max_length=128
            ),
            batched=True,
            batch_size=batch_size,
            remove_columns=[src_lang, tgt_lang],
            num_proc=num_proc
        )

        # lang_2 is source language, lang_1 is target language
        datasets_test_2 = datasets_test_2.map(
            lambda examples: preprocess_function(
                examples, tokenizer, lang_1=tgt_lang, lang_2=src_lang,
                mix=False, max_length=128
            ),
            batched=True,
            batch_size=batch_size,
            remove_columns=[src_lang, tgt_lang],
            num_proc=num_proc
        )

        test_with_trainer(
            test_datasets=datasets_test_1,
            tgt_lang=tgt_lang
        )

        test_with_trainer(
            test_datasets=datasets_test_2,
            tgt_lang=src_lang
        )

    else:
        test_dataset = datasets["test"].map(
            lambda examples: preprocess_function(
                examples, tokenizer, lang_1=src_lang, lang_2=tgt_lang,
                mix=False, max_length=128
            ),
            batched=True,
            batch_size=batch_size,
            remove_columns=[src_lang, tgt_lang],
            num_proc=num_proc
        )

        test_with_trainer(
            test_datasets=test_dataset,
            tgt_lang=tgt_lang
        )


def evaluate_model(
        datasets, model, tokenizer, src_lang='vi', tgt_lang='lo',
        multilingual=False, batch_size=32, num_proc=8
):
    """
    Evaluate the trained model on the test set.
    Args:
        datasets: The datasets to evaluate on.
        model: The trained M2M100 model.
        tokenizer: The tokenizer for the model, should be `M2M100Tokenizer`.
        src_lang: Source language code (default is 'vi').
        tgt_lang: Target language code (default is 'lo').
        multilingual: Whether to use multilingual evaluation or not. 
            If `True`, the test data will be mixed
            so as to evaluate the model on both directions (src->tgt and tgt->src).
            If `False`, the model will be evaluated only on src to tgt.
        batch_size: Batch size for evaluation (default is 32).
        num_proc: Number of processes for tokenization (default is 8).
    """
    device='cuda' if torch.cuda.is_available() else 'cpu'
    print('Evaluating on: ', device)

    model.to(device)
    model.eval()

    def collate_fn(batch):
        return {
            'input_ids': torch.tensor([example['input_ids'] for example in batch]),
            'attention_mask': torch.tensor([example['attention_mask'] for example in batch]),
            'labels': torch.tensor([example['labels'] for example in batch]),
        }

    def test(test_datasets, tgt_lang):
        dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, collate_fn=collate_fn)

        decoded_preds = []
        decoded_labels = []

        torch.cuda.empty_cache()

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=128,
                    forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
                    decoder_start_token_id = tokenizer.lang_code_to_id[tgt_lang]
                )

                decoded_batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                decoded_batch_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                decoded_preds.extend(decoded_batch_preds)
                decoded_labels.extend(decoded_batch_labels)

        pred_file = f'pred_{tgt_lang}.txt'
        label_file = f'label_{tgt_lang}.txt'
        with open(pred_file, 'w', encoding='utf-8') as f_pred, open(label_file, 'w', encoding='utf-8') as f_label:
            for pred, label in zip(decoded_preds, decoded_labels):
                f_pred.write(pred + '\n')
                f_label.write(label + '\n')

    if multilingual:
        datasets_test_1, datasets_test_2 = datasets["test"].train_test_split(test_size=0.5, seed=42).values()
        # lang_1 is source language, lang_2 is target language
        datasets_test_1 = datasets_test_1.map(
            lambda examples: preprocess_function(
                examples, tokenizer, lang_1=src_lang, lang_2=tgt_lang,
                mix=False, max_length=128
            ),
            batched=True,
            batch_size=batch_size,
            remove_columns=[src_lang, tgt_lang],
            num_proc=num_proc
        )

        # lang_2 is source language, lang_1 is target language
        datasets_test_2 = datasets_test_2.map(
            lambda examples: preprocess_function(
                examples, tokenizer, lang_1=tgt_lang, lang_2=src_lang,
                mix=False, max_length=128
            ),
            batched=True,
            batch_size=batch_size,
            remove_columns=[src_lang, tgt_lang],
            num_proc=num_proc
        )

        test(
            test_datasets=datasets_test_1,
            tgt_lang=tgt_lang
        )

        test(
            test_datasets=datasets_test_2,
            tgt_lang=src_lang
        )
    else:
        test_dataset = datasets["test"].map(
            lambda examples: preprocess_function(
                examples, tokenizer, lang_1=src_lang, lang_2=tgt_lang, mix=False
            ),
            batched=True,
            batch_size=batch_size,
            remove_columns=[src_lang, tgt_lang],
            num_proc=4
        )

        test(
            test_datasets=test_dataset,
            tgt_lang=tgt_lang
        )

@torch.no_grad()
def translate_text(text, model, tokenizer, src_lang="vi", tgt_lang="lo", max_length=128):
    """Translate text using the trained model."""
    current_device = model.device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang

    text = text.lower()
    inputs = tokenizer(text, return_tensors="pt", padding=False).to(device)

    forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang]
    decoder_start_token_id = tokenizer.lang_code_to_id[tgt_lang]

    outputs = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        decoder_start_token_id = decoder_start_token_id,
        num_beams=5,
        early_stopping=True,
        max_new_tokens=max_length,
    )
    translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    # Move model back to original device
    model.to(current_device)
    return translation


def main(output_dir="m2m100_vi_lo_model"):
    """Main function to run the entire process."""
    os.makedirs(output_dir, exist_ok=True)
    torch.cuda.empty_cache()

    # 1. Load datasets
    print("Loading datasets...")
    datasets = create_datasets(
        train_lang1_file=TRAIN_VI_FILE,
        train_lang2_file=TRAIN_LO_FILE,
        dev_lang1_file=DEV_VI_FILE,
        dev_lang2_file=DEV_LO_FILE,
        test_lang1_file=TEST_VI_FILE,
        test_lang2_file=TEST_LO_FILE,
        train_ggtrans_lang1_file=TRAIN_GGTRANS_VI_FILE,
        train_ggtrans_lang2_file=TRAIN_GGTRANS_LO_FILE,
        train_gemini_lang1_file=TRAIN_GEMINI_VI_FILE,
        train_gemini_lang2_file=TRAIN_GEMINI_LO_FILE,
        lang_1='vi',
        lang_2='lo'
    )
    print(f"Loaded {len(datasets['train'])} training examples, {len(datasets['validation'])} validation examples, {len(datasets['test'])} test examples")

    # 2. Train SentencePiece model and create tokenizer
    # print("Training SentencePiece model...")
    # spm_file, vocab_file = train_shared_sentencepiece_model(
    #     datasets,
    #     vocab_size=40000
    # )

    print("Creating tokenizer and model...")
    tokenizer = Tokenizer(
        '/kaggle/input/m2m100-tokenizer-200/transformers/default/1/tokenizer-200/vocab.json',
        '/kaggle/input/m2m100-tokenizer-200/transformers/default/1/tokenizer-200/sentencepiece.model'
    )
    model = Model.from_pretrained('/kaggle/input/m2m100-lo2vi-1-0/model-lo2vi-11730')

    batch_size = 64
    num_proc = 4 # Number of processes for dataset mapping
    trainer = train_model(
        model,
        datasets=datasets,
        tokenizer=tokenizer,
        multilingual=False,
        src_lang='lo',
        tgt_lang='vi',
        batch_size=batch_size,
        num_proc=num_proc
    )

    # print("Evaluating the model...")
    evaluate_model_with_trainer(
        trainer, datasets, tokenizer,
        src_lang='lo', tgt_lang='vi',
        multilingual=False,
        batch_size=batch_size // 2,
        num_proc=num_proc
    )



    print("Translating a sample text...")
    sample_texts = [
        "ມື້​ນີ້​ເປັນ​ມື້​ທີ່​ສວຍ​ງາມ​!",
        "ໂຮງຮຽນແຫ່ງນີ້ ມີນັກຮຽນທັງໝົດ 200 ຄົນ, ເປັນຕົວເລກທີ່ຂ້ອນຂ້າງໜ້ອຍເມື່ອທຽບກັບຕົວເມືອງ.",
        "ລາວ​ໄດ້​ຖືກ​ຕັດ​ສິນ​ໂທດ​ໃນ​ຂໍ້​ຫາ​ບາດ​ເຈັບ​ໂດຍ​ເຈດ​ຕະ​ນາ ແລະ​ຖືກ​ຕັດ​ສິນ​ຈຳ​ຄຸກ​ສາມ​ປີ.",
    ]

    for text in sample_texts:
        translation = translate_text(
            text,
            model=model,
            tokenizer=tokenizer,
            src_lang='lo',
            tgt_lang='vi',
            max_length=128
        )
        print(f"Original: {text}")
        print(f"Translated: {translation}")
        print("-" * 50)
        
    # evaluate_model(
    #     datasets, model, tokenizer, src_lang='lo', tgt_lang='vi',
    #     multilingual=False, batch_size=32, num_proc=8
    # )
    print('Done!')


if __name__ == "__main__":
    main()