from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
import torch


BASE_DIR = Path(__file__).resolve().parents[1]
INTENTS_PATH = BASE_DIR / "backend" / "data" / "intents.json"
OUTPUT_DIR = BASE_DIR / "backend" / "models" / "bert_model"
BASE_MODEL = "distilbert-base-uncased"


def ensure_training_dependencies() -> None:
    try:
        import accelerate  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing training dependency: accelerate>=0.26.0\n"
            "Install backend dependencies again with:\n"
            "  pip install -r requirements.txt\n"
            "Or install only the missing package with:\n"
            "  pip install \"accelerate>=0.26.0\""
        ) from exc


def load_training_rows() -> tuple[list[dict[str, str]], list[str]]:
    with INTENTS_PATH.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)

    labels: list[str] = []
    rows: list[dict[str, str]] = []

    for intent in data["intents"]:
        label = intent["tag"]
        labels.append(label)
        for pattern in intent["patterns"]:
            rows.append({"text": pattern, "label_name": label})

    return rows, sorted(set(labels))


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )
    accuracy = accuracy_score(labels, predictions)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def stratified_split(dataset, test_size=0.2, seed=42):
    """Stratified split to maintain class balance"""
    # Get labels for stratification
    labels = dataset["label"]
    unique_labels = np.unique(labels)
    
    train_indices = []
    test_indices = []
    
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        n_test = max(1, int(len(label_indices) * test_size))
        
        # Shuffle indices
        np.random.seed(seed + int(label))
        np.random.shuffle(label_indices)
        
        test_indices.extend(label_indices[:n_test])
        train_indices.extend(label_indices[n_test:])
    
    train_dataset = dataset.select(train_indices)
    test_dataset = dataset.select(test_indices)
    
    return train_dataset, test_dataset


def main() -> None:
    ensure_training_dependencies()

    rows, labels = load_training_rows()
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}

    # Create dataset
    dataset = Dataset.from_list(
        [{"text": row["text"], "label": label2id[row["label_name"]]} for row in rows]
    )
    
    # Use stratified split for better balance
    train_dataset, eval_dataset = stratified_split(dataset, test_size=0.15, seed=42)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")
    print(f"Number of classes: {len(labels)}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    tokenized_train = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
    tokenized_eval = eval_dataset.map(tokenize, batched=True, remove_columns=["text"])
    
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    # Optimized training arguments for higher accuracy
    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="steps",
        eval_steps=200,  # Evaluate every 200 steps
        save_strategy="steps",
        save_steps=400,
        learning_rate=3e-5,  # Slightly higher learning rate
        per_device_train_batch_size=16,  # Increased batch size
        per_device_eval_batch_size=32,   # Larger eval batch
        num_train_epochs=6,  # More epochs for better convergence
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=50,
        warmup_steps=100,  # Warmup for stable training
        fp16=True,  # Use mixed precision for faster training
        dataloader_num_workers=2,  # Parallel data loading
        gradient_accumulation_steps=1,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Final evaluation
    print("\nFinal evaluation results:")
    eval_results = trainer.evaluate()
    print(eval_results)
    
    # Save the best model
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    
    print(f"Model saved to {OUTPUT_DIR}")
    print(f"Final accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"Final F1: {eval_results['eval_f1']:.4f}")


if __name__ == "__main__":
    main()
