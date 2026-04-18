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
)


BASE_DIR = Path(__file__).resolve().parents[1]
INTENTS_PATH = BASE_DIR / "data" / "intents.json"
OUTPUT_DIR = BASE_DIR / "models" / "bert_model"
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


def main() -> None:
    ensure_training_dependencies()

    rows, labels = load_training_rows()
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}

    dataset = Dataset.from_list(
        [{"text": row["text"], "label": label2id[row["label_name"]]} for row in rows]
    ).train_test_split(test_size=0.2, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True)

    tokenized = dataset.map(tokenize, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=4,
        weight_decay=0.01,
        load_best_model_at_end=True,
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))


if __name__ == "__main__":
    main()
