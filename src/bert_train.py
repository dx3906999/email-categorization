"""
BERT training & evaluation for email categorization.

Usage:
    # Train
    python src/bert_train.py --train

    # Evaluate only (quick test on existing model)
    python src/bert_train.py --eval-only

    # Predict a single email
    python src/bert_train.py --predict "Verify your account" "Click here to login..."
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# --- config ---

CATEGORIES = ['adult_violence', 'commercial', 'phishing', 'finance', 'academic']
ID2LABEL = {i: c for i, c in enumerate(CATEGORIES)}
LABEL2ID = {c: i for i, c in enumerate(CATEGORIES)}

DEFAULT_MODEL = "hfl/chinese-roberta-wwm-ext"
DEFAULT_DATA = "data/processed"
DEFAULT_OUTPUT = "models/bert_classifier"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    """macro-F1 is the primary metric (data is imbalanced)."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
    }


class WeightedTrainer(Trainer):
    """Trainer with class-weighted loss for imbalanced data."""

    def __init__(self, class_weights: Optional[torch.Tensor] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(logits.device)
            loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    data_dir: str = DEFAULT_DATA,
    model_name: str = DEFAULT_MODEL,
    output_dir: str = DEFAULT_OUTPUT,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 2e-5,
    warmup_ratio: float = 0.1,
    max_length: int = 256,
    seed: int = 42,
):
    """Train BERT classifier."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    # ---- Load data ----
    if not data_dir.exists():
        print(f"[ERROR] 数据目录不存在: {data_dir}")
        print("  请先运行: python src/bert_dataset.py")
        sys.exit(1)

    dataset = load_from_disk(str(data_dir))
    print(f"Dataset loaded: {data_dir}")
    print(f"  train: {len(dataset['train'])}  val: {len(dataset['val'])}  test: {len(dataset['test'])}")

    # ---- Load class weights ----
    cw_path = data_dir / "class_weights.json"
    class_weights = None
    if cw_path.exists():
        cw = json.loads(cw_path.read_text())
        class_weights = torch.tensor([cw.get(c, 1.0) for c in CATEGORIES], dtype=torch.float32)
        print(f"Class weights: {dict(zip(CATEGORIES, class_weights.tolist()))}")

    # ---- Load model ----
    print(f"\nLoading model: {model_name}")
    num_labels = len(CATEGORIES)
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ---- Training args ----
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name="email-categorization",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=0.01,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        report_to="none",
        seed=seed,
        # disable wandb etc.
        logging_dir=str(output_dir / "logs"),
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # ---- Train ----
    print(f"\nTraining ({epochs} epochs, batch={batch_size}, lr={lr})...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  FP16: {torch.cuda.is_available()}")
    trainer.train()

    # ---- Save ----
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    # Also save label map for inference
    (output_dir / "label_map.json").write_text(
        json.dumps({"id2label": ID2LABEL, "label2id": LABEL2ID}, ensure_ascii=False, indent=2)
    )
    print(f"\nModel saved: {output_dir}")

    # ---- Evaluate on test set ----
    print(f"\n{'='*55}")
    print("  Test set evaluation")
    print(f"{'='*55}")
    test_result = trainer.evaluate(dataset["test"])
    for k, v in test_result.items():
        print(f"  {k}: {v:.4f}")

    # Classification report
    preds = trainer.predict(dataset["test"])
    y_pred = np.argmax(preds.predictions, axis=-1)
    y_true = preds.label_ids

    print(f"\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=CATEGORIES,
        zero_division=0,
    ))

    print(f"Confusion Matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_true, y_pred)
    header = "         " + "".join(f"{c:>6s}" for c in CATEGORIES)
    print(header)
    for i, cat in enumerate(CATEGORIES):
        print(f"  {cat:6s}  " + "".join(f"{cm[i][j]:>6d}" for j in range(len(CATEGORIES))))

    return trainer


# ---------------------------------------------------------------------------
# Inference (for integration with classifier.py)
# ---------------------------------------------------------------------------

class BertPredictor:
    """Lightweight predictor for use in EmailClassifier.classify_fused()."""

    def __init__(self, model_path: str = DEFAULT_OUTPUT,
                 model_name: str = DEFAULT_MODEL):
        self.model_path = Path(model_path)
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        print(f"Loading BERT model from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
        self.model.eval()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self._device = device
        self._loaded = True

    def predict(self, subject: str, content: str,
                from_name: str = "",
                max_length: int = 256) -> tuple[str, float]:
        """Predict category + confidence for a single email."""
        self._ensure_loaded()
        text = f"{from_name} [SEP] {subject} [SEP] {content}"
        inputs = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=max_length, return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        max_idx = int(torch.argmax(probs))
        confidence = float(probs[max_idx])
        category = ID2LABEL[max_idx]
        return category, confidence


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    p = argparse.ArgumentParser(description="BERT 训练与评估")
    p.add_argument("--train", action="store_true", help="训练模型")
    p.add_argument("--eval-only", action="store_true", help="仅评估已有模型")
    p.add_argument("--predict", nargs=2, metavar=("SUBJECT", "CONTENT"),
                   help="预测单封邮件")

    p.add_argument("--model", default=DEFAULT_MODEL, help="预训练模型")
    p.add_argument("--data-dir", default=DEFAULT_DATA, help="处理后的数据集目录")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT, help="模型保存目录")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    if args.predict:
        import os
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        predictor = BertPredictor(model_path=args.output_dir, model_name=args.model)
        cat, conf = predictor.predict(args.predict[0], args.predict[1])
        print(f"Category: {cat}  ({ID2LABEL.get(cat, '?')})")
        print(f"Confidence: {conf:.4f}")
        return

    if args.eval_only:
        dataset = load_from_disk(args.data_dir)
        print(f"Dataset: {args.data_dir}")
        print(f"  test: {len(dataset['test'])}")

        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer

        model = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir)

        trainer = Trainer(
            model=model,
            compute_metrics=compute_metrics,
        )
        result = trainer.evaluate(dataset["test"])
        print(f"\nTest results:")
        for k, v in result.items():
            print(f"  {k}: {v:.4f}")
        return

    if args.train:
        train(
            data_dir=args.data_dir,
            model_name=args.model,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
        )
        return

    p.print_help()


if __name__ == "__main__":
    main()
