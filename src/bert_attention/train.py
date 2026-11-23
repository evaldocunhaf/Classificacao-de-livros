import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from src.evaluate.metrics import save_eval

from .model import AttentionCDDClassifier, BACKBONE

BASE = Path(__file__).resolve().parents[2]
MAX_LEN = 128


def _texts(df: pd.DataFrame) -> list[str]:
    rows = []
    for _, r in df.iterrows():
        t = str(r["titulo"]).strip()
        res = str(r["resumo"]).strip()
        res = "" if res.lower() in ("nan", "none") else res
        rows.append(f"{t}\n{res}" if res else t)
    return rows


class CDDDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, labels: list[str]):
        self.input_ids, self.attention_mask = [], []
        enc = tokenizer(
            _texts(df),
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        index = {c: i for i, c in enumerate(labels)}
        self.labels = torch.tensor([index.get(c, -1) for c in df["cdd"].tolist()])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.input_ids[i], self.attention_mask[i], self.labels[i]


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_dirs(run_dir: str | None) -> tuple[Path, Path, Path]:
    if run_dir is None:
        base = BASE / "output"
        return base / "prepared", base / "results", base / "models"
    root = Path(run_dir)
    if not root.is_absolute():
        root = BASE / root
    return root / "prepared", root / "results", root / "models"


@torch.no_grad()
def predict_probabilities(
    df: pd.DataFrame,
    checkpoint_dir: Path,
    batch_size: int = 16,
) -> tuple[list[str], np.ndarray]:
    """Carrega um checkpoint F e retorna probabilidades alinhadas aos rótulos.

    Não treina nem altera o checkpoint: é usado para combinações posteriores,
    como o ensemble com o baseline kNN.
    """
    checkpoint_dir = Path(checkpoint_dir)
    labels = json.loads((checkpoint_dir / "labels.json").read_text(encoding="utf-8"))
    device = _device()
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    model = AttentionCDDClassifier(n_classes=len(labels)).to(device)
    checkpoint = checkpoint_dir / "model.pt"
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:  # compatibilidade com versões anteriores do PyTorch
        state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    loader = DataLoader(CDDDataset(df, tokenizer, labels), batch_size=batch_size)
    probabilities = []
    for input_ids, mask, _ in loader:
        logits = model(input_ids.to(device), mask.to(device))
        probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return labels, np.concatenate(probabilities, axis=0)


def prediction_frame(df: pd.DataFrame, labels: list[str], probabilities: np.ndarray) -> pd.DataFrame:
    """Converte probabilidades do XLM-R em previsões auditáveis."""
    if len(df) != len(probabilities):
        raise ValueError("número de probabilidades não corresponde ao split")
    columns = ["titulo", "source", "edition_id", "work_id", "idioma", "cdd"]
    result = df[[column for column in columns if column in df]].copy()
    result = result.rename(columns={"cdd": "cdd_true"})
    winner = probabilities.argmax(axis=1)
    result["cdd_pred"] = [labels[index] for index in winner]
    result["method"] = "xlmr_attention"
    result["score_xlmr"] = probabilities.max(axis=1).round(6)
    return result.reset_index(drop=True)


@torch.no_grad()
def _eval_step(model, loader, device) -> tuple[float, float]:
    model.eval()
    tot, hits = 0, 0
    loss = 0.0
    crit = torch.nn.CrossEntropyLoss()
    for input_ids, mask, labels in loader:
        input_ids, mask, labels = input_ids.to(device), mask.to(device), labels.to(device)
        logits = model(input_ids, mask)
        valid = labels >= 0
        if valid.any():
            loss += crit(logits[valid], labels[valid]).item() * valid.sum().item()
        hits += (logits.argmax(dim=-1) == labels).sum().item()
        tot += len(labels)
    return loss / tot, hits / tot


def train(args) -> str:
    _set_seed(args.seed)
    device = _device()
    print(f"device: {device}")

    prepared, results_dir, models_dir = _run_dirs(args.run_dir)
    train_df = pd.read_csv(prepared / "train.csv", dtype=str)
    val_df = pd.read_csv(prepared / "val.csv", dtype=str)
    labels = sorted(train_df["cdd"].unique())
    print(f"labels ({len(labels)}): {labels}")

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    model = AttentionCDDClassifier(
        n_classes=len(labels), unfreeze_last=args.unfreeze_last
    ).to(device)
    if args.use_weights:
        weights = compute_class_weight(
            "balanced", classes=np.array(labels), y=train_df["cdd"].values
        )
        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        print("pesos das classes:", dict(zip(labels, [round(float(w), 2) for w in weights])))
    else:
        weight_tensor = None
    criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)

    opt = torch.optim.AdamW(
        model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    train_loader = DataLoader(
        CDDDataset(train_df, tokenizer, labels), batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(CDDDataset(val_df, tokenizer, labels), batch_size=args.batch_size)
    majority = val_df["cdd"].value_counts().index[0]
    ref = float((val_df["cdd"] == majority).mean())
    print(f"baseline val (constante={majority}): acc {ref:.4f} — precisamos superar isso")

    best_acc, best_state, patience = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = 0.0
        for input_ids, mask, y in train_loader:
            input_ids, mask, y = input_ids.to(device), mask.to(device), y.to(device)
            opt.zero_grad()
            logits = model(input_ids, mask)
            valid = y >= 0
            loss = criterion(logits[valid], y[valid]) if valid.any() else torch.zeros((), device=device)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * valid.sum().item()
        tr_loss = ep_loss / len(train_loader.dataset)
        va_loss, va_acc = _eval_step(model, val_loader, device)
        print(
            f"epoch {epoch:02d} | train_loss {tr_loss:.4f} | val_loss {va_loss:.4f} | val_acc {va_acc:.4f}"
        )
        if va_acc > best_acc:
            best_acc, patience = va_acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"  * novo best val_acc {best_acc:.4f}")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"early stop em {epoch} (patience {args.patience})")
                break

    out_dir = models_dir / "bert_attention"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / "model.pt")
    (out_dir / "labels.json").write_text(json.dumps(labels, ensure_ascii=False))
    print(f"checkpoint salvo em {out_dir} (best val_acc {best_acc:.4f})")

    results_dir.mkdir(parents=True, exist_ok=True)
    val_labels, val_probabilities = predict_probabilities(val_df, out_dir, args.batch_size)
    if labels != val_labels:
        raise ValueError("rótulos do checkpoint divergiram durante a inferência de validação")
    val_result = prediction_frame(val_df, labels, val_probabilities)
    val_result.to_csv(results_dir / "exp_F_val.csv", index=False)
    val_report = save_eval(val_result, str(results_dir / "avaliacao_F_val"))
    print("validação:", json.dumps(val_report, ensure_ascii=False, indent=2))

    if args.evaluate_test:
        test_df = pd.read_csv(prepared / "test.csv", dtype=str)
        test_labels, test_probabilities = predict_probabilities(test_df, out_dir, args.batch_size)
        if labels != test_labels:
            raise ValueError("rótulos do checkpoint divergiram durante a inferência de teste")
        test_result = prediction_frame(test_df, labels, test_probabilities)
        test_result.to_csv(results_dir / "exp_F_test.csv", index=False)
        test_report = save_eval(test_result, str(results_dir / "avaliacao_F_test"))
        print("teste:", json.dumps(test_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERT + attention — Experimento F")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--use-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--unfreeze-last", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="aplica o checkpoint no teste; use somente após fechar a seleção na validação",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="diretório isolado com prepared/, results/ e models/",
    )
    args = parser.parse_args()
    train(args)
