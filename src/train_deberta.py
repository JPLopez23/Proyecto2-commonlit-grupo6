"""Ajuste fino de DeBERTa-v3 con salida doble (content, wording) y validación
Leave-One-Prompt-Out. Pensado para correr en GPU (Kaggle / Colab); también funciona en CPU
con un modelo diminuto para pruebas.

Genera:
  <out_dir>/deberta_<tag>_oof.csv       predicciones fuera de muestra (una por resumen)
  <out_dir>/deberta_<tag>_metrics.json  MCRMSE por texto, tiempos y configuración
  <out_dir>/deberta_<tag>_final/        modelo ajustado con los 4 textos (para la app)

Decisiones (ver informe): salida de regresión de 2 neuronas, pérdida MSE, mean pooling,
dropout desactivado (recomendado en regresión), número de épocas FIJO (no se elige la
mejor época mirando el texto que se evalúa, para no filtrar información).
"""
from __future__ import annotations

import contextlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

TARGETS = ["content", "wording"]


@dataclass
class Config:
    model_name: str = "microsoft/deberta-v3-small"
    tag: str = "small"
    data_dir: str = "data"
    out_dir: str = "results"
    max_len: int = 512
    epochs: int = 3
    batch_size: int = 8
    lr: float = 3e-5            # backbone
    head_lr: float = 1e-3       # capa de regresión
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    include_source: bool = False  # True: añade el texto fuente (truncado) a la entrada
    seed: int = 42
    fp16: bool = True
    subsample: int = 0            # >0: usar solo N filas por texto (pruebas rápidas)
    only_folds: str = ""          # p.ej. "ebad26" para depurar un solo pliegue
    skip_final: bool = False


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    """GPU NVIDIA (CUDA) si hay; si no, GPU de Apple (MPS); si no, CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_ctx(device: torch.device, enabled: bool):
    """Precisión mixta solo en CUDA; en MPS/CPU se entrena en float32."""
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def clean(t: str) -> str:
    return " ".join(str(t).replace("\r", " ").replace("\n", " ").split())


class Regressor(nn.Module):
    def __init__(self, model_name: str, from_pretrained: bool = True):
        super().__init__()
        cfg = AutoConfig.from_pretrained(model_name)
        cfg.hidden_dropout_prob = 0.0
        cfg.attention_probs_dropout_prob = 0.0
        self.backbone = AutoModel.from_pretrained(model_name, config=cfg) if from_pretrained \
            else AutoModel.from_config(cfg)
        self.backbone = self.backbone.float()   # evita "Half and Float" si el checkpoint viene en fp16
        self.head = nn.Linear(cfg.hidden_size, 2)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(pooled)


def make_collate(tok):
    """Rellena input_ids/attention_mask y apila las etiquetas aparte."""
    def collate(batch):
        labels = [b.pop("labels") for b in batch if "labels" in b]
        out = tok.pad(batch, padding=True, return_tensors="pt")
        if labels:
            out["labels"] = torch.stack(labels)
        return out
    return collate


class TextData(torch.utils.data.Dataset):
    def __init__(self, enc, y=None):
        self.enc, self.y = enc, y

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items() if k != "token_type_ids"}
        if self.y is not None:
            item["labels"] = torch.tensor(self.y[i], dtype=torch.float)
        return item


def encode(tok, df: pd.DataFrame, cfg: Config):
    if cfg.include_source:
        pair = (df["prompt_question"].map(clean) + f" {tok.sep_token} " + df["prompt_text"].map(clean)).tolist()
        trunc = "only_second"
    else:
        pair = df["prompt_question"].map(clean).tolist()
        trunc = "longest_first"
    return tok(df["text"].map(clean).tolist(), pair, truncation=trunc, max_length=cfg.max_len)


@torch.no_grad()
def predict(model, tok, enc, device, cfg: Config) -> np.ndarray:
    order = np.argsort([len(x) for x in enc["input_ids"]])
    ds = TextData({k: [enc[k][i] for i in order] for k in enc})
    dl = DataLoader(ds, batch_size=cfg.batch_size * 2, collate_fn=make_collate(tok))
    model.eval()
    out = []
    for b in dl:
        b = {k: v.to(device) for k, v in b.items()}
        with amp_ctx(device, cfg.fp16):
            out.append(model(**b).float().cpu().numpy())
    pred = np.concatenate(out)
    res = np.zeros_like(pred)
    res[order] = pred
    return res


def train_one(train_df: pd.DataFrame, cfg: Config, tok, device, log_prefix: str = ""):
    seed_everything(cfg.seed)
    model = Regressor(cfg.model_name, from_pretrained=True).to(device)
    enc = encode(tok, train_df, cfg)
    y = train_df[TARGETS].to_numpy(dtype=np.float32)
    dl = DataLoader(TextData(enc, y), batch_size=cfg.batch_size, shuffle=True,
                    collate_fn=make_collate(tok), drop_last=False)
    no_decay = ("bias", "LayerNorm.weight")
    groups = [
        {"params": [p for n, p in model.backbone.named_parameters() if not any(k in n for k in no_decay)],
         "lr": cfg.lr, "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in model.backbone.named_parameters() if any(k in n for k in no_decay)],
         "lr": cfg.lr, "weight_decay": 0.0},
        {"params": model.head.parameters(), "lr": cfg.head_lr, "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups)
    total = cfg.epochs * len(dl)
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_ratio * total), total)
    use_amp = cfg.fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = nn.MSELoss()
    step = 0
    for ep in range(cfg.epochs):
        model.train()
        run = 0.0
        for i, b in enumerate(dl):
            labels = b.pop("labels").to(device)
            b = {k: v.to(device) for k, v in b.items()}
            with amp_ctx(device, use_amp):
                pred = model(**b)
            loss = loss_fn(pred.float(), labels)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
            step += 1
        print(f"{log_prefix} época {ep + 1}/{cfg.epochs}  pérdida media={run / len(dl):.4f}", flush=True)
    return model


def mcrmse(y, p) -> float:
    return float(np.mean(np.sqrt(np.mean((y - p) ** 2, axis=0))))


def run(cfg: Config) -> dict:
    device = pick_device()
    print("dispositivo:", device, "| modelo:", cfg.model_name)
    data, out = Path(cfg.data_dir), Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    S = pd.read_csv(data / "summaries_train.csv")
    P = pd.read_csv(data / "prompts_train.csv")
    df = S.merge(P, on="prompt_id", how="left").reset_index(drop=True)
    if cfg.subsample:
        df = df.groupby("prompt_id", group_keys=False).head(cfg.subsample).reset_index(drop=True)
    tok = AutoTokenizer.from_pretrained(cfg.model_name)

    oof = df[["student_id", "prompt_id"] + TARGETS].copy()
    oof[[f"deberta_{cfg.tag}__content", f"deberta_{cfg.tag}__wording"]] = np.nan
    per_prompt, times = {}, {}
    folds = [p for p in sorted(df["prompt_id"].unique()) if not cfg.only_folds or p in cfg.only_folds.split(",")]
    for g in folds:
        t0 = time.time()
        tr, te = df[df["prompt_id"] != g], df[df["prompt_id"] == g]
        model = train_one(tr, cfg, tok, device, log_prefix=f"[fuera {g}]")
        pred = predict(model, tok, encode(tok, te, cfg), device, cfg)
        oof.loc[te.index, [f"deberta_{cfg.tag}__content", f"deberta_{cfg.tag}__wording"]] = pred
        per_prompt[g] = mcrmse(te[TARGETS].to_numpy(), pred)
        times[g] = time.time() - t0
        print(f"[fuera {g}] MCRMSE={per_prompt[g]:.4f}  ({times[g]:.0f}s)", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()
        oof.to_csv(out / f"deberta_{cfg.tag}_oof.csv", index=False)  # guarda tras cada pliegue

    done = oof.dropna()
    metrics = {
        "config": asdict(cfg),
        "mcrmse_mean_prompts": float(np.mean(list(per_prompt.values()))),
        "mcrmse_pooled": mcrmse(done[TARGETS].to_numpy(),
                                done[[f"deberta_{cfg.tag}__content", f"deberta_{cfg.tag}__wording"]].to_numpy()),
        "per_prompt": per_prompt,
        "fold_seconds": times,
    }
    if not cfg.skip_final:
        t0 = time.time()
        model = train_one(df, cfg, tok, device, log_prefix="[final]")
        final = out / f"deberta_{cfg.tag}_final"
        final.mkdir(exist_ok=True)
        model.backbone.save_pretrained(final / "backbone")
        torch.save(model.head.state_dict(), final / "head.pt")
        tok.save_pretrained(final / "tokenizer")
        (final / "config.json").write_text(json.dumps(
            {"model_name": cfg.model_name, "max_len": cfg.max_len, "include_source": cfg.include_source}))
        metrics["final_seconds"] = time.time() - t0
    (out / f"deberta_{cfg.tag}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ("mcrmse_mean_prompts", "mcrmse_pooled", "per_prompt")}, indent=2))
    return metrics


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    for f, v in asdict(Config()).items():
        t = type(v)
        if t is bool:
            ap.add_argument(f"--{f}", type=lambda s: s.lower() in ("1", "true", "yes"), default=v)
        else:
            ap.add_argument(f"--{f}", type=t, default=v)
    run(Config(**vars(ap.parse_args())))
