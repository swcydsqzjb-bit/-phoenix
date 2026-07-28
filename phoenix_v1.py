from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.neighbors import NearestNeighbors

PROJECT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT / "phoenix_cache"
MODEL_FILE = PROJECT / "phoenix_model.joblib"
META_FILE = PROJECT / "phoenix_model_meta.json"
TRAIN_FILE = PROJECT / "phoenix_training_rows.csv"
REPORT_FILE = PROJECT / "phoenix_predictions.csv"

TARGET_RETURN = 0.09
RANDOM_STATE = 42


@dataclass
class DataBundle:
    features: pd.DataFrame
    latest: pd.DataFrame


def safe_div(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray, eps: float = 1e-12):
    return a / np.where(np.abs(b) < eps, np.nan, b)


def normalize_symbol(raw: str) -> str:
    s = str(raw).strip().upper()
    if not s or s == "NAN":
        return ""
    return s if s.endswith(".IS") else f"{s}.IS"


def load_symbols(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Sembol dosyası bulunamadı: {path}. Bir sütunlu symbols.csv oluşturun."
        )
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", header=None)
    except Exception:
        df = pd.read_csv(path, encoding="utf-8-sig")
    values = df.astype(str).stack().tolist()
    symbols = sorted({normalize_symbol(v) for v in values if normalize_symbol(v)})
    if not symbols:
        raise ValueError("Sembol dosyasında geçerli hisse kodu bulunamadı.")
    return symbols


def download_one(symbol: str, start: str, end: str | None, refresh: bool) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{symbol.replace('.', '_')}.csv"
    if cache.exists() and not refresh:
        try:
            df = pd.read_csv(cache, parse_dates=["Date"])
            if len(df) >= 20:
                return df.set_index("Date")
        except Exception:
            pass

    df = yf.download(
        symbol,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
        timeout=25,
    )
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    wanted = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in df.columns for c in wanted):
        return pd.DataFrame()
    df = df[wanted].copy().dropna(subset=["Open", "High", "Low", "Close"])
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    df.reset_index().to_csv(cache, index=False)
    return df


def day_features(df: pd.DataFrame, lag: int) -> dict[str, pd.Series]:
    o = df["Open"].shift(lag)
    h = df["High"].shift(lag)
    l = df["Low"].shift(lag)
    c = df["Close"].shift(lag)
    v = df["Volume"].shift(lag).astype(float)
    prev_c = df["Close"].shift(lag + 1)
    rng = (h - l).replace(0, np.nan)
    body = c - o
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    tag = f"d{lag}"
    return {
        f"{tag}_ret": c / prev_c - 1.0,
        f"{tag}_gap": o / prev_c - 1.0,
        f"{tag}_range": rng / prev_c,
        f"{tag}_body": body / rng,
        f"{tag}_upper_wick": upper / rng,
        f"{tag}_lower_wick": lower / rng,
        f"{tag}_close_pos": (c - l) / rng,
        f"{tag}_open_pos": (o - l) / rng,
        f"{tag}_log_volume": np.log1p(v),
    }


def build_symbol_rows(symbol: str, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tarih t satırı yalnızca t-3, t-2, t-1 verilerini kullanır; hedef t günüdür."""
    if len(df) < 12:
        return pd.DataFrame(), pd.DataFrame()

    out = pd.DataFrame(index=df.index)
    for lag in (3, 2, 1):
        for name, series in day_features(df, lag).items():
            out[name] = series

    # Tam üç günlük dizinin ilişkisel özellikleri. Başka hiçbir güne bakılmaz.
    c3, c2, c1 = (df["Close"].shift(3), df["Close"].shift(2), df["Close"].shift(1))
    h3, h2, h1 = (df["High"].shift(3), df["High"].shift(2), df["High"].shift(1))
    l3, l2, l1 = (df["Low"].shift(3), df["Low"].shift(2), df["Low"].shift(1))
    v3, v2, v1 = (df["Volume"].shift(3), df["Volume"].shift(2), df["Volume"].shift(1))
    r3 = (h3 - l3).replace(0, np.nan)
    r2 = (h2 - l2).replace(0, np.nan)
    r1 = (h1 - l1).replace(0, np.nan)

    out["close_slope_3d"] = c1 / c3 - 1.0
    out["higher_high_count"] = (h2 > h3).astype(float) + (h1 > h2).astype(float)
    out["higher_low_count"] = (l2 > l3).astype(float) + (l1 > l2).astype(float)
    out["range_compression_32"] = safe_div(r2, r3)
    out["range_compression_21"] = safe_div(r1, r2)
    out["volume_change_32"] = safe_div(v2, v3) - 1.0
    out["volume_change_21"] = safe_div(v1, v2) - 1.0
    out["volume_acceleration"] = safe_div(v1 * v3, np.square(v2)) - 1.0
    out["three_green_count"] = (
        (df["Close"].shift(3) > df["Open"].shift(3)).astype(float)
        + (df["Close"].shift(2) > df["Open"].shift(2)).astype(float)
        + (df["Close"].shift(1) > df["Open"].shift(1)).astype(float)
    )
    out["three_day_max_drawdown"] = np.minimum.reduce([
        np.zeros(len(df)),
        (c2 / c3 - 1.0).to_numpy(),
        (c1 / c3 - 1.0).to_numpy(),
    ])
    out["three_day_spread"] = (np.maximum.reduce([h3, h2, h1]) - np.minimum.reduce([l3, l2, l1])) / c3

    today_ret = df["Close"] / df["Close"].shift(1) - 1.0
    out["target"] = (today_ret >= TARGET_RETURN).astype(int)
    out["target_return"] = today_ret
    out["symbol"] = symbol.replace(".IS", "")
    out["date"] = out.index

    feature_cols = [c for c in out.columns if c not in {"target", "target_return", "symbol", "date"}]
    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan)

    train = out.dropna(subset=feature_cols + ["target_return"]).copy()
    latest = out.dropna(subset=feature_cols).tail(1).copy()
    return train, latest


def create_dataset(symbols: Iterable[str], start: str, end: str | None, refresh: bool) -> DataBundle:
    rows: list[pd.DataFrame] = []
    latest_rows: list[pd.DataFrame] = []
    symbols = list(symbols)
    for i, symbol in enumerate(symbols, start=1):
        try:
            df = download_one(symbol, start, end, refresh)
            train, latest = build_symbol_rows(symbol, df)
            if not train.empty:
                rows.append(train)
            if not latest.empty:
                latest_rows.append(latest)
        except Exception as exc:
            print(f"[{i}/{len(symbols)}] {symbol}: hata: {exc}")
        if i % 25 == 0 or i == len(symbols):
            print(f"Veri: {i}/{len(symbols)} sembol işlendi.")
        time.sleep(0.05)

    if not rows:
        raise RuntimeError("Eğitim için veri üretilemedi.")
    return DataBundle(pd.concat(rows, ignore_index=True), pd.concat(latest_rows, ignore_index=True))


def choose_threshold(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, dict]:
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    candidates = []
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        predicted = int((prob >= t).sum())
        if predicted >= 10 and r >= 0.03:
            # Öncelik yanlış alarmı azaltmak; eşitlikte daha fazla yakalama.
            candidates.append((float(p), float(r), float(t), predicted))
    if candidates:
        best = max(candidates, key=lambda x: (x[0], x[1]))
        return best[2], {"precision": best[0], "recall": best[1], "signals": best[3]}
    return 0.50, {"precision": 0.0, "recall": 0.0, "signals": 0}


def train_model(dataset: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    dataset = dataset.sort_values(["date", "symbol"]).reset_index(drop=True)
    feature_cols = [c for c in dataset.columns if c not in {"target", "target_return", "symbol", "date"}]

    unique_dates = np.array(sorted(pd.to_datetime(dataset["date"]).unique()))
    if len(unique_dates) < 100:
        raise RuntimeError("Tarih sıralı test için yeterli veri yok.")
    split_date = unique_dates[int(len(unique_dates) * 0.80)]
    train_mask = pd.to_datetime(dataset["date"]) < split_date
    valid_mask = ~train_mask

    X_train = dataset.loc[train_mask, feature_cols]
    y_train = dataset.loc[train_mask, "target"].astype(int)
    X_valid = dataset.loc[valid_mask, feature_cols]
    y_valid = dataset.loc[valid_mask, "target"].astype(int)

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=14,
        min_samples_leaf=8,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    valid_prob = model.predict_proba(X_valid)[:, 1]
    threshold, threshold_stats = choose_threshold(y_valid.to_numpy(), valid_prob)

    metrics = {
        "split_date": str(pd.Timestamp(split_date).date()),
        "train_rows": int(train_mask.sum()),
        "valid_rows": int(valid_mask.sum()),
        "train_positive_rate": float(y_train.mean()),
        "valid_positive_rate": float(y_valid.mean()),
        "average_precision": float(average_precision_score(y_valid, valid_prob)),
        "roc_auc": float(roc_auc_score(y_valid, valid_prob)) if y_valid.nunique() > 1 else None,
        "threshold": float(threshold),
        "threshold_precision": threshold_stats["precision"],
        "threshold_recall": threshold_stats["recall"],
        "threshold_signals": threshold_stats["signals"],
    }

    # Nihai model, doğrulama ölçümünden sonra tüm geçmiş veriyle yeniden eğitilir.
    model.fit(dataset[feature_cols], dataset["target"].astype(int))

    positives = dataset[dataset["target"] == 1].copy()
    if len(positives) < 20:
        raise RuntimeError("%9+ pozitif örnek sayısı çok az.")
    nn = NearestNeighbors(n_neighbors=min(15, len(positives)), metric="euclidean")
    # RF ölçekten görece bağımsızdır ama komşuluk değildir; robust standardizasyon.
    med = dataset[feature_cols].median()
    iqr = (dataset[feature_cols].quantile(0.75) - dataset[feature_cols].quantile(0.25)).replace(0, 1.0)
    nn.fit(((positives[feature_cols] - med) / iqr).fillna(0.0))

    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "threshold": threshold,
        "metrics": metrics,
        "positive_rows": positives[["symbol", "date", "target_return"] + feature_cols],
        "nn": nn,
        "median": med,
        "iqr": iqr,
    }
    joblib.dump(artifact, MODEL_FILE)
    META_FILE.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return artifact, dataset


def reason_text(row: pd.Series) -> str:
    reasons: list[str] = []
    if row.get("higher_low_count", 0) >= 2:
        reasons.append("üç günde dipler yükseldi")
    if row.get("range_compression_21", 99) < 0.80:
        reasons.append("son gün fiyat aralığı belirgin daraldı")
    if row.get("volume_change_21", -99) > 0.35:
        reasons.append("son gün hacim önceki güne göre arttı")
    if row.get("d1_close_pos", 0) > 0.75:
        reasons.append("son kapanış gün içi aralığın üst bölümünde")
    if row.get("d1_upper_wick", 1) < 0.20:
        reasons.append("son mumda üst fitil baskısı sınırlı")
    if row.get("d1_lower_wick", 0) > 0.35:
        reasons.append("son mumda aşağı satış geri alındı")
    if row.get("close_slope_3d", 0) > 0:
        reasons.append("üç günlük kapanış eğimi yukarı")
    return "; ".join(reasons[:4]) or "üç günlük fiyat-hacim imzası geçmiş pozitif örneklere benziyor"


def predict_latest(artifact: dict, latest: pd.DataFrame) -> pd.DataFrame:
    features = artifact["feature_cols"]
    latest = latest.dropna(subset=features).copy()
    latest["model_probability"] = artifact["model"].predict_proba(latest[features])[:, 1]

    scaled = ((latest[features] - artifact["median"]) / artifact["iqr"]).fillna(0.0)
    distances, indices = artifact["nn"].kneighbors(scaled)
    positives = artifact["positive_rows"].reset_index(drop=True)

    neighbor_dates, neighbor_symbols, similarity = [], [], []
    for ds, idxs in zip(distances, indices):
        chosen = positives.iloc[idxs]
        neighbor_dates.append(", ".join(pd.to_datetime(chosen["date"]).dt.strftime("%Y-%m-%d").head(3)))
        neighbor_symbols.append(", ".join(chosen["symbol"].astype(str).head(3)))
        similarity.append(float(1.0 / (1.0 + np.mean(ds))))

    latest["positive_similarity"] = similarity
    latest["score"] = 100.0 * (0.80 * latest["model_probability"] + 0.20 * latest["positive_similarity"])
    latest["signal"] = np.where(
        latest["model_probability"] >= artifact["threshold"],
        "YARIN %9+ BEKLENTİSİ",
        "İZLEME ADAYI",
    )
    latest["reason"] = latest.apply(reason_text, axis=1)
    latest["similar_symbols"] = neighbor_symbols
    latest["similar_dates"] = neighbor_dates

    cols = [
        "date", "symbol", "signal", "score", "model_probability", "positive_similarity",
        "reason", "similar_symbols", "similar_dates",
    ]
    return latest[cols].sort_values(["signal", "score"], ascending=[True, False]).reset_index(drop=True)


def telegram_send(text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=20).raise_for_status()


def format_telegram(report: pd.DataFrame, metrics: dict, top_n: int) -> str:
    chosen = report.head(top_n)
    lines = [
        "🔥 PHOENIX — 3 GÜNLÜK %9+ TAHMİNİ",
        f"Model test başlangıcı: {metrics['split_date']}",
        f"Test eşik hassasiyeti: %{metrics['threshold_precision'] * 100:.1f}",
        "RSI/EMA/MACD kullanılmadı. Yalnızca önceki 3 günün fiyat-hacim davranışı kullanıldı.",
        "",
    ]
    for _, r in chosen.iterrows():
        lines.extend([
            f"{r['symbol']} — {r['signal']}",
            f"Phoenix skoru: {r['score']:.1f}/100 | Model: %{r['model_probability']*100:.1f}",
            f"Neden: {r['reason']}",
            f"Benzer %9+ örnekler: {r['similar_symbols']} ({r['similar_dates']})",
            "",
        ])
    lines.append("Bu çıktı kesinlik değil, tarih sıralı geçmiş testine dayalı olasılıksal bir araştırma sinyalidir.")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Phoenix: yalnızca önceki 3 günlük ham fiyat-hacim davranışıyla %9+ tahmini")
    p.add_argument("--symbols", default="symbols.csv")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--top", type=int, default=5)
    args = p.parse_args()

    symbols = load_symbols(Path(args.symbols))
    bundle = create_dataset(symbols, args.start, args.end, args.refresh)
    bundle.features.to_csv(TRAIN_FILE, index=False, encoding="utf-8-sig")

    if args.retrain or not MODEL_FILE.exists():
        artifact, _ = train_model(bundle.features)
    else:
        artifact = joblib.load(MODEL_FILE)

    report = predict_latest(artifact, bundle.latest)
    report.to_csv(REPORT_FILE, index=False, encoding="utf-8-sig")
    message = format_telegram(report, artifact["metrics"], args.top)
    print("\n" + message)
    telegram_send(message)
    print(f"\nRapor: {REPORT_FILE}")


if __name__ == "__main__":
    main()
