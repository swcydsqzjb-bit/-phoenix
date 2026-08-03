from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import re
from io import StringIO

import joblib
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

PROJECT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT / "phoenix_cache"
MODEL_FILE = PROJECT / "phoenix_research_model.joblib"
META_FILE = PROJECT / "phoenix_research_metrics.json"
TRAIN_FILE = PROJECT / "phoenix_research_training.csv"
REPORT_FILE = PROJECT / "phoenix_research_predictions.csv"

TARGET_RETURN = 0.09
RANDOM_STATE = 42

NON_FEATURES = {"symbol", "date", "target", "target_return", "outcome_group"}


@dataclass
class DatasetBundle:
    history: pd.DataFrame
    latest: pd.DataFrame


def normalize_symbol(value: object) -> str:
    symbol = str(value).strip().upper()
    if not symbol or symbol in {"NAN", "SYMBOL", "TICKER"}:
        return ""
    return symbol if symbol.endswith(".IS") else f"{symbol}.IS"


def _clean_raw_symbol(value: object) -> str:
    symbol = str(value).strip().upper().replace(".IS", "")
    symbol = re.sub(r"[^A-Z0-9]", "", symbol)
    if not re.fullmatch(r"[A-Z0-9]{2,8}", symbol):
        return ""
    return symbol


def _symbols_from_tables(tables: list[pd.DataFrame]) -> set[str]:
    found: set[str] = set()
    for table in tables:
        frame = table.copy()
        frame.columns = [str(column).strip().lower() for column in frame.columns]
        candidate_columns = [
            column for column in frame.columns
            if any(key in column for key in ("symbol", "ticker", "kod", "code"))
        ]
        if not candidate_columns and len(frame.columns):
            candidate_columns = [frame.columns[0]]
        for column in candidate_columns:
            for value in frame[column].tolist():
                symbol = _clean_raw_symbol(value)
                if symbol:
                    found.add(symbol)
    return found


def fetch_stockanalysis_symbols(max_pages: int = 20) -> list[str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; PhoenixResearch/1.1; +https://github.com/)",
        "Accept-Language": "en-US,en;q=0.9,tr;q=0.8",
    }
    found: set[str] = set()
    empty_pages = 0

    for page in range(1, max_pages + 1):
        url = "https://stockanalysis.com/list/borsa-istanbul/"
        if page > 1:
            url += f"?page={page}"
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code in {404, 410}:
            break
        response.raise_for_status()
        page_symbols = _symbols_from_tables(pd.read_html(StringIO(response.text)))
        page_symbols.update(
            _clean_raw_symbol(value)
            for value in re.findall(r"/quote/ist/([A-Z0-9]{2,8})/", response.text, flags=re.IGNORECASE)
        )
        page_symbols.discard("")
        new_symbols = page_symbols - found
        found.update(page_symbols)
        print(f"Sembol kaynağı StockAnalysis sayfa {page}: +{len(new_symbols)} (toplam {len(found)})")
        if not new_symbols:
            empty_pages += 1
            if empty_pages >= 2:
                break
        else:
            empty_pages = 0

    return sorted(found)


def fetch_kap_symbols() -> list[str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; PhoenixResearch/1.1; +https://github.com/)",
        "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
    }
    urls = ["https://kap.org.tr/tr/Sektorler", "https://kap.org.tr/en/Sektorler"]
    found: set[str] = set()
    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            found.update(_symbols_from_tables(pd.read_html(StringIO(response.text))))
            if len(found) >= 300:
                break
        except Exception as exc:
            print(f"KAP sembol kaynağı kullanılamadı ({url}): {exc}")
    return sorted(found)


def load_symbols(path: Path) -> list[str]:
    """Fetch the current BIST universe; use symbols.csv only as a fallback."""
    discovered: set[str] = set()

    try:
        discovered.update(fetch_stockanalysis_symbols())
    except Exception as exc:
        print(f"StockAnalysis sembol listesi alınamadı: {exc}")

    # KAP is used as an independent backup/augmentation source.
    if len(discovered) < 450:
        try:
            discovered.update(fetch_kap_symbols())
        except Exception as exc:
            print(f"KAP sembol listesi alınamadı: {exc}")

    if discovered:
        symbols = sorted(normalize_symbol(value) for value in discovered)
        symbols = [symbol for symbol in symbols if symbol]
        (PROJECT / "phoenix_symbols_discovered.csv").write_text(
            "\n".join(symbol.removesuffix(".IS") for symbol in symbols) + "\n",
            encoding="utf-8",
        )
        print(f"Otomatik BIST evreni bulundu: {len(symbols)} sembol")
        if len(symbols) < 300:
            print("UYARI: Otomatik listede beklenenden az sembol bulundu; symbols.csv ile birleştirilecek.")
        if path.exists():
            frame = pd.read_csv(path, encoding="utf-8-sig", header=None)
            symbols = sorted(set(symbols) | {normalize_symbol(v) for v in frame.to_numpy().ravel() if normalize_symbol(v)})
        return symbols

    if not path.exists():
        raise FileNotFoundError(f"Otomatik liste alınamadı ve sembol dosyası bulunamadı: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig", header=None)
    symbols = sorted({normalize_symbol(v) for v in frame.to_numpy().ravel()})
    symbols = [symbol for symbol in symbols if symbol]
    if not symbols:
        raise ValueError("Otomatik liste alınamadı ve symbols.csv içinde geçerli hisse kodu bulunamadı.")
    print(f"Yedek symbols.csv kullanılıyor: {len(symbols)} sembol")
    return symbols


def download_symbol(symbol: str, start: str, refresh: bool) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{symbol.replace('.', '_')}.csv"

    if cache_file.exists() and not refresh:
        try:
            cached = pd.read_csv(cache_file, parse_dates=["Date"]).set_index("Date")
            if len(cached) >= 20:
                return cached
        except Exception:
            pass

    frame = yf.download(
        symbol,
        start=start,
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
        timeout=30,
    )
    if frame.empty:
        return pd.DataFrame()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    needed = ["Open", "High", "Low", "Close", "Volume"]
    if any(column not in frame.columns for column in needed):
        return pd.DataFrame()

    frame = frame[needed].copy()
    frame = frame.dropna(subset=["Open", "High", "Low", "Close"])
    frame = frame[(frame[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame.index.name = "Date"
    frame.reset_index().to_csv(cache_file, index=False)
    return frame


def safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def candle_features(frame: pd.DataFrame, lag: int) -> dict[str, pd.Series]:
    """Only uses the OHLCV values of the selected day itself."""
    open_ = frame["Open"].shift(lag)
    high = frame["High"].shift(lag)
    low = frame["Low"].shift(lag)
    close = frame["Close"].shift(lag)
    range_ = (high - low).replace(0, np.nan)
    body = close - open_
    prefix = f"d{lag}"

    return {
        f"{prefix}_close_open": close / open_ - 1.0,
        f"{prefix}_range_open": range_ / open_,
        f"{prefix}_body_range": safe_div(body, range_),
        f"{prefix}_upper_wick": safe_div(high - np.maximum(open_, close), range_),
        f"{prefix}_lower_wick": safe_div(np.minimum(open_, close) - low, range_),
        f"{prefix}_close_position": safe_div(close - low, range_),
        f"{prefix}_open_position": safe_div(open_ - low, range_),
    }


def build_rows(symbol: str, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    A row dated T predicts T using only T-3, T-2 and T-1.
    No value from T-4 or earlier is used in a feature.
    """
    if len(frame) < 12:
        return pd.DataFrame(), pd.DataFrame()

    out = pd.DataFrame(index=frame.index)
    for lag in (3, 2, 1):
        for name, values in candle_features(frame, lag).items():
            out[name] = values

    o3, o2, o1 = (frame["Open"].shift(3), frame["Open"].shift(2), frame["Open"].shift(1))
    h3, h2, h1 = (frame["High"].shift(3), frame["High"].shift(2), frame["High"].shift(1))
    l3, l2, l1 = (frame["Low"].shift(3), frame["Low"].shift(2), frame["Low"].shift(1))
    c3, c2, c1 = (frame["Close"].shift(3), frame["Close"].shift(2), frame["Close"].shift(1))
    v3, v2, v1 = (frame["Volume"].shift(3).astype(float), frame["Volume"].shift(2).astype(float), frame["Volume"].shift(1).astype(float))
    r3, r2, r1 = (h3 - l3, h2 - l2, h1 - l1)

    # Relationships contained strictly inside the three-day window.
    out["close_move_32"] = c2 / c3 - 1.0
    out["close_move_21"] = c1 / c2 - 1.0
    out["three_day_close_move"] = c1 / c3 - 1.0
    out["gap_32"] = o2 / c3 - 1.0
    out["gap_21"] = o1 / c2 - 1.0
    out["higher_high_count"] = (h2 > h3).astype(float) + (h1 > h2).astype(float)
    out["higher_low_count"] = (l2 > l3).astype(float) + (l1 > l2).astype(float)
    out["range_ratio_32"] = safe_div(r2, r3)
    out["range_ratio_21"] = safe_div(r1, r2)
    out["range_acceleration"] = safe_div(r1 * r3, r2.pow(2))
    out["volume_ratio_32"] = safe_div(v2, v3)
    out["volume_ratio_21"] = safe_div(v1, v2)
    out["volume_acceleration"] = safe_div(v1 * v3, v2.pow(2))

    volume_sum = (v3 + v2 + v1).replace(0, np.nan)
    out["volume_share_d3"] = v3 / volume_sum
    out["volume_share_d2"] = v2 / volume_sum
    out["volume_share_d1"] = v1 / volume_sum

    out["green_count"] = (
        (c3 > o3).astype(float) + (c2 > o2).astype(float) + (c1 > o1).astype(float)
    )
    out["inside_day_d2"] = ((h2 <= h3) & (l2 >= l3)).astype(float)
    out["inside_day_d1"] = ((h1 <= h2) & (l1 >= l2)).astype(float)
    out["three_day_range"] = (pd.concat([h3, h2, h1], axis=1).max(axis=1) - pd.concat([l3, l2, l1], axis=1).min(axis=1)) / c3
    out["close_consistency"] = pd.concat([c3 / o3 - 1, c2 / o2 - 1, c1 / o1 - 1], axis=1).std(axis=1)

    target_return = frame["Close"] / frame["Close"].shift(1) - 1.0
    out["target_return"] = target_return
    out["target"] = (target_return >= TARGET_RETURN).astype(int)
    out["outcome_group"] = pd.cut(
        target_return,
        bins=[-np.inf, 0.02, 0.05, 0.09, np.inf],
        labels=["<=%2", "%2-%5", "%5-%9", "%9+"],
        right=False,
    ).astype(str)
    out["symbol"] = symbol.removesuffix(".IS")
    out["date"] = out.index

    feature_columns = [c for c in out.columns if c not in NON_FEATURES]
    out[feature_columns] = out[feature_columns].replace([np.inf, -np.inf], np.nan)

    history = out.dropna(subset=feature_columns + ["target_return"]).copy()
    latest = out.dropna(subset=feature_columns).tail(1).copy()
    return history, latest


def create_dataset(symbols: Iterable[str], start: str, refresh: bool) -> DatasetBundle:
    history_parts: list[pd.DataFrame] = []
    latest_parts: list[pd.DataFrame] = []
    symbols = list(symbols)

    for index, symbol in enumerate(symbols, start=1):
        try:
            prices = download_symbol(symbol, start, refresh)
            history, latest = build_rows(symbol, prices)
            if not history.empty:
                history_parts.append(history)
            if not latest.empty:
                latest_parts.append(latest)
        except Exception as exc:
            print(f"[{index}/{len(symbols)}] {symbol}: {exc}")

        if index % 25 == 0 or index == len(symbols):
            print(f"Veri ilerlemesi: {index}/{len(symbols)}")
        time.sleep(0.03)

    if not history_parts:
        raise RuntimeError("Eğitim verisi oluşturulamadı.")

    return DatasetBundle(
        history=pd.concat(history_parts, ignore_index=True),
        latest=pd.concat(latest_parts, ignore_index=True),
    )


def wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    p = successes / total
    denom = 1.0 + z * z / total
    centre = p + z * z / (2.0 * total)
    margin = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
    return float((centre - margin) / denom)


def select_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict[str, float | int]]:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    choices: list[tuple[float, float, float, int, float]] = []

    for p, r, threshold in zip(precision[:-1], recall[:-1], thresholds):
        mask = probabilities >= threshold
        signal_count = int(mask.sum())
        true_count = int(y_true[mask].sum()) if signal_count else 0
        if signal_count >= 25 and r >= 0.01:
            lower = wilson_lower_bound(true_count, signal_count)
            choices.append((float(p), float(r), float(threshold), signal_count, lower))

    if not choices:
        return 0.50, {"precision": 0.0, "recall": 0.0, "signals": 0, "precision_lower_bound": 0.0}

    # Küçük örneklerde şişen hassasiyet yerine güven aralığının alt sınırını önceliklendir.
    best = max(choices, key=lambda item: (item[4], item[0], item[1]))
    return best[2], {
        "precision": best[0],
        "recall": best[1],
        "signals": best[3],
        "precision_lower_bound": best[4],
    }


def train(history: pd.DataFrame) -> dict:
    history = history.sort_values(["date", "symbol"]).reset_index(drop=True)
    feature_columns = [c for c in history.columns if c not in NON_FEATURES]
    dates = pd.to_datetime(history["date"])
    unique_dates = np.array(sorted(dates.unique()))
    if len(unique_dates) < 250:
        raise RuntimeError("Tarih sıralı yürüyen test için yeterli işlem günü yok.")

    # Üç ayrı ileri-zaman testi. Her doğrulama bölümü yalnızca kendisinden önceki tarihlerle eğitilir.
    fold_points = [(0.60, 0.72), (0.72, 0.84), (0.84, 1.00)]
    oof_probabilities = np.full(len(history), np.nan, dtype=float)
    fold_metrics: list[dict] = []

    for fold_no, (train_end_ratio, valid_end_ratio) in enumerate(fold_points, start=1):
        train_end = pd.Timestamp(unique_dates[max(1, int(len(unique_dates) * train_end_ratio)) - 1])
        valid_end = pd.Timestamp(unique_dates[max(2, int(len(unique_dates) * valid_end_ratio)) - 1])
        train_mask = dates <= train_end
        valid_mask = (dates > train_end) & (dates <= valid_end)
        if int(valid_mask.sum()) == 0 or history.loc[train_mask, "target"].nunique() < 2:
            continue

        fold_model = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=18,
            min_samples_leaf=8,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE + fold_no,
        )
        fold_model.fit(history.loc[train_mask, feature_columns], history.loc[train_mask, "target"].astype(int))
        raw = fold_model.predict_proba(history.loc[valid_mask, feature_columns])[:, 1]
        oof_probabilities[valid_mask.to_numpy()] = raw
        y_fold = history.loc[valid_mask, "target"].astype(int).to_numpy()
        fold_metrics.append({
            "fold": fold_no,
            "train_end": str(train_end.date()),
            "valid_end": str(valid_end.date()),
            "validation_rows": int(valid_mask.sum()),
            "positive_rate": float(y_fold.mean()),
            "average_precision_raw": float(average_precision_score(y_fold, raw)),
        })

    oof_mask = np.isfinite(oof_probabilities)
    if int(oof_mask.sum()) < 1000:
        raise RuntimeError("Yürüyen test için yeterli doğrulama tahmini üretilemedi.")

    y_oof = history.loc[oof_mask, "target"].astype(int).to_numpy()
    raw_oof = oof_probabilities[oof_mask]

    # Ağaç olasılıkları aşırı yüksek görünebildiği için zaman dışı tahminlerle kalibrasyon.
    calibrator = LogisticRegression(solver="lbfgs", random_state=RANDOM_STATE)
    calibrator.fit(raw_oof.reshape(-1, 1), y_oof)
    calibrated_oof = calibrator.predict_proba(raw_oof.reshape(-1, 1))[:, 1]
    threshold, threshold_stats = select_threshold(y_oof, calibrated_oof)

    metrics = {
        "test_start_date": str(pd.Timestamp(history.loc[oof_mask, "date"].min()).date()),
        "validation_rows": int(oof_mask.sum()),
        "positive_rate_validation": float(y_oof.mean()),
        "average_precision": float(average_precision_score(y_oof, calibrated_oof)),
        "roc_auc": float(roc_auc_score(y_oof, calibrated_oof)) if len(np.unique(y_oof)) > 1 else None,
        "threshold": float(threshold),
        "threshold_precision": float(threshold_stats["precision"]),
        "threshold_precision_lower_bound": float(threshold_stats["precision_lower_bound"]),
        "threshold_recall": float(threshold_stats["recall"]),
        "threshold_signals": int(threshold_stats["signals"]),
        "folds": fold_metrics,
    }

    final_model = ExtraTreesClassifier(
        n_estimators=800,
        max_depth=18,
        min_samples_leaf=8,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    final_model.fit(history[feature_columns], history["target"].astype(int))

    scaler = RobustScaler()
    scaled_history = scaler.fit_transform(history[feature_columns])
    # Aynı hisseyi ve çok yakın tarihleri ayıklayabilmek için daha geniş komşu havuzu.
    query_neighbor_count = min(250, len(history))
    neighbors = NearestNeighbors(n_neighbors=query_neighbor_count, metric="euclidean", n_jobs=-1)
    neighbors.fit(scaled_history)

    importances = sorted(
        zip(feature_columns, final_model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )[:15]
    metrics["top_features"] = [{"feature": name, "importance": float(value)} for name, value in importances]

    artifact = {
        "model": final_model,
        "calibrator": calibrator,
        "scaler": scaler,
        "neighbors": neighbors,
        "history_reference": history[["symbol", "date", "target_return", "outcome_group"] + feature_columns],
        "feature_columns": feature_columns,
        "threshold": threshold,
        "metrics": metrics,
    }
    joblib.dump(artifact, MODEL_FILE)
    META_FILE.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return artifact

def describe_pattern(row: pd.Series) -> str:
    reasons: list[str] = []
    if row["higher_low_count"] >= 2:
        reasons.append("üç gün boyunca dipler yükseldi")
    if row["higher_high_count"] >= 2:
        reasons.append("tepeler kademeli yükseldi")
    if row["range_ratio_21"] < 0.82:
        reasons.append("son gün hareket alanı daraldı")
    if row["volume_ratio_21"] > 1.30:
        reasons.append("son gün hacim önceki güne göre güçlendi")
    if row["d1_close_position"] > 0.75:
        reasons.append("son kapanış gün içi aralığın üst bölümünde kaldı")
    if row["d1_upper_wick"] < 0.18:
        reasons.append("son mumdaki üst fitil baskısı düşük")
    if row["d1_lower_wick"] > 0.35:
        reasons.append("aşağı satış son mum içinde geri alındı")
    if row["inside_day_d1"] == 1:
        reasons.append("son gün önceki günün içinde sıkıştı")
    if row["three_day_close_move"] > 0:
        reasons.append("üç günlük kapanış yönü yukarı")
    return "; ".join(reasons[:4]) or "üç günlük fiyat-hacim imzası geçmiş örneklere yakın"


def predict(artifact: dict, latest: pd.DataFrame) -> pd.DataFrame:
    features = artifact["feature_columns"]
    latest = latest.dropna(subset=features).copy()
    raw_probability = artifact["model"].predict_proba(latest[features])[:, 1]
    latest["model_probability"] = artifact["calibrator"].predict_proba(raw_probability.reshape(-1, 1))[:, 1]

    scaled_latest = artifact["scaler"].transform(latest[features])
    distances, indices = artifact["neighbors"].kneighbors(scaled_latest)
    reference = artifact["history_reference"].reset_index(drop=True)
    base_rate = float(artifact["metrics"].get("positive_rate_validation", 0.0))
    neighbor_floor = max(0.10, base_rate * 2.5)

    rows: list[dict] = []
    for row_index, (_, source_row) in enumerate(latest.iterrows()):
        candidate = reference.iloc[indices[row_index]].copy()
        candidate["distance"] = distances[row_index]
        candidate["date"] = pd.to_datetime(candidate["date"])
        source_date = pd.Timestamp(source_row["date"])

        # Aynı hissenin kendi geçmişini ve son 45 gündeki çok yakın tekrarları kanıt listesinden çıkar.
        filtered = candidate[
            (candidate["symbol"] != source_row["symbol"])
            & (candidate["date"] <= source_date - pd.Timedelta(days=45))
        ].head(75)
        if len(filtered) < 40:
            filtered = candidate[candidate["date"] <= source_date - pd.Timedelta(days=10)].head(75)
        nearby = filtered

        positive_count = int((nearby["target_return"] >= 0.09).sum())
        positive_rate = float(positive_count / len(nearby)) if len(nearby) else 0.0
        five_to_nine_rate = float(((nearby["target_return"] >= 0.05) & (nearby["target_return"] < 0.09)).mean()) if len(nearby) else 0.0
        two_to_five_rate = float(((nearby["target_return"] >= 0.02) & (nearby["target_return"] < 0.05)).mean()) if len(nearby) else 0.0
        failure_rate = max(0.0, 1.0 - positive_rate - five_to_nine_rate - two_to_five_rate)
        mean_distance = float(nearby["distance"].mean()) if len(nearby) else float("inf")
        similarity = 1.0 / (1.0 + mean_distance) if np.isfinite(mean_distance) else 0.0
        neighbor_lower = wilson_lower_bound(positive_count, len(nearby))

        closest = nearby.sort_values("distance").head(5)
        examples = " | ".join(
            f"{r.symbol} {pd.Timestamp(r.date).date()} ({r.target_return * 100:+.1f}%)"
            for r in closest.itertuples()
        )
        probability = float(source_row.get("model_probability", latest.loc[source_row.name, "model_probability"]))
        score = 100.0 * (0.55 * probability + 0.25 * positive_rate + 0.10 * neighbor_lower + 0.10 * similarity)
        strong = (
            probability >= artifact["threshold"]
            and positive_rate >= neighbor_floor
            and positive_count >= 8
        )
        signal = "YARIN %9+ BEKLENTİSİ" if strong else "İZLEME ADAYI"
        confidence = "YÜKSEK" if strong and neighbor_lower >= 0.12 else ("ORTA" if strong else "DÜŞÜK")

        rows.append({
            "date": source_row["date"],
            "symbol": source_row["symbol"],
            "signal": signal,
            "confidence": confidence,
            "phoenix_score": score,
            "model_probability": probability,
            "neighbor_count": int(len(nearby)),
            "neighbor_9plus_count": positive_count,
            "neighbor_9plus_rate": positive_rate,
            "neighbor_9plus_lower_bound": neighbor_lower,
            "neighbor_5to9_rate": five_to_nine_rate,
            "neighbor_2to5_rate": two_to_five_rate,
            "neighbor_failure_rate": failure_rate,
            "similarity": similarity,
            "reason": describe_pattern(source_row),
            "closest_examples": examples,
        })

    report = pd.DataFrame(rows)
    signal_rank = report["signal"].eq("YARIN %9+ BEKLENTİSİ").astype(int)
    report = report.assign(_signal_rank=signal_rank).sort_values(
        ["_signal_rank", "phoenix_score", "neighbor_9plus_lower_bound"], ascending=[False, False, False]
    ).drop(columns="_signal_rank")
    return report.reset_index(drop=True)

def send_telegram(message: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        print("Telegram bilgileri bulunmadı; mesaj gönderilmedi.")
        return

    # Telegram tek mesaj sınırının altında güvenli parçalar gönder.
    chunks: list[str] = []
    current = ""
    for block in message.split("\n\n"):
        proposed = block if not current else current + "\n\n" + block
        if len(proposed) <= 3600:
            current = proposed
        else:
            if current:
                chunks.append(current)
            current = block
    if current:
        chunks.append(current)

    for index, chunk in enumerate(chunks, start=1):
        prefix = f"[{index}/{len(chunks)}]\n" if len(chunks) > 1 else ""
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": prefix + chunk},
            timeout=30,
        )
        if not response.ok:
            raise RuntimeError(f"Telegram hatası {response.status_code}: {response.text}")

def format_message(report: pd.DataFrame, metrics: dict, top_n: int) -> str:
    selected = report.head(top_n)
    lines = [
        "🔥 PHOENIX RESEARCH — 3 GÜNLÜK %9+ TAHMİNİ",
        f"Yürüyen test başlangıcı: {metrics['test_start_date']}",
        f"Yürüyen test hassasiyeti: %{metrics['threshold_precision'] * 100:.1f} (alt güven sınırı %{metrics['threshold_precision_lower_bound'] * 100:.1f})",
        f"Yürüyen test yakalama oranı: %{metrics['threshold_recall'] * 100:.1f}",
        "RSI, EMA, MACD ve eski bot skorları kullanılmadı.",
        "Yalnızca son 3 tamamlanmış günün OHLCV davranışı kullanıldı.",
        "",
    ]

    for row in selected.itertuples():
        lines.extend([
            f"{row.symbol} — {row.signal} | Güven: {row.confidence}",
            f"Phoenix: {row.phoenix_score:.1f}/100 | Kalibre olasılık: %{row.model_probability * 100:.1f}",
            f"Benzer {row.neighbor_count} olay: %9+ {row.neighbor_9plus_count} adet (%{row.neighbor_9plus_rate * 100:.1f}; alt sınır %{row.neighbor_9plus_lower_bound * 100:.1f}) | %5-9 %{row.neighbor_5to9_rate * 100:.1f} | %2-5 %{row.neighbor_2to5_rate * 100:.1f}",
            f"Neden: {row.reason}",
            f"En yakın örnekler: {row.closest_examples}",
            "",
        ])

    lines.append("Bu çıktı yatırım garantisi değil; tarihsel benzerlik ve tarih sıralı test kullanan araştırma sinyalidir.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="PHOENIX Research V2: yalnızca önceki 3 günlük OHLCV ile yürüyen test ve kalibre %9+ araştırması")
    parser.add_argument("--symbols", default="symbols.csv")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    symbols = load_symbols(Path(args.symbols))
    print(f"PHOENIX taraması başlayacak: {len(symbols)} sembol")
    bundle = create_dataset(symbols, args.start, args.refresh)
    bundle.history.to_csv(TRAIN_FILE, index=False, encoding="utf-8-sig")

    if args.retrain or not MODEL_FILE.exists():
        artifact = train(bundle.history)
    else:
        artifact = joblib.load(MODEL_FILE)

    report = predict(artifact, bundle.latest)
    report.to_csv(REPORT_FILE, index=False, encoding="utf-8-sig")
    message = format_message(report, artifact["metrics"], args.top)
    print(message)
    send_telegram(message)
    print(f"Rapor oluşturuldu: {REPORT_FILE}")


if __name__ == "__main__":
    main()
