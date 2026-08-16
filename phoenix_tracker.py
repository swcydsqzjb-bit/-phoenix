from __future__ import annotations

from pathlib import Path
from datetime import datetime
import os

import pandas as pd
import requests
import yfinance as yf

PROJECT = Path(__file__).resolve().parent
TRACK_FILE = PROJECT / "phoenix_tracking.csv"
PREDICTIONS_FILE = PROJECT / "phoenix_research_predictions.csv"

COLUMNS = [
    "signal_date", "symbol", "signal_close", "phoenix_score",
    "calibrated_probability", "signal_type", "confidence",
    "day1_close_pct", "day2_close_pct", "day3_close_pct",
    "max_high_pct_3d", "hit_day", "status", "last_checked",
]


def normalize_symbol(symbol: str) -> str:
    symbol = str(symbol).strip().upper()
    return symbol if symbol.endswith(".IS") else f"{symbol}.IS"


def clean_symbol(symbol: str) -> str:
    return str(symbol).strip().upper().replace(".IS", "")


def load_tracking() -> pd.DataFrame:
    if TRACK_FILE.exists():
        try:
            df = pd.read_csv(TRACK_FILE)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=COLUMNS)
    else:
        df = pd.DataFrame(columns=COLUMNS)

    for col in COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[COLUMNS].copy()


def save_tracking(df: pd.DataFrame) -> None:
    df[COLUMNS].to_csv(TRACK_FILE, index=False, encoding="utf-8-sig")


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def _fetch_signal_close(symbol: str, signal_date: pd.Timestamp) -> float | None:
    try:
        hist = yf.download(
            normalize_symbol(symbol),
            start=(signal_date - pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
            end=(signal_date + pd.Timedelta(days=3)).strftime("%Y-%m-%d"),
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        print(f"{symbol}: sinyal kapanışı indirilemedi: {exc}")
        return None

    if hist is None or hist.empty:
        return None

    hist = _flatten_columns(hist)
    hist.index = pd.to_datetime(hist.index).tz_localize(None)
    upto = hist[hist.index <= signal_date]
    if upto.empty:
        return None

    try:
        return float(upto.iloc[-1]["Close"])
    except Exception:
        return None


def _active_symbol_exists(tracking: pd.DataFrame, symbol: str) -> bool:
    if tracking.empty:
        return False
    symbol_mask = tracking["symbol"].astype(str).str.upper().eq(symbol)
    active_mask = tracking["status"].astype(str).str.upper().isin(
        {"TAKIPTE", "HEDEF_GORDU_TAKIPTE"}
    )
    return bool((symbol_mask & active_mask).any())


def capture_latest_predictions(limit: int = 5) -> int:
    if not PREDICTIONS_FILE.exists():
        print("Tahmin dosyası bulunamadı; yeni aday kaydı yapılmadı.")
        return 0

    predictions = pd.read_csv(PREDICTIONS_FILE)
    if predictions.empty:
        print("Tahmin dosyası boş; yeni aday kaydı yapılmadı.")
        return 0

    required = {"date", "symbol"}
    if not required.issubset(predictions.columns):
        missing = sorted(required - set(predictions.columns))
        print(f"Tahmin dosyasında gerekli kolonlar yok: {missing}")
        return 0

    tracking = load_tracking()
    added = 0

    for _, row in predictions.iterrows():
        if added >= limit:
            break

        symbol = clean_symbol(row["symbol"])
        signal_date = pd.to_datetime(row["date"]).tz_localize(None).normalize()

        if _active_symbol_exists(tracking, symbol):
            print(f"{symbol}: mevcut aktif takip kaydı var; yeni sinyal açılmadı.")
            continue

        same_day_duplicate = (
            (tracking["signal_date"].astype(str) == signal_date.strftime("%Y-%m-%d"))
            & (tracking["symbol"].astype(str).str.upper() == symbol)
        )
        if same_day_duplicate.any():
            print(f"{symbol} {signal_date.date()}: zaten kayıtlı.")
            continue

        signal_close = _fetch_signal_close(symbol, signal_date)
        if signal_close is None:
            print(f"{symbol}: sinyal kapanışı bulunamadı, kayıt atlandı.")
            continue

        new_row = {
            "signal_date": signal_date.strftime("%Y-%m-%d"),
            "symbol": symbol,
            "signal_close": round(signal_close, 6),
            "phoenix_score": float(row.get("phoenix_score", 0.0)),
            "calibrated_probability": float(row.get("model_probability", 0.0)),
            "signal_type": str(row.get("signal", "")),
            "confidence": str(row.get("confidence", "")),
            "day1_close_pct": pd.NA,
            "day2_close_pct": pd.NA,
            "day3_close_pct": pd.NA,
            "max_high_pct_3d": pd.NA,
            "hit_day": pd.NA,
            "status": "TAKIPTE",
            "last_checked": pd.NA,
        }

        tracking = pd.concat([tracking, pd.DataFrame([new_row])], ignore_index=True)
        added += 1
        print(f"YENI KAYIT: {symbol} | {signal_date.date()} | kapanış={signal_close:.4f}")

    save_tracking(tracking)
    print(f"Yeni takip kaydı: {added}")
    print(f"Toplam takip kaydı: {len(tracking)}")
    return added


def _pct(value: float, base: float) -> float:
    if not base:
        return 0.0
    return ((float(value) / float(base)) - 1.0) * 100.0


def update_tracking() -> pd.DataFrame:
    tracking = load_tracking()
    if tracking.empty:
        save_tracking(tracking)
        return tracking

    today = pd.Timestamp.now().normalize()

    for idx, row in tracking.iterrows():
        if str(row.get("status", "")).upper() in {"BASARILI", "BASARISIZ"}:
            continue

        symbol = clean_symbol(row["symbol"])
        signal_date = pd.to_datetime(row["signal_date"]).tz_localize(None).normalize()

        try:
            signal_close = float(row["signal_close"])
        except Exception:
            print(f"{symbol}: signal_close okunamadı.")
            continue

        try:
            hist = yf.download(
                normalize_symbol(symbol),
                start=(signal_date - pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
                end=(today + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception as exc:
            print(f"{symbol}: takip verisi indirilemedi: {exc}")
            continue

        if hist is None or hist.empty:
            continue

        hist = _flatten_columns(hist)
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        future = hist[hist.index > signal_date].sort_index().head(3).copy()

        if future.empty:
            tracking.loc[idx, "last_checked"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            continue

        for day_number in range(1, 4):
            if len(future) >= day_number:
                close_value = float(future.iloc[day_number - 1]["Close"])
                tracking.loc[idx, f"day{day_number}_close_pct"] = round(
                    _pct(close_value, signal_close), 2
                )

        hit_day = None
        max_high_pct = None
        for day_number in range(1, len(future) + 1):
            high_value = float(future.iloc[day_number - 1]["High"])
            high_pct = round(_pct(high_value, signal_close), 2)
            if max_high_pct is None or high_pct > max_high_pct:
                max_high_pct = high_pct
            if hit_day is None and high_pct >= 9.0:
                hit_day = day_number

        tracking.loc[idx, "max_high_pct_3d"] = max_high_pct
        tracking.loc[idx, "hit_day"] = hit_day if hit_day is not None else pd.NA
        tracking.loc[idx, "last_checked"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if len(future) < 3:
            tracking.loc[idx, "status"] = (
                "HEDEF_GORDU_TAKIPTE" if hit_day is not None else "TAKIPTE"
            )
        else:
            tracking.loc[idx, "status"] = "BASARILI" if hit_day is not None else "BASARISIZ"

    save_tracking(tracking)
    return tracking


def build_tracking_summary(limit: int = 12) -> str:
    tracking = load_tracking()
    if tracking.empty:
        return "📊 PHOENIX TAKİP\nHenüz kayıtlı aday yok."

    recent = tracking.tail(limit).copy()
    lines = ["📊 PHOENIX ADAY TAKİP"]

    for _, row in recent.iterrows():
        lines.append("")
        lines.append(f"{row['symbol']} — {row['signal_date']}")
        lines.append(f"Durum: {row['status']}")
        if pd.notna(row["day1_close_pct"]):
            lines.append(f"1. gün kapanış: %{float(row['day1_close_pct']):+.2f}")
        if pd.notna(row["day2_close_pct"]):
            lines.append(f"2. gün kapanış: %{float(row['day2_close_pct']):+.2f}")
        if pd.notna(row["day3_close_pct"]):
            lines.append(f"3. gün kapanış: %{float(row['day3_close_pct']):+.2f}")
        if pd.notna(row["max_high_pct_3d"]):
            lines.append(f"3 günlük maksimum: %{float(row['max_high_pct_3d']):+.2f}")
        if pd.notna(row["hit_day"]):
            lines.append(f"🎯 %9+ hedefi: {int(float(row['hit_day']))}. işlem günü")

    return "\n".join(lines)


def send_telegram(message: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        print("Telegram bilgileri yok; takip özeti sadece loga yazıldı.")
        return

    for i in range(0, len(message), 3500):
        chunk = message[i:i + 3500]
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": chunk},
            timeout=30,
        )
        if not response.ok:
            print(f"Telegram takip özeti gönderilemedi: {response.status_code} {response.text}")


def main() -> None:
    update_tracking()
    capture_latest_predictions(limit=5)
    summary = build_tracking_summary(limit=12)
    print(summary)
    if not load_tracking().empty:
        send_telegram(summary)
    print(f"Takip dosyası: {TRACK_FILE}")


if __name__ == "__main__":
    main()
