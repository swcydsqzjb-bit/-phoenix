from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd
import yfinance as yf

TRACK_FILE = Path("phoenix_tracking.csv")


def normalize_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    return symbol if symbol.endswith(".IS") else f"{symbol}.IS"


def init_tracking_file() -> pd.DataFrame:
    columns = [
        "signal_date",
        "symbol",
        "signal_close",
        "phoenix_score",
        "calibrated_probability",
        "signal_type",
        "day1_close_pct",
        "day2_close_pct",
        "day3_close_pct",
        "max_high_pct_3d",
        "status",
        "last_checked",
    ]

    if TRACK_FILE.exists():
        df = pd.read_csv(TRACK_FILE)
        for col in columns:
            if col not in df.columns:
                df[col] = None
        return df[columns]

    return pd.DataFrame(columns=columns)


def add_signal(
    symbol: str,
    signal_date: str,
    signal_close: float,
    phoenix_score: float,
    calibrated_probability: float,
    signal_type: str,
):
    df = init_tracking_file()

    duplicate = (
        (df["signal_date"].astype(str) == str(signal_date))
        & (df["symbol"].astype(str) == str(symbol))
    )

    if duplicate.any():
        return

    row = {
        "signal_date": signal_date,
        "symbol": symbol,
        "signal_close": signal_close,
        "phoenix_score": phoenix_score,
        "calibrated_probability": calibrated_probability,
        "signal_type": signal_type,
        "day1_close_pct": None,
        "day2_close_pct": None,
        "day3_close_pct": None,
        "max_high_pct_3d": None,
        "status": "TAKIPTE",
        "last_checked": None,
    }

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(TRACK_FILE, index=False)


def _pct(value: float, base: float) -> float:
    if base == 0:
        return 0.0
    return ((value / base) - 1.0) * 100.0


def update_tracking():
    df = init_tracking_file()

    if df.empty:
        df.to_csv(TRACK_FILE, index=False)
        return df

    for idx, row in df.iterrows():

        if str(row.get("status", "")).upper() in {"BASARILI", "BASARISIZ"}:
            continue

        symbol = str(row["symbol"])
        signal_date = pd.to_datetime(row["signal_date"])
        signal_close = float(row["signal_close"])

        yf_symbol = normalize_symbol(symbol)

        try:
            hist = yf.download(
                yf_symbol,
                start=(signal_date - pd.Timedelta(days=3)).strftime("%Y-%m-%d"),
                end=(pd.Timestamp.today() + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception:
            continue

        if hist is None or hist.empty:
            continue

        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)

        hist.index = pd.to_datetime(hist.index).tz_localize(None)

        future = hist[hist.index > signal_date].copy()

        if future.empty:
            continue

        future = future.iloc[:3]

        for day_number in range(1, 4):
            if len(future) >= day_number:
                close_value = float(future.iloc[day_number - 1]["Close"])
                df.loc[idx, f"day{day_number}_close_pct"] = round(
                    _pct(close_value, signal_close), 2
                )

        max_high = float(future["High"].max())
        max_high_pct = round(_pct(max_high, signal_close), 2)

        df.loc[idx, "max_high_pct_3d"] = max_high_pct
        df.loc[idx, "last_checked"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if max_high_pct >= 9.0:
            df.loc[idx, "status"] = "BASARILI"
        elif len(future) >= 3:
            df.loc[idx, "status"] = "BASARISIZ"
        else:
            df.loc[idx, "status"] = "TAKIPTE"

    df.to_csv(TRACK_FILE, index=False)
    return df


def build_tracking_summary(limit: int = 10) -> str:
    df = update_tracking()

    if df.empty:
        return "📊 PHOENIX TAKİP\nHenüz kayıtlı aday yok."

    recent = df.tail(limit).copy()

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

    return "\n".join(lines)


if __name__ == "__main__":
    result = update_tracking()
    print(build_tracking_summary())
