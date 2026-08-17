from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os

import pandas as pd
import requests
import yfinance as yf


PROJECT = Path(__file__).resolve().parent
TRACK_FILE = PROJECT / "phoenix_tracking.csv"
PREDICTIONS_FILE = PROJECT / "phoenix_research_predictions.csv"

COLUMNS = [
    "signal_date",
    "symbol",
    "candidate_group",
    "signal_close",
    "phoenix_score",
    "calibrated_probability",
    "signal_type",
    "confidence",
    "day1_close_pct",
    "day2_close_pct",
    "day3_close_pct",
    "max_high_pct_3d",
    "hit_day",
    "status",
    "last_checked",
]

ACTIVE_STATUSES = {
    "TAKIPTE",
    "HEDEF_GORDU_TAKIPTE",
}

FINAL_STATUSES = {
    "BASARILI",
    "BASARISIZ",
}


def normalize_symbol(symbol: str) -> str:
    symbol = str(symbol).strip().upper()
    return symbol if symbol.endswith(".IS") else f"{symbol}.IS"


def clean_symbol(symbol: str) -> str:
    return str(symbol).strip().upper().replace(".IS", "")


def infer_candidate_group(signal_type: object, confidence: object) -> str:
    signal_text = str(signal_type).upper()
    confidence_text = str(confidence).upper()

    if (
        "YARIN %9+" in signal_text
        or confidence_text in {"ORTA", "YÜKSEK"}
    ):
        return "KESIN_ADAY"

    return "YAKIN_TAKIP"


def load_tracking() -> pd.DataFrame:
    if TRACK_FILE.exists():
        try:
            df = pd.read_csv(TRACK_FILE)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=COLUMNS)
    else:
        df = pd.DataFrame(columns=COLUMNS)

    # Eski tracking dosyalarıyla geriye dönük uyumluluk.
    if "candidate_group" not in df.columns:
        if not df.empty:
            df["candidate_group"] = [
                infer_candidate_group(
                    row.get("signal_type", ""),
                    row.get("confidence", ""),
                )
                for _, row in df.iterrows()
            ]
        else:
            df["candidate_group"] = pd.Series(dtype="object")

    for col in COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    if not df.empty:
        df["symbol"] = df["symbol"].astype(str).map(clean_symbol)
        df["candidate_group"] = (
            df["candidate_group"]
            .astype(str)
            .str.upper()
            .replace(
                {
                    "KESİN_ADAY": "KESIN_ADAY",
                    "YAKIN TAKİP": "YAKIN_TAKIP",
                    "YAKIN_TAKIP": "YAKIN_TAKIP",
                }
            )
        )

    return df[COLUMNS].copy()


def save_tracking(df: pd.DataFrame) -> None:
    output = df.copy()

    for col in COLUMNS:
        if col not in output.columns:
            output[col] = pd.NA

    output[COLUMNS].to_csv(
        TRACK_FILE,
        index=False,
        encoding="utf-8-sig",
    )


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def _fetch_signal_close(
    symbol: str,
    signal_date: pd.Timestamp,
) -> float | None:
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


def _active_mask(
    tracking: pd.DataFrame,
    symbol: str,
    candidate_group: str | None = None,
) -> pd.Series:
    if tracking.empty:
        return pd.Series([], dtype=bool)

    mask = (
        tracking["symbol"]
        .astype(str)
        .str.upper()
        .eq(symbol)
        & tracking["status"]
        .astype(str)
        .str.upper()
        .isin(ACTIVE_STATUSES)
    )

    if candidate_group is not None:
        mask &= (
            tracking["candidate_group"]
            .astype(str)
            .str.upper()
            .eq(candidate_group)
        )

    return mask


def _same_day_group_duplicate(
    tracking: pd.DataFrame,
    symbol: str,
    signal_date: pd.Timestamp,
    candidate_group: str,
) -> bool:
    if tracking.empty:
        return False

    mask = (
        tracking["signal_date"]
        .astype(str)
        .eq(signal_date.strftime("%Y-%m-%d"))
        & tracking["symbol"]
        .astype(str)
        .str.upper()
        .eq(symbol)
        & tracking["candidate_group"]
        .astype(str)
        .str.upper()
        .eq(candidate_group)
    )

    return bool(mask.any())


def capture_latest_predictions(
    strict_limit: int = 5,
    watch_limit: int = 5,
) -> int:
    """
    V3.3 araştırma çıktısındaki iki grubu ayrı ayrı kaydeder:

    KESIN_ADAY:
        Gerçek PHOENIX %9+ sinyalleri.

    YAKIN_TAKIP:
        Katı eşiğin altında kalan en güçlü deneysel grup.

    Aynı sembol aynı grup içinde hâlâ aktif takipteyse yeni kayıt açılmaz.
    YAKIN_TAKIP'teki bir sembol daha sonra KESIN_ADAY olursa,
    yeni KESIN_ADAY kaydı açılmasına izin verilir.
    """
    if not PREDICTIONS_FILE.exists():
        print("Tahmin dosyası bulunamadı; yeni aday kaydı yapılmadı.")
        return 0

    predictions = pd.read_csv(PREDICTIONS_FILE)

    if predictions.empty:
        print("Tahmin dosyası boş; yeni aday kaydı yapılmadı.")
        return 0

    required = {
        "date",
        "symbol",
        "candidate_group",
    }

    if not required.issubset(predictions.columns):
        missing = sorted(required - set(predictions.columns))
        print(
            "Tahmin dosyasında V3.3 için gerekli kolonlar yok: "
            f"{missing}"
        )
        return 0

    predictions["candidate_group"] = (
        predictions["candidate_group"]
        .astype(str)
        .str.upper()
    )

    strict_predictions = (
        predictions[
            predictions["candidate_group"].eq("KESIN_ADAY")
        ]
        .head(strict_limit)
        .copy()
    )

    watch_predictions = (
        predictions[
            predictions["candidate_group"].eq("YAKIN_TAKIP")
        ]
        .head(watch_limit)
        .copy()
    )

    selected = pd.concat(
        [strict_predictions, watch_predictions],
        ignore_index=True,
    )

    if selected.empty:
        print("Kaydedilecek kesin aday veya yakın takip adayı yok.")
        return 0

    tracking = load_tracking()
    added = 0

    for _, row in selected.iterrows():
        symbol = clean_symbol(row["symbol"])
        signal_date = (
            pd.to_datetime(row["date"])
            .tz_localize(None)
            .normalize()
        )
        candidate_group = str(row["candidate_group"]).upper()

        # Aynı grup içinde açık takip varsa tekrar sinyal açma.
        same_group_active = _active_mask(
            tracking,
            symbol,
            candidate_group,
        )

        if len(same_group_active) and same_group_active.any():
            print(
                f"{symbol}: {candidate_group} grubunda aktif takip var; "
                "yeni kayıt açılmadı."
            )
            continue

        # Daha güçlü kesin aday zaten aktifse, aynı sembolü ayrıca
        # yakın takip olarak açma.
        if candidate_group == "YAKIN_TAKIP":
            strict_active = _active_mask(
                tracking,
                symbol,
                "KESIN_ADAY",
            )

            if len(strict_active) and strict_active.any():
                print(
                    f"{symbol}: aktif KESIN_ADAY kaydı var; "
                    "YAKIN_TAKIP kaydı açılmadı."
                )
                continue

        if _same_day_group_duplicate(
            tracking,
            symbol,
            signal_date,
            candidate_group,
        ):
            print(
                f"{symbol} {signal_date.date()} {candidate_group}: "
                "zaten kayıtlı."
            )
            continue

        signal_close = _fetch_signal_close(
            symbol,
            signal_date,
        )

        if signal_close is None:
            print(
                f"{symbol}: sinyal kapanışı bulunamadı, kayıt atlandı."
            )
            continue

        new_row = {
            "signal_date": signal_date.strftime("%Y-%m-%d"),
            "symbol": symbol,
            "candidate_group": candidate_group,
            "signal_close": round(signal_close, 6),
            "phoenix_score": float(row.get("phoenix_score", 0.0)),
            "calibrated_probability": float(
                row.get("model_probability", 0.0)
            ),
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

        tracking = pd.concat(
            [
                tracking,
                pd.DataFrame([new_row]),
            ],
            ignore_index=True,
        )

        added += 1

        print(
            f"YENI KAYIT: {symbol} | {candidate_group} | "
            f"{signal_date.date()} | kapanış={signal_close:.4f}"
        )

    save_tracking(tracking)

    print(f"Yeni takip kaydı: {added}")
    print(f"Toplam takip kaydı: {len(tracking)}")

    return added


def _pct(
    value: float,
    base: float,
) -> float:
    if not base:
        return 0.0

    return (
        (float(value) / float(base)) - 1.0
    ) * 100.0


def update_tracking() -> pd.DataFrame:
    tracking = load_tracking()

    if tracking.empty:
        save_tracking(tracking)
        return tracking

    today = pd.Timestamp.now().normalize()

    for idx, row in tracking.iterrows():
        status = str(
            row.get("status", "")
        ).upper()

        if status in FINAL_STATUSES:
            continue

        symbol = clean_symbol(
            row["symbol"]
        )

        signal_date = (
            pd.to_datetime(
                row["signal_date"]
            )
            .tz_localize(None)
            .normalize()
        )

        try:
            signal_close = float(
                row["signal_close"]
            )
        except Exception:
            print(
                f"{symbol}: signal_close okunamadı."
            )
            continue

        try:
            hist = yf.download(
                normalize_symbol(symbol),
                start=(
                    signal_date
                    - pd.Timedelta(days=2)
                ).strftime("%Y-%m-%d"),
                end=(
                    today
                    + pd.Timedelta(days=2)
                ).strftime("%Y-%m-%d"),
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception as exc:
            print(
                f"{symbol}: takip verisi indirilemedi: {exc}"
            )
            continue

        if hist is None or hist.empty:
            continue

        hist = _flatten_columns(
            hist
        )

        hist.index = pd.to_datetime(
            hist.index
        ).tz_localize(None)

        # Yalnızca sinyal tarihinden sonraki gerçek işlem günleri.
        future = (
            hist[
                hist.index > signal_date
            ]
            .sort_index()
            .head(3)
            .copy()
        )

        if future.empty:
            tracking.loc[
                idx,
                "last_checked",
            ] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            continue

        for day_number in range(1, 4):
            if len(future) >= day_number:
                close_value = float(
                    future.iloc[
                        day_number - 1
                    ]["Close"]
                )

                tracking.loc[
                    idx,
                    f"day{day_number}_close_pct",
                ] = round(
                    _pct(
                        close_value,
                        signal_close,
                    ),
                    2,
                )

        hit_day = None
        max_high_pct = None

        for day_number in range(
            1,
            len(future) + 1,
        ):
            high_value = float(
                future.iloc[
                    day_number - 1
                ]["High"]
            )

            high_pct = round(
                _pct(
                    high_value,
                    signal_close,
                ),
                2,
            )

            if (
                max_high_pct is None
                or high_pct > max_high_pct
            ):
                max_high_pct = high_pct

            if (
                hit_day is None
                and high_pct >= 9.0
            ):
                hit_day = day_number

        tracking.loc[
            idx,
            "max_high_pct_3d",
        ] = max_high_pct

        tracking.loc[
            idx,
            "hit_day",
        ] = (
            hit_day
            if hit_day is not None
            else pd.NA
        )

        tracking.loc[
            idx,
            "last_checked",
        ] = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # %9 görülse bile 3 işlem günü tamamlanana kadar
        # izlemeye devam ediyoruz.
        if len(future) < 3:
            tracking.loc[
                idx,
                "status",
            ] = (
                "HEDEF_GORDU_TAKIPTE"
                if hit_day is not None
                else "TAKIPTE"
            )
        else:
            tracking.loc[
                idx,
                "status",
            ] = (
                "BASARILI"
                if hit_day is not None
                else "BASARISIZ"
            )

    save_tracking(tracking)

    return tracking


def _format_one_tracking_row(
    lines: list[str],
    row: pd.Series,
) -> None:
    lines.append("")
    lines.append(
        f"{row['symbol']} — {row['signal_date']}"
    )
    lines.append(
        f"Durum: {row['status']}"
    )

    if pd.notna(
        row["day1_close_pct"]
    ):
        lines.append(
            f"1. gün kapanış: "
            f"%{float(row['day1_close_pct']):+.2f}"
        )

    if pd.notna(
        row["day2_close_pct"]
    ):
        lines.append(
            f"2. gün kapanış: "
            f"%{float(row['day2_close_pct']):+.2f}"
        )

    if pd.notna(
        row["day3_close_pct"]
    ):
        lines.append(
            f"3. gün kapanış: "
            f"%{float(row['day3_close_pct']):+.2f}"
        )

    if pd.notna(
        row["max_high_pct_3d"]
    ):
        lines.append(
            f"3 günlük maksimum: "
            f"%{float(row['max_high_pct_3d']):+.2f}"
        )

    if pd.notna(
        row["hit_day"]
    ):
        lines.append(
            f"🎯 %9+ hedefi: "
            f"{int(float(row['hit_day']))}. işlem günü"
        )


def _group_statistics(
    tracking: pd.DataFrame,
    candidate_group: str,
) -> str:
    group = tracking[
        tracking[
            "candidate_group"
        ].astype(str).eq(
            candidate_group
        )
    ].copy()

    if group.empty:
        return "Henüz kayıt yok."

    finished = group[
        group[
            "status"
        ].astype(str).isin(
            FINAL_STATUSES
        )
    ]

    if finished.empty:
        return (
            f"Toplam {len(group)} kayıt; "
            "henüz tamamlanmış 3 günlük takip yok."
        )

    successful = int(
        finished[
            "status"
        ].astype(str).eq(
            "BASARILI"
        ).sum()
    )

    success_rate = (
        successful
        / len(finished)
        * 100.0
    )

    return (
        f"Tamamlanan {len(finished)} kayıt | "
        f"Başarılı {successful} | "
        f"Başarı %{success_rate:.1f}"
    )


def build_tracking_summary(
    per_group_limit: int = 8,
) -> str:
    tracking = load_tracking()

    if tracking.empty:
        return (
            "📊 PHOENIX TAKİP\n"
            "Henüz kayıtlı aday yok."
        )

    lines = [
        "📊 PHOENIX ADAY TAKİP",
        "",
        "🔥 KESİN ADAYLAR",
        _group_statistics(
            tracking,
            "KESIN_ADAY",
        ),
    ]

    strict = tracking[
        tracking[
            "candidate_group"
        ].astype(str).eq(
            "KESIN_ADAY"
        )
    ].tail(
        per_group_limit
    )

    if strict.empty:
        lines.append(
            "Henüz kesin aday kaydı yok."
        )
    else:
        for _, row in strict.iterrows():
            _format_one_tracking_row(
                lines,
                row,
            )

    lines.extend(
        [
            "",
            "👀 YAKIN TAKİP",
            _group_statistics(
                tracking,
                "YAKIN_TAKIP",
            ),
        ]
    )

    watch = tracking[
        tracking[
            "candidate_group"
        ].astype(str).eq(
            "YAKIN_TAKIP"
        )
    ].tail(
        per_group_limit
    )

    if watch.empty:
        lines.append(
            "Henüz yakın takip kaydı yok."
        )
    else:
        for _, row in watch.iterrows():
            _format_one_tracking_row(
                lines,
                row,
            )

    return "\n".join(
        lines
    )


def send_telegram(
    message: str,
) -> None:
    token = os.getenv(
        "TELEGRAM_BOT_TOKEN",
        "",
    ).strip()

    chat_id = os.getenv(
        "TELEGRAM_CHAT_ID",
        "",
    ).strip()

    if not token or not chat_id:
        print(
            "Telegram bilgileri yok; "
            "takip özeti sadece loga yazıldı."
        )
        return

    # Telegram mesaj sınırının altında güvenli parçalara ayır.
    for i in range(
        0,
        len(message),
        3500,
    ):
        chunk = message[
            i : i + 3500
        ]

        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": chunk,
            },
            timeout=30,
        )

        if not response.ok:
            print(
                "Telegram takip özeti gönderilemedi: "
                f"{response.status_code} "
                f"{response.text}"
            )


def main() -> None:
    # 1) Daha önce kaydedilen adayların sonuçlarını güncelle.
    update_tracking()

    # 2) Bugünün kesin aday + yakın takip listesini ayrı gruplarla ekle.
    capture_latest_predictions(
        strict_limit=5,
        watch_limit=5,
    )

    # 3) İki grubun özetini ayrı ayrı göster.
    summary = build_tracking_summary(
        per_group_limit=8,
    )

    print(
        summary
    )

    if not load_tracking().empty:
        send_telegram(
            summary
        )

    print(
        f"Takip dosyası: {TRACK_FILE}"
    )


if __name__ == "__main__":
    main()
