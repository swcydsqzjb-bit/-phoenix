# PHOENIX Research v1

PHOENIX, bir hissenin **yalnızca son üç tamamlanmış işlem günündeki** fiyat ve hacim davranışını kullanarak ertesi işlem gününde kapanıştan kapanışa `%9 veya üzeri` hareket ihtimalini araştırır.

## Kesin sınırlar

Kullanılan pencere yalnızca `T-3`, `T-2`, `T-1` günleridir. RSI, EMA, MACD, Bollinger, haber, KAP, sektör ve eski bot skorları kullanılmaz.

Modelin gördüğü bilgiler: açık, yüksek, düşük, kapanış ve hacimden türetilen mum gövdesi, fitiller, kapanış konumu, üç günlük dip/tepe ilişkisi, aralık daralması ve üç gün içindeki hacim oranlarıdır.

## Dosyalar

- `phoenix_research.py`: veri indirme, özellik çıkarma, tarih sıralı test, eğitim, benzerlik ve Telegram raporu
- `symbols.csv`: taranacak BIST kodları; `.IS` eklemek gerekmez
- `requirements.txt`: Python paketleri
- `.github/workflows/phoenix.yml`: manuel ve hafta içi otomatik çalışma

## İlk çalıştırma

GitHub'da **Actions → PHOENIX Research → Run workflow** yolunu açın.

İlk çalıştırmada:

- `retrain`: `true`
- `refresh`: `true`

seçili olmalıdır. İlk eğitim, sembol sayısına ve Yahoo Finance hızına göre uzun sürebilir.

## Telegram ayarları

Repository içinde **Settings → Secrets and variables → Actions → New repository secret** bölümüne iki secret ekleyin:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

Bu secret'lar eklenmezse sistem yine çalışır; rapor yalnızca Actions ekranında ve artifact dosyalarında kalır.

## Üretilen çıktılar

- `phoenix_research_predictions.csv`: güncel sıralama
- `phoenix_research_metrics.json`: tarih sıralı test ölçümleri ve sinyal eşiği
- `phoenix_research_training.csv`: araştırma veri seti
- `phoenix_research_model.joblib`: eğitilmiş model ve benzerlik hafızası

## Ölçümlerin anlamı

`threshold_precision`, geçmişte modelin sinyal verdiği olayların ne kadarının gerçekten `%9+` olduğudur. `threshold_recall`, gerçekleşen `%9+` olayların ne kadarının yakalandığını gösterir. Bu değerler yeterli değilse sistem başarılı kabul edilmemelidir.

## Uyarı

Bu proje yatırım tavsiyesi veya getiri garantisi değildir. Nadir bir olayı tarihsel veriler üzerinde araştıran karar destek deneyidir. Canlı kullanım öncesinde ileri tarihli kâğıt üzerinde test edilmelidir.
