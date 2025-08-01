# 🏥 Kurulum Rehberi

## Hızlı Kurulum

```bash
# 1. Gereksinimleri yükle
pip3 install -r requirements.txt

# 2. Tüm analizi çalıştır
python3 run_analysis.py
```

## Sistem Gereksinimleri

- **Python**: 3.8+
- **RAM**: En az 4GB (8GB önerilir)
- **Disk**: 2GB boş alan
- **İnternet**: Veri seti indirmesi için

## Paket Gereksinimleri

```bash
pip3 install pandas numpy scikit-learn xgboost matplotlib seaborn imbalanced-learn shap joblib
```

## Çıktılar

Analiz tamamlandığında şu dosyalar oluşur:
- `models/` - Eğitilmiş modeller
- `visualizations/` - Grafikler
- `reports/` - Analiz raporu
- `data/` - İndirilen veri seti

## Sorun Giderme

**Paket kurulum hatası:**
```bash
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

**Bellek hatası:**
- Daha az veri kullanmak için `run_analysis.py` dosyasında `sample_size` değerini azaltın

**İnternet bağlantısı:**
- UCI veri seti otomatik indirilir, internet bağlantısı gereklidir