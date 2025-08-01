# 🏥 Hastane Yeniden Yatış Risk Tahmini

Bu proje, hastaların taburcu olduktan sonraki **30 gün içinde yeniden hastaneye yatış** riskini makine öğrenmesi ile tahmin eden kapsamlı bir sistemdir.

## 🎯 Proje Amacı

- Yüksek riskli hastaları önceden belirlemek
- Proaktif bakım sağlamak
- Gereksiz tekrar yatışları azaltarak kaynak tasarrufu sağlamak
- Sağlık kurumlarına klinik karar desteği sunmak

## 📊 Veri Kümesi

**UCI Diabetes 130-US Hospitals Dataset (1999-2008)**
- 100,000+ hasta ziyareti
- 130 hastaneden 10 yıllık veriler
- Demografik bilgiler, tanı kodları, ilaçlar, prosedürler

## 🤖 Makine Öğrenmesi Modelleri

- **XGBoost**: En iyi performans gösteren model
- **Random Forest**: Ensemble yöntemi
- **Logistic Regression**: Yorumlanabilir temel model

## 🚀 Tek Komutla Çalıştırma

```bash
# Gereksinimler
pip3 install -r requirements.txt

# Tüm analizi çalıştır
python3 run_analysis.py
```

Bu komut otomatik olarak:
- UCI veri setini indirir
- Verileri işler ve temizler
- 3 makine öğrenmesi modelini eğitir
- Performans grafiklerini oluşturur
- SHAP ile model açıklamasını yapar
- Kapsamlı rapor hazırlar

## 📁 Çıktı Dosyaları

### 🤖 Eğitilmiş Modeller
- `models/xgboost_model.pkl` - En iyi model
- `models/random_forest_model.pkl` - Orman modeli
- `models/logistic_regression_model.pkl` - Lojistik regresyon
- `models/data_preprocessor.pkl` - Veri ön işleme pipeline

### 📊 Analizler
- `visualizations/model_evaluation.png` - Performans grafikleri
- `models/feature_importance.csv` - Özellik önem sıralaması
- `models/model_performance_summary.csv` - Model karşılaştırması

### 📄 Raporlar
- `reports/analysis_report.md` - Kapsamlı analiz raporu

## 🎯 Model Performansı

| Model | ROC-AUC | Doğruluk | F1-Skor |
|-------|---------|----------|---------|
| XGBoost | 0.6314 | 87.79% | 0.0907 |
| Random Forest | 0.5601 | 81.12% | 0.0721 |
| Logistic Regression | 0.5702 | 87.87% | 0.0905 |

## 🔍 En Önemli Risk Faktörleri

1. **İlaç Değişiklikleri** (change) - En güçlü gösterge
2. **Yaş** (age_numeric) - Yaşlı hastalar daha riskli
3. **Hastanede Kalış Süresi** (los_category) - Uzun kalış riski artırır
4. **İnsülin Kullanımı** (insulin) - Diyabet yönetimi kritik
5. **Toplam Hastane Günü** (time_in_hospital) - Süre ile risk artar

## 🏥 Klinik Öneriler

### Yüksek Riskli Hastalar için:
- Gelişmiş taburcu planlaması
- 48-72 saat içinde takip
- İlaç uyumluluğu kontrolü
- Bakım koordinasyonu

### Sistem Entegrasyonu:
- EHR sistemleri ile entegrasyon
- Gerçek zamanlı risk değerlendirmesi
- Üç aylık model yenileme
- Veri kayması izleme

## 📋 Gereksinimler

```
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.6.0
matplotlib>=3.5.0
seaborn>=0.11.0
imbalanced-learn>=0.8.0
shap>=0.40.0
joblib>=1.1.0
```

## 🎉 Başarı Kriterleri

- ✅ **Doğruluk ≥ 70%**: 87.79% (Hedef aşıldı)
- ⚠️ **ROC-AUC ≥ 75%**: 63.14% (Geliştirme gerekli)
- ⚠️ **F1-Skor ≥ 68%**: 9.07% (Sınıf dengesizliği etkisi)

## 📞 Destek

Proje ile ilgili sorularınız için issue açabilir veya katkıda bulunabilirsiniz.

---
*Bu proje sağlık sektöründe yapay zeka uygulamaları için geliştirilmiştir. Klinik kararlar alırken her zaman sağlık profesyonellerine danışınız.*