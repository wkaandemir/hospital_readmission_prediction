# Hastane Yeniden Yatış Risk Tahmini - Analiz Raporu

Oluşturulma tarihi: 2025-08-01 17:20:57

## Yönetici Özeti
Bu analiz, 30 günlük hastane yeniden yatış riskini tahmin etmek için makine öğrenmesi modelleri geliştirdi.

### Temel Sonuçlar:
- En İyi Model: XGBoost
- ROC-AUC: 0.6314
- Accuracy: 0.8779
- F1-Score: 0.0907

### Veri Seti Genel Bakışı:
- Toplam örnek: 101,763
- Özellikler: 48
- Yeniden yatış oranı: 0.112

## Model Performansı

| Model | ROC-AUC | Accuracy | F1-Score |
|-------|---------|----------|----------|
| XGBoost | 0.6314 | 0.8779 | 0.0907 |
| Random Forest | 0.5601 | 0.8112 | 0.1393 |
| Logistic Regression | 0.5702 | 0.8787 | 0.0818 |

## En Önemli Risk Faktörleri
1. change: 0.9092
2. age_numeric: 0.6824
3. age: 0.6397
4. los_category: 0.4755
5. insulin: 0.4054
6. time_in_hospital: 0.3999
7. metformin: 0.2890
8. discharge_disposition_id: 0.2471
9. payer_code: 0.2466
10. admission_source_id: 0.2277

## Oluşturulan Dosyalar
- `models/`: Eğitilmiş modeller ve ön işleyici
- `visualizations/model_evaluation.png`: Performans grafikleri
- `models/feature_importance.csv`: Özellik sıralamaları
- `models/model_performance_summary.csv`: Performans metrikleri

## Hedef Başarımı
❌ ROC-AUC hedefi kaçırıldı: 0.6314 < 0.75
✅ Doğruluk ≥ 0.70 başarıldı
❌ F1-Skor hedefi kaçırıldı: 0.0907 < 0.68

**Genel: 1/3 hedef başarıldı**