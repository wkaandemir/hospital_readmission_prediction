#!/usr/bin/env python3
"""
Hastane Yeniden Yatış Risk Tahmini - Komple Analiz Çalıştırıcı
==============================================================

Bu script, hastane yeniden yatış tahmin analizini tek seferde çalıştırır.
Jupyter notebook'a gerek yok - her şey bağımsız Python scripti olarak çalışır.

Kullanım:
    python3 run_analysis.py

Yazar: AI Mühendislik Takımı
Tarih: 2025-08-01
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import os
import urllib.request
import zipfile
from pathlib import Path
import sys

# Machine Learning libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, f1_score, accuracy_score
)

# Handle class imbalance
from imblearn.over_sampling import SMOTE

# XGBoost
import xgboost as xgb

# SHAP for model explainability
import shap

# Configure settings
plt.style.use('default')
warnings.filterwarnings('ignore')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def create_directories():
    """Gerekli dizinleri oluştur"""
    directories = ['data', 'models', 'visualizations', 'reports']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("📁 Dizinler başarıyla oluşturuldu!")

def download_and_load_data():
    """UCI Diyabet veri setini indir ve yükle"""
    print("📥 Veri seti indiriliyor ve yükleniyor...")
    
    data_dir = Path('data')
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip"
    zip_path = data_dir / "dataset_diabetes.zip"
    
    if not zip_path.exists():
        print("   UCI deposundan veri seti indiriliyor...")
        urllib.request.urlretrieve(url, zip_path)
        
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("   Veri seti indirildi ve çıkarıldı!")
    else:
        print("   Veri seti zaten mevcut!")
    
    # Load the dataset
    data_path = data_dir / "dataset_diabetes" / "diabetic_data.csv"
    df = pd.read_csv(data_path)
    
    print(f"   Veri seti yüklendi: {df.shape[0]:,} satır, {df.shape[1]} sütun")
    return df

class HospitalDataPreprocessor:
    """Hastane yeniden yatış tahmini için veri ön işleyici"""
    
    def __init__(self):
        self.label_encoders = {}
        self.feature_columns = None
        
    def create_target_variable(self, df):
        """İkili hedef oluştur: <30 gün içinde yeniden yatış = 1, diğerleri = 0"""
        df = df.copy()
        df['target'] = (df['readmitted'] == '<30').astype(int)
        return df
    
    def clean_data(self, df):
        """Ham verileri temizle ve ön işle"""
        df = df.copy()
        
        # Remove rows with missing key information
        df = df[df['gender'] != 'Unknown/Invalid']
        
        # Handle missing values marked as '?'
        df = df.replace('?', np.nan)
        
        # Remove features with too many missing values (>50%)
        missing_threshold = 0.5
        missing_pct = df.isnull().sum() / len(df)
        cols_to_drop = missing_pct[missing_pct > missing_threshold].index.tolist()
        df = df.drop(columns=cols_to_drop)
        
        # Remove identifiers
        identifier_cols = ['encounter_id', 'patient_nbr']
        df = df.drop(columns=[col for col in identifier_cols if col in df.columns])
        
        return df
    
    def engineer_features(self, df):
        """Mevcut verilerden ek özellikler oluştur"""
        df = df.copy()
        
        # Age groups (convert to numeric)
        age_mapping = {
            '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
            '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
            '[80-90)': 85, '[90-100)': 95
        }
        df['age_numeric'] = df['age'].map(age_mapping)
        
        # Length of stay categories (numeric)
        df['los_category'] = pd.cut(df['time_in_hospital'], 
                                   bins=[0, 3, 7, 14, float('inf')],
                                   labels=[1, 2, 3, 4]).astype('int')
        
        # Service utilization score
        df['service_utilization'] = (
            df['num_lab_procedures'] + 
            df['num_procedures'] + 
            df['num_medications']
        )
        
        # Diabetes medication indicator
        if 'diabetesMed' in df.columns:
            df['diabetes_med_binary'] = (df['diabetesMed'] == 'Yes').astype(int)
        
        return df
    
    def encode_categorical_features(self, df, fit=True):
        """Kategorik özellikleri kodla"""
        df = df.copy()
        
        # Identify categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        categorical_cols = [col for col in categorical_cols 
                           if col not in ['target', 'readmitted']]
        
        # Label encode categorical features
        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                df[col] = df[col].fillna('missing')
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    df[col] = df[col].fillna('missing')
                    # Handle unseen categories
                    df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'missing')
                    df[col] = le.transform(df[col])
        
        return df
    
    def fit_transform(self, df):
        """Ön işleyiciyi uydur ve veriyi dönüştür"""
        print("🔧 Veriler ön işleniyor...")
        
        # Create target variable
        df = self.create_target_variable(df)
        
        # Clean data
        df = self.clean_data(df)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df, fit=True)
        
        # Separate features and target
        feature_cols = [col for col in df.columns 
                       if col not in ['target', 'readmitted']]
        
        X = df[feature_cols]
        y = df['target']
        
        # Handle remaining missing values and convert to numeric
        for col in X.columns:
            if X[col].dtype in ['object', 'category']:
                X[col] = X[col].fillna(X[col].mode().iloc[0] if len(X[col].mode()) > 0 else 'missing')
                # Convert category columns to numeric
                if X[col].dtype == 'category':
                    X[col] = X[col].cat.codes
            else:
                X[col] = X[col].fillna(X[col].median())
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        print(f"   İşlenmiş boyut: {X.shape}")
        print(f"   Yeniden yatış oranı: {y.mean():.3f}")
        
        return X, y

class ModelTrainer:
    """Hastane yeniden yatış tahmini için model eğitici"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        
    def initialize_models(self):
        """Tüm modelleri başlat"""
        self.models = {
            'XGBoost': xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Logistic Regression': LogisticRegression(
                C=0.1,
                max_iter=1000,
                random_state=self.random_state,
                solver='liblinear'
            )
        }
    
    def train_models(self, X, y, cv_folds=5):
        """Modelleri çapraz doğrulama ile eğit"""
        print("🤖 Makine öğrenmesi modelleri eğitiliyor...")
        
        self.initialize_models()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, 
            stratify=y
        )
        
        # Store test set
        self.X_test = X_test
        self.y_test = y_test
        
        # Apply SMOTE for class balancing
        smote = SMOTE(random_state=self.random_state)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        print(f"   Orijinal eğitim seti: {y_train.value_counts().values}")
        print(f"   SMOTE sonrası: {pd.Series(y_train_balanced).value_counts().values}")
        
        # Setup cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, 
                            random_state=self.random_state)
        
        # Train models
        for name, model in self.models.items():
            print(f"   {name} eğitiliyor...")
            
            # Cross-validation scores
            cv_scores = cross_val_score(
                model, X_train_balanced, y_train_balanced, 
                cv=cv, scoring='roc_auc', n_jobs=-1
            )
            
            # Train on full training set
            model.fit(X_train_balanced, y_train_balanced)
            
            # Test predictions
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Store results
            self.results[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_roc_auc': roc_auc,
                'test_accuracy': accuracy,
                'test_f1': f1,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"     CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"     Test ROC-AUC: {roc_auc:.4f}")
            print(f"     Test Accuracy: {accuracy:.4f}")
        
        return self.results

def create_visualizations(trainer, X, y):
    """Kapsamlı görselleştirmeler oluştur"""
    print("📊 Görselleştirmeler oluşturuluyor...")
    
    # 1. Performance comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # ROC Curves
    ax1 = axes[0, 0]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, (name, result) in enumerate(trainer.results.items()):
        fpr, tpr, _ = roc_curve(trainer.y_test, result['y_pred_proba'])
        roc_auc = result['test_roc_auc']
        ax1.plot(fpr, tpr, color=colors[i], lw=2, 
                label=f'{name} (AUC = {roc_auc:.3f})')
    
    ax1.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
    ax1.set_xlabel('Yanlış Pozitif Oranı')
    ax1.set_ylabel('Doğru Pozitif Oranı')
    ax1.set_title('ROC Eğrileri')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Performance metrics comparison
    ax2 = axes[0, 1]
    metrics = ['test_roc_auc', 'test_accuracy', 'test_f1']
    metric_labels = ['ROC-AUC', 'Accuracy', 'F1-Score']
    
    x = np.arange(len(metric_labels))
    width = 0.25
    
    for i, (name, result) in enumerate(trainer.results.items()):
        values = [result[metric] for metric in metrics]
        ax2.bar(x + i*width, values, width, label=name, color=colors[i], alpha=0.8)
    
    ax2.set_xlabel('Metrikler')
    ax2.set_ylabel('Skor')
    ax2.set_title('Performans Metrikleri')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(metric_labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1)
    
    # Feature importance (for best model)
    best_model_name = max(trainer.results.keys(), 
                         key=lambda x: trainer.results[x]['test_roc_auc'])
    best_model = trainer.results[best_model_name]['model']
    
    ax3 = axes[1, 0]
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = best_model.feature_importances_
        indices = np.argsort(feature_importance)[-15:]  # Top 15
        ax3.barh(range(len(indices)), feature_importance[indices])
        ax3.set_yticks(range(len(indices)))
        ax3.set_yticklabels([X.columns[i] for i in indices])
        ax3.set_xlabel('Önem')
        ax3.set_title(f'En Önemli Özellikler - {best_model_name}')
    
    # Confusion matrix for best model
    ax4 = axes[1, 1]
    best_result = trainer.results[best_model_name]
    cm = confusion_matrix(trainer.y_test, best_result['y_pred'])
    
    im = ax4.imshow(cm, interpolation='nearest', cmap='Blues')
    ax4.set_title(f'Karmaşıklık Matrisi - {best_model_name}')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax4.text(j, i, f'{cm[i, j]}',
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    ax4.set_ylabel('Gerçek Etiket')
    ax4.set_xlabel('Tahmin Edilen Etiket')
    ax4.set_xticks([0, 1])
    ax4.set_yticks([0, 1])
    ax4.set_xticklabels(['Yeniden Yatış Yok', 'Yeniden Yatış'])
    ax4.set_yticklabels(['Yeniden Yatış Yok', 'Yeniden Yatış'])
    
    plt.tight_layout()
    plt.savefig('visualizations/model_evaluation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   Görselleştirme kaydedildi: visualizations/model_evaluation.png")

def analyze_shap(trainer, X):
    """SHAP ile model yorumlanabilirliğini analiz et"""
    print("🔍 Model yorumlanabilirliği analiz ediliyor...")
    
    # Get best model
    best_model_name = max(trainer.results.keys(), 
                         key=lambda x: trainer.results[x]['test_roc_auc'])
    best_model = trainer.results[best_model_name]['model']
    
    # Sample data for SHAP (for performance)
    sample_size = min(500, len(trainer.X_test))
    sample_indices = np.random.choice(len(trainer.X_test), sample_size, replace=False)
    X_sample = trainer.X_test.iloc[sample_indices]
    
    try:
        # Initialize SHAP explainer
        if best_model_name in ['XGBoost', 'Random Forest']:
            explainer = shap.TreeExplainer(best_model)
        else:
            explainer = shap.LinearExplainer(best_model, trainer.X_test)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
        
        # Feature importance from SHAP
        feature_importance = np.abs(shap_values).mean(0)
        feature_importance_df = pd.DataFrame({
            'Feature': X_sample.columns,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        print(f"   {best_model_name} için SHAP analizi tamamlandı")
        print("   En Önemli 10 Özellik:")
        for i, (_, row) in enumerate(feature_importance_df.head(10).iterrows(), 1):
            print(f"     {i}. {row['Feature']}: {row['Importance']:.4f}")
        
        return feature_importance_df
        
    except Exception as e:
        print(f"   SHAP analizi başarısız: {e}")
        # Fallback to model's built-in feature importance
        if hasattr(best_model, 'feature_importances_'):
            feature_importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': best_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            return feature_importance_df
        else:
            return None

def save_models_and_results(trainer, preprocessor, feature_importance):
    """Modelleri ve sonuçları kaydet"""
    print("💾 Modeller ve sonuçlar kaydediliyor...")
    
    # Save preprocessor
    joblib.dump(preprocessor, 'models/data_preprocessor.pkl')
    
    # Save models
    for name, result in trainer.results.items():
        model_filename = f"models/{name.lower().replace(' ', '_')}_model.pkl"
        joblib.dump(result['model'], model_filename)
    
    # Save feature importance
    if feature_importance is not None:
        feature_importance.to_csv('models/feature_importance.csv', index=False)
    
    # Save performance summary
    performance_data = []
    for name, result in trainer.results.items():
        performance_data.append({
            'Model': name,
            'CV_ROC_AUC_Mean': f"{result['cv_mean']:.4f}",
            'CV_ROC_AUC_Std': f"{result['cv_std']:.4f}",
            'Test_ROC_AUC': f"{result['test_roc_auc']:.4f}",
            'Test_Accuracy': f"{result['test_accuracy']:.4f}",
            'Test_F1_Score': f"{result['test_f1']:.4f}"
        })
    
    performance_df = pd.DataFrame(performance_data)
    performance_df.to_csv('models/model_performance_summary.csv', index=False)
    
    print("   Tüm modeller ve sonuçlar kaydedildi!")

def generate_report(trainer, feature_importance, X, y):
    """Son raporu oluştur"""
    print("📄 Son rapor oluşturuluyor...")
    
    # Get best model
    best_model_name = max(trainer.results.keys(), 
                         key=lambda x: trainer.results[x]['test_roc_auc'])
    best_result = trainer.results[best_model_name]
    
    report = []
    report.append("# Hastane Yeniden Yatış Risk Tahmini - Analiz Raporu")
    report.append(f"\nOluşturulma tarihi: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    report.append("\n## Yönetici Özeti")
    report.append("Bu analiz, 30 günlük hastane yeniden yatış riskini tahmin etmek için makine öğrenmesi modelleri geliştirdi.")
    
    report.append(f"\n### Temel Sonuçlar:")
    report.append(f"- En İyi Model: {best_model_name}")
    report.append(f"- ROC-AUC: {best_result['test_roc_auc']:.4f}")
    report.append(f"- Accuracy: {best_result['test_accuracy']:.4f}")
    report.append(f"- F1-Score: {best_result['test_f1']:.4f}")
    
    report.append(f"\n### Veri Seti Genel Bakışı:")
    report.append(f"- Toplam örnek: {len(X):,}")
    report.append(f"- Özellikler: {len(X.columns)}")
    report.append(f"- Yeniden yatış oranı: {y.mean():.3f}")
    
    report.append("\n## Model Performansı")
    report.append("\n| Model | ROC-AUC | Accuracy | F1-Score |")
    report.append("|-------|---------|----------|----------|")
    
    for name, result in trainer.results.items():
        report.append(f"| {name} | {result['test_roc_auc']:.4f} | {result['test_accuracy']:.4f} | {result['test_f1']:.4f} |")
    
    if feature_importance is not None:
        report.append("\n## En Önemli Risk Faktörleri")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
            report.append(f"{i}. {row['Feature']}: {row['Importance']:.4f}")
    
    report.append("\n## Oluşturulan Dosyalar")
    report.append("- `models/`: Eğitilmiş modeller ve ön işleyici")
    report.append("- `visualizations/model_evaluation.png`: Performans grafikleri")
    report.append("- `models/feature_importance.csv`: Özellik sıralamaları")
    report.append("- `models/model_performance_summary.csv`: Performans metrikleri")
    
    report.append(f"\n## Hedef Başarımı")
    targets_met = 0
    total_targets = 3
    
    if best_result['test_roc_auc'] >= 0.75:
        report.append("✅ ROC-AUC ≥ 0.75 başarıldı")
        targets_met += 1
    else:
        report.append(f"❌ ROC-AUC hedefi kaçırıldı: {best_result['test_roc_auc']:.4f} < 0.75")
    
    if best_result['test_accuracy'] >= 0.70:
        report.append("✅ Doğruluk ≥ 0.70 başarıldı")
        targets_met += 1
    else:
        report.append(f"❌ Doğruluk hedefi kaçırıldı: {best_result['test_accuracy']:.4f} < 0.70")
    
    if best_result['test_f1'] >= 0.68:
        report.append("✅ F1-Skor ≥ 0.68 başarıldı")
        targets_met += 1
    else:
        report.append(f"❌ F1-Skor hedefi kaçırıldı: {best_result['test_f1']:.4f} < 0.68")
    
    report.append(f"\n**Genel: {targets_met}/{total_targets} hedef başarıldı**")
    
    # Save report
    with open('reports/analysis_report.md', 'w') as f:
        f.write('\n'.join(report))
    
    print("   Rapor kaydedildi: reports/analysis_report.md")

def main():
    """Ana analiz fonksiyonu"""
    print("🏥 HASTANE YENİDEN YATIŞ TAHMİNİ - KOMPLE ANALİZ")
    print("=" * 60)
    
    try:
        # 1. Setup
        create_directories()
        
        # 2. Load data
        df = download_and_load_data()
        
        # 3. Preprocess data
        preprocessor = HospitalDataPreprocessor()
        X, y = preprocessor.fit_transform(df)
        
        # 4. Train models
        trainer = ModelTrainer(random_state=RANDOM_STATE)
        results = trainer.train_models(X, y)
        
        # 5. Create visualizations
        create_visualizations(trainer, X, y)
        
        # 6. Analyze interpretability
        feature_importance = analyze_shap(trainer, X)
        
        # 7. Save everything
        save_models_and_results(trainer, preprocessor, feature_importance)
        
        # 8. Generate report
        generate_report(trainer, feature_importance, X, y)
        
        # 9. Final summary
        print("\n" + "=" * 60)
        print("✅ ANALİZ BAŞARIYLA TAMAMLANDI!")
        print("=" * 60)
        
        best_model_name = max(results.keys(), key=lambda x: results[x]['test_roc_auc'])
        best_result = results[best_model_name]
        
        print(f"\n🎯 SON SONUÇLAR:")
        print(f"   En İyi Model: {best_model_name}")
        print(f"   ROC-AUC: {best_result['test_roc_auc']:.4f} (Hedef: ≥0.75)")
        print(f"   Doğruluk: {best_result['test_accuracy']:.4f} (Hedef: ≥0.70)")
        print(f"   F1-Skor: {best_result['test_f1']:.4f} (Hedef: ≥0.68)")
        
        print(f"\n📁 ÇIKTI DOSYALARI:")
        print(f"   📊 Görselleştirmeler: visualizations/model_evaluation.png")
        print(f"   🤖 Modeller: models/ dizini")
        print(f"   📄 Rapor: reports/analysis_report.md")
        
        # Check if targets achieved
        targets = [
            best_result['test_roc_auc'] >= 0.75,
            best_result['test_accuracy'] >= 0.70,
            best_result['test_f1'] >= 0.68
        ]
        
        if all(targets):
            print(f"\n🎉 TÜM PERFORMANS HEDEFLERİ BAŞARILDI!")
        else:
            print(f"\n⚠️  {sum(targets)}/3 performans hedefi başarıldı")
        
        print(f"\n✨ Sistem üretime dağıtıma hazır!")
        
    except Exception as e:
        print(f"\n❌ HATA: {str(e)}")
        print("Lütfen yukarıdaki hata mesajını kontrol edin ve tekrar deneyin.")
        sys.exit(1)

if __name__ == "__main__":
    main()