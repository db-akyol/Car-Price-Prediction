# Car Price Prediction — Araç Fiyat Tahmini için Regresyon Modeli Karşılaştırması

Bu proje, ikinci el araçların marka, model, motor hacmi, kilometre, yakıt tipi, şanzıman ve yaş bilgilerinden yola çıkarak **satış fiyatını tahmin eden** bir makine öğrenmesi çalışmasıdır. Kaggle üzerindeki "Car Price Prediction" veri seti kullanılarak uçtan uca bir regresyon akışı kurulmuş; veri ön işleme, kategorik kodlama, keşifsel veri analizi (EDA), altı farklı regresyon modelinin karşılaştırılması ve hiperparametre optimizasyonu adımları tek bir Jupyter Notebook içinde uygulanmıştır. En iyi sonuç, hiperparametre ayarlı **Lasso Regression** ile **R² = 0.858** olarak elde edilmiştir.

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-Data%20Analysis-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numeric-013243?style=flat&logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=flat)
![seaborn](https://img.shields.io/badge/seaborn-EDA-4C72B0?style=flat)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat)

---

## Veri Seti

**Kaynak:** [Kaggle — Car Price Prediction (amjadzhour)](https://www.kaggle.com/datasets/amjadzhour/car-price-prediction/data)
**Dosya:** `Car_Price_Prediction.csv` · **Boyut:** 1.000 satır × 8 sütun · **Eksik değer yok, tekrarlı kayıt yok.**

| Sütun | Tip | Açıklama |
|---|---|---|
| `Make` | kategorik | Marka — Audi, BMW, Ford, Honda, Toyota |
| `Model` | kategorik | Model — Model A … Model E |
| `Year` | sayısal | Üretim yılı (2000–2021) |
| `Engine Size` | sayısal | Motor hacmi, litre (1.0–4.5) |
| `Mileage` | sayısal | Kilometre (56–199.867) |
| `Fuel Type` | kategorik | Yakıt tipi — Diesel, Electric, Petrol |
| `Transmission` | kategorik | Şanzıman — Manual, Automatic |
| **`Price`** | **sayısal** | **Hedef değişken** — araç fiyatı (6.704,95 – 41.780,50; ortalama ≈ 25.136) |

---

## Yöntem / İş Akışı

### 1. Veri İnceleme
- `df.info()`, `df.head()` ile yapı ve veri tipi kontrolü; `Year`, `Make` için benzersiz değer analizi.
- `groupby("Model")["Price"].mean()` ile model bazlı ortalama fiyat karşılaştırması.

### 2. Ön İşleme ve Feature Engineering
- **Ordinal/binary encoding:** `Transmission` sütunu `Manual → 0`, `Automatic → 1` olarak eşlendi.
- **One-hot encoding:** `Make`, `Fuel Type` ve `Model` sütunları `pd.get_dummies()` ile dummy değişkenlere dönüştürüldü.
- **Yeni özellik — `Age`:** Aracın yaşı `2022 - Year` formülüyle türetildi; ham `Year` sütunu veri setinden çıkarıldı.
- Tüm sütunlar modelleme öncesi `float` tipine çevrildi.

### 3. Keşifsel Veri Analizi (EDA)
`matplotlib` ve `seaborn` ile üretilen görseller:
- Araç yaşı (`Age`) dağılımı — histogram + KDE
- Fiyat (`Price`) dağılımı — histogram + KDE
- Tüm sayısal değişkenler arası **korelasyon ısı haritası**
- Motor hacmi (`Engine Size`) dağılımı
- Motor hacmi ↔ fiyat ilişkisi — scatter plot

### 4. Modelleme
- **Train/test ayrımı:** `train_test_split(test_size=0.25, random_state=15)` → 750 eğitim / 250 test
- **Ölçekleme:** `StandardScaler` yalnızca eğitim setine `fit` edilip test setine `transform` uygulandı (veri sızıntısı önlendi).
- **Karşılaştırılan modeller:** Linear Regression, Lasso, Ridge, K-Neighbors Regressor, Decision Tree Regressor, Random Forest Regressor
- **Değerlendirme metrikleri:** özel `calculate_model_metrics()` fonksiyonu ile MAE, RMSE ve R² — hem eğitim hem test seti üzerinde (overfitting tespiti için).

### 5. Hiperparametre Optimizasyonu
Doğrusal modeller için `RandomizedSearchCV` (n_iter=100) ve `GridSearchCV`, 3 katlı çapraz doğrulama (`cv=3`) ile çalıştırıldı:

| Model | Aranan parametreler | En iyi kombinasyon (GridSearchCV) |
|---|---|---|
| Lasso | `alpha`, `max_iter`, `tol` | `alpha=100`, `max_iter=1000`, `tol=1e-4` |
| Ridge | `alpha`, `solver`, `tol` | `alpha=0.1`, `solver="sag"`, `tol=1e-2` |
| Linear Regression | `fit_intercept`, `positive` | `fit_intercept=True`, `positive=False` |

---

## Sonuçlar

### Temel (varsayılan parametreli) modeller — Test seti

| Model | R² | RMSE | MAE |
|---|---:|---:|---:|
| **Lasso** | **0.8528** | 1975.79 | 1584.41 |
| Ridge | 0.8527 | 1976.29 | 1584.60 |
| Linear Regression | 0.8526 | 1976.79 | 1585.27 |
| Random Forest Regressor | 0.8068 | 2263.42 | 1807.97 |
| Decision Tree Regressor | 0.6194 | 3176.86 | 2523.06 |
| K-Neighbors Regressor | 0.4840 | 3699.18 | 3031.94 |

> **Overfitting gözlemi:** Decision Tree eğitim setinde R² = 1.000 / RMSE = 0.0, Random Forest ise R² = 0.971 değerine ulaşırken test performansları belirgin biçimde düştü. Doğrusal modeller ise eğitim (R² ≈ 0.835) ve test (R² ≈ 0.853) skorları arasında tutarlılık göstererek bu veri seti için daha genellenebilir sonuç verdi.

### Hiperparametre optimizasyonu sonrası — Test seti

| Model | R² | RMSE | MAE |
|---|---:|---:|---:|
| **Lasso Regression** (tuned) | **0.8582** | **1939.13** | **1557.61** |
| Ridge Regression (tuned) | 0.8532 | 1972.79 | 1579.22 |
| Linear Regression (tuned) | 0.8526 | 1976.79 | 1585.27 |

**En iyi model:** Hiperparametre ayarlı **Lasso Regression** — test setinde **R² = 0.8582**, **RMSE = 1939.13**, **MAE = 1557.61**. Ortalama fiyatın ≈ 25.136 olduğu bir veri setinde MAE'nin ~1.558 olması, tahminlerin ortalama fiyatın yaklaşık **%6'sı** kadar bir sapmayla üretildiği anlamına gelir.

---

## Kurulum ve Çalıştırma

### Gereksinimler
Python 3.13 (notebook bu sürümle çalıştırılmıştır) ve aşağıdaki kütüphaneler:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Adımlar

```bash
# 1. Depoyu klonlayın
git clone https://github.com/db-akyol/Car-Price-Prediction.git
cd Car-Price-Prediction

# 2. (Önerilir) Sanal ortam oluşturun
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Bağımlılıkları kurun
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# 4. Jupyter'i başlatın
jupyter notebook "Car Price Prediction.ipynb"
```

Notebook'taki hücreleri sırayla çalıştırmanız yeterlidir; veri seti (`Car_Price_Prediction.csv`) depo kök dizininde yer aldığı için ek bir indirme gerekmez.

> **Not:** `Random Forest` ve `Decision Tree` modellerinde `random_state` sabitlenmediği için bu iki modelin skorları çalıştırmalar arasında küçük farklılıklar gösterebilir. Doğrusal modeller ise deterministiktir.

---

## Dosya Yapısı

```
Car-Price-Prediction/
├── Car Price Prediction.ipynb   # Tüm analiz: ön işleme, EDA, modelleme, hiperparametre optimizasyonu
├── Car_Price_Prediction.csv     # Veri seti (1.000 satır × 8 sütun)
├── .gitignore                   # Jupyter/Python/editör dosyaları için hariç tutma kuralları
├── LICENSE                      # MIT Lisansı
└── README.md                    # Bu dosya
```

---

## Lisans

Bu proje **MIT Lisansı** ile lisanslanmıştır. Ayrıntılar için [LICENSE](LICENSE) dosyasına bakınız.

© 2025 Deniz Akyol
