# Customer Churn Prediction & Customer Feedback Analysis

Otomotiv sektöründe müşteri terk (customer churn) davranışlarını analiz etmek, makine öğrenmesi modelleriyle müşteri terk riskini tahmin etmek ve müşteri geri bildirimlerini Büyük Dil Modelleri (LLM) kullanarak sınıflandırmak amacıyla geliştirilmiş bir veri analizi projesidir.

## Proje Hakkında

Bu projede müşteri, araç, servis kayıtları ve müşteri geri bildirimlerinden oluşan veri setleri kullanılarak iki temel çalışma gerçekleştirilmiştir:

- **Customer Churn Analysis:** Müşterilerin geçmiş servis davranışları analiz edilerek churn tahmini için özellikler oluşturulmuş ve farklı makine öğrenmesi modelleri değerlendirilmiştir.
- **Customer Feedback Analysis:** Müşteri geri bildirimleri Hugging Face Transformers kullanılarak duygu ve konu kategorilerine göre otomatik olarak sınıflandırılmıştır.

Proje kapsamında veri ön işleme, veri kalitesi kontrolleri, keşifsel veri analizi (EDA), özellik mühendisliği, model değerlendirme ve otomatik metin etiketleme adımları uygulanmıştır.

## Kullanılan Teknolojiler

| Teknoloji / Kütüphane | Kullanım Alanı |
|---|---|
| Python | Veri analizi ve model geliştirme |
| Pandas | Veri işleme ve analizi |
| NumPy | Sayısal işlemler |
| Matplotlib | Veri görselleştirme |
| Scikit-learn | Makine öğrenmesi ve model değerlendirme |
| Hugging Face Transformers | LLM tabanlı metin sınıflandırma |
| Qwen2.5-1.5B-Instruct | Müşteri geri bildirimlerini otomatik etiketleme |
| PyTorch | Dil modelinin çalıştırılması |

## Proje Klasör Yapısı

```text
case_study/
│
├── veri/
│   ├── musteriler.csv
│   ├── araclar.csv
│   ├── servis_kayitlari.csv
│   └── geri_bildirimler.csv
│
├── kod/
│   ├── pipeline.py
│   └── llm_etiketleme.py
│
├── ciktilar/
│   ├── altin_set.csv
│   ├── llm_etiketleri.csv
│   └── ...
│
├── Rapor/
│   └── ...
│
├── README.md
└── ...
```

### Klasör ve Dosyaların Açıklamaları

**`veri/`**

Projede kullanılan ham veri setlerini içerir.

- `musteriler.csv`: Müşteri bilgileri
- `araclar.csv`: Müşterilere ait araç bilgileri
- `servis_kayitlari.csv`: Servis işlem kayıtları
- `geri_bildirimler.csv`: Müşteri geri bildirimleri

**`kod/`**

Projenin analiz ve modelleme süreçlerinde kullanılan Python dosyalarını içerir.

- `pipeline.py`: Veri ön işleme, keşifsel veri analizi, özellik mühendisliği, churn analizi ve makine öğrenmesi süreçlerini yürütür.
- `llm_etiketleme.py`: Müşteri geri bildirimlerinin LLM kullanılarak duygu ve konu kategorilerine göre sınıflandırılmasını ve etiketleme sonuçlarının değerlendirilmesini sağlar.

**`ciktilar/`**

Kodların çalıştırılması sonucunda oluşturulan dosyaları içerir. Analiz sonuçları, model çıktıları, otomatik etiketlenmiş geri bildirimler ve manuel değerlendirme için kullanılan veri setleri bu klasörde saklanır.

- `altin_set.csv`: LLM etiketleme performansını değerlendirmek için manuel olarak etiketlenen örnekler.
- `llm_etiketleri.csv`: Otomatik etiketleme sonucunda oluşturulan geri bildirim etiketleri.

**`Rapor/`**

Projenin yöntemlerini, analizlerini, elde edilen bulguları ve önerileri ayrıntılı olarak açıklayan raporu içerir.

## Kurulum

Projeyi yerel bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz.

### 1. Projeyi klonlayın

```bash
git clone <repository-url>
cd case_study
```

### 2. Sanal ortam oluşturun

```bash
python -m venv venv
```

Sanal ortamı etkinleştirin.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 3. Gerekli kütüphaneleri yükleyin

Projede kullanılan kütüphaneleri yükleyin:

```bash
pip install pandas numpy matplotlib scikit-learn transformers torch
```

## Çalıştırma

Proje, iki temel Python dosyası üzerinden yürütülmektedir. Komutlar proje ana dizininden çalıştırılmalıdır.

### 1. Churn Analizi ve Makine Öğrenmesi

```bash
python kod/pipeline.py
```

Bu dosya aşağıdaki süreçleri yürütür:

- Veri kontrolleri ve ön işleme
- Keşifsel veri analizi
- Müşteri seviyesinde özellik mühendisliği
- Churn analizi
- Makine öğrenmesi modellerinin eğitilmesi ve değerlendirilmesi
- Analiz ve modelleme çıktılarının oluşturulması

### 2. LLM ile Müşteri Geri Bildirimlerinin Etiketlenmesi

```bash
python kod/llm_etiketleme.py
```

Bu dosya müşteri geri bildirimlerini iki temel kategoride sınıflandırır:

**Duygu etiketleri:**
- Olumlu
- Olumsuz
- Karışık
- Alakasız

**Konu etiketleri:**
- randevu
- fiyat
- süre
- personel
- iş_kalitesi
- yedek_parça
- temizlik
- diğer

Otomatik etiketleme sonuçları `ciktilar/llm_etiketleri.csv` dosyasına kaydedilir. Model performansı, manuel olarak etiketlenmiş altın veri seti (`altin_set.csv`) ile karşılaştırılabilir.

## Model Değerlendirme Sonuçları

Churn tahmini için Dummy Classifier, Logistic Regression ve Gradient Boosting modelleri değerlendirilmiştir.

Gradient Boosting modeli, 5 katlı Stratified Cross-Validation sonuçlarında en yüksek performansı göstermiştir.

| Metrik | Gradient Boosting |
|---|---:|
| ROC-AUC | 0.835 |
| Average Precision | 0.608 |
| Recall@Top 10% | 0.415 |
| Lift@Top 10% | 4.143 |

### Zamansal Test

Modelin farklı bir zaman dönemindeki genelleme başarısını değerlendirmek amacıyla ayrıca zamansal test uygulanmıştır.

| Metrik | Sonuç |
|---|---:|
| ROC-AUC | 0.516 |
| Average Precision | 0.173 |

Çapraz doğrulamada elde edilen yüksek performans, zamansal testte korunamamıştır. Bu nedenle modelin farklı dönemlerdeki tahmin başarısının geliştirilmesi gerekmektedir ve mevcut haliyle doğrudan operasyonel kullanıma hazır kabul edilmemiştir.

## LLM Etiketleme Değerlendirmesi

Otomatik etiketleme performansı, model çıktılarından bağımsız olarak manuel etiketlenen altın veri seti (gold set) ile değerlendirilmiştir.

| Değerlendirme | Sonuç |
|---|---:|
| Altın veri setindeki örnek | 40 |
| Değerlendirilebilen kayıt | 35 |
| Duygu doğruluğu | %68,6 |
| Konu tam eşleşmesi | %5,7 |

Duygu doğruluğu, modelin ürettiği duygu etiketinin manuel etiketle aynı olma oranıdır. Konu tam eşleşmesi ise tahmin edilen konu etiketlerinin tamamının manuel etiketlerle birebir aynı olmasını gerektirir.

Değerlendirme sonuçları, özellikle konu sınıflandırmasının iyileştirilmesi gerektiğini göstermektedir. Sonuçlar yalnızca değerlendirilebilen 35 kayıt üzerinden hesaplanmıştır.

## Proje Kapsamı

- Otomotiv müşteri ve servis verilerinin analizi
- Veri kalitesi kontrolleri ve veri ön işleme
- Müşteri davranışlarına yönelik özellik mühendisliği
- Makine öğrenmesi ile churn tahmini
- Çapraz doğrulama ve zamansal model değerlendirmesi
- Veri sızıntısı (data leakage) analizi
- LLM ile müşteri geri bildirimlerinin otomatik sınıflandırılması
- Manuel etiketleme ile LLM performansının değerlendirilmesi
