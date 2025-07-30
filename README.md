# 🚀 Hibrit Makine Öğrenmesi Modelleri ile Stok Optimizasyonu

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Libraries](https://img.shields.io/badge/Libraries-Pandas%20%7C%20Scikit--learn%20%7C%20LightGBM%20%7C%20Prophet%20%7C%20Optuna-orange)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red)


> Sipariş bazlı (Make-to-Order) üretim senaryoları için geliştirilmiş, talep tahminlerini optimize ederek stok maliyetlerini düşüren ve üretim devamlılığını sağlayan otonom bir karar destek sistemi.

Bu proje,  Teknoloji firmalarının karmaşık tedarik zinciri ve üretim süreçleri göz önünde bulundurularak, sentetik veriler üzerinde bir Kavram Kanıtlama (PoC) çalışması olarak geliştirilmiştir.

---

## 🖼️ İnteraktif Dashboard

Sistem, tüm analiz sonuçlarını ve politika önerilerini kullanıcı dostu bir arayüzde sunar. Kullanıcılar, farklı tahmin modellerini (LightGBM, Prophet, Ensemble) anlık olarak karşılaştırabilir ve sonuçları görsel olarak inceleyebilir.

---

## 📋 İçindekiler

* [Ana Özellikler](#-ana-özellikler)
* [Teknoloji Mimarisi](#-teknoloji-mimarisi)
* [Proje Yapısı](#-proje-yapısı)
* [Kurulum ve Çalıştırma](#-kurulum-ve-çalıştırma)

---

## 🎯 Ana Özellikler

Bu projenin temel yetenekleri şunlardır:

🧠 **Hibrit Modelleme:** Sektör standardı iki güçlü modeli birleştirir:
  - **LightGBM:** Gelişmiş özellik mühendisliği (lag/rolling features) ile yüksek doğruluk sağlar.
  - **Prophet:** Trend, mevsimsellik ve tatil etkilerini otomatik olarak modelleyen bir zaman serisi uzmanı.
  - **Ensemble (Ortalama):** Her iki modelin gücünü birleştirerek daha sağlam ve güvenilir bir "süper model" oluşturur.

⚙️ **Otomatik Optimizasyon:** **Optuna** kütüphanesi kullanılarak LightGBM modelinin hiperparametreleri, veri setine özel en iyi performansı verecek şekilde otomatik olarak bulunur.

📊 **İnteraktif Arayüz:** **Streamlit** ile geliştirilen dashboard, teknik olmayan kullanıcıların bile sistemi kolayca kullanmasına, farklı senaryoları test etmesine ve sonuçları anında görmesine olanak tanır.

🤖 **Akıllı Stok Politikaları:** Her bir hammadde için Endüstri Mühendisliği'nin temel prensiplerini kullanarak bilimsel stok politikaları hesaplar:
  - **Emniyet Stoğu (Safety Stock):** Talepteki belirsizliğe karşı koruma sağlar.
  - **Yeniden Sipariş Noktası (ROP):** "Ne zaman sipariş verilmeli?" sorusunu cevaplar.
  - **Ekonomik Sipariş Miktarı (EOQ):** "Ne kadar sipariş verilmeli?" sorusunu cevaplayarak maliyetleri minimize eder.

📧 **Proaktif Uyarı Sistemi:** Arka planda çalışarak stok seviyelerini otonom olarak kontrol eder ve ROP'un altına düşen hammaddeler için otomatik olarak uyarı e-postası gönderir.

---

## 🛠️ Teknoloji Mimarisi

![Pandas](https://img.shields.io/badge/Pandas-2.0.3-150458?style=for-the-badge&logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-1.26.4-013243?style=for-the-badge&logo=numpy)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-F7931E?style=for-the-badge&logo=scikit-learn)
![LightGBM](https://img.shields.io/badge/LightGBM-4.1.0-8A2BE2?style=for-the-badge)
![Prophet](https://img.shields.io/badge/Prophet-1.1.5-0078D4?style=for-the-badge&logo=facebook)
![Optuna](https://img.shields.io/badge/Optuna-3.5.0-8A2BE2?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-FF4B4B?style=for-the-badge&logo=streamlit)

---

## 📂 Proje Yapısı

Proje, yönetimi kolay ve modüler bir yapıda tasarlanmıştır:
```
Stok_Optimizasyon_Projesi/
├── config.py              # Tüm ayarların ve parametrelerin merkezi
├── 00_simulate_inventory.py # Stok durumunu simüle eden yardımcı script
├── 01_generate_data.py    # Yapay veri setlerini üreten script
├── 03_daily_alert_check.py  # Otomatik e-posta uyarı sistemi    -----> Sonradan eklenecek
├── inventory_toolkit.py   # Tüm analiz fonksiyonlarını içeren araç kutusu
├── dashboard.py           # Streamlit interaktif dashboard uygulaması
├── requirements.txt       # Gerekli Python kütüphaneleri
└── README.md              # Bu dosya
```
---

## 🚀 Kurulum ve Çalıştırma

Bu projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin.

### 1. Projeyi Klonlama
```bash
git clone [https://github.com/](https://github.com/)[KULLANICI_ADINIZ]/[REPO_ADINIZ].git
cd [REPO_ADINIZ]
```

### 2. Gerekli Kütüphaneleri Yükleme
```bash
pip install -r requirements.txt
```

### 3. Veri Setlerini Oluşturma
Proje, sentetik veriler üzerinde çalışmaktadır. Aşağıdaki komutlarla gerekli CSV dosyalarını oluşturun.
```bash
python 01_generate_data.py
python 00_simulate_inventory.py
```

### 4. İnteraktif Dashboard'u Başlatma
```bash
streamlit run dashboard.py
```
Bu komut, tarayıcınızda projenin web arayüzünü otomatik olarak açacaktır.

---


