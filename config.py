"""Amacı: Projemizdeki tüm ayarları, parametreleri ve "sihirli sayıları" tek bir yerde toplamak. 
Bir parametreyi (örneğin hizmet seviyesini) değiştirmek istediğinizde, 
kodun derinliklerinde kaybolmak yerine sadece bu dosyayı düzenlersiniz."""

# config.py

# --- VERİ ÜRETİM AYARLARI ---
NUM_RAW_MATERIALS = 50
NUM_PRODUCTS = 20
START_DATE = "2023-01-01"
END_DATE = "2025-07-29"

# --- DOSYA YOLLARI ---
MATERIALS_DATA_PATH = 'synthetic_materials.csv'
BOM_DATA_PATH = 'synthetic_bom.csv'
ORDERS_DATA_PATH = 'synthetic_orders.csv'

# --- MODELLEME AYARLARI ---
TEST_DAYS = 90
FUTURE_FORECAST_DAYS = 30
MODEL_FEATURES = [
    'year', 'month', 'day_of_year', 'day',  # <-- 'day_of_month' BU ŞEKİLDE DEĞİŞTİ
    'day_of_week', 'week_of_year',
    'lag_7', 'lag_14', 'rolling_mean_7', 'rolling_std_7'
]
TARGET_COLUMN = 'total_material_needed'

# --- STOK POLİTİKASI AYARLARI ---
SERVICE_LEVEL = 0.95
ORDER_COST = 450.0  # Sipariş başına maliyet (TL)

# --- OTOMASYON AYARLARI ---
# Analizini otomatikleştirmek istediğimiz kritik hammaddelerin listesi
CRITICAL_MATERIALS = ['RAW_011', 'RAW_003', 'RAW_042', 'RAW_025', 'RAW_018']