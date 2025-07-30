# 01_generate_data.py

import pandas as pd
import numpy as np
import config  # Ayarları config dosyasından alıyoruz

def generate_and_save_data():
    """Tüm sentetik verileri üretir ve ilgili CSV dosyalarına kaydeder."""
    print("--- Yapay Veri Üretimi Başlatıldı ---")

    # Hammadde Verileri
    material_ids = [f"RAW_{i:03d}" for i in range(1, config.NUM_RAW_MATERIALS + 1)]
    df_materials = pd.DataFrame({
        'raw_material_id': material_ids,
        'supplier_lead_time_days': np.random.randint(5, 25, size=config.NUM_RAW_MATERIALS),
        'cost_per_unit': np.random.uniform(10.5, 500.0, size=config.NUM_RAW_MATERIALS).round(2),
        'holding_cost_rate': np.random.uniform(0.15, 0.30, size=config.NUM_RAW_MATERIALS).round(2)
    })
    df_materials.to_csv(config.MATERIALS_DATA_PATH, index=False)
    print(f"'{config.MATERIALS_DATA_PATH}' oluşturuldu.")

    # Ürün Ağacı (BOM)
    product_ids = [f"PROD_{chr(65+i)}" for i in range(config.NUM_PRODUCTS)]
    bom_list = []
    for prod_id in product_ids:
        num_components = np.random.randint(2, 9)
        components = np.random.choice(df_materials['raw_material_id'], size=num_components, replace=False)
        for comp_id in components:
            bom_list.append({'product_id': prod_id, 'raw_material_id': comp_id, 'usage_quantity': np.random.randint(1, 6)})
    df_bom = pd.DataFrame(bom_list)
    df_bom.to_csv(config.BOM_DATA_PATH, index=False)
    print(f"'{config.BOM_DATA_PATH}' oluşturuldu.")

    # Sipariş Geçmişi
    date_range = pd.to_datetime(pd.date_range(start=config.START_DATE, end=config.END_DATE, freq='D'))
    num_days = len(date_range)
    trend = np.linspace(10, 30, num_days)
    weekly_seasonality = 10 * np.sin(2 * np.pi * (date_range.dayofweek / 6))
    noise = np.random.normal(0, 5, num_days)
    daily_order_counts = np.maximum(0, trend + weekly_seasonality + noise).astype(int)
    product_popularity = 1 / np.arange(1, config.NUM_PRODUCTS + 1)
    product_popularity /= product_popularity.sum()
    orders_list = []
    order_id_counter = 1
    for date, num_orders in zip(date_range, daily_order_counts):
        if num_orders > 0:
            ordered_products = np.random.choice(product_ids, size=num_orders, p=product_popularity)
            for product in ordered_products:
                orders_list.append({'order_id': order_id_counter, 'order_date': date, 'product_id': product, 'quantity': np.random.randint(1, 21)})
                order_id_counter += 1
    df_orders = pd.DataFrame(orders_list)
    df_orders.to_csv(config.ORDERS_DATA_PATH, index=False)
    print(f"'{config.ORDERS_DATA_PATH}' oluşturuldu.")
    print("--- Veri Üretimi Tamamlandı ---")

if __name__ == "__main__":
    generate_and_save_data()