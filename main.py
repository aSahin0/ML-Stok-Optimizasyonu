# main.py

import pandas as pd
import matplotlib.pyplot as plt
import config
import inventory_toolkit as itk # Araç kutumuzu itk kısaltmasıyla import ediyoruz

def main():
    """
    Projenin ana iş akışını yönetir.
    Belirlenen tüm kritik hammaddeler için stok politikalarını hesaplar ve raporlar.
    """
    print("### STOK OPTİMİZASYON ANALİZİ BAŞLATILIYOR ###")

    # Adım 1: Veriyi yükle ve işle (Bu işlem bir kere yapılır)
    try:
        df_demand = itk.load_and_process_data()
        print("Veri yüklendi ve işlendi.")
    except FileNotFoundError:
        print("\nHATA: Veri dosyaları bulunamadı. Lütfen önce '01_generate_data.py' script'ini çalıştırın.")
        return

    # Adım 2: Kritik hammaddeler için döngüyü başlat
    final_policies = []
    print(f"\nAnaliz edilecek kritik hammaddeler: {config.CRITICAL_MATERIALS}")

    for material_to_analyze in config.CRITICAL_MATERIALS:
        print(f"--- Analiz ediliyor: {material_to_analyze} ---")
        
        # O anki hammaddeye ait veriyi filtrele
        df_model_data = df_demand[df_demand['raw_material_id'] == material_to_analyze].copy()
        
        if df_model_data.empty:
            print(f"UYARI: {material_to_analyze} için veri bulunamadı. Atlanıyor.")
            continue

        # Talep olmayan günler için 0 değeri ekle
        date_range_full = pd.date_range(start=df_model_data['order_date'].min(), end=df_model_data['order_date'].max(), freq='D')
        df_model_data = df_model_data.set_index('order_date').reindex(date_range_full, fill_value=0).reset_index(names='order_date')

        # Modeli eğit
        model, mae, _, _ = itk.train_demand_model(df_model_data)
        
        # Gelecek için tahmin yap
        future_df = itk.forecast_future_demand(model)
        avg_daily_forecast = future_df['predicted_demand'].mean()
        
        # Stok politikasını hesapla
        rop, ss, eoq, _ = itk.calculate_inventory_policy(material_to_analyze, avg_daily_forecast, mae)
        
        # Sonuçları listeye ekle
        final_policies.append({
            'Hammadde ID': material_to_analyze,
            'ROP (Birim)': int(rop),
            'EOQ (Birim)': eoq,
            'Emniyet Stoğu (Birim)': int(ss),
            'Ort. Günlük Talep': round(avg_daily_forecast, 2)
        })

    # Adım 3: Nihai toplu raporu sun
    if final_policies:
        report_df = pd.DataFrame(final_policies)
        print("\n" + "="*70)
        print("### KRİTİK HAMMADDELER İÇİN STOK YÖNETİM POLİTİKASI RAPORU ###")
        print("="*70)
        print(report_df.to_string(index=False))
        print("="*70)
    else:
        print("\nAnaliz edilecek hammadde bulunamadı veya hiçbirine ait veri yoktu.")

    print("\n### ANALİZ BAŞARIYLA TAMAMLANDI ###")


if __name__ == "__main__":
    main()