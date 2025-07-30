# dashboard.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import inventory_toolkit as itk
import config

st.set_page_config(layout="wide", page_title="Stok Optimizasyon Sistemi")

@st.cache_data
def load_full_data():
    """Tüm veriyi yükler ve işler."""
    try:
        return itk.load_and_process_data()
    except FileNotFoundError:
        return None

st.title("🚀 Hibrit Stok Optimizasyon Sistemi")
st.markdown("Bu dashboard, LightGBM ve Prophet modellerini tek tek veya birleştirerek (Ensemble) talep tahmini yapar ve stok politikaları önerir.")

df_demand_full = load_full_data()

if df_demand_full is None:
    st.error("Veri dosyaları bulunamadı. Lütfen önce `01_generate_data.py` script'ini çalıştırın.")
else:
    st.sidebar.header("Analiz Ayarları")
    material_list = config.CRITICAL_MATERIALS
    selected_material = st.sidebar.selectbox("Analiz Edilecek Hammaddeyi Seçin:", material_list)
    
    model_choice = st.sidebar.radio(
        "Kullanılacak Tahmin Modelini Seçin:",
        ('LightGBM', 'Prophet', 'Ensemble (Ortalama)')
    )
    
    st.header(f"Detaylı Analiz: {selected_material} (Model: {model_choice})")
    
    if selected_material:
        df_model_data = df_demand_full[df_demand_full['raw_material_id'] == selected_material].copy()
        if df_model_data.empty:
            st.warning(f"{selected_material} için talep verisi bulunamadı.")
        else:
            date_range_full = pd.date_range(start=df_model_data['order_date'].min(), end=df_model_data['order_date'].max(), freq='D')
            df_model_data_full = df_model_data.set_index('order_date').reindex(date_range_full, fill_value=0).reset_index(names='order_date')
            df_model_data_full['raw_material_id'] = selected_material
            
            with st.spinner(f'{model_choice} modeli çalıştırılıyor... Lütfen bekleyin.'):
                # --- MODEL ÇALIŞTIRMA MANTIĞI ---
                lgbm_model, lgbm_mae, y_test_lgbm, lgbm_preds_array = itk.train_demand_model(df_model_data_full)
                lgbm_future_df = itk.forecast_future_demand(lgbm_model)
                
                # --- DÜZELTME BURADA: LGBM tahminlerine doğru tarih indeksini atıyoruz ---
                test_dates_lgbm = df_model_data_full.loc[y_test_lgbm.index, 'order_date']
                lgbm_predictions_series = pd.Series(lgbm_preds_array, index=test_dates_lgbm)
                y_test_lgbm.index = test_dates_lgbm # y_test'in de indeksini tarihlerle güncelliyoruz.
                # -------------------------------------------------------------------------

                if model_choice in ['Prophet', 'Ensemble (Ortalama)']:
                    prophet_model, prophet_mae, prophet_future_only, prophet_test_forecast, df_test_prophet, full_forecast = itk.train_and_forecast_with_prophet(df_model_data_full)
                    y_test_prophet = df_test_prophet.set_index('ds')['y']
                    prophet_predictions_series = prophet_test_forecast.set_index('ds')['yhat']

                # --- SEÇİME GÖRE SONUÇLARI HAZIRLAMA ---
                if model_choice == 'LightGBM':
                    mae = lgbm_mae
                    final_predictions = lgbm_predictions_series
                    y_test = y_test_lgbm
                    avg_daily_forecast = lgbm_future_df['predicted_demand'].mean()

                elif model_choice == 'Prophet':
                    mae = prophet_mae
                    final_predictions = prophet_predictions_series
                    y_test = y_test_prophet
                    avg_daily_forecast = prophet_future_only['yhat'].mean()
                
                elif model_choice == 'Ensemble (Ortalama)':
                    # Artık her iki seri de aynı tarih indeksine sahip olduğu için doğru birleşecekler.
                    final_predictions = (lgbm_predictions_series + prophet_predictions_series) / 2
                    y_test = y_test_lgbm
                    mae = mean_absolute_error(y_test, final_predictions)
                    
                    lgbm_future_demand = lgbm_future_df.set_index('order_date')['predicted_demand']
                    prophet_future_demand = prophet_future_only.set_index('ds')['yhat']
                    ensemble_future_demand = (lgbm_future_demand + prophet_future_demand) / 2
                    avg_daily_forecast = ensemble_future_demand.mean()

            # Stok politikasını hesapla
            rop, ss, eoq, info = itk.calculate_inventory_policy(selected_material, avg_daily_forecast, mae)
            
            st.subheader(f"📈 Sonuçlar ve Stok Politikası")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Model Hatası (MAE)", f"{mae:.2f}")
            col2.metric("Yeniden Sipariş Noktası (ROP)", f"{int(rop)}")
            col3.metric("Ekonomik Sipariş Miktarı (EOQ)", f"{int(eoq)}")
            col4.metric("Emniyet Stoğu", f"{int(ss)}")

            st.subheader("Gerçekleşen Talep vs. Model Tahmini (Test Seti)")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(y_test.index, y_test, label='Gerçek Talep', marker='.', linestyle='-')
            ax.plot(final_predictions.index, final_predictions, label=f'{model_choice} Tahmini', linestyle='--')
            ax.legend()
            ax.set_ylabel("Talep Miktarı")
            ax.set_title(f"{selected_material} Talep Grafiği")
            st.pyplot(fig)

            if model_choice == 'Prophet':
                st.subheader("Prophet Modelinin Bileşen Analizi")
                fig_prophet = prophet_model.plot_components(full_forecast)
                st.pyplot(fig_prophet)