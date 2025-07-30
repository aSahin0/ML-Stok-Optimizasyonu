# inventory_toolkit.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from scipy.stats import norm
import config
import optuna


def load_and_process_data():
    """CSV dosyalarını yükler, günlük talebi hesaplar ve gelişmiş özellikler üretir."""
    df_orders = pd.read_csv(config.ORDERS_DATA_PATH, parse_dates=['order_date'])
    df_bom = pd.read_csv(config.BOM_DATA_PATH)
    
    df_merged = pd.merge(df_orders, df_bom, on='product_id')
    df_merged['total_material_needed'] = df_merged['quantity'] * df_merged['usage_quantity']
    df_demand = df_merged.groupby(['order_date', 'raw_material_id'])['total_material_needed'].sum().reset_index()
    
    # --- TEMEL ÖZELLİK MÜHENDİSLİĞİ ---
    df_demand['year'] = df_demand['order_date'].dt.year
    df_demand['month'] = df_demand['order_date'].dt.month
    df_demand['day_of_year'] = df_demand['order_date'].dt.dayofyear
    df_demand['day'] = df_demand['order_date'].dt.day # <-- 'day_of_month' DEĞİŞTİ
    df_demand['day_of_week'] = df_demand['order_date'].dt.dayofweek
    df_demand['week_of_year'] = df_demand['order_date'].dt.isocalendar().week.astype(int)


    # --- GELİŞMİŞ ÖZELLİK MÜHENDİSLİĞİ (YENİ) ---
    # Önemli: groupby('raw_material_id') kullanımı, bu hesaplamaların her hammadde için
    # ayrı ayrı yapılmasını sağlar, böylece bir hammaddenin verisi diğerini etkilemez.
    
    # Lag Features (Geçmiş Talep Bilgileri)
    df_demand['lag_7'] = df_demand.groupby('raw_material_id')[config.TARGET_COLUMN].shift(7)
    df_demand['lag_14'] = df_demand.groupby('raw_material_id')[config.TARGET_COLUMN].shift(14)
    
    # Rolling Window Features (Hareketli Pencere İstatistikleri)
    df_demand['rolling_mean_7'] = df_demand.groupby('raw_material_id')[config.TARGET_COLUMN].rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True)
    df_demand['rolling_std_7'] = df_demand.groupby('raw_material_id')[config.TARGET_COLUMN].rolling(window=7, min_periods=1).std().reset_index(level=0, drop=True)

    # Hesaplamalar sonrası oluşabilecek boş (NaN) değerleri 0 ile doldur
    df_demand.fillna(0, inplace=True)
    
    print("Gelişmiş özellikler (lag, rolling) eklendi.")
    return df_demand

def objective(trial, X_train, y_train, X_test, y_test):
    """Optuna'nın her denemede çalıştıracağı hedef fonksiyonu."""
    # Hiperparametreler için arama uzayını tanımla
    params = {
        'objective': 'regression_l1',  # MAE için optimize eder
        'metric': 'mae',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'verbose': -1 # Logları kapatır
    }
    
    model = lgb.LGBMRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    return mae

def train_demand_model(df_model_data):
    """
    Optuna ile en iyi hiperparametreleri bulur ve iki model eğitir.
    """
    # --- Veri Hazırlığı ---
    X_full = df_model_data[config.MODEL_FEATURES]
    y = df_model_data[config.TARGET_COLUMN]
    split_date = df_model_data['order_date'].max() - pd.Timedelta(days=config.TEST_DAYS)
    train_indices = df_model_data['order_date'] <= split_date
    test_indices = df_model_data['order_date'] > split_date
    X_train_full, y_train = X_full[train_indices], y[train_indices]
    X_test_full, y_test = X_full[test_indices], y[test_indices]

    # --- Optuna ile Hiperparametre Optimizasyonu ---
    # `lambda` kullanarak objective fonksiyonuna ek parametreler gönderiyoruz
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train_full, y_train, X_test_full, y_test), n_trials=50) # 50 deneme yapacak

    print(f"Optimizasyon tamamlandı. En iyi MAE: {study.best_value:.2f}")
    print(f"En iyi parametreler: {study.best_params}")
    
    best_params = study.best_params
    
    # --- Nihai Modelleri En İyi Parametrelerle Eğitme ---
    # Model 1 (Kompleks): En iyi MAE'yi ve tahminleri döndürmek için
    model_full = lgb.LGBMRegressor(**best_params, random_state=42)
    model_full.fit(X_train_full, y_train)
    predictions = model_full.predict(X_test_full)
    final_mae = mean_absolute_error(y_test, predictions)

    # Model 2 (Basit): Gelecek tahmini için
    basic_features = ['year', 'month', 'day_of_year', 'day', 'day_of_week', 'week_of_year']
    X_simple = df_model_data[basic_features]
    X_train_simple, _ = X_simple[train_indices], y[train_indices]
    model_simple = lgb.LGBMRegressor(**best_params, random_state=42)
    model_simple.fit(X_train_simple, y_train)

    return model_simple, final_mae, y_test, predictions

def forecast_future_demand(model_simple): # Artık basit modeli alıyor
    """Eğitilmiş BASİT bir modeli kullanarak gelecekteki talebi tahmin eder."""
    last_date = pd.to_datetime(config.END_DATE)
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=config.FUTURE_FORECAST_DAYS, freq='D')
    future_df = pd.DataFrame({'order_date': future_dates})

    # Sadece basit özellikleri oluştur
    basic_features = ['year', 'month', 'day_of_year', 'day', 'day_of_week', 'week_of_year'] # <-- 'day_of_month' DEĞİSTİ
    for col in basic_features:
        if col == 'week_of_year':
            future_df[col] = future_df['order_date'].dt.isocalendar().week.astype(int)
        else:
            future_df[col] = getattr(future_df['order_date'].dt, col)
    
    # Basit model ile basit özellikler üzerinden tahmin yap
    future_predictions = model_simple.predict(future_df[basic_features])
    future_df['predicted_demand'] = future_predictions
    return future_df

def calculate_inventory_policy(material_id, avg_daily_forecast, mae):
    """ROP ve EOQ değerlerini NaN ve negatif değerlere karşı sağlam bir şekilde hesaplar."""
    df_materials = pd.read_csv(config.MATERIALS_DATA_PATH)
    material_info = df_materials[df_materials['raw_material_id'] == material_id].iloc[0]
    lead_time = material_info['supplier_lead_time_days']
    
    # Girdilerin geçerli olup olmadığını kontrol et
    if pd.isna(avg_daily_forecast) or pd.isna(mae):
        # Eğer girdilerden biri bile NaN ise, hesaplama yapılamaz.
        # Her şeyi 0 veya mantıklı varsayılan değerler olarak döndür.
        return 0, 0, 0, material_info

    # --- ROP HESAPLAMASI ---
    z_score = norm.ppf(config.SERVICE_LEVEL)
    error_std_dev = 1.25 * mae
    uncertainty_during_lead_time = error_std_dev * np.sqrt(lead_time)
    
    # ÖNEMLİ DÜZELTME: safety_stock burada tanımlanmalıydı.
    safety_stock = z_score * uncertainty_during_lead_time
    
    # ROP hesaplarken negatif talebi 0 olarak kabul et
    demand_during_lead_time = max(0, avg_daily_forecast) * lead_time
    reorder_point = demand_during_lead_time + safety_stock
    
    # --- EOQ HESAPLAMASI ---
    annual_demand_forecast = avg_daily_forecast * 365
    holding_cost_per_unit = material_info['cost_per_unit'] * material_info['holding_cost_rate']
    
    # ÖNEMLİ DÜZELTME: Hem negatifliği hem de NaN olma durumunu kontrol et
    if annual_demand_forecast <= 0 or holding_cost_per_unit <= 0:
        economic_order_quantity = 0
    else:
        eoq_value = np.sqrt((2 * annual_demand_forecast * config.ORDER_COST) / holding_cost_per_unit)
        # eoq_value'nun da NaN olup olmadığını kontrol et
        if pd.isna(eoq_value):
            economic_order_quantity = 0
        else:
            economic_order_quantity = int(eoq_value)
            
    return reorder_point, safety_stock, economic_order_quantity, material_info

from prophet import Prophet

def train_and_forecast_with_prophet(df_model_data):
    """Prophet modelini kullanarak talep tahmini yapar."""
    
    material_id = df_model_data['raw_material_id'].iloc[0]
    print(f"Prophet modeli {material_id} için çalıştırılıyor...")
    
    df_prophet = df_model_data[['order_date', 'total_material_needed']].rename(
        columns={'order_date': 'ds', 'total_material_needed': 'y'}
    )
    
    split_date = df_prophet['ds'].max() - pd.Timedelta(days=config.TEST_DAYS)
    df_train = df_prophet[df_prophet['ds'] <= split_date]
    df_test = df_prophet[df_prophet['ds'] > split_date]
    
    model_prophet = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model_prophet.add_country_holidays(country_name='TR')
    model_prophet.fit(df_train)
    
    # TEST verisi üzerinde tahmin yaparak MAE hesapla
    test_forecast = model_prophet.predict(df_test[['ds']])
    mae = mean_absolute_error(df_test['y'], test_forecast['yhat'])
    print(f"Prophet modeli MAE: {mae:.2f}")

    # GELECEK için tahmin yap
    # --- DÜZELTME BURADA: Prophet'in hem test periyodunu hem de gelecek periyodunu tahmin etmesini sağlıyoruz ---
    future_periods_to_forecast = config.TEST_DAYS + config.FUTURE_FORECAST_DAYS
    future_dates = model_prophet.make_future_dataframe(periods=future_periods_to_forecast)
    full_forecast = model_prophet.predict(future_dates)
    future_forecast_only = full_forecast[full_forecast['ds'] > df_prophet['ds'].max()] # Sadece gerçek gelecek günlerini al
    
    return model_prophet, mae, future_forecast_only, test_forecast, df_test, full_forecast

# inventory_toolkit.py dosyasının sonuna ekleyin

def run_inventory_simulation(daily_demands, policy, lead_time, holding_cost_per_unit, order_cost):
    """
    Belirli bir stok politikası altında envanter sisteminin simülasyonunu çalıştırır.
    
    :param daily_demands: Günlük talepleri içeren bir pandas Series.
    :param policy: (ROP, EOQ) değerlerini içeren bir tuple.
    :param lead_time: Tedarik süresi (gün).
    :param holding_cost_per_unit: Birim başına YILLIK stok tutma maliyeti.
    :param order_cost: Sipariş başına maliyet.
    :return: Performans metriklerini içeren bir sözlük.
    """
    rop, eoq = policy
    
    # EOQ 0 ise simülasyon anlamsızdır, maliyetleri 0 kabul et.
    if eoq == 0:
        return {
            "Toplam Maliyet (TL)": 0, "Stok Tutma Maliyeti (TL)": 0,
            "Sipariş Maliyeti (TL)": 0, "Hizmet Seviyesi (%)": 0
        }

    # Başlangıç durumu
    inventory_level = rop + eoq  # Simülasyona dolu stokla başlayalım
    on_order_count = 0
    order_arrival_days = []
    
    total_holding_cost = 0
    total_order_cost = 0
    total_demand = 0
    unmet_demand = 0
    
    # Simülasyon periyodu kadar döngü
    for day, demand in enumerate(daily_demands):
        # 1. Gelen sipariş var mı kontrol et
        if day in order_arrival_days:
            inventory_level += eoq
            on_order_count -= 1
            order_arrival_days.remove(day)

        # 2. O günün talebini karşıla
        total_demand += demand
        units_to_ship = min(demand, inventory_level)
        inventory_level -= units_to_ship
        if units_to_ship < demand:
            unmet_demand += (demand - units_to_ship)

        # 3. Gün sonu stok tutma maliyetini hesapla
        # Yıllık maliyeti günlüğe çeviriyoruz
        total_holding_cost += inventory_level * (holding_cost_per_unit / 365) 

        # 4. Stok seviyesini kontrol et ve sipariş ver
        # Sadece bekleyen sipariş yoksa yenisini ver (basit bir sipariş politikası)
        if inventory_level <= rop and on_order_count == 0:
            on_order_count += 1
            total_order_cost += order_cost
            arrival_day = day + lead_time
            order_arrival_days.append(arrival_day)
            
    # Simülasyon sonuçlarını hesapla
    service_level = ((total_demand - unmet_demand) / total_demand) * 100 if total_demand > 0 else 100
    total_cost = total_holding_cost + total_order_cost
    
    return {
        "Toplam Maliyet (TL)": total_cost,
        "Stok Tutma Maliyeti (TL)": total_holding_cost,
        "Sipariş Maliyeti (TL)": total_order_cost,
        "Hizmet Seviyesi (%)": service_level,
    }