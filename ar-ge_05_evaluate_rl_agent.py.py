# 05_evaluate_rl_agent.py

import pandas as pd
import numpy as np
import config
import inventory_toolkit as itk
from rl_environment import InventoryEnv
from sklearn.metrics import mean_absolute_error

def evaluate_rl_policy(q_table, env):
    """
    Eğitilmiş bir Q-tablosunu kullanarak bir simülasyon çalıştırır ve performansı ölçer.
    """
    state = env.reset()
    done = False
    
    total_holding_cost = 0
    total_order_cost = 0
    total_demand = 0
    unmet_demand = 0
    inventory_level = env.inventory_level
    on_order = []
    
    for day, demand in enumerate(env.daily_demands):
        # Sipariş karşılama
        arrivals_today = [order for order in on_order if order['arrival_day'] == day]
        for order in arrivals_today:
            inventory_level += order['quantity']
            on_order.remove(order)
            
        # Talebi karşılama
        total_demand += demand
        units_to_ship = min(demand, inventory_level)
        inventory_level -= units_to_ship
        if units_to_ship < demand:
            unmet_demand += (demand - units_to_ship)
        
        # Stok tutma maliyetini hesapla
        total_holding_cost += inventory_level * env.holding_cost_per_unit
            
        # Aksiyon alma (sadece en iyi bilinen aksiyonu kullan, keşfetme yok)
        current_state = int(inventory_level / config.STATE_BIN_SIZE)
        # Durumun Q-tablosu sınırları içinde olduğundan emin ol
        if current_state >= q_table.shape[0]:
            current_state = q_table.shape[0] - 1
            
        action_idx = np.argmax(q_table[current_state, :])
        action_quantity = config.ACTION_SPACE[action_idx]
        
        if action_quantity > 0:
            is_order_placed = any(o['arrival_day'] > day for o in on_order)
            if not is_order_placed:
                 on_order.append({'arrival_day': day + env.lead_time, 'quantity': action_quantity})
                 total_order_cost += env.order_cost

    service_level = ((total_demand - unmet_demand) / total_demand) * 100 if total_demand > 0 else 100
    stockout_cost = unmet_demand * env.stockout_penalty_cost
    total_cost = total_holding_cost + total_order_cost + stockout_cost

    return {
        "Politika": "Pekiştirmeli Öğrenme (RL)",
        "Toplam Maliyet (TL)": total_cost,
        "Stok Tutma Maliyeti (TL)": total_holding_cost,
        "Sipariş Maliyeti (TL)": total_order_cost,
        "Hizmet Seviyesi (%)": service_level
    }


def get_ensemble_policy_results(df_model_data_full, test_demands, material_to_evaluate):
    """Ensemble modeli için ROP/EOQ politikasını ve simülasyon sonucunu hesaplar."""
    # LightGBM
    lgbm_model, lgbm_mae, y_test_lgbm, lgbm_preds_array = itk.train_demand_model(df_model_data_full)
    lgbm_future_df = itk.forecast_future_demand(lgbm_model)
    test_dates_lgbm = df_model_data_full.loc[y_test_lgbm.index, 'order_date']
    lgbm_predictions_series = pd.Series(lgbm_preds_array, index=test_dates_lgbm)

    # Prophet
    _, prophet_mae, prophet_future_only, _, _, _ = itk.train_and_forecast_with_prophet(df_model_data_full)
    prophet_predictions_series = _ # Bu değişkene ihtiyacımız yok ama unpack için gerekli
    
    # Ensemble
    # Not: Prophet test tahminlerini de almamız lazım, toolkit fonksiyonunu güncelleyebiliriz veya burada yeniden yaparız.
    # Şimdilik basitlik adına iki modelin MAE ve tahminlerinin ortalamasını alalım.
    avg_mae = (lgbm_mae + prophet_mae) / 2
    lgbm_future_demand = lgbm_future_df.set_index('order_date')['predicted_demand']
    prophet_future_demand = prophet_future_only.set_index('ds')['yhat']
    ensemble_future_demand = (lgbm_future_demand + prophet_future_demand) / 2
    avg_daily_forecast = ensemble_future_demand.mean()
    
    rop, _, eoq, material_info = itk.calculate_inventory_policy(material_to_evaluate, avg_daily_forecast, avg_mae)
    
    # Ensemble Politikasını Simüle Et
    policy = (rop, eoq)
    holding_cost = material_info['cost_per_unit'] * material_info['holding_cost_rate']
    lead_time = material_info['supplier_lead_time_days']
    
    results = itk.run_inventory_simulation(
        daily_demands=test_demands,
        policy=policy,
        lead_time=lead_time,
        holding_cost_per_unit=holding_cost,
        order_cost=config.ORDER_COST
    )
    results['Politika'] = 'Ensemble (ROP/EOQ)'
    return results


def main():
    print("--- Politikalar Değerlendiriliyor: RL vs. Ensemble (ROP/EOQ) ---")
    
    # 1. Veriyi ve Ortamı Hazırla
    df_demand_full = itk.load_and_process_data()
    material_to_evaluate = config.CRITICAL_MATERIALS[0]
    
    df_model_data = df_demand_full[df_demand_full['raw_material_id'] == material_to_evaluate].copy()
    date_range_full = pd.date_range(start=df_model_data['order_date'].min(), end=df_model_data['order_date'].max(), freq='D')
    df_model_data_full = df_model_data.set_index('order_date').reindex(date_range_full, fill_value=0).reset_index(names='order_date')

    split_date = df_model_data['order_date'].max() - pd.Timedelta(days=config.TEST_DAYS)
    test_demands = df_model_data[df_model_data['order_date'] > split_date]['total_material_needed']

    # 2. Ensemble Politikasının Sonuçlarını Al
    print("\nEnsemble (ROP/EOQ) politikası simüle ediliyor...")
    ensemble_results = get_ensemble_policy_results(df_model_data_full, test_demands, material_to_evaluate)

    # 3. RL Ajanının Politikasının Sonuçlarını Al
    print("Pekiştirmeli Öğrenme (RL) politikası simüle ediliyor...")
    q_table_path = f'q_table_{material_to_evaluate}.npy'
    try:
        q_table = np.load(q_table_path)
        
        # RL için ortamı kur
        material_info = pd.read_csv(config.MATERIALS_DATA_PATH)
        material_info = material_info[material_info['raw_material_id'] == material_to_evaluate].iloc[0]
        env_rl = InventoryEnv(
            daily_demands=test_demands.values, 
            lead_time=material_info['supplier_lead_time_days'], 
            holding_cost_per_unit=material_info['cost_per_unit'] * material_info['holding_cost_rate'], 
            order_cost=config.ORDER_COST, 
            stockout_penalty_cost=10000.0 # Eğitimdekiyle aynı ceza maliyeti
        )
        rl_results = evaluate_rl_policy(q_table, env_rl)
        
        # 4. Sonuçları Karşılaştır
        comparison_df = pd.DataFrame([ensemble_results, rl_results])
        print("\n" + "="*70)
        print(f"### {material_to_evaluate} İÇİN POLİTİKA KARŞILAŞTIRMA SONUÇLARI ###")
        print("="*70)
        print(comparison_df.to_string(index=False))
        print("="*70)

    except FileNotFoundError:
        print(f"\nHATA: '{q_table_path}' bulunamadı. Lütfen önce '04_train_rl_agent.py' script'ini çalıştırın.")

if __name__ == "__main__":
    main()