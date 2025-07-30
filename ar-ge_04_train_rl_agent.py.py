# 04_train_rl_agent.py

import pandas as pd
import numpy as np
from tqdm import tqdm
import config
import inventory_toolkit as itk
from rl_environment import InventoryEnv
from rl_agent import QLearningAgent

def train():
    print("--- RL Ajanı Eğitimi Başlatılıyor ---")
    
    # 1. Eğitim için veri hazırla (bir hammadde seçelim)
    df_demand_full = itk.load_and_process_data()
    material_to_train = config.CRITICAL_MATERIALS[0]
    df_model_data = df_demand_full[df_demand_full['raw_material_id'] == material_to_train].copy()
    
    # Eğitim için test verisini kullanalım (veya eğitim verisinin bir kısmını)
    split_date = df_model_data['order_date'].max() - pd.Timedelta(days=config.TEST_DAYS)
    training_demands = df_model_data[df_model_data['order_date'] > split_date]['total_material_needed'].values
    
    # 2. Ortam ve Ajanı oluştur
    # Basit bir örnek için parametreleri manuel girelim
    env = InventoryEnv(
        daily_demands=training_demands, 
        lead_time=10, 
        holding_cost_per_unit=50.0, 
        order_cost=500.0, 
        stockout_penalty_cost=10000.0 # Stoksuz kalma cezası, en önemli parametrelerden biri
    )
    
    # Durum sayısını belirle (örn: max stok 5000 ise 500 durum)
    num_states = 3000
    num_actions = len(config.ACTION_SPACE)
    agent = QLearningAgent(num_states, num_actions)

    print(f"{config.RL_TRAINING_EPISODES} bölüm (episode) boyunca eğitim yapılacak...")
    
    # 3. Eğitim döngüsü
    for episode in tqdm(range(config.RL_TRAINING_EPISODES)):
        state = env.reset()
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            
        agent.decay_epsilon()

    print("\n--- Eğitim Tamamlandı ---")
    
    # 4. Öğrenilen Q-tablosunu kaydet
    q_table_path = f'q_table_{material_to_train}.npy'
    np.save(q_table_path, agent.q_table)
    print(f"Öğrenilen politika (Q-Tablosu) '{q_table_path}' dosyasına kaydedildi.")


if __name__ == "__main__":
    train()