# rl_environment.py

import numpy as np
import config

class InventoryEnv:
    def __init__(self, daily_demands, lead_time, holding_cost_per_unit, order_cost, stockout_penalty_cost):
        self.daily_demands = daily_demands
        self.lead_time = lead_time
        self.holding_cost_per_unit = holding_cost_per_unit / 365 # Günlük maliyet
        self.order_cost = order_cost
        self.stockout_penalty_cost = stockout_penalty_cost # Stoksuz kalmanın birim başına cezası
        self.simulation_length = len(daily_demands)
        self.reset()

    def reset(self):
        """Simülasyonu başlangıç durumuna sıfırlar."""
        self.current_day = 0
        self.inventory_level = 200 # Başlangıç stoğu
        self.on_order = [] # (varış günü, miktar)
        return self._get_state()

    def _get_state(self):
        """Mevcut stok seviyesini durum olarak döndürür (gruplanmış)."""
        return int(self.inventory_level / config.STATE_BIN_SIZE)

    def step(self, action_index):
        """Ajanın seçtiği aksiyonu uygular ve bir gün ilerler."""
        action = config.ACTION_SPACE[action_index]
        
        # 1. Sipariş ver (eğer aksiyon 0'dan büyükse)
        daily_cost = 0
        if action > 0:
            arrival_day = self.current_day + self.lead_time
            self.on_order.append({'arrival_day': arrival_day, 'quantity': action})
            daily_cost += self.order_cost

        # 2. Gelen siparişleri karşıla
        arrivals_today = [order for order in self.on_order if order['arrival_day'] == self.current_day]
        for order in arrivals_today:
            self.inventory_level += order['quantity']
            self.on_order.remove(order)

        # 3. Talebi karşıla
        demand = self.daily_demands[self.current_day]
        units_to_ship = min(demand, self.inventory_level)
        self.inventory_level -= units_to_ship
        
        stockout_units = demand - units_to_ship
        if stockout_units > 0:
            daily_cost += stockout_units * self.stockout_penalty_cost
        
        # 4. Stok tutma maliyetini ekle
        daily_cost += self.inventory_level * self.holding_cost_per_unit
        
        # 5. Ödülü hesapla (negatif maliyet)
        reward = -daily_cost

        # 6. Bir sonraki güne geç
        self.current_day += 1
        done = self.current_day >= self.simulation_length
        next_state = self._get_state()
        
        return next_state, reward, done