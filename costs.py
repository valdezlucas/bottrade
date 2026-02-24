import numpy as np


class TradingCosts:
    """
    Simula costos reales de Forex por operación.
    Todos los valores en unidades de precio (no pips).
    """

    def __init__(
        self,
        spread_pips=1.5,
        max_slippage_pips=1.0,
        swap_per_night=0.0,
        pip_value=0.0001,
    ):
        """
        Args:
            spread_pips: Spread fijo del broker en pips
            max_slippage_pips: Slippage máximo aleatorio en pips
            swap_per_night: Costo de swap por noche en unidades de precio
            pip_value: Valor de 1 pip (0.0001 para EUR/USD, 0.01 para JPY pairs)
        """
        self.spread = spread_pips * pip_value
        self.max_slippage = max_slippage_pips * pip_value
        self.swap_per_night = swap_per_night
        self.pip_value = pip_value

    def entry_cost(self):
        """Costo al entrar: spread + slippage aleatorio."""
        slippage = np.random.uniform(0, self.max_slippage)
        return self.spread + slippage

    def exit_cost(self):
        """Costo al salir: slippage aleatorio."""
        return np.random.uniform(0, self.max_slippage)

    def holding_cost(self, nights=0):
        """Costo por mantener posición overnight."""
        return self.swap_per_night * nights

    def total_cost(self, nights=0):
        """Costo total de una operación completa."""
        return self.entry_cost() + self.exit_cost() + self.holding_cost(nights)

    def apply_to_pnl(self, gross_pnl, nights=0):
        """Aplica costos a un PnL bruto y devuelve PnL neto."""
        return gross_pnl - self.total_cost(nights)
