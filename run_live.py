import os
import time

def main():
    print("="*60)
    print("  🚀 INICIANDO ESCÁNER MULTI-TEMPORALIDAD A MT5")
    print("="*60)
    print("\n[1/2] Ejecutando Escáner Diario (1D)...")
    os.system("python mt5_trader.py --tf 1d")
    
    # Pausa breve para evitar saturar la conexión
    time.sleep(2)
    
    print("\n[2/2] Ejecutando Escáner 4 Horas (4H)...")
    os.system("python mt5_trader.py --tf 4h")
    
    print("\n" + "="*60)
    print("  ✅ ESCANEO COMPLETADO CON ÉXITO")
    print("  Las órdenes viables se enviaron a MetaTrader 5.")
    print("="*60)
    time.sleep(5)

if __name__ == "__main__":
    main()
