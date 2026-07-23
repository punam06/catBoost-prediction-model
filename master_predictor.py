"""
master_predictor.py
===================
A unified terminal menu to run all EV prediction models.

Usage:
  source venv/bin/activate
  python3 master_predictor.py
"""

import os
import sys

def main():
    while True:
        print("\n" + "="*60)
        print(" ⚡ MULYTIC EV AI PREDICTION HUB ⚡")
        print("="*60)
        print("Select a prediction model to run:")
        print("  [1] Energy Consumption Predictor (Charger Level)")
        print("  [2] State of Charge (SoC %) Predictor (Charger Level)")
        print("  [3] Station Congestion Predictor (Station Level)")
        print("  [4] Station Demand Forecaster (Station Level)")
        print("  [5] Leftover Energy Predictor (Charger Level)")
        print("  [6] Start REST API Server (FastAPI)")
        print("  [q] Quit")
        print("-" * 60)
        
        choice = input("Enter your choice (1/2/3/4/5/q): ").strip().lower()
        
        if choice == '1':
            os.system("python3 src/01_energy_model/predict.py")
        elif choice == '2':
            os.system("python3 src/02_soc_model/predict.py")
        elif choice == '3':
            os.system("python3 src/03_congestion_model/predict.py")
        elif choice == '4':
            os.system("python3 src/04_demand_model/predict.py")
        elif choice == '5':
            os.system("python3 src/05_leftover_model/predict.py")
        elif choice == '6':
            print("\nStarting FastAPI server at http://127.0.0.1:8000")
            print("Swagger UI: http://127.0.0.1:8000/docs")
            print("Press CTRL+C to stop the server.\n")
            os.system("uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload")
        elif choice == 'q' or choice == 'quit':
            print("Exiting Prediction Hub. Have a great day!")
            sys.exit(0)
        else:
            print("[ERROR] Invalid choice. Please enter 1, 2, 3, 4, 5, or q.")
        
        # Pause before showing the menu again
        input("\nPress ENTER to return to the main menu...")

if __name__ == "__main__":
    main()
