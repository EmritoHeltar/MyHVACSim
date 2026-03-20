import sys
import time
import argparse
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEMO_DIR = SCRIPT_DIR / "Start_Hack_26_H4ckerbros" / "Belimo-START-Hack-2026-main" / "Belimo-START-Hack-2026-main" / "demo"
if DEMO_DIR.exists():
    sys.path.append(str(DEMO_DIR))

try:
    from interface.influx.api import get_measurement_data, set_process_data
except ImportError:
    print("Warning: Belimo InfluxDB API not found. Running in simulation mode only.")
    def get_measurement_data(n=1): return None
    def set_process_data(setpoint, test_number=-1): pass

class MockEnvironment:
    def __init__(self, initial_temp=24.0):
        self.east_temp = initial_temp
        self.west_temp = initial_temp + 1.2
        self.outdoor_temp = 30.0
        
    def update(self, east_cw, west_cw, east_fan, west_fan, east_oa, west_oa):
        base_heating = 0.5 
        
        east_cooling = (east_cw / 10.0) * (east_fan / 10.0) * 1.5
        west_cooling = (west_cw / 10.0) * (west_fan / 10.0) * 1.4
        
        east_oa_impact = (east_oa / 10.0) * (self.outdoor_temp - self.east_temp) * 0.05
        west_oa_impact = (west_oa / 10.0) * (self.outdoor_temp - self.west_temp) * 0.05
        
        self.east_temp += base_heating - east_cooling + east_oa_impact + random.uniform(-0.05, 0.05)
        self.west_temp += (base_heating * 1.1) - west_cooling + west_oa_impact + random.uniform(-0.05, 0.05)
        
        self.east_temp = max(15.0, min(35.0, self.east_temp))
        self.west_temp = max(15.0, min(35.0, self.west_temp))
        
        return self.east_temp, self.west_temp

class HVACAgent:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.dummy_state = 50.0
        self.last_valid_oa_voltage = None
        self.model = None
        
        if self.model_path:
            print(f"Loading SB3 PPO from {self.model_path}...")
            try:
                from stable_baselines3 import PPO
                self.model = PPO.load(self.model_path, device="cpu")
                print("Model loaded successfully!")
            except ImportError:
                print("Error: stable_baselines3 is not installed.")
            except Exception as e:
                print(f"Failed to load SB3 model: {e}")
        else:
            print("No model path provided. Running the ML fallback/dummy logic.")

    def predict(self, actuator_state, mock_sensors_state):
        east_zone_temp = mock_sensors_state['east_zone_temp']
        west_zone_temp = mock_sensors_state['west_zone_temp']
        room_thermostat_avg = (east_zone_temp + west_zone_temp) / 2.0
        
        if self.model is not None:
            import numpy as np
            
            obs_array = np.zeros(37, dtype=np.float32)
            obs_array[3] = 32.0
            obs_array[4] = 50.0
            obs_array[26] = 0.5
            
            obs_array[17] = 13.0
            obs_array[19] = 13.0
            obs_array[16] = east_zone_temp + 1.0
            obs_array[18] = west_zone_temp + 1.2
            
            obs_array[9] = east_zone_temp 
            obs_array[10] = west_zone_temp
            
            action, _states = self.model.predict(obs_array, deterministic=False)
            
            raw_east_cw = float(action[0]) if len(action) > 0 else 0.0
            raw_west_cw = float(action[1]) if len(action) > 1 else raw_east_cw
            raw_east_fan = float(action[2]) if len(action) > 2 else raw_east_cw
            raw_west_fan = float(action[3]) if len(action) > 3 else raw_east_fan
            raw_east_oa = float(action[4]) if len(action) > 4 else raw_east_cw
            raw_west_oa = float(action[5]) if len(action) > 5 else raw_east_oa
            
            def unnormalize(val):
                return max(0.0, min(10.0, 0.0 + 0.5 * (val + 1.0) * 10.0))
                
            east_cw = unnormalize(raw_east_cw)
            west_cw = unnormalize(raw_west_cw)
            east_fan = unnormalize(raw_east_fan)
            west_fan = unnormalize(raw_west_fan)
            east_oa = unnormalize(raw_east_oa)
            west_oa = unnormalize(raw_west_oa)
            
            if 4.95 <= east_oa <= 5.05:
                if self.last_valid_oa_voltage is not None:
                    east_oa = self.last_valid_oa_voltage
            else:
                self.last_valid_oa_voltage = east_oa
            
            return {
                'east_cw': east_cw, 'west_cw': west_cw,
                'east_fan': east_fan, 'west_fan': west_fan,
                'east_oa': east_oa, 'west_oa': west_oa
            }
            
        if room_thermostat_avg > 22.5:
            self.dummy_state += 10.0
        elif room_thermostat_avg < 21.5:
            self.dummy_state -= 10.0
            
        fallback_v = (max(0.0, min(100.0, self.dummy_state)) / 100.0) * 10.0
        return {
            'east_cw': fallback_v, 'west_cw': fallback_v,
            'east_fan': fallback_v, 'west_fan': fallback_v,
            'east_oa': fallback_v, 'west_oa': fallback_v
        }


def main():
    parser = argparse.ArgumentParser(description="Live 6-Channel Actuator Demo Loop")
    parser.add_argument("--model-path", type=str, help="Path to your trained model weights")
    parser.add_argument("--test-number", type=int, default=1001, help="InfluxDB experiment tag")
    parser.add_argument("--mock-temp", type=float, default=24.0, help="Initial mock room temperature")
    parser.add_argument("--offline", action="store_true", help="Run offline dry-run (bypasses InfluxDB timeout freezing)")
    args = parser.parse_args()

    mock_env = MockEnvironment(initial_temp=args.mock_temp)
    agent = HVACAgent(model_path=args.model_path)

    print("\n--- Starting Full 6-Channel Live Inference Loop ---")
    if args.offline:
        print(">>> RUNNING IN OFFLINE DRY-RUN MODE (InfluxDB calls bypassed)")
    print("Press Ctrl+C to exit.\n")
    
    current_state = {
        'east_cw': 5.0, 'west_cw': 5.0,
        'east_fan': 5.0, 'west_fan': 5.0,
        'east_oa': 5.0, 'west_oa': 5.0
    }
    
    try:
        while True:
            east_temp, west_temp = mock_env.update(**current_state)
            
            mock_sensors_state = {
                'east_zone_temp': east_temp,
                'west_zone_temp': west_temp
            }
            
            if not args.offline:
                try:
                    df = get_measurement_data(n=1)
                    actuator_state = df.iloc[0].to_dict() if (df is not None and not df.empty) else {}
                except Exception:
                    print("[Warning] InfluxDB Spotty Connection - Faking Hardware Readout!")
                    actuator_state = {'feedback_position_%': (current_state['east_oa'] / 10.0) * 100.0, 'internal_temperature_deg_C': 25.0}
            else:
                actuator_state = {'feedback_position_%': (current_state['east_oa'] / 10.0) * 100.0, 'internal_temperature_deg_C': 25.0}

            pos = actuator_state.get('feedback_position_%', 50.0)
            
            current_state = agent.predict(actuator_state, mock_sensors_state)
            
            east_oa_percentage = (current_state['east_oa'] / 10.0) * 100.0
            
            if not args.offline and 'get_measurement_data' in globals() and actuator_state:
                try:
                    set_process_data(east_oa_percentage, test_number=args.test_number)
                except Exception as e:
                    print(f"[Warning] Failed to push data to physical Actuator API (Timeout).")
            
            avg_temp = (east_temp + west_temp) / 2.0
            print(f"[Hardware  ] Belimo Desk Actuator Pos (East OA): {pos:.1f}%")
            print(f"[Mock Env  ] Thermostat: {avg_temp:.2f}°C (Outdoor: {mock_env.outdoor_temp:.1f}°C)")
            
            print(f"[ML Outputs]")
            print(f"  CW Valves   -> East: {current_state['east_cw']:5.2f}V | West: {current_state['west_cw']:5.2f}V")
            print(f"  Fan Drives  -> East: {current_state['east_fan']:5.2f}V | West: {current_state['west_fan']:5.2f}V")
            print(f"  OA Dampers  -> East: {current_state['east_oa']:5.2f}V | West: {current_state['west_oa']:5.2f}V")
            
            print("-" * 65)
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nLive control loop stopped safely.")

if __name__ == "__main__":
    main()
