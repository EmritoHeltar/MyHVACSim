import sys
import os
import time
import numpy as np
from types import ModuleType
import shutil
import glob

# ==========================================
# WINDOWS STABILITY HACKS
# ==========================================
mock_fcntl = ModuleType("fcntl")
mock_fcntl.LOCK_EX = 2; mock_fcntl.LOCK_SH = 1; mock_fcntl.LOCK_NB = 4; mock_fcntl.LOCK_UN = 8
def dummy_flock(fd, operation): pass 
mock_fcntl.flock = dummy_flock
sys.modules["fcntl"] = mock_fcntl

import platform

# ==========================================
# WINDOWS & LINUX PATH HANDLING
# ==========================================
if platform.system() == "Windows":
    eplus_path = r'C:\EnergyPlusV25-2-0'
else:
    # Linux (Docker) standard installation path
    eplus_path = os.environ.get('EPLUS_PATH', '/usr/local/EnergyPlus-24-1-0')

os.environ['EPLUS_PATH'] = eplus_path
if eplus_path not in sys.path:
    sys.path.append(eplus_path)

import gymnasium as gym
import sinergym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# ==========================================
# 1. THE NATIVE HARDWARE REWARD SYSTEM (REBALANCED)
# ==========================================
class NativeHardwareRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_actions = None
        self.MAX_IT_LOAD_WATTS = 30000.0
        
        # Reward weights
        self.w_power = 0.5        # Priority: Minimize HVAC power
        self.w_comfort = 0.5      # Priority: Maintain server temperature
        self.w_voltage_jitter = 0.0 # Temporarily disabled
        self.w_synergy = 0.1      # Penalty for inefficient action pairs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_actions = np.zeros(6, dtype=np.float32)
        return obs, info

    def step(self, action):
        east_coil_v = action[0]
        west_coil_v = action[1]
        east_fan_v = action[2]
        west_fan_v = action[3]
        east_damper_v = action[4]
        west_damper_v = action[5]
        
        # Scale voltages [0, 10] to appropriate EnergyPlus physical bounds
        # Coils: 0.0 to 1.0 (Valve fraction)
        # Fans: VFD mapping 0V -> 3.0 kg/s (prevent evaporator freeze), 10V -> 10.0 kg/s
        # Dampers: 0V -> 0 kg/s, 10V -> 5.0 kg/s max OA flow
        MIN_FAN_FLOW = 3.0
        MAX_FAN_FLOW = 10.0
        east_fan_flow = MIN_FAN_FLOW + (east_fan_v / 10.0) * (MAX_FAN_FLOW - MIN_FAN_FLOW)
        west_fan_flow = MIN_FAN_FLOW + (west_fan_v / 10.0) * (MAX_FAN_FLOW - MIN_FAN_FLOW)

        sim_action = np.array([
            east_coil_v / 10.0, west_coil_v / 10.0,
            east_fan_flow, west_fan_flow,
            east_damper_v / 2.0, west_damper_v / 2.0
        ], dtype=np.float32)
        
        obs, reward, terminated, truncated, info = self.env.step(sim_action)

        # 1. Absolute Power Penalty
        hvac_power = obs[36] 
        cpu_load = obs[26]
        # Calculate PUE just for logging purposes
        it_power = self.MAX_IT_LOAD_WATTS * max(0.3, cpu_load)
        total_power = it_power + hvac_power
        pue = total_power / it_power
        
        # The penalty is raw HVAC power
        power_penalty = hvac_power / 10000.0 
        
        # 2. Comfort Penalty (Target deadband: 21C to 24C)
        avg_zone_temp = (obs[9] + obs[10]) / 2.0
        if 21.0 <= avg_zone_temp <= 24.0:
            comfort_penalty = 0.0
        elif avg_zone_temp > 24.0:
            comfort_penalty = avg_zone_temp - 24.0
        else:
            comfort_penalty = 21.0 - avg_zone_temp
        
        # 3. Voltage Jitter Penalty (Disabled right now)
        if self.prev_actions is not None:
            voltage_jitter = np.mean(np.abs(action - self.prev_actions))
        else:
            voltage_jitter = 0.0
            
        self.prev_actions = action.copy()

        # 4. Synergistic Action Penalty (Blowing Hot Air)
        # If fan voltage > 5.0 but cooling coil voltage < 1.0, penalize heavily.
        east_synergy = max(0.0, east_fan_v - 5.0) * max(0.0, 1.0 - east_coil_v)
        west_synergy = max(0.0, west_fan_v - 5.0) * max(0.0, 1.0 - west_coil_v)
        synergy_penalty = east_synergy + west_synergy

        # Final Grade
        custom_reward = - (self.w_power * power_penalty) - (self.w_comfort * comfort_penalty) - (self.w_synergy * synergy_penalty) - (self.w_voltage_jitter * voltage_jitter)
        
        info['hardware_signals'] = {
            'east_coil_v': east_coil_v,
            'west_coil_v': west_coil_v,
            'east_fan_v': east_fan_v,
            'west_fan_v': west_fan_v,
            'east_damper_v': east_damper_v,
            'west_damper_v': west_damper_v,
            'v_jitter': voltage_jitter,
            'pue': pue
        }
        
        return obs, custom_reward, terminated, truncated, info
# ==========================================
# 2. THE HARDWARE LOGGER CALLBACK
# ==========================================
class BelimoHardwareCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        obs = self.locals["new_obs"][0] 
        
        hw_data = info.get('hardware_signals', {})
        
        self.logger.record("belimo_hardware/1_East_Coil_Volts", hw_data.get('east_coil_v', 0.0))
        self.logger.record("belimo_hardware/2_West_Coil_Volts", hw_data.get('west_coil_v', 0.0))
        self.logger.record("belimo_hardware/3_East_Fan_Volts", hw_data.get('east_fan_v', 0.0))
        self.logger.record("belimo_hardware/4_West_Fan_Volts", hw_data.get('west_fan_v', 0.0))
        self.logger.record("belimo_hardware/5_East_Damper_Volts", hw_data.get('east_damper_v', 0.0))
        self.logger.record("belimo_hardware/6_West_Damper_Volts", hw_data.get('west_damper_v', 0.0))
        self.logger.record("belimo_hardware/7_Voltage_Jitter", hw_data.get('v_jitter', 0.0))
        
        self.logger.record("belimo_kpis/pue_ratio", hw_data.get('pue', 1.0))
        self.logger.record("environment/indoor_server_temp", (obs[9] + obs[10]) / 2.0)
        self.logger.record("environment/cpu_heat_load", obs[26])
        return True

# ==========================================
# 3. LAUNCHING THE SWARM 
# ==========================================
def make_env(rank):
    def _init():
        time.sleep(rank * 0.7) 
        
        env = gym.make(
            'Eplus-datacenter_dx-mixed-continuous-stochastic-v1',
            actuators={
                'East_Coil_Voltage': ('Plant Component Coil:Cooling:Water', 'On/Off Supervisory', 'EAST ZONE CW COOLING COIL'),
                'West_Coil_Voltage': ('Plant Component Coil:Cooling:Water', 'On/Off Supervisory', 'WEST ZONE CW COOLING COIL'),
                'East_Fan_Voltage': ('Fan', 'Fan Air Mass Flow Rate', 'EAST ZONE SUPPLY FAN'),
                'West_Fan_Voltage': ('Fan', 'Fan Air Mass Flow Rate', 'WEST ZONE SUPPLY FAN'),
                'East_Damper_Voltage': ('Outdoor Air Controller', 'Air Mass Flow Rate', 'EAST DATA CENTER OA CONTROLLER'),
                'West_Damper_Voltage': ('Outdoor Air Controller', 'Air Mass Flow Rate', 'WEST DATA CENTER OA CONTROLLER')
            },
            action_space=gym.spaces.Box(low=0.0, high=10.0, shape=(6,), dtype=np.float32)
        )
        
        env = NativeHardwareRewardWrapper(env)   
        return env
    return _init

if __name__ == '__main__':
    print("Starting native hardware continuous control training...")
    
    # Clean up old Sinergym simulation output directories
    print("Cleaning old simulation run folders...")
    for folder in glob.glob("Eplus-datacenter_dx*"):
        if os.path.isdir(folder):
            try:
                shutil.rmtree(folder)
            except Exception as e:
                pass # Ignore locked files on Windows
    print("Cleanup complete.")

    envs = SubprocVecEnv([make_env(i) for i in range(16)])
    brain_path = "hvac_belimo_native_hardware.zip"
    
    # Load existing model if available
    if os.path.exists(brain_path):
        print(f"Loading existing model from {brain_path}...")
        model = PPO.load(brain_path, env=envs, device="cuda")
        reset_logs = False
    else:
        print("Initializing new PPO model...")
        model = PPO("MlpPolicy", envs, verbose=1, device="cuda", tensorboard_log="./hvac_tensorboard/", n_steps=2048, batch_size=64)
        reset_logs = True

    print("\nPress Ctrl+C to pause and save the model.")
    try:
        model.learn(
            total_timesteps=100000000, 
            callback=BelimoHardwareCallback(), 
            tb_log_name="run_belimo_native_hardware",
            reset_num_timesteps=reset_logs
        )
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Saving current model state...")

    model.save(brain_path)
    print(f"Model successfully saved to {brain_path}")
