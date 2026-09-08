#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert phase_energy_analysis.mat to CSV across all condition directories
========================================================================
"""

import os
import glob
import scipy.io as sio
import numpy as np
import pandas as pd

base_dir = r"D:\codes\Volvocine_PicoV2\VolBotVideo"
conditions = ['baseline', 'sinz', 'msinz', 'moptz']

converted = 0
for cond in conditions:
    cond_dir = os.path.join(base_dir, cond)
    if not os.path.exists(cond_dir):
        continue
    trials = sorted([d for d in os.listdir(cond_dir) if os.path.isdir(os.path.join(cond_dir, d)) and d.startswith("GX")])
    for t_name in trials:
        td = os.path.join(cond_dir, t_name)
        mat_path = os.path.join(td, "phase_energy_analysis.mat")
        if not os.path.exists(mat_path):
            continue
        
        try:
            mat = sio.loadmat(mat_path)
            t = mat['t_common'].flatten()
            z_abs = mat['Z_abs']  # N x 4
            
            data_dict = {
                "time_sec": t,
                "order_param_mode1_Z1": z_abs[:, 0],
                "order_param_mode2_Z2": z_abs[:, 1],
                "order_param_mode3_Z3": z_abs[:, 2],
                "order_param_mode4_Z4": z_abs[:, 3],
            }
            
            # Extract phase_series
            ps_cell = mat['phase_series']
            entries = ps_cell[0, 0].flatten()
            total_power = np.zeros(len(t))
            
            for entry in entries:
                ag_id = int(entry['agent_id'][0, 0])
                ph = entry['phase'].flatten()
                ph_unwrapped = entry['absolute_phase_unwrapped'].flatten()
                pw = entry['power_w'].flatten() if 'power_w' in entry.dtype.names and len(entry['power_w']) > 0 else np.zeros(len(t))
                
                n = min(len(t), len(ph))
                data_dict[f"relative_phase_rad_agent_{ag_id}"] = ph[:len(t)] if len(ph) >= len(t) else np.pad(ph, (0, len(t)-len(ph)), constant_values=np.nan)
                data_dict[f"abs_phase_unwrapped_rad_agent_{ag_id}"] = ph_unwrapped[:len(t)] if len(ph_unwrapped) >= len(t) else np.pad(ph_unwrapped, (0, len(t)-len(ph_unwrapped)), constant_values=np.nan)
                if len(pw) >= len(t):
                    data_dict[f"power_w_agent_{ag_id}"] = pw[:len(t)]
                    total_power += np.nan_to_num(pw[:len(t)])
                    
            data_dict["total_power_w"] = total_power
            
            df_out = pd.DataFrame(data_dict)
            csv_path = os.path.join(td, "phase_energy_analysis.csv")
            df_out.to_csv(csv_path, index=False)
            
            # Clean up .mat
            os.remove(mat_path)
            
            converted += 1
            print(f"[CONVERTED] [{cond}] {t_name}: Saved phase_energy_analysis.csv ({len(df_out)} rows, {len(df_out.columns)} cols)")
        except Exception as e:
            print(f"[ERROR] [{cond}] {t_name} failed conversion: {e}")

print(f"\nSuccessfully converted {converted} trial analysis files from .mat to .csv!")
