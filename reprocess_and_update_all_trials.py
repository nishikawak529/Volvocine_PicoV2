#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recompute filtered velocities and re-generate plots across all 57 trials
with robust outlier suppression (no spikes > 400 px/s).
"""

import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path

from track_robot import compute_filtered_velocity, plot_swimming_results

BASE_DIR = Path(r"D:\codes\Volvocine_PicoV2\VolBotVideo")
CONDITIONS = ['baseline', 'sinz', 'msinz', 'moptz']

def main():
    updated_records = []
    fps = 59.94
    stroke_freq = 1.25
    
    print("=" * 80)
    print("  REPROCESSING ALL 57 TRIALS WITH ROBUST VELOCITY FILTERING")
    print("=" * 80)
    
    for cond in CONDITIONS:
        cond_dir = BASE_DIR / cond
        if not cond_dir.exists():
            continue
        trials = sorted([d for d in cond_dir.iterdir() if d.is_dir() and d.name.startswith("GX")])
        
        for t_dir in trials:
            t_name = t_dir.name
            csv_path = t_dir / f"{t_name}_trajectory_velocity.csv"
            fig_path = t_dir / f"{t_name}_swimming_analysis.png"
            
            if not csv_path.exists():
                continue
                
            df = pd.read_csv(csv_path)
            
            # 1. Truncate prolonged lost gaps (>= 20 frames in 2nd half of video)
            is_lost = (df['status'] == 'lost').to_numpy()
            lost_runs = []
            in_run, start_r = False, 0
            for i, val in enumerate(is_lost):
                if val and not in_run:
                    in_run, start_r = True, i
                elif not val and in_run:
                    in_run = False
                    if (i - start_r) >= 20:
                        lost_runs.append((start_r, i))
            if in_run and (len(is_lost) - start_r) >= 20:
                lost_runs.append((start_r, len(is_lost)))
                
            cutoff_idx = len(df)
            for s_r, e_r in lost_runs:
                if s_r > len(df) * 0.5:
                    cutoff_idx = min(cutoff_idx, s_r)
                    
            if cutoff_idx < len(df):
                print(f"[TRUNCATE] [{cond}] {t_name}: Truncating at index {cutoff_idx} (was {len(df)}) due to end-of-swim lost gap.")
                df = df.iloc[:cutoff_idx].copy().reset_index(drop=True)
                
            # 2. Recompute filtered velocities with new Hampel outlier filter and short-window instantaneous speed
            df_analyzed, win_sec, win_frames = compute_filtered_velocity(
                df=df,
                fps=fps,
                stroke_freq=stroke_freq
            )
            
            # Overwrite CSV
            df_analyzed.to_csv(csv_path, index=False)
            
            # Re-generate swimming analysis PNG and PDF
            peak_freq, mean_speed = plot_swimming_results(
                df=df_analyzed,
                fps=fps,
                window_sec=win_sec,
                window_frames=win_frames,
                target_freq=stroke_freq,
                output_img_path=str(fig_path)
            )
            
            max_raw = df_analyzed['speed_raw_px_s'].max()
            max_sg = df_analyzed['speed_sg_px_s'].max()
            mean_sg = df_analyzed['speed_sg_px_s'].mean()
            
            updated_records.append({
                'cond': cond,
                'trial': t_name,
                'mean_speed': mean_sg,
                'max_raw': max_raw,
                'max_sg': max_sg,
                'over_400': (df_analyzed['speed_raw_px_s'] > 400).sum(),
                'over_200': (df_analyzed['speed_raw_px_s'] > 200).sum(),
                'over_100': (df_analyzed['speed_raw_px_s'] > 100).sum()
            })
            
            print(f"[{cond}] {t_name}: Mean = {mean_sg:.2f} px/s, Max Raw = {max_raw:.1f} px/s, Max SG = {max_sg:.1f} px/s")

    df_summary = pd.DataFrame(updated_records)
    print("\n" + "=" * 80)
    print("  SUMMARY OF UPDATED VELOCITY METRICS ACROSS ALL 57 TRIALS")
    print("=" * 80)
    print(f"Total trials processed: {len(df_summary)}")
    print(f"Trials with speed_raw > 100: {(df_summary['over_100'] > 0).sum()}")
    print(f"Maximum raw instantaneous speed across ALL trials: {df_summary['max_raw'].max():.1f} px/s")
    print(f"Maximum stroke-filtered SG speed across ALL trials: {df_summary['max_sg'].max():.1f} px/s")

if __name__ == "__main__":
    main()
