#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trim 5 seconds from start and 2 seconds from end across all trials in VolBotVideo,
recompute stroke-filtered velocities, and regenerate swimming analysis plots.
"""

import os
import sys
import glob
import time
import shutil
import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(r"D:\codes\Volvocine_PicoV2")
from track_robot import compute_filtered_velocity, plot_swimming_results

BASE_DIR = Path(r"D:\codes\Volvocine_PicoV2\VolBotVideo")
CONDITIONS = ['baseline', 'sinz', 'm5sinz', 'm10sinz', 'moptz']

POOL_HEIGHT_PX_4K = 1585.0
POOL_HEIGHT_M = 2.0
SCALE_M_PER_PX = POOL_HEIGHT_M / POOL_HEIGHT_PX_4K
SCALE_CM_PER_PX = SCALE_M_PER_PX * 100.0


def process_single_trial(info):
    """
    Process a single trial: backup original CSV, trim 5s start and 2s end,
    recompute velocities, overwrite CSV, and regenerate swimming analysis PNG/PDF.
    """
    cond = info["condition"]
    t_name = info["trial"]
    t_dir = Path(info["trial_dir"])
    
    csv_path = t_dir / f"{t_name}_trajectory_velocity.csv"
    backup_path = t_dir / f"{t_name}_trajectory_velocity_backup.csv"
    fig_path = t_dir / f"{t_name}_swimming_analysis.png"
    bg_path = t_dir / f"{t_name}_background.jpg"
    
    if not csv_path.exists() and not backup_path.exists():
        return None
        
    # Backup original CSV if not already backed up
    if not backup_path.exists():
        shutil.copy2(csv_path, backup_path)
        
    # Always load from the pristine backup to ensure idempotence
    df_orig = pd.read_csv(backup_path)
    if len(df_orig) < 30:
        return {"cond": cond, "trial": t_name, "status": "too_short_orig"}
        
    t_orig_start = float(df_orig['time_sec'].iloc[0])
    t_orig_end = float(df_orig['time_sec'].iloc[-1])
    orig_dur = t_orig_end - t_orig_start
    
    # Trim 5.0 seconds from start and 2.0 seconds from end
    t_cut_start = t_orig_start + 5.0
    t_cut_end = t_orig_end - 2.0
    
    # Safe boundary check
    if t_cut_end <= t_cut_start + 2.0:
        # If trial is very short, keep at least middle section
        t_cut_start = t_orig_start + min(3.0, orig_dur * 0.2)
        t_cut_end = t_orig_end - min(1.5, orig_dur * 0.1)
        
    df_trim = df_orig[(df_orig['time_sec'] >= t_cut_start) & (df_orig['time_sec'] <= t_cut_end)].copy().reset_index(drop=True)
    
    if len(df_trim) < 30:
        return {"cond": cond, "trial": t_name, "status": "too_short_after_trim"}
        
    t_trim_start = float(df_trim['time_sec'].iloc[0])
    t_trim_end = float(df_trim['time_sec'].iloc[-1])
    trim_dur = t_trim_end - t_trim_start
    
    # Determine FPS from actual median time interval
    dt = float(np.median(np.diff(df_trim['time_sec'])))
    fps = 1.0 / dt if dt > 0 else 59.94
    
    # Recompute filtered velocities with full metrics
    df_analyzed, win_sec, win_frames = compute_filtered_velocity(
        df=df_trim,
        fps=fps,
        stroke_freq=1.25,
        scale_m_per_px=SCALE_M_PER_PX
    )
    
    # Overwrite the primary CSV with trimmed, re-filtered data
    df_analyzed.to_csv(csv_path, index=False)
    
    # Load background if present
    bg_img = None
    if bg_path.exists():
        bg_img = cv2.imread(str(bg_path))
        
    # Re-generate swimming analysis plot (both PNG and PDF are saved by this function)
    peak_freq, mean_speed = plot_swimming_results(
        df=df_analyzed,
        fps=fps,
        window_sec=win_sec,
        window_frames=win_frames,
        target_freq=1.25,
        output_img_path=str(fig_path),
        bg_image=bg_img
    )
    
    mean_speed_cm_s = mean_speed * SCALE_CM_PER_PX
    max_sg_px_s = float(df_analyzed['speed_sg_px_s'].max())
    max_raw_px_s = float(df_analyzed['speed_raw_px_s'].max())
    
    return {
        "cond": cond,
        "trial": t_name,
        "status": "success",
        "fps": fps,
        "orig_start": t_orig_start,
        "orig_end": t_orig_end,
        "orig_dur": orig_dur,
        "trim_start": t_trim_start,
        "trim_end": t_trim_end,
        "trim_dur": trim_dur,
        "orig_frames": len(df_orig),
        "trim_frames": len(df_analyzed),
        "mean_speed_px_s": mean_speed,
        "mean_speed_cm_s": mean_speed_cm_s,
        "max_sg_px_s": max_sg_px_s,
        "max_raw_px_s": max_raw_px_s,
        "peak_freq": peak_freq
    }


def main():
    parser = argparse.ArgumentParser(description="Trim 5s start and 2s end across all trials and reprocess.")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel processes (default: 8)")
    args = parser.parse_args()
    
    print("=" * 80)
    print("  TRIMMING 5s START & 2s END ACROSS ALL TRIALS AND REPROCESSING")
    print("=" * 80)
    
    # Gather trials
    trials_to_process = []
    for cond in CONDITIONS:
        cond_dir = BASE_DIR / cond
        if not cond_dir.exists():
            continue
        subdirs = sorted([d for d in cond_dir.iterdir() if d.is_dir() and d.name.startswith("GX")])
        for t_dir in subdirs:
            trials_to_process.append({
                "condition": cond,
                "trial": t_dir.name,
                "trial_dir": str(t_dir)
            })
            
    print(f"[INFO] Found {len(trials_to_process)} trials across {len(CONDITIONS)} conditions.")
    
    t0 = time.time()
    results = []
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_trial, t): t for t in trials_to_process}
        for future in as_completed(futures):
            res = future.result()
            if res is not None:
                results.append(res)
                if res.get("status") == "success":
                    print(f"[{res['cond']}] {res['trial']}: trimmed {res['orig_dur']:.1f}s -> {res['trim_dur']:.1f}s, speed={res['mean_speed_cm_s']:.2f} cm/s ({res['mean_speed_px_s']:.1f} px/s), peak={res['peak_freq']:.2f} Hz")
                else:
                    print(f"[WARN] [{res.get('cond')}] {res.get('trial')}: status={res.get('status')}")
                    
    elapsed = time.time() - t0
    df_res = pd.DataFrame([r for r in results if r.get("status") == "success"])
    
    print("\n" + "=" * 80)
    print(f"  FINISHED REPROCESSING {len(df_res)} / {len(trials_to_process)} TRIALS IN {elapsed:.1f}s")
    print("=" * 80)
    if not df_res.empty:
        print("\nSummary by condition:")
        summary = df_res.groupby('cond').agg({
            'trial': 'count',
            'orig_dur': 'mean',
            'trim_dur': 'mean',
            'mean_speed_cm_s': ['mean', 'std'],
            'peak_freq': 'mean'
        })
        print(summary)


if __name__ == "__main__":
    main()
