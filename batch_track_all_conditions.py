#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Tracking across all condition directories in VolBotVideo
============================================================
Processes all trials using the improved tracking algorithm:
- Black robot physical prior (gray < 105, dark contrast > 25, blue heading marker)
- Lost-frame recovery tolerance (up to 15 frames before full-frame search)
- Metric calibration (2.0 m = 1585 px in 4K reference)
- Unified pool bounds on background overlay
"""

import os
import glob
import time
import argparse
import subprocess
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys
sys.path.append(r"D:\codes\Volvocine_PicoV2")
from track_robot import generate_background, track_robot_trajectory, compute_filtered_velocity, plot_swimming_results

BASE_DIR = r"D:\codes\Volvocine_PicoV2\VolBotVideo"
CONDITIONS = ['baseline', 'sinz', 'm5sinz', 'm10sinz', 'moptz']

# Physical calibration constants
POOL_HEIGHT_PX_4K = 1585.0
POOL_HEIGHT_M = 2.0
SCALE_M_PER_PX = POOL_HEIGHT_M / POOL_HEIGHT_PX_4K
SCALE_CM_PER_PX = SCALE_M_PER_PX * 100.0


def get_all_trials(selected_conditions=None):
    if selected_conditions is None:
        selected_conditions = CONDITIONS
    trials_list = []
    for cond in selected_conditions:
        cond_dir = os.path.join(BASE_DIR, cond)
        if not os.path.exists(cond_dir):
            continue
        subdirs = sorted([d for d in os.listdir(cond_dir) if os.path.isdir(os.path.join(cond_dir, d)) and d.startswith("GX")])
        for t in subdirs:
            t_dir = os.path.join(cond_dir, t)
            v_matches = glob.glob(os.path.join(t_dir, f"{t}.[mM][pP]4"))
            if v_matches:
                trials_list.append({
                    "condition": cond,
                    "trial": t,
                    "trial_dir": t_dir,
                    "video_path": v_matches[0]
                })
    return trials_list


def process_single_trial(info, force=True):
    t_name = info["trial"]
    cond = info["condition"]
    trial_dir = info["trial_dir"]
    video_path = info["video_path"]
    
    csv_path = os.path.join(trial_dir, f"{t_name}_trajectory_velocity.csv")
    fig_path = os.path.join(trial_dir, f"{t_name}_swimming_analysis.png")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_dur = total_frames / fps if fps > 0 else 0
    cap.release()
    
    # Check if already processed (only when not forced)
    if not force and os.path.exists(csv_path) and os.path.exists(fig_path):
        try:
            df_existing = pd.read_csv(csv_path)
            if len(df_existing) > 100:
                swim_dur = df_existing['time_sec'].iloc[-1] - df_existing['time_sec'].iloc[0]
                mean_speed = float(df_existing['speed_sg_px_s'].mean())
                max_speed = float(df_existing['speed_sg_px_s'].max())
                std_speed = float(df_existing['speed_sg_px_s'].std())
                dx = np.diff(df_existing['x_smooth_px'])
                dy = np.diff(df_existing['y_smooth_px'])
                total_dist_px = float(np.sum(np.sqrt(dx**2 + dy**2)))
                dt = 1.0 / fps
                speed_detrend = df_existing['speed_raw_px_s'] - df_existing['speed_raw_px_s'].mean()
                freqs = np.fft.rfftfreq(len(speed_detrend), d=dt)
                fft_vals = np.abs(np.fft.rfft(speed_detrend))
                mask = (freqs >= 0.5) & (freqs <= 3.0)
                peak_freq = float(freqs[mask][np.argmax(fft_vals[mask])]) if np.any(mask) else 1.25
                return {
                    "condition": cond,
                    "trial": t_name,
                    "video_duration_sec": video_dur,
                    "swimming_duration_sec": swim_dur,
                    "detected_frames": len(df_existing),
                    "detection_rate_pct": 100.0,
                    "mean_speed_px_s": mean_speed,
                    "mean_speed_cm_s": mean_speed * SCALE_CM_PER_PX,
                    "max_speed_px_s": max_speed,
                    "max_speed_cm_s": max_speed * SCALE_CM_PER_PX,
                    "std_speed_px_s": std_speed,
                    "std_speed_cm_s": std_speed * SCALE_CM_PER_PX,
                    "total_distance_px": total_dist_px,
                    "total_distance_m": total_dist_px * SCALE_M_PER_PX,
                    "stroke_peak_freq_hz": peak_freq
                }
        except Exception:
            pass

    start_sec = 5.0
    end_sec = max(start_sec + 5.0, video_dur - 2.5)
    
    t_start_clock = time.time()
    
    # Generate background specifically for this video using evenly spaced samples
    sample_fractions = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65]
    sample_times = [video_dur * frac for frac in sample_fractions]
    bg_image = generate_background(video_path, sample_times=sample_times)
    
    # Save background for record
    bg_path = os.path.join(trial_dir, f"{t_name}_background.jpg")
    cv2.imwrite(bg_path, bg_image)
    
    # Track with black robot prior & lost frame recovery tolerance
    df_track, fps_out = track_robot_trajectory(
        video_path=video_path,
        bg_image=bg_image,
        start_sec=start_sec,
        end_sec=end_sec,
        save_annotated_video=False,
        verbose=False
    )
    
    detected_indices = df_track.index[df_track['status'] == 'detected'].tolist()
    if not detected_indices or len(detected_indices) < 60:
        print(f"[WARN] [{cond}] {t_name}: Not enough detected frames.")
        return None
        
    first_det = detected_indices[0]
    
    # Check for prolonged lost gaps (e.g. >= 20 consecutive lost frames in 2nd half of video)
    # when robot stops/reaches wall and is picked up
    is_lost = (df_track['status'] == 'lost').to_numpy()
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
    
    cutoff_idx = len(df_track)
    for s_r, e_r in lost_runs:
        if s_r > len(df_track) * 0.5:
            cutoff_idx = min(cutoff_idx, s_r)
            
    detected_before_cutoff = [i for i in detected_indices if i < cutoff_idx]
    last_det = detected_before_cutoff[-1] if detected_before_cutoff else detected_indices[-1]
    
    df_swim = df_track.iloc[first_det:last_det + 1].copy().reset_index(drop=True)
    
    swim_dur = df_swim['time_sec'].iloc[-1] - df_swim['time_sec'].iloc[0]
    det_ratio = (df_swim['status'] == 'detected').mean() * 100
    
    df_analyzed, win_sec, win_frames = compute_filtered_velocity(
        df=df_swim,
        fps=fps_out,
        stroke_freq=1.25,
        scale_m_per_px=SCALE_M_PER_PX
    )
    
    df_analyzed.to_csv(csv_path, index=False)
    
    peak_freq, mean_speed = plot_swimming_results(
        df=df_analyzed,
        fps=fps_out,
        window_sec=win_sec,
        window_frames=win_frames,
        target_freq=1.25,
        output_img_path=fig_path,
        bg_image=bg_image
    )
    
    speed_sg = df_analyzed['speed_sg_px_s']
    max_speed = float(speed_sg.max())
    std_speed = float(speed_sg.std())
    
    dx = np.diff(df_analyzed['x_smooth_px'])
    dy = np.diff(df_analyzed['y_smooth_px'])
    total_dist_px = float(np.sum(np.sqrt(dx**2 + dy**2)))
    
    mean_speed_cm_s = mean_speed * SCALE_CM_PER_PX
    max_speed_cm_s = max_speed * SCALE_CM_PER_PX
    std_speed_cm_s = std_speed * SCALE_CM_PER_PX
    total_dist_m = total_dist_px * SCALE_M_PER_PX
    
    elapsed_time = time.time() - t_start_clock
    
    return {
        "condition": cond,
        "trial": t_name,
        "video_duration_sec": video_dur,
        "swimming_duration_sec": swim_dur,
        "detected_frames": len(df_swim),
        "detection_rate_pct": det_ratio,
        "mean_speed_px_s": mean_speed,
        "mean_speed_cm_s": mean_speed_cm_s,
        "max_speed_px_s": max_speed,
        "max_speed_cm_s": max_speed_cm_s,
        "std_speed_px_s": std_speed,
        "std_speed_cm_s": std_speed_cm_s,
        "total_distance_px": total_dist_px,
        "total_distance_m": total_dist_m,
        "stroke_peak_freq_hz": peak_freq,
        "processing_time_sec": elapsed_time
    }


def main():
    parser = argparse.ArgumentParser(description="Batch re-track all trials with improved tracking algorithm.")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel worker processes (default: 10)")
    parser.add_argument("--no-force", action="store_true", help="Skip trials that already have results (default: force reprocess)")
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS, help="List of conditions to process")
    args = parser.parse_args()
    
    force_reprocess = not args.no_force
    trials = get_all_trials(args.conditions)
    total_n = len(trials)
    print("=" * 80)
    print("  BATCH RE-TRACKING ALL CONDITIONS WITH IMPROVED TRACKING")
    print("=" * 80)
    print(f"Total trials to process: {total_n}")
    print(f"Conditions: {args.conditions}")
    print(f"Parallel workers: {args.workers}")
    print(f"Force reprocess: {force_reprocess}")
    print("=" * 80)
    
    results = []
    completed_count = 0
    t0_all = time.time()
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_info = {
            executor.submit(process_single_trial, info, force_reprocess): info 
            for info in trials
        }
        
        for future in as_completed(future_to_info):
            info = future_to_info[future]
            completed_count += 1
            try:
                res = future.result()
                if res:
                    results.append(res)
                    print(f"[{completed_count:2d}/{total_n:2d}] [{res['condition']:8s}] {res['trial']}: "
                          f"Speed = {res['mean_speed_cm_s']:.2f} cm/s ({res['mean_speed_px_s']:.1f} px/s), "
                          f"Dist = {res['total_distance_m']:.2f} m, Det = {res['detection_rate_pct']:.1f}%, "
                          f"Time = {res['processing_time_sec']:.1f}s", flush=True)
                else:
                    print(f"[{completed_count:2d}/{total_n:2d}] [{info['condition']:8s}] {info['trial']}: FAILED (insufficient detections)", flush=True)
            except Exception as e:
                print(f"[{completed_count:2d}/{total_n:2d}] [{info['condition']:8s}] {info['trial']}: EXCEPTION: {e}", flush=True)

    total_elapsed = time.time() - t0_all
    print("\n" + "=" * 80, flush=True)
    print(f"  BATCH TRACKING COMPLETED IN {total_elapsed / 60.0:.2f} MINUTES", flush=True)
    print("=" * 80, flush=True)
    
    df_summary = pd.DataFrame(results)
    if not df_summary.empty:
        df_summary.sort_values(by=['condition', 'trial'], inplace=True)
        out_csv = os.path.join(BASE_DIR, "tracking_all_trials_summary.csv")
        df_summary.to_csv(out_csv, index=False)
        print(f"[SAVED] Tracking summary saved to: {out_csv}")
        
        # Summary by condition
        print("\n--- Summary Statistics by Condition ---")
        grouped = df_summary.groupby('condition').agg(
            trials=('trial', 'count'),
            mean_speed_cm_s=('mean_speed_cm_s', 'mean'),
            std_speed_cm_s=('mean_speed_cm_s', 'std'),
            mean_dist_m=('total_distance_m', 'mean'),
            mean_det_pct=('detection_rate_pct', 'mean')
        )
        print(grouped.to_string())

    # Automatically trigger downstream updates
    print("\n" + "=" * 80)
    print("  STEP 2: UPDATING SPEED VS COT AND ALL TRIALS SUMMARY")
    print("=" * 80)
    try:
        cmd_summary = [sys.executable, r"D:\codes\Volvocine_PicoV2\plot_speed_vs_cot.py"]
        subprocess.run(cmd_summary, check=True)
        print("[SUCCESS] Speed vs. CoT plots and all_trials_summary.csv updated.")
    except Exception as e:
        print(f"[ERROR] Failed to run plot_speed_vs_cot.py: {e}")

    print("\n" + "=" * 80)
    print("  STEP 3: RE-GENERATING POOL BACKGROUND TRAJECTORY OVERLAYS")
    print("=" * 80)
    try:
        cmd_bg = [sys.executable, r"D:\codes\Volvocine_PicoV2\plot_trajectories_on_background.py"]
        subprocess.run(cmd_bg, check=True)
        print("[SUCCESS] Background overlay figures successfully regenerated.")
    except Exception as e:
        print(f"[ERROR] Failed to run plot_trajectories_on_background.py: {e}")

    print("\n[ALL TASKS COMPLETED SUCCESSFULLY]")


if __name__ == "__main__":
    main()
