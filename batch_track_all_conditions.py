#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Tracking across all condition directories in VolBotVideo
============================================================
"""

import os
import glob
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys
sys.path.append(r"D:\codes\Volvocine_PicoV2")
from track_robot import generate_background, track_robot_trajectory, compute_filtered_velocity, plot_swimming_results

BASE_DIR = r"D:\codes\Volvocine_PicoV2\VolBotVideo"
CONDITIONS = ['baseline', 'sinz', 'm5sinz', 'm10sinz', 'moptz', 'msinz']


def get_all_trials():
    trials_list = []
    for cond in CONDITIONS:
        cond_dir = os.path.join(BASE_DIR, cond)
        if not os.path.exists(cond_dir):
            continue
        subdirs = sorted([d for d in os.listdir(cond_dir) if os.path.isdir(os.path.join(cond_dir, d)) and d.startswith("GX")])
        for t in subdirs:
            t_dir = os.path.join(cond_dir, t)
            v_path = os.path.join(t_dir, f"{t}.MP4")
            if os.path.exists(v_path):
                trials_list.append({
                    "condition": cond,
                    "trial": t,
                    "trial_dir": t_dir,
                    "video_path": v_path
                })
    return trials_list


def process_single_trial(info):
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
    
    # Check if already processed
    if os.path.exists(csv_path) and os.path.exists(fig_path):
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
                print(f"[ALREADY DONE] [{cond}] {t_name}: Mean Speed = {mean_speed:.2f} px/s")
                return {
                    "condition": cond,
                    "trial": t_name,
                    "video_duration_sec": video_dur,
                    "swimming_duration_sec": swim_dur,
                    "detected_frames": len(df_existing),
                    "detection_rate_pct": 100.0,
                    "mean_speed_px_s": mean_speed,
                    "max_speed_px_s": max_speed,
                    "std_speed_px_s": std_speed,
                    "total_distance_px": total_dist_px,
                    "stroke_peak_freq_hz": peak_freq
                }
        except Exception as e:
            print(f"[WARN] Error reading existing data for {t_name}: {e}. Re-processing...")

    start_sec = 5.0
    end_sec = max(start_sec + 5.0, video_dur - 2.5)
    
    print(f"[START] [{cond}] {t_name} (Dur: {video_dur:.1f}s, Window: {start_sec:.1f}s - {end_sec:.1f}s) ...")
    
    # Generate background specifically for this video using evenly spaced samples
    sample_fractions = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65]
    sample_times = [video_dur * frac for frac in sample_fractions]
    bg_image = generate_background(video_path, sample_times=sample_times)
    
    # Save background for record
    bg_path = os.path.join(trial_dir, f"{t_name}_background.jpg")
    cv2.imwrite(bg_path, bg_image)
    
    df_track, fps_out = track_robot_trajectory(
        video_path=video_path,
        bg_image=bg_image,
        start_sec=start_sec,
        end_sec=end_sec,
        save_annotated_video=False
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
        stroke_freq=1.25
    )
    
    df_analyzed.to_csv(csv_path, index=False)
    
    peak_freq, mean_speed = plot_swimming_results(
        df=df_analyzed,
        fps=fps_out,
        window_sec=win_sec,
        window_frames=win_frames,
        target_freq=1.25,
        output_img_path=fig_path
    )
    
    speed_sg = df_analyzed['speed_sg_px_s']
    max_speed = float(speed_sg.max())
    std_speed = float(speed_sg.std())
    
    dx = np.diff(df_analyzed['x_smooth_px'])
    dy = np.diff(df_analyzed['y_smooth_px'])
    total_dist_px = float(np.sum(np.sqrt(dx**2 + dy**2)))
    
    print(f"[FINISHED] [{cond}] {t_name}: Mean Speed = {mean_speed:.2f} px/s, Peak = {peak_freq:.2f} Hz, Detected = {det_ratio:.1f}%")
    
    return {
        "condition": cond,
        "trial": t_name,
        "video_duration_sec": video_dur,
        "swimming_duration_sec": swim_dur,
        "detected_frames": len(df_swim),
        "detection_rate_pct": det_ratio,
        "mean_speed_px_s": mean_speed,
        "max_speed_px_s": max_speed,
        "std_speed_px_s": std_speed,
        "total_distance_px": total_dist_px,
        "stroke_peak_freq_hz": peak_freq
    }


def main():
    trials = get_all_trials()
    print(f"Found total {len(trials)} trials across conditions: {CONDITIONS}")
    
    max_workers = 6
    print(f"Starting parallel tracking across {max_workers} worker processes...")
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_trial, info): info for info in trials}
        for future in as_completed(futures):
            info = futures[future]
            try:
                res = future.result()
                if res:
                    results.append(res)
            except Exception as e:
                print(f"[ERROR] {info['trial']} raised exception: {e}")

    df_summary = pd.DataFrame(results)
    df_summary.sort_values(by=['condition', 'trial'], inplace=True)
    out_csv = os.path.join(BASE_DIR, "tracking_all_trials_summary.csv")
    df_summary.to_csv(out_csv, index=False)
    print(f"\nAll tracking completed! Summary saved to: {out_csv}")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
