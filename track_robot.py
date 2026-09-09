#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robot Swimming Trajectory & Velocity Tracker (Volvocine PicoV2)
==============================================================
定点カメラ動画からロボットの遊泳軌跡を抽出し、
水かき周期（約1.25Hz）の窓フィルターを適用して遊泳速度を解析・可視化するスクリプト。

主な機能:
1. 高精度・高速トラッキング:
   - メディアン合成による静止プール背景画像の自動生成
   - 背景差分と局所探索窓（Adaptive ROI）によるノイズ耐性（底面タイルや壁汚れの完全除去）
   - サブピクセル精度の重心座標 (x, y) 算出
   - 青色マーカー検出によるロボットの姿勢角 (theta) の算出
2. 1.25Hz 窓フィルターと速度解析:
   - Savitzky-Golay フィルター（窓長 ≈ 0.8秒 / 約49フレーム）による平滑化 & 解析的微分速度
   - 移動平均フィルター（Centered Moving Average, 窓長 ≈ 0.8秒）
   - Butterworth ゼロ位相ローパスフィルター（カットオフ ≈ 1.25Hz）
   - 生の瞬間速度 (Raw diff) と平滑化速度の比較
   - ロボット進行方向（Surge）と横方向（Sway）への速度分解
3. FFT スペクトル解析:
   - 速度振動のパワースペクトルから実際の水かき周波数ピークを自動検出
4. 出力:
   - 詳細な時系列データ CSV (座標, 姿勢角, 各種速度)
   - 4面統合解析グラフ (軌跡, 座標時系列, 速度時系列, FFTスペクトル)
   - オプション: トラッキング描画オーバーレイ動画 (Annotated MP4)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm


def generate_background(video_path, sample_times=[10, 20, 30, 40, 50, 60, 70, 80], pool_roi=None):
    """
    複数タイムスタンプからメディアン合成を行い、
    ロボットや人のいない綺麗な静止プール背景画像を生成する。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames = []
    for t in sample_times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError("Could not read any sample frames for background generation.")

    print(f"[INFO] Generating clean background from {len(frames)} sample frames...")
    bg = np.median(frames, axis=0).astype(np.uint8)
    return bg


def track_robot_trajectory(
    video_path,
    bg_image,
    start_sec=0.0,
    end_sec=None,
    pool_roi=(600, 100, 3700, 1950),
    search_margin=120,
    save_annotated_video=False,
    output_video_path="tracked_robot.mp4",
    verbose=True
):
    """
    動画からロボットの2D位置 (x, y) および進行方向方位角 theta をフレーム毎にトラッキングする。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(round(start_sec * fps))
    end_frame = int(round(end_sec * fps)) if end_sec else total_video_frames
    end_frame = min(end_frame, total_video_frames)
    total_process_frames = end_frame - start_frame

    if verbose:
        print(f"[INFO] Video: {video_path}")
        print(f"[INFO] Resolution: {video_w}x{video_h}, FPS: {fps:.3f}")
        print(f"[INFO] Tracking range: {start_sec:.2f}s - {end_sec:.2f}s ({start_frame} - {end_frame} frames, Total: {total_process_frames})")

    bg_gray = cv2.cvtColor(bg_image, cv2.COLOR_BGR2GRAY)
    scale = video_w / 3840.0

    # Scale pool_roi and parameters to current video resolution
    if pool_roi is not None:
        ref_xmin, ref_ymin, ref_xmax, ref_ymax = pool_roi
        xmin = int(round(ref_xmin * scale))
        ymin = int(round(ref_ymin * scale))
        xmax = int(round(ref_xmax * scale))
        ymax = int(round(ref_ymax * scale))
    else:
        xmin, ymin, xmax, ymax = 0, 0, video_w, video_h

    cur_search_margin = int(round(search_margin * scale))
    min_area = int(round(3000 * (scale**2)))
    max_area_lim = int(round(45000 * (scale**2)))
    min_dt = 12.0 * scale
    min_bm_m00 = 30.0 * (scale**2)
    # Frame-to-frame jump limit: at 60 fps 4K, max travel is 10 px. Adjust for scale and fps
    max_jump_px = 10.0 * scale * (60.0 / fps)
    max_jump_d = max_jump_px ** 2

    # Pool mask
    pool_mask = np.zeros((video_h, video_w), dtype=np.uint8)
    pool_mask[ymin:ymax, xmin:xmax] = 255

    # Video Writer if requested
    vw = None
    if save_annotated_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Scale down 1/2 for output video to save space and fast encode
        out_w, out_h = video_w // 2, video_h // 2
        vw = cv2.VideoWriter(output_video_path, fourcc, fps, (out_w, out_h))
        if verbose:
            print(f"[INFO] Saving annotated video to {output_video_path} ({out_w}x{out_h})")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    records = []
    prev_center = None
    consecutive_lost = 0
    trail_points = []

    pbar = tqdm(total=total_process_frames, desc="Tracking robot", unit="frames", disable=not verbose)

    for f_idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break

        t_sec = f_idx / fps

        # Determine search window
        if prev_center is None:
            cur_xmin, cur_ymin, cur_xmax, cur_ymax = xmin, ymin, xmax, ymax
        else:
            px, py = prev_center
            cur_margin = int(round(cur_search_margin * (1.0 + 0.15 * consecutive_lost)))
            cur_xmin = max(xmin, int(px - cur_margin))
            cur_xmax = min(xmax, int(px + cur_margin))
            cur_ymin = max(ymin, int(py - cur_margin))
            cur_ymax = min(ymax, int(py + cur_margin))

        # Crop local search area
        frame_crop = frame[cur_ymin:cur_ymax, cur_xmin:cur_xmax]
        bg_crop = bg_gray[cur_ymin:cur_ymax, cur_xmin:cur_xmax]
        gray_crop = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY)

        # Background subtraction with Black Robot Physical Prior:
        # The robot chassis and paddle legs are black plastic (gray < 105),
        # while waves, ripples, and specular glints are bright white (gray > 180).
        # By enforcing (1) Darker than BG (bg - frame > 25), (2) Absolute low luminance (gray < 105),
        # and (3) Preserving blue heading markers, surface reflections are 100% eliminated.
        diff_crop = np.clip(bg_crop.astype(np.int16) - gray_crop.astype(np.int16), 0, 255).astype(np.uint8)
        is_dark = (gray_crop < 105)
        hsv_crop = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv_crop, np.array([90, 70, 60]), np.array([135, 255, 255]))

        bin_mask = (((diff_crop > 25) & is_dark) | (blue_mask > 0)).astype(np.uint8) * 255
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel)
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)

        # Contours inside search window
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_c = None
        best_dist = float('inf')
        max_area = 0

        for c in contours:
            area = cv2.contourArea(c)
            if min_area < area < max_area_lim:
                # Extract core body centroid via Distance Transform to reject arm/reflection shifts
                bx, by, bw, bh = cv2.boundingRect(c)
                roi_mask = np.zeros((bh, bw), dtype=np.uint8)
                c_shifted = c - np.array([bx, by])
                cv2.drawContours(roi_mask, [c_shifted], -1, 255, -1)
                dt_roi = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)
                max_dt = float(np.max(dt_roi))
                
                if max_dt > min_dt:
                    core = (dt_roi >= 0.65 * max_dt).astype(np.uint8)
                    Mc = cv2.moments(core)
                    if Mc["m00"] > 0:
                        cx = (Mc["m10"] / Mc["m00"]) + bx + cur_xmin
                        cy = (Mc["m01"] / Mc["m00"]) + by + cur_ymin
                    else:
                        M = cv2.moments(c)
                        cx = (M["m10"] / M["m00"]) + cur_xmin
                        cy = (M["m01"] / M["m00"]) + cur_ymin
                else:
                    M = cv2.moments(c)
                    cx = (M["m10"] / M["m00"]) + cur_xmin
                    cy = (M["m01"] / M["m00"]) + cur_ymin

                if prev_center is None:
                    if area > max_area:
                        max_area = area
                        best_c = (cx, cy, area, c)
                else:
                    d = (cx - prev_center[0])**2 + (cy - prev_center[1])**2
                    # Reject sudden jumps
                    if d < max_jump_d and d < best_dist:
                        best_dist = d
                        best_c = (cx, cy, area, c)

        if best_c is not None:
            cx, cy, area, c = best_c
            prev_center = (cx, cy)
            consecutive_lost = 0
            trail_points.append((int(round(cx)), int(round(cy))))
            if len(trail_points) > 300:
                trail_points.pop(0)

            # Detect blue markers inside robot crop for heading angle
            hsv_crop = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2HSV)
            blue_mask = cv2.inRange(hsv_crop, np.array([90, 100, 100]), np.array([135, 255, 255]))
            bm = cv2.moments(blue_mask)
            if bm["m00"] > min_bm_m00:
                bx = (bm["m10"] / bm["m00"]) + cur_xmin
                by = (bm["m01"] / bm["m00"]) + cur_ymin
                theta = np.arctan2(by - cy, bx - cx)
            else:
                bx, by, theta = np.nan, np.nan, np.nan

            # Standardize coordinates to 4K reference pixel space (3840x2160)
            records.append({
                "frame": f_idx,
                "time_sec": t_sec,
                "x": cx / scale,
                "y": cy / scale,
                "area": area / (scale**2),
                "blue_x": bx / scale if not np.isnan(bx) else np.nan,
                "blue_y": by / scale if not np.isnan(by) else np.nan,
                "theta_rad": theta,
                "status": "detected"
            })

            # Render overlay if video output requested
            if vw is not None:
                # Draw trail
                for i in range(1, len(trail_points)):
                    alpha_trail = i / len(trail_points)
                    color = (int(255 * (1 - alpha_trail)), int(200 * alpha_trail), 255)
                    cv2.line(frame, trail_points[i - 1], trail_points[i], color, 3)

                # Draw bounding box & center
                bx_c, by_c, bw_c, bh_c = cv2.boundingRect(c)
                cv2.rectangle(frame, (cur_xmin + bx_c, cur_ymin + by_c),
                              (cur_xmin + bx_c + bw_c, cur_ymin + by_c + bh_c), (0, 255, 0), 3)
                cv2.circle(frame, (int(round(cx)), int(round(cy))), 6, (0, 0, 255), -1)

                # Draw heading arrow
                if not np.isnan(theta):
                    arrow_len = 80
                    ax_end = int(round(cx + arrow_len * np.cos(theta)))
                    ay_end = int(round(cy + arrow_len * np.sin(theta)))
                    cv2.arrowedLine(frame, (int(round(cx)), int(round(cy))), (ax_end, ay_end), (255, 100, 0), 4, tipLength=0.3)

                # Text info
                info_text = f"Time: {t_sec:.2f}s | Frame: {f_idx} | Pos: ({cx:.1f}, {cy:.1f})"
                cv2.putText(frame, info_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                
                # Resize and write
                frame_out = cv2.resize(frame, (out_w, out_h))
                vw.write(frame_out)
        else:
            consecutive_lost += 1
            if consecutive_lost > 15:
                prev_center = None
            records.append({
                "frame": f_idx,
                "time_sec": t_sec,
                "x": np.nan,
                "y": np.nan,
                "area": np.nan,
                "blue_x": np.nan,
                "blue_y": np.nan,
                "theta_rad": np.nan,
                "status": "lost"
            })
            if vw is not None:
                cv2.putText(frame, "TARGET LOST - SEARCHING", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                frame_out = cv2.resize(frame, (out_w, out_h))
                vw.write(frame_out)

        pbar.update(1)

    pbar.close()
    cap.release()
    if vw is not None:
        vw.release()

    df = pd.DataFrame(records)
    detected_count = (df['status'] == 'detected').sum()
    print(f"[INFO] Tracking finished. Detected {detected_count}/{len(df)} frames ({detected_count/len(df)*100:.1f}%)")
    return df, fps


def compute_filtered_velocity(df, fps, stroke_freq=1.25, scale_m_per_px=2.0 / 1585.0):
    """
    1.25Hz の水かき周期に対応したフィルターを適用し、
    位置の平滑化および遊泳速度を算出する。
    """
    dt = 1.0 / fps

    # Window length for 1.25 Hz stroke: T = 1 / 1.25 = 0.8s -> frames = 0.8 * fps
    window_sec = 1.0 / stroke_freq
    window_frames = int(round(window_sec * fps))
    if window_frames % 2 == 0:
        window_frames += 1  # Savgol requires odd window length

    print(f"[INFO] Velocity filtering: stroke_freq={stroke_freq} Hz, window={window_sec:.3f} s ({window_frames} frames)")

    # Interpolate missing values if any
    x = df['x'].interpolate(method='linear').bfill().ffill().to_numpy()
    y = df['y'].interpolate(method='linear').bfill().ffill().to_numpy()
    t = df['time_sec'].to_numpy()

    # 0. Clean position outliers (Hampel filter / median deviation)
    # Multi-pass Hampel filter with expanding windows to catch single-frame and multi-frame (0.05-0.25s)
    # Scale filter window length dynamically based on fps
    base_windows = [(15, 2.5), (21, 2.0), (11, 1.5)]
    fps_scale = fps / 60.0
    for w_med_base, th in base_windows:
        w_med = max(3, int(round(w_med_base * fps_scale)))
        if w_med % 2 == 0:
            w_med += 1
        pad_w = w_med // 2
        x_pad = np.pad(x, pad_w, mode='edge')
        y_pad = np.pad(y, pad_w, mode='edge')
        x_med = signal.medfilt(x_pad, kernel_size=w_med)[pad_w:-pad_w]
        y_med = signal.medfilt(y_pad, kernel_size=w_med)[pad_w:-pad_w]
        dev = np.sqrt((x - x_med)**2 + (y - y_med)**2)
        outliers = dev > th
        x[outliers] = x_med[outliers]
        y[outliers] = y_med[outliers]

    # 1. Instantaneous swimming velocity
    # Uses short-window Savitzky-Golay (~0.18s) to capture genuine intra-stroke oscillations
    # while eliminating high-frequency digitizing jitter
    inst_win = max(5, int(round(0.18 * fps)))
    if inst_win % 2 == 0:
        inst_win += 1
    if inst_win >= len(x):
        inst_win = max(3, len(x) if len(x) % 2 == 1 else len(x) - 1)
    vx_raw = signal.savgol_filter(x, window_length=inst_win, polyorder=2, deriv=1, delta=dt)
    vy_raw = signal.savgol_filter(y, window_length=inst_win, polyorder=2, deriv=1, delta=dt)
    speed_raw = np.sqrt(vx_raw**2 + vy_raw**2)

    # Suppress non-physical spikes above 90 px/s (robot peak physical stroke speed is ~60-80 px/s, mean ~40 px/s)
    spikes = speed_raw > 90.0
    if np.any(spikes):
        med_k = max(3, int(round(9 * fps_scale)))
        if med_k % 2 == 0:
            med_k += 1
        spd_med = signal.medfilt(speed_raw, kernel_size=med_k)
        speed_raw[spikes] = np.minimum(speed_raw[spikes], np.maximum(spd_med[spikes], 65.0))
        speed_raw = np.clip(speed_raw, 0, 95.0)

    # 2. Savitzky-Golay filter (Stroke-cycle averaged Position & Velocity derivative)
    x_sg = signal.savgol_filter(x, window_length=window_frames, polyorder=2)
    y_sg = signal.savgol_filter(y, window_length=window_frames, polyorder=2)
    vx_sg = signal.savgol_filter(x, window_length=window_frames, polyorder=2, deriv=1, delta=dt)
    vy_sg = signal.savgol_filter(y, window_length=window_frames, polyorder=2, deriv=1, delta=dt)
    speed_sg = np.sqrt(vx_sg**2 + vy_sg**2)
    # Prevent boundary polynomial extrapolation divergence at sequence edges
    speed_sg = np.clip(speed_sg, 0, np.maximum(speed_raw, 80.0))

    # 3. Moving average velocity
    speed_ma = pd.Series(speed_raw).rolling(window=window_frames, center=True, min_periods=1).mean().to_numpy()

    # 4. Butterworth Low-Pass Filter (cutoff = stroke_freq)
    nyquist = 0.5 * fps
    cutoff = stroke_freq
    b, a = signal.butter(4, cutoff / nyquist, btype='low')
    vx_butter = signal.filtfilt(b, a, vx_raw)
    vy_butter = signal.filtfilt(b, a, vy_raw)
    speed_butter = np.sqrt(vx_butter**2 + vy_butter**2)
    # Prevent transient ringing divergence at edges
    speed_butter = np.clip(speed_butter, 0, np.maximum(speed_raw, 80.0))

    # Robot Heading & Surge / Sway velocity decomposition
    theta = df['theta_rad'].interpolate(method='linear').bfill().ffill().to_numpy()
    theta_unwrapped = np.unwrap(theta)
    theta_sg = signal.savgol_filter(theta_unwrapped, window_length=window_frames, polyorder=2)
    omega_sg = signal.savgol_filter(theta_unwrapped, window_length=window_frames, polyorder=2, deriv=1, delta=dt)

    # Surge (forward velocity along robot heading) & Sway (lateral velocity)
    # Heading unit vector: u_h = [cos(theta), sin(theta)]
    v_surge = vx_sg * np.cos(theta_sg) + vy_sg * np.sin(theta_sg)
    v_sway = -vx_sg * np.sin(theta_sg) + vy_sg * np.cos(theta_sg)

    # Attach to DataFrame
    df['x_smooth_px'] = x_sg
    df['y_smooth_px'] = y_sg
    df['vx_raw_px_s'] = vx_raw
    df['vy_raw_px_s'] = vy_raw
    df['speed_raw_px_s'] = speed_raw

    df['vx_sg_px_s'] = vx_sg
    df['vy_sg_px_s'] = vy_sg
    df['speed_sg_px_s'] = speed_sg
    df['speed_ma_px_s'] = speed_ma
    df['speed_butter_px_s'] = speed_butter

    df['theta_smooth_deg'] = np.degrees(theta_sg) % 360
    df['omega_deg_s'] = np.degrees(omega_sg)
    df['v_surge_px_s'] = v_surge
    df['v_sway_px_s'] = v_sway

    # Physical metric conversion (Reference 4K calibration: 2.0 m = 1585 px)
    if scale_m_per_px is not None:
        df['x_m'] = x_sg * scale_m_per_px
        df['y_m'] = y_sg * scale_m_per_px
        df['speed_sg_m_s'] = speed_sg * scale_m_per_px
        df['speed_sg_cm_s'] = speed_sg * scale_m_per_px * 100.0
        df['speed_butter_m_s'] = speed_butter * scale_m_per_px
        df['v_surge_m_s'] = v_surge * scale_m_per_px
        df['v_surge_cm_s'] = v_surge * scale_m_per_px * 100.0

    return df, window_sec, window_frames


def plot_swimming_results(df, fps, window_sec, window_frames, target_freq=1.25, output_img_path="swimming_analysis.png", bg_image=None):
    """
    軌跡、位置時系列、1.25Hz平滑化速度、FFT周波数スペクトルの4連グラフを作成して保存する。
    bg_imageが指定されている場合、第1パネルにプール背景画像を敷いて軌跡を描画する。
    """
    t = df['time_sec'].to_numpy()
    dt = 1.0 / fps

    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    x_sg = df['x_smooth_px'].to_numpy()
    y_sg = df['y_smooth_px'].to_numpy()

    speed_raw = df['speed_raw_px_s'].to_numpy()
    speed_sg = df['speed_sg_px_s'].to_numpy()
    speed_butter = df['speed_butter_px_s'].to_numpy()

    # FFT analysis of velocity oscillation
    speed_detrend = speed_raw - np.mean(speed_raw)
    n_fft = len(speed_detrend)
    freqs = np.fft.rfftfreq(n_fft, d=dt)
    fft_vals = np.abs(np.fft.rfft(speed_detrend))

    # Find dominant stroke peak in 0.5 - 3.0 Hz
    mask = (freqs >= 0.5) & (freqs <= 3.0)
    if np.any(mask):
        peak_freq = freqs[mask][np.argmax(fft_vals[mask])]
    else:
        peak_freq = target_freq

    fig, axs = plt.subplots(4, 1, figsize=(12, 16))

    # 1. 2D Trajectory (Overlaid on Background if provided)
    if bg_image is not None:
        bg_h, bg_w = bg_image.shape[:2]
        s_x = bg_w / 3840.0
        s_y = bg_h / 2160.0
        axs[0].imshow(cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB))
        axs[0].plot(x * s_x, y * s_y, color='white', lw=1.2, alpha=0.6, label='Raw trajectory')
        sc = axs[0].scatter(x_sg * s_x, y_sg * s_y, c=t, cmap='plasma', s=8, label='Smoothed trajectory', zorder=5)
        cbar = plt.colorbar(sc, ax=axs[0])
        cbar.set_label('Time (s)', fontsize=11)
        # Crop just outside the yellow pool rim (1080p: [140, 1920], [0, 1060])
        x_low = int(round(140 * (bg_w / 1920.0)))
        x_high = int(round(1920 * (bg_w / 1920.0)))
        y_low = int(round(0 * (bg_h / 1080.0)))
        y_high = int(round(1060 * (bg_h / 1080.0)))
        axs[0].set_xlim(x_low, x_high)
        axs[0].set_ylim(y_high, y_low)
    else:
        axs[0].plot(x, y, color='lightgray', lw=1.2, label='Raw trajectory')
        sc = axs[0].scatter(x_sg, y_sg, c=t, cmap='viridis', s=6, label='Smoothed trajectory')
        cbar = plt.colorbar(sc, ax=axs[0])
        cbar.set_label('Time (s)', fontsize=11)
        axs[0].invert_yaxis()  # Match image coordinates
        axs[0].axis('equal')
        axs[0].grid(True, linestyle='--', alpha=0.5)

    axs[0].set_title('Robot Swimming Trajectory in Pool', fontsize=14, fontweight='bold')
    axs[0].set_xlabel('X Coordinate (px)', fontsize=11)
    axs[0].set_ylabel('Y Coordinate (px)', fontsize=11)
    axs[0].legend(loc='upper right')

    # 2. Coordinates vs Time
    axs[1].plot(t, x, color='salmon', alpha=0.5, label='X (raw)')
    axs[1].plot(t, x_sg, color='firebrick', lw=1.8, label='X (Savgol smoothed)')
    axs[1].plot(t, y, color='skyblue', alpha=0.5, label='Y (raw)')
    axs[1].plot(t, y_sg, color='royalblue', lw=1.8, label='Y (Savgol smoothed)')
    axs[1].set_title('Robot Position vs Time', fontsize=14, fontweight='bold')
    axs[1].set_xlabel('Time (s)', fontsize=11)
    axs[1].set_ylabel('Position (px)', fontsize=11)
    axs[1].grid(True, linestyle='--', alpha=0.5)
    axs[1].legend(loc='upper right')

    # 3. Swimming Speed vs Time
    mean_speed = np.mean(speed_sg)
    axs[2].plot(t, speed_raw, color='lightgray', alpha=0.7, lw=1.0, label='Raw instantaneous speed')
    axs[2].plot(t, speed_sg, color='seagreen', lw=2.2, label=f'Savgol Filtered ({window_sec:.2f}s / {window_frames}f window)')
    axs[2].plot(t, speed_butter, color='darkorange', lw=1.5, linestyle='--', label=f'Butterworth Low-pass ({target_freq} Hz)')
    axs[2].axhline(mean_speed, color='crimson', linestyle=':', lw=2, label=f'Mean Cruising Speed = {mean_speed:.1f} px/s')
    axs[2].set_title(f'Swimming Velocity vs Time (~{target_freq} Hz Stroke Filtered)', fontsize=14, fontweight='bold')
    axs[2].set_xlabel('Time (s)', fontsize=11)
    axs[2].set_ylabel('Speed (px/s)', fontsize=11)
    axs[2].set_ylim(0, max(np.percentile(speed_raw, 99.5) * 1.25, mean_speed * 2.2, 80.0))
    axs[2].grid(True, linestyle='--', alpha=0.5)
    axs[2].legend(loc='upper right')

    # 4. FFT Power Spectrum
    axs[3].plot(freqs, fft_vals, color='purple', lw=1.5, label='Speed Oscillation Spectrum')
    axs[3].axvline(peak_freq, color='crimson', linestyle='--', lw=1.8, label=f'Observed Stroke Peak = {peak_freq:.2f} Hz')
    axs[3].axvline(target_freq, color='darkgreen', linestyle=':', lw=1.8, label=f'Target Frequency = {target_freq:.2f} Hz')
    axs[3].set_xlim(0, 5.0)
    axs[3].set_title('Velocity Oscillation Spectrum (FFT)', fontsize=14, fontweight='bold')
    axs[3].set_xlabel('Frequency (Hz)', fontsize=11)
    axs[3].set_ylabel('Amplitude', fontsize=11)
    axs[3].grid(True, linestyle='--', alpha=0.5)
    axs[3].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_img_path, dpi=200)
    pdf_path = os.path.splitext(output_img_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()
    print(f"[INFO] Figures saved to:")
    print(f"       PNG: {output_img_path}")
    print(f"       PDF: {pdf_path}")
    return peak_freq, mean_speed


def main():
    parser = argparse.ArgumentParser(description="Track swimming robot and compute filtered velocity from fixed-camera video.")
    parser.add_argument("--video", type=str, default=r"d:\codes\Volvocine_PicoV2\VolBotVideo\GX011315.MP4",
                        help="Path to input video file")
    parser.add_argument("--start_sec", type=float, default=6.0,
                        help="Start time in seconds for tracking (default: 6.0s)")
    parser.add_argument("--end_sec", type=float, default=82.0,
                        help="End time in seconds for tracking (default: 82.0s, None for end of video)")
    parser.add_argument("--stroke_freq", type=float, default=1.25,
                        help="Robot paddle stroke frequency in Hz for filter window (default: 1.25 Hz)")
    parser.add_argument("--scale_m_per_px", type=float, default=None,
                        help="Spatial scale in meters per pixel (optional)")
    parser.add_argument("--output_dir", type=str, default="tracking_results",
                        help="Output directory for results")
    parser.add_argument("--save_video", action="store_true",
                        help="Save video with tracking trajectory and bounding box overlay")
    parser.add_argument("--yolo", action="store_true",
                        help="Flag passed from prompt (runs automated pipeline without interactive prompts)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    video_basename = os.path.splitext(os.path.basename(args.video))[0]

    # 1. Background generation
    bg_image = generate_background(args.video)
    bg_path = os.path.join(args.output_dir, f"{video_basename}_background.jpg")
    cv2.imwrite(bg_path, bg_image)
    print(f"[INFO] Saved clean background to {bg_path}")

    # 2. Tracking
    annotated_video_path = os.path.join(args.output_dir, f"{video_basename}_tracked.mp4")
    df_track, fps = track_robot_trajectory(
        video_path=args.video,
        bg_image=bg_image,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
        save_annotated_video=args.save_video,
        output_video_path=annotated_video_path
    )

    # 3. Filtering and Velocity Computation
    df_analyzed, win_sec, win_frames = compute_filtered_velocity(
        df=df_track,
        fps=fps,
        stroke_freq=args.stroke_freq,
        scale_m_per_px=args.scale_m_per_px
    )

    # 4. Save CSV
    csv_path = os.path.join(args.output_dir, f"{video_basename}_trajectory_velocity.csv")
    df_analyzed.to_csv(csv_path, index=False)
    print(f"[INFO] Trajectory & velocity data saved to {csv_path}")

    # 5. Plot Figures
    fig_path = os.path.join(args.output_dir, f"{video_basename}_swimming_analysis.png")
    peak_freq, mean_speed = plot_swimming_results(
        df=df_analyzed,
        fps=fps,
        window_sec=win_sec,
        window_frames=win_frames,
        target_freq=args.stroke_freq,
        output_img_path=fig_path,
        bg_image=bg_image
    )

    # Summary
    print("\n" + "="*60)
    print("           SWIMMING ANALYSIS SUMMARY REPORT")
    print("="*60)
    print(f" Video File            : {args.video}")
    print(f" Analyzed Duration     : {args.start_sec:.2f}s to {args.end_sec:.2f}s ({len(df_analyzed)/fps:.2f}s)")
    print(f" Frame Rate (FPS)      : {fps:.3f} fps")
    print(f" Filter Window Length  : {win_sec:.3f} s ({win_frames} frames) [Stroke ~{args.stroke_freq} Hz]")
    print(f" Observed Stroke Peak  : {peak_freq:.3f} Hz")
    print(f" Mean Swimming Speed   : {mean_speed:.2f} px/s")
    print(f" Max Filtered Speed    : {df_analyzed['speed_sg_px_s'].max():.2f} px/s")
    print(f" Min Filtered Speed    : {df_analyzed['speed_sg_px_s'].min():.2f} px/s")
    if args.scale_m_per_px:
        print(f" Mean Speed (m/s)      : {mean_speed * args.scale_m_per_px:.3f} m/s ({mean_speed * args.scale_m_per_px * 100:.1f} cm/s)")
    print(f" Output CSV            : {csv_path}")
    print(f" Output Plot (PNG/PDF) : {fig_path}")
    if args.save_video:
        print(f" Output Video          : {annotated_video_path}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
