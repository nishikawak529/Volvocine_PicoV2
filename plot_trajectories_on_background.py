#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robot Trajectory Overlay on Pool Background (Volvocine PicoV2)
=============================================================
This script composites the robot at the start, time midpoint, and end of the swimming trajectory
onto the clean static pool background image, and visualizes the trajectory with physical scale calibration.

Outputs:
1. Individual trial trajectory overlay: <trial_dir>/<trial>_trajectory_overlay.png / .pdf
2. 2x2 Multi-condition representative comparison: VolBotVideo/trajectories_representative_grid.png / .pdf
3. Single-frame multi-condition trajectory comparison: VolBotVideo/trajectories_all_conditions_overlay.png / .pdf
"""

import os
import cv2
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patheffects as pe

BASE_DIR = r"D:\codes\Volvocine_PicoV2\VolBotVideo"
POOL_HEIGHT_PX_4K = 1585.0
POOL_HEIGHT_M = 2.0
SCALE_M_PER_PX_4K = POOL_HEIGHT_M / POOL_HEIGHT_PX_4K  # 1.26183e-3 m/px
SCALE_CM_PER_PX_4K = SCALE_M_PER_PX_4K * 100.0

COND_CONFIG = {
    'baseline': {'label': r'Baseline ($\kappa = 0$)', 'color': '#118AB2', 'marker': 'o'},
    'sinz':     {'label': r'$\kappa = 5$',   'color': '#E07A5F', 'marker': 's'},
    'm5sinz':   {'label': r'$\kappa = -5$',  'color': '#06D6A0', 'marker': '^'},
    'm10sinz':  {'label': r'$\kappa = -10$', 'color': '#E71D36', 'marker': 'D'},
}

# Unified Pool Cropping Boundaries (Just outside the yellow rim in 1080p coordinates)
# 1080p: x in [140, 1920], y in [0, 1060] (4K: x in [280, 3840], y in [0, 2120])
UNIFIED_CROP_1080 = {
    'x_min': 140,
    'x_max': 1920,
    'y_min': 0,
    'y_max': 1060
}


def get_unified_pool_bounds(w_bg, h_bg):
    """
    Returns unified bounding coordinates trimmed just outside the yellow pool rim,
    scaled to the resolution of the background image.
    """
    scale_w = w_bg / 1920.0
    scale_h = h_bg / 1080.0
    x_low = int(round(UNIFIED_CROP_1080['x_min'] * scale_w))
    x_high = int(round(UNIFIED_CROP_1080['x_max'] * scale_w))
    y_low = int(round(UNIFIED_CROP_1080['y_min'] * scale_h))
    y_high = int(round(UNIFIED_CROP_1080['y_max'] * scale_h))
    return x_low, x_high, y_low, y_high


def get_or_create_background(trial_dir, video_path):
    t_name = os.path.basename(trial_dir)
    bg_path = os.path.join(trial_dir, f"{t_name}_background.jpg")
    if os.path.exists(bg_path):
        bg = cv2.imread(bg_path)
        if bg is not None and bg.size > 0:
            return bg, bg_path
            
    # Check if any other trial in the same directory has a background
    parent = os.path.dirname(trial_dir)
    sibling_bgs = glob.glob(os.path.join(parent, "*", "*_background.jpg"))
    if sibling_bgs:
        bg = cv2.imread(sibling_bgs[0])
        if bg is not None and bg.size > 0:
            cv2.imwrite(bg_path, bg)
            return bg, bg_path
            
    # Generate background using sample frames from video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_dur = total_frames / fps if fps > 0 else 0
    sample_fractions = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65]
    sample_times = [video_dur * frac for frac in sample_fractions]
    
    frames = []
    for t in sample_times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    
    if not frames:
        raise RuntimeError(f"Could not extract background frames from {video_path}")
        
    bg = np.median(frames, axis=0).astype(np.uint8)
    cv2.imwrite(bg_path, bg)
    return bg, bg_path


def composite_robot_snapshots(bg, video_path, df, vid_w, vid_h):
    """
    Extracts robot at start, mid, and end timestamps and composites them onto the background.
    """
    cap = cv2.VideoCapture(video_path)
    scale_x = vid_w / 3840.0
    scale_y = vid_h / 2160.0
    
    indices = [0, len(df) // 2, len(df) - 1]
    labels = ["Start", "Mid", "End"]
    marker_colors = ["#06D6A0", "#FFD166", "#EF476F"]  # Cyan, Yellow, Pink
    
    composite = bg.copy()
    robot_snapshots = []
    
    for idx, lbl, col in zip(indices, labels, marker_colors):
        row = df.iloc[idx]
        f = int(row['frame'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if not ret:
            continue
            
        cx_vid = int(round(row['x'] * scale_x))
        cy_vid = int(round(row['y'] * scale_y))
        
        diff = cv2.absdiff(frame, bg)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        r = int(round(68 * (vid_w / 1920.0)))
        y1, y2 = max(0, cy_vid - r), min(vid_h, cy_vid + r)
        x1, x2 = max(0, cx_vid - r), min(vid_w, cx_vid + r)
        
        frame_crop = frame[y1:y2, x1:x2]
        bg_crop = bg[y1:y2, x1:x2]
        gray_crop = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY)
        bg_gray_crop = cv2.cvtColor(bg_crop, cv2.COLOR_BGR2GRAY)
        
        # 1. Dark difference: robot is darker than background (rejects white waves/glints)
        dark_diff = np.clip(bg_gray_crop.astype(np.int16) - gray_crop.astype(np.int16), 0, 255).astype(np.uint8)
        
        # 2. Absolute darkness constraint: robot body and paddle arms are black plastic (I < 105)
        is_dark = (gray_crop < 105)
        
        # 3. Blue marker preservation (for front heading markers)
        hsv_crop = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv_crop, np.array([90, 70, 60]), np.array([135, 255, 255]))
        
        # Combined mask: purely dark object or blue marker (100% immune to white water ripples/reflections)
        robot_mask = (((dark_diff > 20) & is_dark) | (blue_mask > 0)).astype(np.uint8) * 255
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_clean = cv2.morphologyEx(robot_mask, cv2.MORPH_CLOSE, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)
        mask_soft = cv2.GaussianBlur(mask_clean.astype(float) / 255.0, (7, 7), 2.0)[:, :, None]
        
        bg_target_crop = composite[y1:y2, x1:x2]
        blended = (frame_crop.astype(float) * mask_soft + bg_target_crop.astype(float) * (1.0 - mask_soft)).astype(np.uint8)
        composite[y1:y2, x1:x2] = blended
        
        robot_snapshots.append({
            'label': lbl,
            'time_rel': row['time_sec'] - df['time_sec'].iloc[0],
            'time_abs': row['time_sec'],
            'cx': cx_vid,
            'cy': cy_vid,
            'color': col
        })
        
    cap.release()
    return composite, robot_snapshots


def render_single_trial_trajectory(trial_dir, output_png=None, title_prefix="", cot_val=None):
    """
    Renders trajectory overlaid on the background image for a single trial.
    """
    t_name = os.path.basename(trial_dir)
    csv_path = os.path.join(trial_dir, f"{t_name}_trajectory_velocity.csv")
    vid_path = os.path.join(trial_dir, f"{t_name}.MP4")
    
    if not os.path.exists(csv_path) or not os.path.exists(vid_path):
        return None
        
    df = pd.read_csv(csv_path)
    if len(df) < 50:
        return None
        
    bg, bg_path = get_or_create_background(trial_dir, vid_path)
    h_bg, w_bg = bg.shape[:2]
    
    cap = cv2.VideoCapture(vid_path)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    s_bg_x = w_bg / 3840.0
    s_bg_y = h_bg / 2160.0
    
    composite, robot_snapshots = composite_robot_snapshots(bg, vid_path, df, vid_w, vid_h)
    
    # Trajectory points in background image space
    x_tr = (df['x_smooth_px'] * s_bg_x).to_numpy()
    y_tr = (df['y_smooth_px'] * s_bg_y).to_numpy()
    t_rel = (df['time_sec'] - df['time_sec'].iloc[0]).to_numpy()
    
    v_mean_cm_s = df['speed_sg_px_s'].mean() * SCALE_CM_PER_PX_4K
    dur_s = t_rel[-1]
    dx = np.diff(df['x_smooth_px'])
    dy = np.diff(df['y_smooth_px'])
    dist_m = float(np.sum(np.sqrt(dx**2 + dy**2))) * SCALE_M_PER_PX_4K
    
    fig, ax = plt.subplots(figsize=(15, 9), dpi=200)
    ax.imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
    
    # Draw trajectory line with glow
    points = np.array([x_tr, y_tr]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    lc_bg = LineCollection(segments, color='white', linewidth=5.5, alpha=0.9, zorder=4)
    ax.add_collection(lc_bg)
    
    norm = plt.Normalize(t_rel.min(), t_rel.max())
    lc = LineCollection(segments, cmap='plasma', norm=norm, linewidth=3.4, alpha=0.95, zorder=5)
    lc.set_array(t_rel)
    line = ax.add_collection(lc)
    
    cbar = fig.colorbar(line, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Elapsed Swimming Time [s]", fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=11)
    
    # Annotate robot points
    for r in robot_snapshots:
        cx, cy = r['cx'], r['cy']
        lbl = r['label']
        t_val = r['time_rel']
        col = r['color']
        
        ax.scatter(cx, cy, s=150, facecolor=col, edgecolor='white', linewidth=2.2, zorder=10)
        offset_y = -65 if cy > 180 else 65
        ax.annotate(
            f"{lbl} ({t_val:.1f}s)",
            xy=(cx, cy),
            xytext=(cx, cy + offset_y),
            ha='center', va='center',
            fontsize=11, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.35', facecolor=col, alpha=0.95, edgecolor='white', linewidth=1.4),
            arrowprops=dict(arrowstyle='->', lw=1.8, color='white'),
            zorder=12,
            path_effects=[pe.withStroke(linewidth=2.5, foreground='black')]
        )
        
    # Crop to unified pool bounds (just outside the yellow rim)
    x_low, x_high, y_low, y_high = get_unified_pool_bounds(w_bg, h_bg)
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(y_high, y_low)
    
    # Physical scale bar (50 cm) placed on bottom-left yellow rim
    scale_1080_px_per_m = (POOL_HEIGHT_PX_4K * (h_bg / 2160.0)) / POOL_HEIGHT_M
    bar_50cm_px = 0.5 * scale_1080_px_per_m
    bar_x0 = x_low + 50
    bar_y0 = y_high - 45
    ax.plot([bar_x0, bar_x0 + bar_50cm_px], [bar_y0, bar_y0], color='white', lw=5.0, solid_capstyle='butt', zorder=15)
    ax.plot([bar_x0, bar_x0 + bar_50cm_px], [bar_y0, bar_y0], color='black', lw=3.0, solid_capstyle='butt', zorder=16)
    ax.text(bar_x0 + bar_50cm_px / 2, bar_y0 - 12, "50 cm", ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='white', zorder=17,
            path_effects=[pe.withStroke(linewidth=2.5, foreground='black')])
            
    stat_line = f"Speed: {v_mean_cm_s:.2f} cm/s  |  Distance: {dist_m:.2f} m  |  Duration: {dur_s:.1f} s"
    if cot_val is not None:
        stat_line += f"  |  CoT: {cot_val:.2f} J/m"
    ax.set_title(f"{title_prefix}{t_name} Swimming Trajectory on Pool Background\n{stat_line}",
                 fontsize=13.5, fontweight='bold', pad=12)
    ax.axis('off')
    
    plt.tight_layout()
    if output_png is None:
        output_png = os.path.join(trial_dir, f"{t_name}_trajectory_overlay.png")
    pdf_path = os.path.splitext(output_png)[0] + ".pdf"
    plt.savefig(output_png, dpi=200, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {output_png}")
    print(f"[SAVED] {pdf_path}")
    return output_png


def generate_representative_grid_figure(output_dir):
    """
    Figure: 2x2 Grid comparing Representative Swimming Trajectories on Pool Background
    for Baseline, sinz, m5sinz, and m10sinz.
    """
    summary_path = os.path.join(output_dir, "all_trials_summary.csv")
    if not os.path.exists(summary_path):
        print(f"[WARN] Summary file not found: {summary_path}")
        return
        
    df_summary = pd.read_csv(summary_path)
    conds = ['baseline', 'sinz', 'm5sinz', 'm10sinz']
    rep_trials = []
    
    for c in conds:
        sub = df_summary[df_summary['condition'] == c]
        if sub.empty:
            continue
        m_spd = sub['mean_speed_cm_s'].mean()
        best_idx = (sub['mean_speed_cm_s'] - m_spd).abs().idxmin()
        row = sub.loc[best_idx]
        t_name = row['trial']
        t_dir = os.path.join(output_dir, c, t_name)
        rep_trials.append({
            'condition': c,
            'trial': t_name,
            'trial_dir': t_dir,
            'mean_speed_cm_s': row['mean_speed_cm_s'],
            'cot_J_m': row['cot_J_m'] if 'cot_J_m' in row else None
        })
        
    fig, axes = plt.subplots(2, 2, figsize=(24, 18), dpi=200)
    plt.subplots_adjust(hspace=0.15, wspace=0.08)
    
    for ax, item in zip(axes.flat, rep_trials):
        c = item['condition']
        t_name = item['trial']
        t_dir = item['trial_dir']
        cfg = COND_CONFIG[c]
        
        csv_path = os.path.join(t_dir, f"{t_name}_trajectory_velocity.csv")
        vid_path = os.path.join(t_dir, f"{t_name}.MP4")
        
        if not os.path.exists(csv_path) or not os.path.exists(vid_path):
            continue
            
        df = pd.read_csv(csv_path)
        bg, _ = get_or_create_background(t_dir, vid_path)
        h_bg, w_bg = bg.shape[:2]
        
        cap = cv2.VideoCapture(vid_path)
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        s_bg_x = w_bg / 3840.0
        s_bg_y = h_bg / 2160.0
        
        composite, robot_snapshots = composite_robot_snapshots(bg, vid_path, df, vid_w, vid_h)
        
        x_tr = (df['x_smooth_px'] * s_bg_x).to_numpy()
        y_tr = (df['y_smooth_px'] * s_bg_y).to_numpy()
        t_rel = (df['time_sec'] - df['time_sec'].iloc[0]).to_numpy()
        
        v_mean_cm_s = df['speed_sg_px_s'].mean() * SCALE_CM_PER_PX_4K
        dur_s = t_rel[-1]
        dist_m = float(np.sum(np.sqrt(np.diff(df['x_smooth_px'])**2 + np.diff(df['y_smooth_px'])**2))) * SCALE_M_PER_PX_4K
        cot_str = f", CoT: {item['cot_J_m']:.2f} J/m" if item['cot_J_m'] is not None else ""
        
        ax.imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
        
        # Trajectory line
        points = np.array([x_tr, y_tr]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        lc_bg = LineCollection(segments, color='white', linewidth=5.5, alpha=0.9, zorder=4)
        ax.add_collection(lc_bg)
        
        norm = plt.Normalize(t_rel.min(), t_rel.max())
        lc = LineCollection(segments, cmap='plasma', norm=norm, linewidth=3.4, alpha=0.95, zorder=5)
        lc.set_array(t_rel)
        line = ax.add_collection(lc)
        
        cbar = fig.colorbar(line, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Elapsed Time [s]", fontsize=11, fontweight='bold')
        
        # Robot snapshots
        for r in robot_snapshots:
            cx, cy = r['cx'], r['cy']
            lbl = r['label']
            t_val = r['time_rel']
            col = r['color']
            ax.scatter(cx, cy, s=150, facecolor=col, edgecolor='white', linewidth=2.2, zorder=10)
            offset_y = -65 if cy > 180 else 65
            ax.annotate(
                f"{lbl} ({t_val:.1f}s)",
                xy=(cx, cy),
                xytext=(cx, cy + offset_y),
                ha='center', va='center',
                fontsize=10.5, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.35', facecolor=col, alpha=0.95, edgecolor='white', linewidth=1.4),
                arrowprops=dict(arrowstyle='->', lw=1.8, color='white'),
                zorder=12,
                path_effects=[pe.withStroke(linewidth=2.5, foreground='black')]
            )
            
        # Crop to unified pool bounds (just outside the yellow rim)
        x_low, x_high, y_low, y_high = get_unified_pool_bounds(w_bg, h_bg)
        ax.set_xlim(x_low, x_high)
        ax.set_ylim(y_high, y_low)
        
        # Scale bar (50 cm) placed on bottom-left yellow rim
        scale_1080_px_per_m = (POOL_HEIGHT_PX_4K * (h_bg / 2160.0)) / POOL_HEIGHT_M
        bar_50cm_px = 0.5 * scale_1080_px_per_m
        bar_x0 = x_low + 50
        bar_y0 = y_high - 45
        ax.plot([bar_x0, bar_x0 + bar_50cm_px], [bar_y0, bar_y0], color='white', lw=5.0, solid_capstyle='butt', zorder=15)
        ax.plot([bar_x0, bar_x0 + bar_50cm_px], [bar_y0, bar_y0], color='black', lw=3.0, solid_capstyle='butt', zorder=16)
        ax.text(bar_x0 + bar_50cm_px / 2, bar_y0 - 12, "50 cm", ha='center', va='bottom',
                fontsize=11, fontweight='bold', color='white', zorder=17,
                path_effects=[pe.withStroke(linewidth=2.5, foreground='black')])
                
        panel_title = f"{cfg['label']} ({t_name})\nSpeed: {v_mean_cm_s:.2f} cm/s, Dist: {dist_m:.2f} m{cot_str}"
        ax.set_title(panel_title, fontsize=13.5, fontweight='bold', pad=10, color=cfg['color'])
        ax.axis('off')
        
    plt.suptitle("Representative Swimming Trajectories on Pool Background across Conditions\n(Robot Overlaid at Start, Temporal Midpoint, and End)",
                 fontsize=17, fontweight='bold', y=0.98)
    
    png_path = os.path.join(output_dir, "trajectories_representative_grid.png")
    pdf_path = os.path.join(output_dir, "trajectories_representative_grid.pdf")
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {png_path}")
    print(f"[SAVED] {pdf_path}")


def generate_all_conditions_overlay_figure(output_dir):
    """
    Figure: All 4 Conditions Superimposed on One Shared Pool Background Image.
    Each trajectory is color-coded by condition, with robot snapshots at Start, Mid, and End.
    """
    summary_path = os.path.join(output_dir, "all_trials_summary.csv")
    if not os.path.exists(summary_path):
        return
        
    df_summary = pd.read_csv(summary_path)
    conds = ['baseline', 'sinz', 'm5sinz', 'm10sinz']
    
    rep_trials = []
    for c in conds:
        sub = df_summary[df_summary['condition'] == c]
        if sub.empty:
            continue
        m_spd = sub['mean_speed_cm_s'].mean()
        best_idx = (sub['mean_speed_cm_s'] - m_spd).abs().idxmin()
        row = sub.loc[best_idx]
        t_name = row['trial']
        t_dir = os.path.join(output_dir, c, t_name)
        rep_trials.append({
            'condition': c,
            'trial': t_name,
            'trial_dir': t_dir,
            'mean_speed_cm_s': row['mean_speed_cm_s'],
            'cot_J_m': row['cot_J_m'] if 'cot_J_m' in row else None
        })
        
    # Use clean background from first available trial
    first_dir = rep_trials[0]['trial_dir']
    first_vid = os.path.join(first_dir, f"{rep_trials[0]['trial']}.MP4")
    shared_bg, _ = get_or_create_background(first_dir, first_vid)
    h_bg, w_bg = shared_bg.shape[:2]
    
    s_bg_x = w_bg / 3840.0
    s_bg_y = h_bg / 2160.0
    
    fig, ax = plt.subplots(figsize=(18, 11), dpi=200)
    ax.imshow(cv2.cvtColor(shared_bg, cv2.COLOR_BGR2RGB))
    
    all_x = []
    all_y = []
    
    for item in rep_trials:
        c = item['condition']
        t_name = item['trial']
        t_dir = item['trial_dir']
        cfg = COND_CONFIG[c]
        
        csv_path = os.path.join(t_dir, f"{t_name}_trajectory_velocity.csv")
        vid_path = os.path.join(t_dir, f"{t_name}.MP4")
        if not os.path.exists(csv_path) or not os.path.exists(vid_path):
            continue
            
        df = pd.read_csv(csv_path)
        x_tr = (df['x_smooth_px'] * s_bg_x).to_numpy()
        y_tr = (df['y_smooth_px'] * s_bg_y).to_numpy()
        all_x.extend(x_tr)
        all_y.extend(y_tr)
        
        # Draw trajectory
        ax.plot(x_tr, y_tr, color='white', lw=5.0, alpha=0.85, zorder=4)
        cot_str = f" ({item['mean_speed_cm_s']:.2f} cm/s, {item['cot_J_m']:.1f} J/m)" if item['cot_J_m'] is not None else ""
        ax.plot(x_tr, y_tr, color=cfg['color'], lw=3.2, label=f"{cfg['label']}{cot_str}", zorder=5)
        
        # Mark Start, Mid, End
        idx_mid = len(df) // 2
        ax.scatter(x_tr[0], y_tr[0], s=120, facecolor=cfg['color'], edgecolor='white', linewidth=1.8, zorder=8)
        ax.scatter(x_tr[idx_mid], y_tr[idx_mid], s=80, facecolor='white', edgecolor=cfg['color'], linewidth=2.0, zorder=8)
        ax.scatter(x_tr[-1], y_tr[-1], s=140, marker='s', facecolor=cfg['color'], edgecolor='white', linewidth=1.8, zorder=8)
        
    # Crop to unified pool bounds (just outside the yellow rim)
    x_low, x_high, y_low, y_high = get_unified_pool_bounds(w_bg, h_bg)
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(y_high, y_low)
    
    # Scale bar (50 cm) placed on bottom-left yellow rim
    scale_1080_px_per_m = (POOL_HEIGHT_PX_4K * (h_bg / 2160.0)) / POOL_HEIGHT_M
    bar_50cm_px = 0.5 * scale_1080_px_per_m
    bar_x0 = x_low + 50
    bar_y0 = y_high - 45
    ax.plot([bar_x0, bar_x0 + bar_50cm_px], [bar_y0, bar_y0], color='white', lw=5.0, solid_capstyle='butt', zorder=15)
    ax.plot([bar_x0, bar_x0 + bar_50cm_px], [bar_y0, bar_y0], color='black', lw=3.0, solid_capstyle='butt', zorder=16)
    ax.text(bar_x0 + bar_50cm_px / 2, bar_y0 - 12, "50 cm", ha='center', va='bottom',
            fontsize=11.5, fontweight='bold', color='white', zorder=17,
            path_effects=[pe.withStroke(linewidth=2.5, foreground='black')])
            
    # Legend (placed in upper right to keep scale bar completely unobstructed)
    legend = ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.92, fontsize=12.5)
    legend.set_zorder(20)
    
    ax.set_title("Comparison of Representative Swimming Trajectories across Conditions\nSuperimposed on Pool Background",
                 fontsize=15, fontweight='bold', pad=14)
    ax.axis('off')
    
    plt.tight_layout()
    png_path = os.path.join(output_dir, "trajectories_all_conditions_overlay.png")
    pdf_path = os.path.join(output_dir, "trajectories_all_conditions_overlay.pdf")
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {png_path}")
    print(f"[SAVED] {pdf_path}")


def main():
    print("\n" + "="*80)
    print("  GENERATING POOL BACKGROUND TRAJECTORY OVERLAYS (Volvocine PicoV2)")
    print("="*80)
    
    # 1. Generate 2x2 grid comparison of representative trials
    print("\n[INFO] Generating 2x2 representative trajectories grid on pool background...")
    generate_representative_grid_figure(BASE_DIR)
    
    # 2. Generate superimposed overlay on single background
    print("\n[INFO] Generating all-conditions superimposed trajectory overlay...")
    generate_all_conditions_overlay_figure(BASE_DIR)
    
    # 3. Generate individual overlays for representative trials of each condition
    summary_path = os.path.join(BASE_DIR, "all_trials_summary.csv")
    if os.path.exists(summary_path):
        df_sum = pd.read_csv(summary_path)
        for cond in ['baseline', 'sinz', 'm5sinz', 'm10sinz']:
            sub = df_sum[df_sum['condition'] == cond]
            if sub.empty:
                continue
            m_spd = sub['mean_speed_cm_s'].mean()
            best_idx = (sub['mean_speed_cm_s'] - m_spd).abs().idxmin()
            row = sub.loc[best_idx]
            t_dir = os.path.join(BASE_DIR, cond, row['trial'])
            print(f"\n[INFO] Generating individual overlay for {cond} representative trial {row['trial']}...")
            render_single_trial_trajectory(
                trial_dir=t_dir,
                title_prefix=f"[{COND_CONFIG[cond]['label']}] ",
                cot_val=row['cot_J_m'] if 'cot_J_m' in row else None
            )
            
    print("\n[SUCCESS] All background trajectory overlay figures successfully generated!")


if __name__ == "__main__":
    main()
