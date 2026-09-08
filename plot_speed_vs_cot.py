#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Volvocine PicoV2: Swimming Speed vs. Cost of Transport & Mode Correlation Analysis
===================================================================================
- 各条件ディレクトリ（baseline, sinz, msinz, moptz）配下の各試行フォルダから
  `phase_energy_analysis.csv` と `*_trajectory_velocity.csv` を読み込む。
- 水中定常遊泳区間における各指標（速度、電力、CoT、モード収束値 |Z1|〜|Z4|、受信入力振幅 |Gi|）を集約。
- 条件ごと（baseline: 青, sinz: 赤, msinz: 緑, moptz: 紫）に色・マーカーを分けて可視化。
- 散布図および相関解析（Pearson r, Spearman rho, p値）:
  1. 遊泳速度 vs. Cost of Transport (CoT)
  2. 第1モード収束値 (|Z1|) vs. 遊泳速度 & CoT
  3. 受信入力振幅の総和 (sum |Gi|) vs. 遊泳速度 & CoT
  4. 全モード（Mode 1〜4）秩序パラメータ vs. 遊泳速度 & CoT
- 全試行サマリー CSV (`all_trials_summary.csv`) の出力。
"""

import os
import glob
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0

# Condition definitions, colors, and markers
COND_CONFIG = {
    'baseline': {
        'label': 'Baseline',
        'color': '#118AB2',  # Oceanic Blue
        'marker': 'o',
        'size': 85,
        'zorder': 3
    },
    'sinz': {
        'label': 'sinz',
        'color': '#E63946',  # Crimson Red
        'marker': 'D',
        'size': 110,
        'zorder': 4
    },
    'msinz': {
        'label': 'msinz',
        'color': '#06D6A0',  # Emerald Green
        'marker': 's',
        'size': 95,
        'zorder': 4
    },
    'moptz': {
        'label': 'moptz',
        'color': '#9B5DE5',  # Vivid Purple
        'marker': '^',
        'size': 100,
        'zorder': 5
    }
}


def load_and_aggregate_all_trials(base_dir=r"D:\codes\Volvocine_PicoV2\VolBotVideo"):
    """
    全条件ディレクトリから試行データを読み込み、水中遊泳定常状態指標を集約する。
    """
    # Load SVD weights and singular values for weighted input amplitude calculation
    svd_dir = r"D:\codes\Volvocine_PicoV2\EstimateL\SStick\low_rank_analysis\M10\global_joint_cp_rank1_profile_free_network_svd"
    weights_table = pd.read_csv(os.path.join(svd_dir, "agent_svd_contributions.csv"))
    summary_table = pd.read_csv(os.path.join(svd_dir, "network_svd_modes_summary.csv"))
    agent_map = {9: 8, 8: 10, 11: 7, 12: 9}
    real_agents = [8, 9, 11, 12]
    N = len(real_agents)
    L = 4
    sigma = summary_table["singular_value_sigma"].values[:L]
    U = np.zeros((N, L))
    V = np.zeros((N, L))
    for i, real_ag in enumerate(real_agents):
        mode_ag = agent_map[real_ag]
        row_w = weights_table[weights_table["agent_id"] == mode_ag].iloc[0]
        for l in range(L):
            U[i, l] = row_w[f"receiver_u_mode{l+1}"]
            V[i, l] = row_w[f"sender_v_mode{l+1}"]

    rows = []
    
    for cond in ['baseline', 'sinz', 'msinz', 'moptz']:
        cond_dir = os.path.join(base_dir, cond)
        if not os.path.exists(cond_dir):
            continue
        trial_subdirs = sorted([d for d in os.listdir(cond_dir) 
                                if os.path.isdir(os.path.join(cond_dir, d)) and d.startswith("GX")])
        
        for t_name in trial_subdirs:
            if "11316" in t_name:
                continue  # Exclude aborted trial 11316
                
            td = os.path.join(cond_dir, t_name)
            track_csvs = glob.glob(os.path.join(td, "*_trajectory_velocity.csv"))
            phase_csvs = glob.glob(os.path.join(td, "phase_energy_analysis.csv"))
            
            if not track_csvs or not phase_csvs:
                continue
                
            try:
                df_track = pd.read_csv(track_csvs[0])
                df_phase = pd.read_csv(phase_csvs[0])
            except Exception as e:
                print(f"[WARN] Error reading CSVs in {td}: {e}")
                continue
            
            if len(df_track) < 60 or len(df_phase) < 60:
                continue
                
            # 1. Swimming interval from video tracking
            swim_t_start = df_track['time_sec'].iloc[0]
            swim_t_end = df_track['time_sec'].iloc[-1]
            swim_duration = swim_t_end - swim_t_start
            
            # Mean swimming speed from tracking
            mean_speed = float(df_track['speed_sg_px_s'].mean())
            max_speed = float(df_track['speed_sg_px_s'].max())
            std_speed = float(df_track['speed_sg_px_s'].std())
            
            dx = np.diff(df_track['x_smooth_px'])
            dy = np.diff(df_track['y_smooth_px'])
            total_dist_px = float(np.sum(np.sqrt(dx**2 + dy**2)))
            
            # Peak stroke frequency from FFT
            dt_v = np.median(np.diff(df_track['time_sec']))
            speed_detrend = df_track['speed_raw_px_s'] - df_track['speed_raw_px_s'].mean()
            freqs = np.fft.rfftfreq(len(speed_detrend), d=dt_v)
            fft_vals = np.abs(np.fft.rfft(speed_detrend))
            f_mask = (freqs >= 0.5) & (freqs <= 3.0)
            peak_freq = float(freqs[f_mask][np.argmax(fft_vals[f_mask])]) if np.any(f_mask) else 1.25
            
            # 2. Steady-state Mode convergence evaluation
            # Cut off the final 3.0s before water removal to avoid hand pickup disturbances
            t_phase = df_phase['time_sec']
            t_eval_end = max(swim_t_start + 10.0, swim_t_end - 3.0)
            t_eval_start = max(swim_t_start + 8.0, t_eval_end - 15.0)
            
            mask_steady = (t_phase >= t_eval_start) & (t_phase <= t_eval_end)
            if not np.any(mask_steady):
                t_max_p = t_phase.max()
                mask_steady = (t_phase >= t_max_p * 0.6) & (t_phase <= max(t_max_p * 0.6 + 5.0, t_max_p - 4.0))
                
            mode_metrics = {}
            for m in range(1, 5):
                col_name = f'order_param_mode{m}_Z{m}'
                if col_name in df_phase.columns:
                    z_series = df_phase.loc[mask_steady, col_name]
                    mode_metrics[f'mode{m}_converged_mean'] = float(z_series.mean())
                    mode_metrics[f'mode{m}_steady_std'] = float(z_series.std())
                else:
                    mode_metrics[f'mode{m}_converged_mean'] = np.nan
                    mode_metrics[f'mode{m}_steady_std'] = np.nan
            
            # 3. Weighted input amplitude received by oscillators: G_i(t) = sum_l sigma_l * u_{il} * Z_l(t)
            phases = np.zeros((len(df_phase), N))
            valid_phases = True
            for i, ag in enumerate(real_agents):
                col_ag = f'abs_phase_unwrapped_rad_agent_{ag}'
                if col_ag in df_phase.columns:
                    phases[:, i] = df_phase[col_ag].values
                else:
                    valid_phases = False
                    break
                    
            if valid_phases:
                phasors = np.exp(1j * phases)
                Z_comp = np.dot(phasors, V)
                G_comp = np.dot(Z_comp * sigma, U.T)
                G_abs_matrix = np.abs(G_comp)
                G_steady = G_abs_matrix[mask_steady]
                g_means = np.mean(G_steady, axis=0) if len(G_steady) > 0 else np.zeros(N)
                g_sum_all = float(np.sum(g_means))
                g_mean_all = float(np.mean(g_means))
            else:
                g_means = np.zeros(N)
                g_sum_all = np.nan
                g_mean_all = np.nan
            
            # Mean electrical power during pure swimming
            mask_swim = (t_phase >= swim_t_start) & (t_phase <= swim_t_end - 2.0)
            if np.any(mask_swim) and 'total_power_w' in df_phase.columns:
                mean_power = float(df_phase.loc[mask_swim, 'total_power_w'].mean())
            elif 'total_power_w' in df_phase.columns:
                mean_power = float(df_phase['total_power_w'].mean())
            else:
                mean_power = np.nan
                
            # Cost of Transport (CoT): CoT = Power / Speed
            cot_J_px = mean_power / max(mean_speed, 1e-6) if np.isfinite(mean_power) else np.nan
            cot_mJ_px = cot_J_px * 1000.0 if np.isfinite(cot_J_px) else np.nan
            
            # Cumulative energy during valid swimming
            if np.any(mask_swim) and 'total_power_w' in df_phase.columns:
                total_energy_J = float(np.trapz(df_phase.loc[mask_swim, 'total_power_w'], t_phase[mask_swim]))
            else:
                total_energy_J = np.nan
                
            cfg = COND_CONFIG.get(cond, {'color': '#333333', 'marker': 'o', 'label': cond})
            
            row_data = {
                "trial": t_name,
                "condition": cond,
                "plot_color": cfg['color'],
                "plot_marker": cfg['marker'],
                "swimming_duration_sec": swim_duration,
                "mean_speed_px_s": mean_speed,
                "max_speed_px_s": max_speed,
                "std_speed_px_s": std_speed,
                "total_distance_px": total_dist_px,
                "stroke_peak_freq_hz": peak_freq,
                "mean_power_W": mean_power,
                "mean_power_mW": mean_power * 1000.0 if np.isfinite(mean_power) else np.nan,
                "total_energy_J": total_energy_J,
                "cot_J_px": cot_J_px,
                "cot_mJ_px": cot_mJ_px,
                "mode1_converged_mean": mode_metrics['mode1_converged_mean'],
                "mode1_steady_std": mode_metrics['mode1_steady_std'],
                "mode2_converged_mean": mode_metrics['mode2_converged_mean'],
                "mode2_steady_std": mode_metrics['mode2_steady_std'],
                "mode3_converged_mean": mode_metrics['mode3_converged_mean'],
                "mode3_steady_std": mode_metrics['mode3_steady_std'],
                "mode4_converged_mean": mode_metrics['mode4_converged_mean'],
                "mode4_steady_std": mode_metrics['mode4_steady_std'],
                "G_sum_all": g_sum_all,
                "G_mean_all": g_mean_all,
                "G_ag8": float(g_means[0]),
                "G_ag9": float(g_means[1]),
                "G_ag11": float(g_means[2]),
                "G_ag12": float(g_means[3]),
                "eval_window_start_sec": t_eval_start,
                "eval_window_end_sec": t_eval_end
            }
            rows.append(row_data)

    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values(by=['condition', 'trial'], inplace=True)
    return df


def compute_correlations(x, y, label=""):
    """Compute Pearson and Spearman correlation statistics."""
    mask = np.isfinite(x) & np.isfinite(y)
    x_c = np.array(x)[mask]
    y_c = np.array(y)[mask]
    if len(x_c) < 3:
        return None
    r, p_pearson = stats.pearsonr(x_c, y_c)
    rho, p_spearman = stats.spearmanr(x_c, y_c)
    return {
        "label": label,
        "n": len(x_c),
        "pearson_r": r,
        "pearson_p": p_pearson,
        "spearman_rho": rho,
        "spearman_p": p_spearman
    }


def set_smart_axis_limits(ax, x_vals, y_vals, x_pad=0.08, y_pad=0.08):
    """Set axis limits with intelligent padding to avoid clipping."""
    x_c = np.array(x_vals)[np.isfinite(x_vals)]
    y_c = np.array(y_vals)[np.isfinite(y_vals)]
    if len(x_c) > 0:
        x_min, x_max = x_c.min(), x_c.max()
        x_span = max(x_max - x_min, 1e-3)
        ax.set_xlim(x_min - x_span * x_pad, x_max + x_span * x_pad)
    if len(y_c) > 0:
        y_min, y_max = y_c.min(), y_c.max()
        y_span = max(y_max - y_min, 1e-3)
        ax.set_ylim(y_min - y_span * y_pad, y_max + y_span * y_pad)


def plot_speed_vs_cot(df, output_dir):
    """Figure 1: Swimming Speed vs. Cost of Transport (Color-coded by Condition)"""
    fig, ax = plt.subplots(figsize=(10, 7.5))

    for cond, cfg in COND_CONFIG.items():
        group = df[df['condition'] == cond]
        if group.empty:
            continue
        ax.scatter(group['mean_speed_px_s'], group['cot_mJ_px'],
                   c=cfg['color'], marker=cfg['marker'], s=cfg['size'], label=f"{cfg['label']} (N={len(group)})",
                   edgecolors='black', linewidth=1.1, alpha=0.9, zorder=cfg['zorder'])

    # Annotate trial IDs
    for _, row in df.iterrows():
        t_id = row['trial'].replace('GX01', '')
        cond = row['condition']
        cfg = COND_CONFIG.get(cond, {'color': '#333333'})
        ax.annotate(t_id, (row['mean_speed_px_s'], row['cot_mJ_px']),
                    xytext=(4, 4), textcoords='offset points',
                    fontsize=8.0, fontweight='medium', color=cfg['color'],
                    path_effects=[pe.withStroke(linewidth=2.2, foreground='white')])

    set_smart_axis_limits(ax, df['mean_speed_px_s'], df['cot_mJ_px'], x_pad=0.08, y_pad=0.10)
    
    # Correlation for all trials
    corr_all = compute_correlations(df['mean_speed_px_s'], df['cot_mJ_px'], "All")
    if corr_all:
        stat_str = (f"All Trials (N={corr_all['n']}):\n"
                    f"  Pearson r = {corr_all['pearson_r']:+.3f} (p = {corr_all['pearson_p']:.2e})\n"
                    f"  Spearman ρ = {corr_all['spearman_rho']:+.3f} (p = {corr_all['spearman_p']:.2e})")
        ax.text(0.96, 0.96, stat_str, transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                fontsize=9.5, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.92, edgecolor='#aaaaaa'))

    ax.set_title("Distribution of Swimming Speed vs. Cost of Transport (CoT) by Condition", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Mean Swimming Speed $v$ [px/s]", fontsize=12)
    ax.set_ylabel("Cost of Transport (CoT = $P/v$) [mJ/px]", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.92, fontsize=10.5)

    plt.tight_layout()
    png_path = os.path.join(output_dir, "scatter_speed_vs_cot.png")
    pdf_path = os.path.join(output_dir, "scatter_speed_vs_cot.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"[SAVED] {png_path}")
    print(f"[SAVED] {pdf_path}")


def plot_mode1_correlations(df, output_dir):
    """Figure 2: Mode 1 Convergence Value vs Speed and CoT (Color-coded by Condition)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.5))

    # --- Panel 1: Mode 1 vs. Speed ---
    for cond, cfg in COND_CONFIG.items():
        group = df[df['condition'] == cond]
        if group.empty:
            continue
        ax1.scatter(group['mode1_converged_mean'], group['mean_speed_px_s'],
                    c=cfg['color'], marker=cfg['marker'], s=cfg['size'], label=f"{cfg['label']} (N={len(group)})",
                    edgecolors='black', linewidth=1.1, alpha=0.9, zorder=cfg['zorder'])

    for _, row in df.iterrows():
        t_id = row['trial'].replace('GX01', '')
        cond = row['condition']
        cfg = COND_CONFIG.get(cond, {'color': '#333333'})
        ax1.annotate(t_id, (row['mode1_converged_mean'], row['mean_speed_px_s']),
                     xytext=(4, 4), textcoords='offset points',
                     fontsize=8.0, color=cfg['color'],
                     path_effects=[pe.withStroke(linewidth=2.2, foreground='white')])

    # Baseline regression line
    base_df = df[df['condition'] == 'baseline']
    if len(base_df) >= 3:
        p_base = np.polyfit(base_df['mode1_converged_mean'], base_df['mean_speed_px_s'], deg=1)
        x_line = np.linspace(df['mode1_converged_mean'].min()*0.9, df['mode1_converged_mean'].max()*1.1, 100)
        ax1.plot(x_line, np.polyval(p_base, x_line), color='#118AB2', linestyle='--', lw=1.6, alpha=0.7, label='Baseline Fit')

    set_smart_axis_limits(ax1, df['mode1_converged_mean'], df['mean_speed_px_s'])
    
    corr_base = compute_correlations(base_df['mode1_converged_mean'], base_df['mean_speed_px_s'], "Baseline")
    corr_all = compute_correlations(df['mode1_converged_mean'], df['mean_speed_px_s'], "All")
    stat_lines = []
    if corr_base:
        stat_lines.append(f"Baseline (N={corr_base['n']}): r = {corr_base['pearson_r']:+.3f} (p = {corr_base['pearson_p']:.2e})")
    if corr_all:
        stat_lines.append(f"All Trials (N={corr_all['n']}): r = {corr_all['pearson_r']:+.3f} (p = {corr_all['pearson_p']:.2e})")
    if stat_lines:
        ax1.text(0.96, 0.96, "\n".join(stat_lines), transform=ax1.transAxes, verticalalignment='top', horizontalalignment='right',
                 fontsize=9.5, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.92, edgecolor='#aaaaaa'))

    ax1.set_title("First Mode Convergence Value vs. Swimming Speed", fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel("First Mode Steady Convergence Value $|Z_1|$", fontsize=12)
    ax1.set_ylabel("Mean Swimming Speed $v$ [px/s]", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.92, fontsize=10)

    # --- Panel 2: Mode 1 vs. CoT ---
    for cond, cfg in COND_CONFIG.items():
        group = df[df['condition'] == cond]
        if group.empty:
            continue
        ax2.scatter(group['mode1_converged_mean'], group['cot_mJ_px'],
                    c=cfg['color'], marker=cfg['marker'], s=cfg['size'], label=f"{cfg['label']} (N={len(group)})",
                    edgecolors='black', linewidth=1.1, alpha=0.9, zorder=cfg['zorder'])

    for _, row in df.iterrows():
        t_id = row['trial'].replace('GX01', '')
        cond = row['condition']
        cfg = COND_CONFIG.get(cond, {'color': '#333333'})
        ax2.annotate(t_id, (row['mode1_converged_mean'], row['cot_mJ_px']),
                     xytext=(4, 4), textcoords='offset points',
                     fontsize=8.0, color=cfg['color'],
                     path_effects=[pe.withStroke(linewidth=2.2, foreground='white')])

    if len(base_df) >= 3:
        p_base2 = np.polyfit(base_df['mode1_converged_mean'], base_df['cot_mJ_px'], deg=1)
        x_line = np.linspace(df['mode1_converged_mean'].min()*0.9, df['mode1_converged_mean'].max()*1.1, 100)
        ax2.plot(x_line, np.polyval(p_base2, x_line), color='#118AB2', linestyle='--', lw=1.6, alpha=0.7, label='Baseline Fit')

    set_smart_axis_limits(ax2, df['mode1_converged_mean'], df['cot_mJ_px'])

    corr_base2 = compute_correlations(base_df['mode1_converged_mean'], base_df['cot_mJ_px'], "Baseline")
    corr_all2 = compute_correlations(df['mode1_converged_mean'], df['cot_mJ_px'], "All")
    stat_lines2 = []
    if corr_base2:
        stat_lines2.append(f"Baseline (N={corr_base2['n']}): r = {corr_base2['pearson_r']:+.3f} (p = {corr_base2['pearson_p']:.2e})")
    if corr_all2:
        stat_lines2.append(f"All Trials (N={corr_all2['n']}): r = {corr_all2['pearson_r']:+.3f} (p = {corr_all2['pearson_p']:.2e})")
    if stat_lines2:
        ax2.text(0.96, 0.96, "\n".join(stat_lines2), transform=ax2.transAxes, verticalalignment='top', horizontalalignment='right',
                 fontsize=9.5, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.92, edgecolor='#aaaaaa'))

    ax2.set_title("First Mode Convergence Value vs. Cost of Transport", fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel("First Mode Steady Convergence Value $|Z_1|$", fontsize=12)
    ax2.set_ylabel("Cost of Transport (CoT = $P/v$) [mJ/px]", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.92, fontsize=10)

    plt.tight_layout()
    png_path = os.path.join(output_dir, "scatter_mode1_correlations.png")
    pdf_path = os.path.join(output_dir, "scatter_mode1_correlations.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"[SAVED] {png_path}")
    print(f"[SAVED] {pdf_path}")


def plot_weighted_input_correlations(df, output_dir):
    """Figure 3: Sum of Weighted Input Amplitudes (|G_i|) vs Speed and CoT (Color-coded)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.5))

    base_df = df[df['condition'] == 'baseline']

    # --- Panel 1: Sum of |Gi| vs. Speed ---
    for cond, cfg in COND_CONFIG.items():
        group = df[df['condition'] == cond]
        if group.empty:
            continue
        ax1.scatter(group['G_sum_all'], group['mean_speed_px_s'],
                    c=cfg['color'], marker=cfg['marker'], s=cfg['size'], label=f"{cfg['label']} (N={len(group)})",
                    edgecolors='black', linewidth=1.1, alpha=0.9, zorder=cfg['zorder'])

    for _, row in df.iterrows():
        t_id = row['trial'].replace('GX01', '')
        cond = row['condition']
        cfg = COND_CONFIG.get(cond, {'color': '#333333'})
        ax1.annotate(t_id, (row['G_sum_all'], row['mean_speed_px_s']),
                     xytext=(4, 4), textcoords='offset points',
                     fontsize=8.0, color=cfg['color'],
                     path_effects=[pe.withStroke(linewidth=2.2, foreground='white')])

    if len(base_df) >= 3:
        p_base = np.polyfit(base_df['G_sum_all'], base_df['mean_speed_px_s'], deg=1)
        x_line = np.linspace(df['G_sum_all'].min()*0.9, df['G_sum_all'].max()*1.1, 100)
        ax1.plot(x_line, np.polyval(p_base, x_line), color='#118AB2', linestyle='--', lw=1.6, alpha=0.7)

    set_smart_axis_limits(ax1, df['G_sum_all'], df['mean_speed_px_s'])
    
    corr1_base = compute_correlations(base_df['G_sum_all'], base_df['mean_speed_px_s'], "Baseline")
    corr1_all = compute_correlations(df['G_sum_all'], df['mean_speed_px_s'], "All")
    stat_lines = []
    if corr1_base:
        stat_lines.append(f"Baseline (N={corr1_base['n']}): r = {corr1_base['pearson_r']:+.3f} (p = {corr1_base['pearson_p']:.2e})")
    if corr1_all:
        stat_lines.append(f"All Trials (N={corr1_all['n']}): r = {corr1_all['pearson_r']:+.3f} (p = {corr1_all['pearson_p']:.2e})")
    if stat_lines:
        ax1.text(0.96, 0.96, "\n".join(stat_lines), transform=ax1.transAxes, verticalalignment='top', horizontalalignment='right',
                 fontsize=9.5, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.92, edgecolor='#aaaaaa'))

    ax1.set_title("Total Weighted Input Amplitude vs. Swimming Speed", fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel(r"Total Received Input Amplitude $\sum_{i} |G_i|$", fontsize=12)
    ax1.set_ylabel("Mean Swimming Speed $v$ [px/s]", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.92, fontsize=10)

    # --- Panel 2: Sum of |Gi| vs. CoT ---
    for cond, cfg in COND_CONFIG.items():
        group = df[df['condition'] == cond]
        if group.empty:
            continue
        ax2.scatter(group['G_sum_all'], group['cot_mJ_px'],
                    c=cfg['color'], marker=cfg['marker'], s=cfg['size'], label=f"{cfg['label']} (N={len(group)})",
                    edgecolors='black', linewidth=1.1, alpha=0.9, zorder=cfg['zorder'])

    for _, row in df.iterrows():
        t_id = row['trial'].replace('GX01', '')
        cond = row['condition']
        cfg = COND_CONFIG.get(cond, {'color': '#333333'})
        ax2.annotate(t_id, (row['G_sum_all'], row['cot_mJ_px']),
                     xytext=(4, 4), textcoords='offset points',
                     fontsize=8.0, color=cfg['color'],
                     path_effects=[pe.withStroke(linewidth=2.2, foreground='white')])

    if len(base_df) >= 3:
        p_base2 = np.polyfit(base_df['G_sum_all'], base_df['cot_mJ_px'], deg=1)
        x_line = np.linspace(df['G_sum_all'].min()*0.9, df['G_sum_all'].max()*1.1, 100)
        ax2.plot(x_line, np.polyval(p_base2, x_line), color='#118AB2', linestyle='--', lw=1.6, alpha=0.7)

    set_smart_axis_limits(ax2, df['G_sum_all'], df['cot_mJ_px'])

    corr2_base = compute_correlations(base_df['G_sum_all'], base_df['cot_mJ_px'], "Baseline")
    corr2_all = compute_correlations(df['G_sum_all'], df['cot_mJ_px'], "All")
    stat_lines2 = []
    if corr2_base:
        stat_lines2.append(f"Baseline (N={corr2_base['n']}): r = {corr2_base['pearson_r']:+.3f} (p = {corr2_base['pearson_p']:.2e})")
    if corr2_all:
        stat_lines2.append(f"All Trials (N={corr2_all['n']}): r = {corr2_all['pearson_r']:+.3f} (p = {corr2_all['pearson_p']:.2e})")
    if stat_lines2:
        ax2.text(0.96, 0.96, "\n".join(stat_lines2), transform=ax2.transAxes, verticalalignment='top', horizontalalignment='right',
                 fontsize=9.5, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.92, edgecolor='#aaaaaa'))

    ax2.set_title("Total Weighted Input Amplitude vs. Cost of Transport", fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel(r"Total Received Input Amplitude $\sum_{i} |G_i|$", fontsize=12)
    ax2.set_ylabel("Cost of Transport (CoT = $P/v$) [mJ/px]", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.92, fontsize=10)

    plt.tight_layout()
    png_path = os.path.join(output_dir, "scatter_weighted_input_correlations.png")
    pdf_path = os.path.join(output_dir, "scatter_weighted_input_correlations.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"[SAVED] {png_path}")
    print(f"[SAVED] {pdf_path}")


def plot_all_modes_correlations(df, output_dir):
    """Figure 4: Modes 1-4 Order Parameters vs. Swimming Speed & CoT (2x4 Grid, Color-coded)"""
    fig, axes = plt.subplots(2, 4, figsize=(26, 12))
    plt.subplots_adjust(hspace=0.28, wspace=0.24)

    base_df = df[df['condition'] == 'baseline']
    modes = [1, 2, 3, 4]

    for m in modes:
        col_m = f'mode{m}_converged_mean'
        
        # --- Row 0: Mode m vs. Speed ---
        ax_speed = axes[0, m - 1]
        for cond, cfg in COND_CONFIG.items():
            group = df[df['condition'] == cond]
            if group.empty:
                continue
            ax_speed.scatter(group[col_m], group['mean_speed_px_s'],
                             c=cfg['color'], marker=cfg['marker'], s=cfg['size'], label=f"{cfg['label']} (N={len(group)})",
                             edgecolors='black', linewidth=1.1, alpha=0.9, zorder=cfg['zorder'])
                             
        for _, row in df.iterrows():
            t_id = row['trial'].replace('GX01', '')
            cond = row['condition']
            cfg = COND_CONFIG.get(cond, {'color': '#333333'})
            ax_speed.annotate(t_id, (row[col_m], row['mean_speed_px_s']),
                              xytext=(4, 4), textcoords='offset points',
                              fontsize=7.5, color=cfg['color'],
                              path_effects=[pe.withStroke(linewidth=2.0, foreground='white')])
                              
        if len(base_df) >= 3:
            p_s = np.polyfit(base_df[col_m], base_df['mean_speed_px_s'], deg=1)
            x_s = np.linspace(df[col_m].min() * 0.9, df[col_m].max() * 1.1, 100)
            ax_speed.plot(x_s, np.polyval(p_s, x_s), color='#118AB2', linestyle='--', lw=1.5, alpha=0.7)

        set_smart_axis_limits(ax_speed, df[col_m], df['mean_speed_px_s'])
        
        corr_s_base = compute_correlations(base_df[col_m], base_df['mean_speed_px_s'], f"Mode{m}")
        corr_s_all = compute_correlations(df[col_m], df['mean_speed_px_s'], f"Mode{m}")
        stat_lines = []
        if corr_s_base:
            stat_lines.append(f"Baseline: r={corr_s_base['pearson_r']:+.3f}")
        if corr_s_all:
            stat_lines.append(f"All: r={corr_s_all['pearson_r']:+.3f}")
        if stat_lines:
            ax_speed.text(0.96, 0.96, "\n".join(stat_lines), transform=ax_speed.transAxes,
                          verticalalignment='top', horizontalalignment='right',
                          fontsize=8.5, bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.92, edgecolor='#aaaaaa'))

        ax_speed.set_title(f"Mode {m} ($|Z_{m}|$) vs. Speed", fontsize=12.5, fontweight='bold', pad=10)
        ax_speed.set_xlabel(f"Mode {m} Convergence Value $|Z_{m}|$", fontsize=11)
        ax_speed.set_ylabel("Mean Speed $v$ [px/s]", fontsize=11)
        ax_speed.grid(True, linestyle='--', alpha=0.5)
        if m == 1:
            ax_speed.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.92, fontsize=9.5)

        # --- Row 1: Mode m vs. CoT ---
        ax_cot = axes[1, m - 1]
        for cond, cfg in COND_CONFIG.items():
            group = df[df['condition'] == cond]
            if group.empty:
                continue
            ax_cot.scatter(group[col_m], group['cot_mJ_px'],
                           c=cfg['color'], marker=cfg['marker'], s=cfg['size'], label=f"{cfg['label']} (N={len(group)})",
                           edgecolors='black', linewidth=1.1, alpha=0.9, zorder=cfg['zorder'])

        for _, row in df.iterrows():
            t_id = row['trial'].replace('GX01', '')
            cond = row['condition']
            cfg = COND_CONFIG.get(cond, {'color': '#333333'})
            ax_cot.annotate(t_id, (row[col_m], row['cot_mJ_px']),
                            xytext=(4, 4), textcoords='offset points',
                            fontsize=7.5, color=cfg['color'],
                            path_effects=[pe.withStroke(linewidth=2.0, foreground='white')])

        if len(base_df) >= 3:
            p_c = np.polyfit(base_df[col_m], base_df['cot_mJ_px'], deg=1)
            x_c = np.linspace(df[col_m].min() * 0.9, df[col_m].max() * 1.1, 100)
            ax_cot.plot(x_c, np.polyval(p_c, x_c), color='#118AB2', linestyle='--', lw=1.5, alpha=0.7)

        set_smart_axis_limits(ax_cot, df[col_m], df['cot_mJ_px'])

        corr_c_base = compute_correlations(base_df[col_m], base_df['cot_mJ_px'], f"Mode{m}")
        corr_c_all = compute_correlations(df[col_m], df['cot_mJ_px'], f"Mode{m}")
        stat_lines_cot = []
        if corr_c_base:
            stat_lines_cot.append(f"Baseline: r={corr_c_base['pearson_r']:+.3f}")
        if corr_c_all:
            stat_lines_cot.append(f"All: r={corr_c_all['pearson_r']:+.3f}")
        if stat_lines_cot:
            ax_cot.text(0.96, 0.96, "\n".join(stat_lines_cot), transform=ax_cot.transAxes,
                        verticalalignment='top', horizontalalignment='right',
                        fontsize=8.5, bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.92, edgecolor='#aaaaaa'))

        ax_cot.set_title(f"Mode {m} ($|Z_{m}|$) vs. Cost of Transport", fontsize=12.5, fontweight='bold', pad=10)
        ax_cot.set_xlabel(f"Mode {m} Convergence Value $|Z_{m}|$", fontsize=11)
        ax_cot.set_ylabel("Cost of Transport [mJ/px]", fontsize=11)
        ax_cot.grid(True, linestyle='--', alpha=0.5)
        if m == 1:
            ax_cot.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.92, fontsize=9.5)

    plt.tight_layout()
    png_path = os.path.join(output_dir, "scatter_all_modes_correlations.png")
    pdf_path = os.path.join(output_dir, "scatter_all_modes_correlations.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"[SAVED] {png_path}")
    print(f"[SAVED] {pdf_path}")


def main():
    base_dir = r"D:\codes\Volvocine_PicoV2\VolBotVideo"
    print("\n[INFO] Loading all trial CSVs across condition directories...")
    df = load_and_aggregate_all_trials(base_dir)
    
    if df.empty:
        print("[WARN] No trials found to aggregate!")
        return

    # Save combined summary
    summary_path = os.path.join(base_dir, "all_trials_summary.csv")
    cols_to_save = [
        'trial', 'condition', 'mean_speed_px_s', 'mean_power_W', 'cot_mJ_px', 'cot_J_px',
        'mode1_converged_mean', 'mode1_steady_std',
        'mode2_converged_mean', 'mode2_steady_std',
        'mode3_converged_mean', 'mode3_steady_std',
        'mode4_converged_mean', 'mode4_steady_std',
        'G_sum_all', 'G_mean_all',
        'G_ag8', 'G_ag9', 'G_ag11', 'G_ag12', 'stroke_peak_freq_hz',
        'total_energy_J', 'total_distance_px', 'swimming_duration_sec',
        'eval_window_start_sec', 'eval_window_end_sec'
    ]
    avail_cols = [c for c in cols_to_save if c in df.columns]
    df[avail_cols].to_csv(summary_path, index=False)
    print(f"[INFO] Combined summary saved to: {summary_path} ({len(df)} trials)")

    # Generate plots
    print("\n[INFO] Generating Speed vs. CoT scatter plot...")
    plot_speed_vs_cot(df, base_dir)

    print("\n[INFO] Generating Mode 1 correlation plots...")
    plot_mode1_correlations(df, base_dir)

    print("\n[INFO] Generating Weighted Input Amplitude correlation plots...")
    plot_weighted_input_correlations(df, base_dir)

    print("\n[INFO] Generating All Modes (1-4) correlation plots (2x4 Grid)...")
    plot_all_modes_correlations(df, base_dir)

    # Print summary table by condition
    print("\n" + "="*145)
    print("                                            ALL TRIALS ANALYSIS SUMMARY TABLE")
    print("="*145)
    disp_cols = ['trial', 'condition', 'mean_speed_px_s', 'mean_power_W', 'cot_mJ_px', 'mode1_converged_mean', 'mode2_converged_mean', 'mode3_converged_mean', 'mode4_converged_mean', 'G_sum_all']
    disp_df = df[[c for c in disp_cols if c in df.columns]]
    print(disp_df.to_string(index=False))
    print("="*145 + "\n")


if __name__ == "__main__":
    main()
