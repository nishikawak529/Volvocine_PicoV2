#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Volvocine PicoV2: Swimming Speed vs. Cost of Transport & Mode 1 Correlation Analysis
===================================================================================
- 各個別ディレクトリから `phase_energy_analysis.csv` と `*_trajectory_velocity.csv` を読み込む。
- 水からの取り上げ時の過渡期（末尾数秒）をカットし、純粋な水中遊泳定常区間における
  第1モード収束値 (|Z1|) と平均電力を正確に算出する。
- 試行 11315（赤）と 11317（緑）を個別色、その他（青）を同条件としてプロット。
- 散布図および相関解析（Pearson r, Spearman rho, p値）:
  1. 遊泳速度 vs. Cost of Transport (CoT)
  2. 第1モード収束値 (|Z1|) vs. 遊泳速度
  3. 第1モード収束値 (|Z1|) vs. Cost of Transport (CoT)
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


def load_and_aggregate_all_trials(base_dir=r"D:\codes\Volvocine_PicoV2\VolBotVideo"):
    """
    全試行フォルダ内の CSV から、取り上げ過渡期を排除した真の遊泳定常状態指標を集約する。
    """
    trial_dirs = sorted([d for d in glob.glob(os.path.join(base_dir, "GX*")) if os.path.isdir(d)])

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
    for td in trial_dirs:
        t_name = os.path.basename(td)
        if "11316" in t_name:
            continue  # Exclude aborted trial 11316
            
        track_csvs = glob.glob(os.path.join(td, "*_trajectory_velocity.csv"))
        phase_csvs = glob.glob(os.path.join(td, "phase_energy_analysis.csv"))
        
        if not track_csvs or not phase_csvs:
            continue
            
        df_track = pd.read_csv(track_csvs[0])
        df_phase = pd.read_csv(phase_csvs[0])
        
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
        
        # 2. Steady-state Mode 1 convergence evaluation
        # IMPORTANT: Cut off the final 3.0s before water removal to avoid hand pickup disturbances
        t_phase = df_phase['time_sec']
        t_eval_end = max(swim_t_start + 10.0, swim_t_end - 3.0)
        t_eval_start = max(swim_t_start + 8.0, t_eval_end - 15.0)
        
        mask_steady = (t_phase >= t_eval_start) & (t_phase <= t_eval_end)
        if not np.any(mask_steady):
            # Fallback to last 20% excluding last 4s
            t_max_p = t_phase.max()
            mask_steady = (t_phase >= t_max_p * 0.6) & (t_phase <= max(t_max_p * 0.6 + 5.0, t_max_p - 4.0))
            
        z1_series = df_phase.loc[mask_steady, 'order_param_mode1_Z1']
        mode1_converged = float(z1_series.mean())
        mode1_std = float(z1_series.std())

        z2_series = df_phase.loc[mask_steady, 'order_param_mode2_Z2']
        mode2_converged = float(z2_series.mean())
        mode2_std = float(z2_series.std())

        z3_series = df_phase.loc[mask_steady, 'order_param_mode3_Z3']
        mode3_converged = float(z3_series.mean())
        mode3_std = float(z3_series.std())

        z4_series = df_phase.loc[mask_steady, 'order_param_mode4_Z4']
        mode4_converged = float(z4_series.mean())
        mode4_std = float(z4_series.std())

        # 3. Weighted input amplitude received by oscillators: G_i(t) = sum_l sigma_l * u_{il} * Z_l(t)
        phases = np.zeros((len(df_phase), N))
        for i, ag in enumerate(real_agents):
            phases[:, i] = df_phase[f'abs_phase_unwrapped_rad_agent_{ag}'].values
        phasors = np.exp(1j * phases)
        Z_comp = np.dot(phasors, V)
        G_comp = np.dot(Z_comp * sigma, U.T)
        G_abs_matrix = np.abs(G_comp)

        G_steady = G_abs_matrix[mask_steady]
        g_means = np.mean(G_steady, axis=0) if len(G_steady) > 0 else np.zeros(N)
        g_sum_all = float(np.sum(g_means))
        g_mean_all = float(np.mean(g_means))
        
        # Mean electrical power during pure swimming
        mask_swim = (t_phase >= swim_t_start) & (t_phase <= swim_t_end - 2.0)
        if np.any(mask_swim):
            mean_power = float(df_phase.loc[mask_swim, 'total_power_w'].mean())
        else:
            mean_power = float(df_phase['total_power_w'].mean())
            
        # Cost of Transport (CoT): CoT = Power / Speed
        cot_J_px = mean_power / max(mean_speed, 1e-6)
        cot_mJ_px = cot_J_px * 1000.0
        
        # Cumulative energy during valid swimming
        dt_p = np.median(np.diff(t_phase))
        total_energy_J = float(np.trapz(df_phase.loc[mask_swim, 'total_power_w'], t_phase[mask_swim])) if np.any(mask_swim) else 0.0
        
        rows.append({
            "trial": t_name,
            "swimming_duration_sec": swim_duration,
            "mean_speed_px_s": mean_speed,
            "max_speed_px_s": max_speed,
            "std_speed_px_s": std_speed,
            "total_distance_px": total_dist_px,
            "stroke_peak_freq_hz": peak_freq,
            "mean_power_W": mean_power,
            "mean_power_mW": mean_power * 1000.0,
            "total_energy_J": total_energy_J,
            "cot_J_px": cot_J_px,
            "cot_mJ_px": cot_mJ_px,
            "mode1_converged_mean": mode1_converged,
            "mode1_steady_std": mode1_std,
            "mode2_converged_mean": mode2_converged,
            "mode2_steady_std": mode2_std,
            "mode3_converged_mean": mode3_converged,
            "mode3_steady_std": mode3_std,
            "mode4_converged_mean": mode4_converged,
            "mode4_steady_std": mode4_std,
            "G_sum_all": g_sum_all,
            "G_mean_all": g_mean_all,
            "G_ag8": float(g_means[0]),
            "G_ag9": float(g_means[1]),
            "G_ag11": float(g_means[2]),
            "G_ag12": float(g_means[3]),
            "eval_window_start_sec": t_eval_start,
            "eval_window_end_sec": t_eval_end
        })
        
    df = pd.DataFrame(rows)
    df.sort_values(by='trial', inplace=True)
    
    # Assign conditions & styling
    conditions = []
    colors = []
    markers = []
    for t in df['trial']:
        if '11315' in t:
            conditions.append('Trial 11315')
            colors.append('#D90429')  # Crimson Red
            markers.append('D')
        elif '11317' in t:
            conditions.append('Trial 11317')
            colors.append('#06D6A0')  # Emerald Green
            markers.append('s')
        else:
            conditions.append('Standard Condition')
            colors.append('#118AB2')  # Oceanic Blue
            markers.append('o')
            
    df['condition'] = conditions
    df['plot_color'] = colors
    df['plot_marker'] = markers
    
    return df


def compute_correlations(x, y, label=""):
    """Compute Pearson and Spearman correlation statistics."""
    mask = np.isfinite(x) & np.isfinite(y)
    x_c = x[mask]
    y_c = y[mask]
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


def plot_speed_vs_cot(df, output_dir):
    """Figure 1: Swimming Speed vs. Cost of Transport"""
    fig, ax = plt.subplots(figsize=(9, 7))

    for cond in ['Standard Condition', 'Trial 11315', 'Trial 11317']:
        group = df[df['condition'] == cond]
        if group.empty:
            continue
        color = group['plot_color'].iloc[0]
        marker = group['plot_marker'].iloc[0]
        size = 140 if cond != 'Standard Condition' else 90
        zorder = 5 if cond != 'Standard Condition' else 3
        
        ax.scatter(group['mean_speed_px_s'], group['cot_mJ_px'],
                   c=color, marker=marker, s=size, label=cond,
                   edgecolors='black', linewidth=1.2, alpha=0.9, zorder=zorder)

    # Annotate trial IDs
    for _, row in df.iterrows():
        t_id = row['trial'].replace('GX01', '')
        fontweight = 'bold' if row['condition'] != 'Standard Condition' else 'normal'
        fontsize = 9.5 if row['condition'] != 'Standard Condition' else 8.5
        color = row['plot_color'] if row['condition'] != 'Standard Condition' else '#222222'
        ax.annotate(t_id, (row['mean_speed_px_s'], row['cot_mJ_px']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=fontsize, fontweight=fontweight, color=color,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

    ax.set_title("Distribution of Swimming Speed vs. Cost of Transport (CoT)", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Mean Swimming Speed $v$ [px/s]", fontsize=12)
    ax.set_ylabel("Cost of Transport (CoT = $P/v$) [mJ/px]", fontsize=12)
    ax.set_xlim(31, 46)
    ax.set_ylim(47, 72)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)

    plt.tight_layout()
    png_path = os.path.join(output_dir, "scatter_speed_vs_cot.png")
    pdf_path = os.path.join(output_dir, "scatter_speed_vs_cot.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"[SAVED] {png_path}")
    print(f"[SAVED] {pdf_path}")


def plot_mode1_correlations(df, output_dir):
    """Figure 2: Mode 1 Convergence Value vs Speed and CoT"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    std_df = df[df['condition'] == 'Standard Condition']

    # --- Panel 1: Mode 1 vs. Speed ---
    for cond in ['Standard Condition', 'Trial 11315', 'Trial 11317']:
        group = df[df['condition'] == cond]
        if group.empty:
            continue
        color = group['plot_color'].iloc[0]
        marker = group['plot_marker'].iloc[0]
        size = 140 if cond != 'Standard Condition' else 90
        zorder = 5 if cond != 'Standard Condition' else 3
        
        ax1.scatter(group['mode1_converged_mean'], group['mean_speed_px_s'],
                    c=color, marker=marker, s=size, label=cond,
                    edgecolors='black', linewidth=1.2, alpha=0.9, zorder=zorder)

    for _, row in df.iterrows():
        t_id = row['trial'].replace('GX01', '')
        fontweight = 'bold' if row['condition'] != 'Standard Condition' else 'normal'
        fontsize = 9.5 if row['condition'] != 'Standard Condition' else 8.5
        color = row['plot_color'] if row['condition'] != 'Standard Condition' else '#222222'
        ax1.annotate(t_id, (row['mode1_converged_mean'], row['mean_speed_px_s']),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=fontsize, fontweight=fontweight, color=color,
                     path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

    if len(std_df) >= 3:
        p1 = np.polyfit(std_df['mode1_converged_mean'], std_df['mean_speed_px_s'], deg=1)
        x_line = np.linspace(df['mode1_converged_mean'].min()*0.95, df['mode1_converged_mean'].max()*1.05, 100)
        ax1.plot(x_line, np.polyval(p1, x_line), color='#118AB2', linestyle='--', lw=1.6, alpha=0.7)

    corr1_std = compute_correlations(std_df['mode1_converged_mean'], std_df['mean_speed_px_s'], "Standard")
    corr1_all = compute_correlations(df['mode1_converged_mean'], df['mean_speed_px_s'], "All")
    if corr1_std and corr1_all:
        stat_text1 = (f"Standard Condition (N={corr1_std['n']}):\n"
                      f"  Pearson r = {corr1_std['pearson_r']:.3f} (p = {corr1_std['pearson_p']:.2e})\n"
                      f"  Spearman ρ = {corr1_std['spearman_rho']:.3f} (p = {corr1_std['spearman_p']:.2e})\n"
                      f"All Trials (N={corr1_all['n']}):\n"
                      f"  Pearson r = {corr1_all['pearson_r']:.3f} (p = {corr1_all['pearson_p']:.2e})")
        ax1.text(0.96, 0.96, stat_text1, transform=ax1.transAxes, verticalalignment='top', horizontalalignment='right',
                 fontsize=9.5, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.92, edgecolor='#aaaaaa'))

    ax1.set_title("First Mode Convergence Value vs. Swimming Speed", fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel("First Mode Steady Convergence Value $|Z_1|$", fontsize=12)
    ax1.set_ylabel("Mean Swimming Speed $v$ [px/s]", fontsize=12)
    ax1.set_ylim(32, 45)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)

    # --- Panel 2: Mode 1 vs. CoT ---
    for cond in ['Standard Condition', 'Trial 11315', 'Trial 11317']:
        group = df[df['condition'] == cond]
        if group.empty:
            continue
        color = group['plot_color'].iloc[0]
        marker = group['plot_marker'].iloc[0]
        size = 140 if cond != 'Standard Condition' else 90
        zorder = 5 if cond != 'Standard Condition' else 3
        
        ax2.scatter(group['mode1_converged_mean'], group['cot_mJ_px'],
                    c=color, marker=marker, s=size, label=cond,
                    edgecolors='black', linewidth=1.2, alpha=0.9, zorder=zorder)

    for _, row in df.iterrows():
        t_id = row['trial'].replace('GX01', '')
        fontweight = 'bold' if row['condition'] != 'Standard Condition' else 'normal'
        fontsize = 9.5 if row['condition'] != 'Standard Condition' else 8.5
        color = row['plot_color'] if row['condition'] != 'Standard Condition' else '#222222'
        ax2.annotate(t_id, (row['mode1_converged_mean'], row['cot_mJ_px']),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=fontsize, fontweight=fontweight, color=color,
                     path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

    if len(std_df) >= 3:
        p2 = np.polyfit(std_df['mode1_converged_mean'], std_df['cot_mJ_px'], deg=1)
        x_line = np.linspace(df['mode1_converged_mean'].min()*0.95, df['mode1_converged_mean'].max()*1.05, 100)
        ax2.plot(x_line, np.polyval(p2, x_line), color='#118AB2', linestyle='--', lw=1.6, alpha=0.7)

    corr2_std = compute_correlations(std_df['mode1_converged_mean'], std_df['cot_mJ_px'], "Standard")
    corr2_all = compute_correlations(df['mode1_converged_mean'], df['cot_mJ_px'], "All")
    if corr2_std and corr2_all:
        stat_text2 = (f"Standard Condition (N={corr2_std['n']}):\n"
                      f"  Pearson r = {corr2_std['pearson_r']:.3f} (p = {corr2_std['pearson_p']:.2e})\n"
                      f"  Spearman ρ = {corr2_std['spearman_rho']:.3f} (p = {corr2_std['spearman_p']:.2e})\n"
                      f"All Trials (N={corr2_all['n']}):\n"
                      f"  Pearson r = {corr2_all['pearson_r']:.3f} (p = {corr2_all['pearson_p']:.2e})")
        ax2.text(0.04, 0.96, stat_text2, transform=ax2.transAxes, verticalalignment='top',
                 fontsize=9.5, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.92, edgecolor='#aaaaaa'))

    ax2.set_title("First Mode Convergence Value vs. Cost of Transport", fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel("First Mode Steady Convergence Value $|Z_1|$", fontsize=12)
    ax2.set_ylabel("Cost of Transport (CoT = $P/v$) [mJ/px]", fontsize=12)
    ax2.set_ylim(48, 72)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)

    plt.tight_layout()
    png_path = os.path.join(output_dir, "scatter_mode1_correlations.png")
    pdf_path = os.path.join(output_dir, "scatter_mode1_correlations.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"[SAVED] {png_path}")
    print(f"[SAVED] {pdf_path}")


def plot_weighted_input_correlations(df, output_dir):
    """Figure 3: Sum of Weighted Input Amplitudes (|G_i|) vs Speed and CoT"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    std_df = df[df['condition'] == 'Standard Condition']

    # --- Panel 1: Sum of |Gi| vs. Speed ---
    for cond in ['Standard Condition', 'Trial 11315', 'Trial 11317']:
        group = df[df['condition'] == cond]
        if group.empty:
            continue
        color = group['plot_color'].iloc[0]
        marker = group['plot_marker'].iloc[0]
        size = 140 if cond != 'Standard Condition' else 90
        zorder = 5 if cond != 'Standard Condition' else 3
        
        ax1.scatter(group['G_sum_all'], group['mean_speed_px_s'],
                    c=color, marker=marker, s=size, label=cond,
                    edgecolors='black', linewidth=1.2, alpha=0.9, zorder=zorder)

    for _, row in df.iterrows():
        t_id = row['trial'].replace('GX01', '')
        fontweight = 'bold' if row['condition'] != 'Standard Condition' else 'normal'
        fontsize = 9.5 if row['condition'] != 'Standard Condition' else 8.5
        color = row['plot_color'] if row['condition'] != 'Standard Condition' else '#222222'
        ax1.annotate(t_id, (row['G_sum_all'], row['mean_speed_px_s']),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=fontsize, fontweight=fontweight, color=color,
                     path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

    if len(std_df) >= 3:
        p1 = np.polyfit(std_df['G_sum_all'], std_df['mean_speed_px_s'], deg=1)
        x_line = np.linspace(df['G_sum_all'].min()*0.95, df['G_sum_all'].max()*1.05, 100)
        ax1.plot(x_line, np.polyval(p1, x_line), color='#118AB2', linestyle='--', lw=1.6, alpha=0.7)

    corr1_std = compute_correlations(std_df['G_sum_all'], std_df['mean_speed_px_s'], "Standard")
    corr1_all = compute_correlations(df['G_sum_all'], df['mean_speed_px_s'], "All")
    if corr1_std and corr1_all:
        stat_text1 = (f"Standard Condition (N={corr1_std['n']}):\n"
                      f"  Pearson r = {corr1_std['pearson_r']:.3f} (p = {corr1_std['pearson_p']:.2e})\n"
                      f"  Spearman ρ = {corr1_std['spearman_rho']:.3f} (p = {corr1_std['spearman_p']:.2e})\n"
                      f"All Trials (N={corr1_all['n']}):\n"
                      f"  Pearson r = {corr1_all['pearson_r']:.3f} (p = {corr1_all['pearson_p']:.2e})\n"
                      f"  Spearman ρ = {corr1_all['spearman_rho']:.3f} (p = {corr1_all['spearman_p']:.2e})")
        ax1.text(0.96, 0.96, stat_text1, transform=ax1.transAxes, verticalalignment='top', horizontalalignment='right',
                 fontsize=9.5, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.92, edgecolor='#aaaaaa'))

    ax1.set_title("Total Weighted Input Amplitude vs. Swimming Speed", fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel(r"Total Received Input Amplitude $\sum_{i} |G_i|$", fontsize=12)
    ax1.set_ylabel("Mean Swimming Speed $v$ [px/s]", fontsize=12)
    ax1.set_xlim(0.10, 0.36)
    ax1.set_ylim(32, 45)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)

    # --- Panel 2: Sum of |Gi| vs. CoT ---
    for cond in ['Standard Condition', 'Trial 11315', 'Trial 11317']:
        group = df[df['condition'] == cond]
        if group.empty:
            continue
        color = group['plot_color'].iloc[0]
        marker = group['plot_marker'].iloc[0]
        size = 140 if cond != 'Standard Condition' else 90
        zorder = 5 if cond != 'Standard Condition' else 3
        
        ax2.scatter(group['G_sum_all'], group['cot_mJ_px'],
                    c=color, marker=marker, s=size, label=cond,
                    edgecolors='black', linewidth=1.2, alpha=0.9, zorder=zorder)

    for _, row in df.iterrows():
        t_id = row['trial'].replace('GX01', '')
        fontweight = 'bold' if row['condition'] != 'Standard Condition' else 'normal'
        fontsize = 9.5 if row['condition'] != 'Standard Condition' else 8.5
        color = row['plot_color'] if row['condition'] != 'Standard Condition' else '#222222'
        ax2.annotate(t_id, (row['G_sum_all'], row['cot_mJ_px']),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=fontsize, fontweight=fontweight, color=color,
                     path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

    if len(std_df) >= 3:
        p2 = np.polyfit(std_df['G_sum_all'], std_df['cot_mJ_px'], deg=1)
        x_line = np.linspace(df['G_sum_all'].min()*0.95, df['G_sum_all'].max()*1.05, 100)
        ax2.plot(x_line, np.polyval(p2, x_line), color='#118AB2', linestyle='--', lw=1.6, alpha=0.7)

    corr2_std = compute_correlations(std_df['G_sum_all'], std_df['cot_mJ_px'], "Standard")
    corr2_all = compute_correlations(df['G_sum_all'], df['cot_mJ_px'], "All")
    if corr2_std and corr2_all:
        stat_text2 = (f"Standard Condition (N={corr2_std['n']}):\n"
                      f"  Pearson r = {corr2_std['pearson_r']:.3f} (p = {corr2_std['pearson_p']:.2e})\n"
                      f"  Spearman ρ = {corr2_std['spearman_rho']:.3f} (p = {corr2_std['spearman_p']:.2e})\n"
                      f"All Trials (N={corr2_all['n']}):\n"
                      f"  Pearson r = {corr2_all['pearson_r']:.3f} (p = {corr2_all['pearson_p']:.2e})\n"
                      f"  Spearman ρ = {corr2_all['spearman_rho']:.3f} (p = {corr2_all['spearman_p']:.2e})")
        ax2.text(0.04, 0.96, stat_text2, transform=ax2.transAxes, verticalalignment='top',
                 fontsize=9.5, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.92, edgecolor='#aaaaaa'))

    ax2.set_title("Total Weighted Input Amplitude vs. Cost of Transport", fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel(r"Total Received Input Amplitude $\sum_{i} |G_i|$", fontsize=12)
    ax2.set_ylabel("Cost of Transport (CoT = $P/v$) [mJ/px]", fontsize=12)
    ax2.set_xlim(0.10, 0.36)
    ax2.set_ylim(48, 72)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)

    plt.tight_layout()
    png_path = os.path.join(output_dir, "scatter_weighted_input_correlations.png")
    pdf_path = os.path.join(output_dir, "scatter_weighted_input_correlations.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"[SAVED] {png_path}")
    print(f"[SAVED] {pdf_path}")


def plot_all_modes_correlations(df, output_dir):
    """Figure 4: Modes 1-4 Order Parameters vs. Swimming Speed & CoT (2x4 Grid)"""
    fig, axes = plt.subplots(2, 4, figsize=(24, 11))
    plt.subplots_adjust(hspace=0.28, wspace=0.22)

    std_df = df[df['condition'] == 'Standard Condition']
    modes = [1, 2, 3, 4]
    xlims = {1: (0.05, 1.85), 2: (0.15, 1.50), 3: (0.10, 1.75), 4: (0.10, 1.70)}

    for m in modes:
        col_m = f'mode{m}_converged_mean'
        
        # --- Row 0: Mode m vs. Speed ---
        ax_speed = axes[0, m - 1]
        for cond in ['Standard Condition', 'Trial 11315', 'Trial 11317']:
            group = df[df['condition'] == cond]
            if group.empty:
                continue
            color = group['plot_color'].iloc[0]
            marker = group['plot_marker'].iloc[0]
            size = 140 if cond != 'Standard Condition' else 90
            zorder = 5 if cond != 'Standard Condition' else 3
            ax_speed.scatter(group[col_m], group['mean_speed_px_s'],
                             c=color, marker=marker, s=size, label=cond,
                             edgecolors='black', linewidth=1.2, alpha=0.9, zorder=zorder)
                             
        for _, row in df.iterrows():
            t_id = row['trial'].replace('GX01', '')
            fontweight = 'bold' if row['condition'] != 'Standard Condition' else 'normal'
            fontsize = 9.0 if row['condition'] != 'Standard Condition' else 8.0
            color = row['plot_color'] if row['condition'] != 'Standard Condition' else '#222222'
            ax_speed.annotate(t_id, (row[col_m], row['mean_speed_px_s']),
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=fontsize, fontweight=fontweight, color=color,
                              path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])
                              
        if len(std_df) >= 3:
            p_s = np.polyfit(std_df[col_m], std_df['mean_speed_px_s'], deg=1)
            x_s = np.linspace(df[col_m].min() * 0.95, df[col_m].max() * 1.05, 100)
            ax_speed.plot(x_s, np.polyval(p_s, x_s), color='#118AB2', linestyle='--', lw=1.6, alpha=0.7)

        corr_s_std = compute_correlations(std_df[col_m], std_df['mean_speed_px_s'], f"Mode{m}")
        corr_s_all = compute_correlations(df[col_m], df['mean_speed_px_s'], f"Mode{m}")
        if corr_s_std and corr_s_all:
            stat_text = (f"Std (N=18): r={corr_s_std['pearson_r']:+.3f} (p={corr_s_std['pearson_p']:.2e})\n"
                         f"               ρ={corr_s_std['spearman_rho']:+.3f} (p={corr_s_std['spearman_p']:.2e})\n"
                         f"All (N=20): r={corr_s_all['pearson_r']:+.3f} (p={corr_s_all['pearson_p']:.2e})")
            box_x = 0.96 if corr_s_std['pearson_r'] < 0 else 0.04
            ha = 'right' if corr_s_std['pearson_r'] < 0 else 'left'
            ax_speed.text(box_x, 0.96, stat_text, transform=ax_speed.transAxes, verticalalignment='top', horizontalalignment=ha,
                          fontsize=8.5, bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.92, edgecolor='#aaaaaa'))

        ax_speed.set_title(f"Mode {m} ($|Z_{m}|$) vs. Speed", fontsize=12.5, fontweight='bold', pad=10)
        ax_speed.set_xlabel(f"Mode {m} Convergence Value $|Z_{m}|$", fontsize=11)
        ax_speed.set_ylabel("Mean Speed $v$ [px/s]", fontsize=11)
        ax_speed.set_xlim(xlims[m])
        ax_speed.set_ylim(31.5, 45)
        ax_speed.grid(True, linestyle='--', alpha=0.5)
        if m == 1:
            ax_speed.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)

        # --- Row 1: Mode m vs. CoT ---
        ax_cot = axes[1, m - 1]
        for cond in ['Standard Condition', 'Trial 11315', 'Trial 11317']:
            group = df[df['condition'] == cond]
            if group.empty:
                continue
            color = group['plot_color'].iloc[0]
            marker = group['plot_marker'].iloc[0]
            size = 140 if cond != 'Standard Condition' else 90
            zorder = 5 if cond != 'Standard Condition' else 3
            ax_cot.scatter(group[col_m], group['cot_mJ_px'],
                           c=color, marker=marker, s=size, label=cond,
                           edgecolors='black', linewidth=1.2, alpha=0.9, zorder=zorder)

        for _, row in df.iterrows():
            t_id = row['trial'].replace('GX01', '')
            fontweight = 'bold' if row['condition'] != 'Standard Condition' else 'normal'
            fontsize = 9.0 if row['condition'] != 'Standard Condition' else 8.0
            color = row['plot_color'] if row['condition'] != 'Standard Condition' else '#222222'
            ax_cot.annotate(t_id, (row[col_m], row['cot_mJ_px']),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=fontsize, fontweight=fontweight, color=color,
                            path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

        if len(std_df) >= 3:
            p_c = np.polyfit(std_df[col_m], std_df['cot_mJ_px'], deg=1)
            x_c = np.linspace(df[col_m].min() * 0.95, df[col_m].max() * 1.05, 100)
            ax_cot.plot(x_c, np.polyval(p_c, x_c), color='#118AB2', linestyle='--', lw=1.6, alpha=0.7)

        corr_c_std = compute_correlations(std_df[col_m], std_df['cot_mJ_px'], f"Mode{m}")
        corr_c_all = compute_correlations(df[col_m], df['cot_mJ_px'], f"Mode{m}")
        if corr_c_std and corr_c_all:
            stat_text = (f"Std (N=18): r={corr_c_std['pearson_r']:+.3f} (p={corr_c_std['pearson_p']:.2e})\n"
                         f"               ρ={corr_c_std['spearman_rho']:+.3f} (p={corr_c_std['spearman_p']:.2e})\n"
                         f"All (N=20): r={corr_c_all['pearson_r']:+.3f} (p={corr_c_all['pearson_p']:.2e})")
            box_x = 0.04 if corr_c_std['pearson_r'] > 0 else 0.96
            ha = 'left' if corr_c_std['pearson_r'] > 0 else 'right'
            ax_cot.text(box_x, 0.96, stat_text, transform=ax_cot.transAxes, verticalalignment='top', horizontalalignment=ha,
                        fontsize=8.5, bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.92, edgecolor='#aaaaaa'))

        ax_cot.set_title(f"Mode {m} ($|Z_{m}|$) vs. Cost of Transport", fontsize=12.5, fontweight='bold', pad=10)
        ax_cot.set_xlabel(f"Mode {m} Convergence Value $|Z_{m}|$", fontsize=11)
        ax_cot.set_ylabel("Cost of Transport [mJ/px]", fontsize=11)
        ax_cot.set_xlim(xlims[m])
        ax_cot.set_ylim(48, 72)
        ax_cot.grid(True, linestyle='--', alpha=0.5)
        if m == 1:
            ax_cot.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)

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
    print("\n[INFO] Loading all trial CSVs and computing steady-state metrics...")
    df = load_and_aggregate_all_trials(base_dir)
    
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
    print(f"[INFO] Combined summary saved to: {summary_path}")

    # Generate plots
    print("\n[INFO] Generating Speed vs. CoT scatter plot...")
    plot_speed_vs_cot(df, base_dir)

    print("\n[INFO] Generating Mode 1 correlation plots...")
    plot_mode1_correlations(df, base_dir)

    print("\n[INFO] Generating Weighted Input Amplitude correlation plots...")
    plot_weighted_input_correlations(df, base_dir)

    print("\n[INFO] Generating All Modes (1-4) correlation plots (2x4 Grid)...")
    plot_all_modes_correlations(df, base_dir)

    # Print summary table
    print("\n" + "="*135)
    print("                                            ALL TRIALS ANALYSIS SUMMARY TABLE")
    print("="*135)
    disp_df = df[['trial', 'condition', 'mean_speed_px_s', 'mean_power_W', 'cot_mJ_px', 'mode1_converged_mean', 'mode2_converged_mean', 'mode3_converged_mean', 'mode4_converged_mean', 'G_sum_all']]
    print(disp_df.to_string(index=False))
    print("="*135 + "\n")


if __name__ == "__main__":
    main()
