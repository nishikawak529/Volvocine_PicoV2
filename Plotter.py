import os
import gc
import sys

if "MPLCONFIGDIR" not in os.environ:
    matplotlib_config_dir = os.path.join("/tmp", "matplotlib-volvocine")
    os.makedirs(matplotlib_config_dir, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = matplotlib_config_dir

import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np
import matplotlib.ticker as ticker  # 目盛りのフォーマット用
from datetime import datetime
import matplotlib.gridspec as gridspec  # 追加


def show_plot_and_cleanup():
    try:
        plt.show()
    finally:
        plt.close("all")
        gc.collect()


def moving_average_ignore_nan(values, window_size):
    if window_size <= 1:
        return values
    return (
        pd.Series(values)
        .rolling(window=window_size, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def sorted_unique_phase_series(sub):
    sub = sub.sort_values("time_pc_sec_abs").copy()
    sub["a0"] = correct_phase_discontinuity(sub["a0"].to_numpy(dtype=float))
    sub = sub.drop_duplicates(subset="time_pc_sec_abs", keep="first")
    return sub


def compute_relative_phase_series(
    df_main,
    base_agent_id=None,
    time_step_sec=0.01,
    filter_window_size=1,
    n_sync=1,
    m_sync=1,
):
    if df_main.empty:
        return base_agent_id, []

    agents = sorted(df_main["agent_id"].unique())
    if len(agents) < 2:
        return base_agent_id, []

    if base_agent_id is None or base_agent_id not in agents:
        base_agent_id = min(agents)

    base = sorted_unique_phase_series(df_main[df_main["agent_id"] == base_agent_id])
    if base.empty:
        return base_agent_id, []

    series = []
    for agent_id in agents:
        if agent_id == base_agent_id:
            continue

        sub = sorted_unique_phase_series(df_main[df_main["agent_id"] == agent_id])
        if sub.empty:
            continue

        start_time = max(base["time_pc_sec_abs"].min(), sub["time_pc_sec_abs"].min())
        end_time = min(base["time_pc_sec_abs"].max(), sub["time_pc_sec_abs"].max())
        if end_time <= start_time:
            continue

        t_abs = np.arange(start_time, end_time, time_step_sec)
        if t_abs.size < 2:
            continue

        base_a0 = np.interp(t_abs, base["time_pc_sec_abs"], base["a0"])
        agent_a0 = np.interp(t_abs, sub["time_pc_sec_abs"], sub["a0"])
        base_a0 = moving_average_ignore_nan(base_a0, filter_window_size)
        agent_a0 = moving_average_ignore_nan(agent_a0, filter_window_size)

        phase_raw = m_sync * base_a0 - n_sync * agent_a0
        phase_diff = ((phase_raw + 128.0) % 256.0) - 128.0
        phase_diff = phase_diff * (2.0 * np.pi / 256.0)

        phase_diff_with_nan = phase_diff.copy()
        jump_indices = np.where(np.abs(np.diff(phase_diff_with_nan)) > np.pi)[0] + 1
        phase_diff_with_nan[jump_indices] = np.nan

        series.append(
            {
                "agent_id": agent_id,
                "time": t_abs,
                "phase": phase_diff_with_nan,
            }
        )

    return base_agent_id, series


def format_phase_axis(ax):
    ax.set_ylim(-np.pi, np.pi)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=np.pi))
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda x, _: "-π" if np.isclose(x, -np.pi)
            else "0" if np.isclose(x, 0)
            else "π" if np.isclose(x, np.pi)
            else ""
        )
    )
    ax.grid(True)


def plot_chunks(file_list):
    # None チェックを追加
    if not file_list:
        print("[INFO] No files provided to plot.")
        return

    # 単一ファイルパスが来たらリストに変換
    if isinstance(file_list, str):
        file_list = [file_list]

    print("[DEBUG] Plotting from files:")
    for f in file_list:
        print(f"  - {f}")

    dfs = []
    for file in file_list:
        if not os.path.isfile(file):
            print(f"[WARN] File not found: {file}")
            continue

        try:
            df = pd.read_csv(file)
            if all(col in df.columns for col in ["agent_id", "chunk_id", "time_pc_sec_abs", "a0", "a1", "a2"]):
                dfs.append(df[["agent_id", "chunk_id", "time_pc_sec_abs", "a0", "a1", "a2"]])
        except Exception as e:
            print(f"[WARN] Failed to load {file}: {e}")

    if not dfs:
        print("[INFO] No valid data to plot.")
        return

    df_all = pd.concat(dfs, ignore_index=True)
    corrected_dfs = []
    for (agent_id, chunk_id), sub in df_all.groupby(["agent_id", "chunk_id"]):
        corrected_dfs.append(correct_large_jump(sub))
    if corrected_dfs:
        df_all = pd.concat(corrected_dfs, ignore_index=True)
        df_all = correct_chunk_start_times(df_all)
    detect_time_anomalies(df_all)

    # agent_id==99のデータを分離
    df_99 = df_all[df_all["agent_id"] == 99].copy()
    df_main = df_all[df_all["agent_id"] != 99]
    base_agent_id, phase_series = compute_relative_phase_series(df_main)

    # a1が170以上の時は-255する
    if not df_99.empty:
        df_99.loc[df_99["a1"] >= 170, "a1"] -= 255

    # a0/a1/a2 + 位相差 + agent99(あれば) を描く
    show_agent99 = not df_99.empty
    n_axes = 5 if show_agent99 else 4
    fig, axs = plt.subplots(n_axes, 1, figsize=(10, 2.0 * n_axes), sharex=True)
    axs = np.atleast_1d(axs)
    colors = {}
    color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    # 通常プロット（agent_id==99以外）
    for (ag_id, _), sub in df_main.groupby(["agent_id", "chunk_id"]):
        if ag_id not in colors:
            colors[ag_id] = next(color_cycle)
        axs[0].plot(sub["time_pc_sec_abs"], sub["a0"], color=colors[ag_id])
        axs[1].plot(sub["time_pc_sec_abs"], sub["a1"], color=colors[ag_id])
        axs[2].plot(sub["time_pc_sec_abs"], sub["a2"], color=colors[ag_id])

    axs[0].set_ylabel("a0")
    axs[1].set_ylabel("a1")
    axs[2].set_ylabel("a2")
    for ax in axs[:3]:
        ax.grid(True)

    handles = [plt.Line2D([0], [0], color=color, lw=2, label=f"Agent {ag_id}")
               for ag_id, color in colors.items()]
    if handles:
        axs[0].legend(handles=handles, title="Agents")

    phase_ax = axs[3]
    for entry in phase_series:
        agent_id = entry["agent_id"]
        phase_ax.plot(
            entry["time"],
            entry["phase"],
            color=colors.get(agent_id),
            linewidth=0.9,
            label=f"Agent {base_agent_id} - {agent_id}",
        )
    phase_ax.axhline(0, color="0.35", linestyle="--", linewidth=0.8)
    phase_ax.set_ylabel("Phase diff\n(rad)")
    format_phase_axis(phase_ax)
    if phase_series:
        phase_ax.legend(title="Relative phase", loc="upper right")
    else:
        phase_ax.text(
            0.5,
            0.5,
            "Need at least 2 agents",
            transform=phase_ax.transAxes,
            ha="center",
            va="center",
            color="0.45",
        )

    if show_agent99:
        imu_ax = axs[4]
        imu_ax.plot(df_99["time_pc_sec_abs"], df_99["a0"], label="Agent 99 a0", color="tab:blue")
        imu_ax.plot(df_99["time_pc_sec_abs"], df_99["a1"], label="Agent 99 a1", color="tab:orange")
        imu_ax.set_ylabel("Agent99\na0/a1")
        imu_ax.legend()
        imu_ax.grid(True)

    axs[-1].set_xlabel("PC time (sec)")

    plt.tight_layout()
    show_plot_and_cleanup()

def correct_phase_discontinuity(phase_data):
    """
    位相データのジャンプを補正する関数。
    急激な変化があった場合に 256 を加算または減算して連続性を保つ。
    """
    corrected_phase = phase_data.copy()
    for i in range(1, len(corrected_phase)):
        diff = corrected_phase[i] - corrected_phase[i - 1]
        if diff < -128:  # 急に128以上小さくなった場合
            corrected_phase[i:] += 256
        elif diff > 128:  # 急に128以上大きくなった場合
            corrected_phase[i:] -= 256
    return corrected_phase

def plot_relativePhase(file_list):
    # None チェックを追加
    if not file_list:
        print("[INFO] No files provided to plot.")
        return

    # 単一ファイルパスが来たらリストに変換
    if isinstance(file_list, str):
        file_list = [file_list]

    print("[DEBUG] Plotting from files:")
    for f in file_list:
        print(f"  - {f}")

    dfs = []
    for file in file_list:
        if not os.path.isfile(file):
            print(f"[WARN] File not found: {file}")
            continue

        try:
            df = pd.read_csv(file)
            if all(col in df.columns for col in ["agent_id", "chunk_id", "time_pc_sec_abs", "a0", "a1", "a2"]):
                dfs.append(df[["agent_id", "chunk_id", "time_pc_sec_abs", "a0", "a1", "a2"]])
        except Exception as e:
            print(f"[WARN] Failed to load {file}: {e}")

    if not dfs:
        print("[INFO] No valid data to plot.")
        return

    df_all = pd.concat(dfs, ignore_index=True)
    # オーバーフロー補正（agent_id, chunk_id ごとに補正）
    corrected_dfs = []
    for (agent_id, chunk_id), sub in df_all.groupby(["agent_id", "chunk_id"]):
        sub_corrected = correct_large_jump(sub)
        corrected_dfs.append(sub_corrected)

    # 補正済みで再結合
    df_all = pd.concat(corrected_dfs, ignore_index=True)

    # 異常チェック（補正後）
    detect_time_anomalies(df_all)

    # チャンクごとの開始・終了UNIX時刻と範囲を算出
    summary = (
        df_all.groupby(["agent_id", "chunk_id"])["time_pc_sec_abs"]
        .agg(start_time="min", end_time="max")
        .assign(duration=lambda x: x["end_time"] - x["start_time"])
        .sort_values("start_time")
        .reset_index()
    )

    # UNIX秒 → datetime に変換（開始時刻のみ表示用）
    summary["start_dt"] = summary["start_time"].apply(lambda t: datetime.fromtimestamp(t))

    # 表示列（日時だけ）
    columns_to_show = ["agent_id", "chunk_id", "start_dt", "duration"]

    # 出力
    print("\n[INFO] Time range per chunk (human-readable):")
    print(summary[columns_to_show])

    # 各チャンクの開始時刻の不一致補正（オーバーフローで未来に飛んだチャンクを戻す）
    df_all = correct_chunk_start_times(df_all)


    # agent_id==99のデータを分離
    df_99 = df_all[df_all["agent_id"] == 99].copy()
    df_main = df_all[df_all["agent_id"] != 99]

    # a1が170以上の時は-255する
    #if not df_99.empty:
    #    df_99.loc[df_99["a1"] >= 170, "a1"] -= 255

    # 新しい時系列を定義 (100Hz)
    min_time = df_all["time_pc_sec_abs"].min()
    max_time = df_all["time_pc_sec_abs"].max()

    for agent_id, sub in df_all.groupby("agent_id"):
        sub = sub.sort_values("time_pc_sec_abs")
        min_time = max(min_time, sub["time_pc_sec_abs"].min())
        max_time = min(max_time, sub["time_pc_sec_abs"].max())

    if min_time >= max_time:
        print(f"[INFO] No overlapping time range for agents. min_time={min_time}, max_time={max_time}")
        return

    new_time_series = np.arange(min_time, max_time, 0.01) - min_time  # 最小値を基準にシフト

    # 線形補間で位相データを再定義（99以外のみ）
    interpolated_data = {}
    for agent_id, sub in df_main.groupby("agent_id"):
        sub = sub.sort_values("time_pc_sec_abs")
        sub["a0"] = correct_phase_discontinuity(sub["a0"].values)
        interpolated_data[agent_id] = {
            "time": new_time_series,
            "a0": np.interp(new_time_series + min_time, sub["time_pc_sec_abs"], sub["a0"])
        }

    # 基準エージェントの選択
    base_agent_id = min(interpolated_data.keys())
    base_agent_a0 = interpolated_data[base_agent_id]["a0"]

    if not df_99.empty:
        # サブプロットを4段＋カラーバー用1列に
        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(2, 2, width_ratios=[20, 1], wspace=0.3)
        # 1列目の4つのAxesをx軸共有で作成
        axs = []
        for i in range(2):
            if i == 0:
                ax = fig.add_subplot(gs[i, 0])
            else:
                ax = fig.add_subplot(gs[i, 0], sharex=axs[0])
            axs.append(ax)
        colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

        # 相対位相差プロット（99以外）
        for agent_id, data in interpolated_data.items():
            if agent_id == base_agent_id:
                continue
            phase_diff = (data["a0"] - base_agent_a0 + 128) % 256 - 128
            phase_diff_with_nan = phase_diff.copy()
            for i in range(1, len(phase_diff)):
                if abs(phase_diff[i] - phase_diff[i - 1]) > 128:
                    phase_diff_with_nan[i] = np.nan
            phase_diff_with_nan = phase_diff_with_nan * (2 * np.pi / 256)
            axs[0].plot(data["time"], phase_diff_with_nan, label=f"Agent {agent_id} - Agent {base_agent_id}", color=next(colors))

        axs[0].set_ylim(-np.pi, np.pi)
        axs[0].yaxis.set_major_locator(ticker.MultipleLocator(base=np.pi / 2))
        axs[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x / np.pi)}π" if x % np.pi == 0 else f"{x / np.pi:.1f}π"))
        axs[0].set_ylabel("Phase Diff (radians)")
        axs[0].legend(title="Relative Phase")
        axs[0].grid(True)

        # agent_id==99 の a0, a1 を下段にプロット
        axs[1].plot(df_99["time_pc_sec_abs"] - min_time, df_99["a0"], label="Agent 99 a0", color="tab:blue")
        axs[1].plot(df_99["time_pc_sec_abs"] - min_time, df_99["a1"], label="Agent 99 a1", color="tab:orange")
        axs[1].set_ylabel("Agent99\na0/a1")
        axs[1].legend()
        axs[1].grid(True)

        
        plt.tight_layout()
        show_plot_and_cleanup()
    else:
        # 99がない場合は普通に1段で表示
        plt.figure(figsize=(9, 4))
        colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        for agent_id, data in interpolated_data.items():
            if agent_id == base_agent_id:
                continue
            phase_diff = (data["a0"] - base_agent_a0 + 128) % 256 - 128
            phase_diff_with_nan = phase_diff.copy()
            for i in range(1, len(phase_diff)):
                if abs(phase_diff[i] - phase_diff[i - 1]) > 128:
                    phase_diff_with_nan[i] = np.nan
            phase_diff_with_nan = phase_diff_with_nan * (2 * np.pi / 256)
            plt.plot(data["time"], phase_diff_with_nan, label=f"Agent {agent_id} - Agent {base_agent_id}", color=next(colors))

        plt.ylim(-np.pi, np.pi)
        plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(base=np.pi / 2))
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x / np.pi)}π" if x % np.pi == 0 else f"{x / np.pi:.1f}π"))
        plt.ylabel("Phase Diff (radians)")
        plt.xlabel("Time (s)")
        plt.legend(title="Relative Phase")
        plt.grid(True)
        plt.xlim(0, new_time_series[-1])
        plt.tight_layout()
        show_plot_and_cleanup()


def detect_time_anomalies(df, threshold_sec=1.0):
    """
    同一チャンク内での時間逆行やジャンプを検出し、ジャンプ秒数も表示。
    """
    for (agent_id, chunk_id), sub in df.groupby(["agent_id", "chunk_id"]):
        sub = sub.sort_values("time_pc_sec_abs").reset_index(drop=True)
        time_diff = sub["time_pc_sec_abs"].diff().fillna(0)

        # 時間が戻った点（diff < 0）
        backward_jumps = sub[time_diff < 0]
        if not backward_jumps.empty:
            print(f"[WARN] Time reversed in agent {agent_id}, chunk {chunk_id}:")
            for idx in backward_jumps.index:
                t_prev = sub.loc[idx - 1, "time_pc_sec_abs"] if idx > 0 else None
                t_curr = sub.loc[idx, "time_pc_sec_abs"]
                print(f"  At index {idx}: {t_prev:.6f} → {t_curr:.6f} (Δ = {t_curr - t_prev:.6f} sec)")

        # 時間ジャンプ（diff > threshold）
        large_jump_indices = sub.index[time_diff > threshold_sec]
        if len(large_jump_indices) > 0:
            print(f"[INFO] Large time jump (> {threshold_sec:.1f}s) in agent {agent_id}, chunk {chunk_id}:")
            for idx in large_jump_indices:
                t_prev = sub.loc[idx - 1, "time_pc_sec_abs"]
                t_curr = sub.loc[idx, "time_pc_sec_abs"]
                delta = t_curr - t_prev
                print(f"  At index {idx}: {t_prev:.6f} → {t_curr:.6f} (Δ = {delta:.6f} sec)")


# 時間ジャンプが大きすぎる場合に補正（例：4294秒前後のジャンプなら修正）
T_OVERFLOW = 2**32 / 1e6  # 約4294.967296秒
T_TOL = 5.0  # 許容誤差（秒）

def correct_large_jump(sub, threshold_sec=T_OVERFLOW - T_TOL, jump_sec=T_OVERFLOW):
    sub = sub.copy()
    time_diff = sub["time_pc_sec_abs"].diff().fillna(0)
    jump_idx = sub.index[time_diff > threshold_sec]
    for idx in jump_idx:
        sub.loc[idx:, "time_pc_sec_abs"] -= jump_sec
        print(f"[FIX] Corrected overflow at index {idx}, subtracted {jump_sec} sec.")
    return sub

def correct_chunk_start_times(df, threshold_sec=4000.0, jump_sec=4294.967296):
    """
    各チャンクの開始時刻を比較し、極端に未来のタイムスタンプがあればジャンプ分だけ補正。
    """
    corrected_chunks = []
    chunk_starts = df.groupby(["agent_id", "chunk_id"])["time_pc_sec_abs"].min().reset_index()
    median_start = chunk_starts["time_pc_sec_abs"].median()

    for (agent_id, chunk_id), sub in df.groupby(["agent_id", "chunk_id"]):
        start_time = sub["time_pc_sec_abs"].min()
        if start_time - median_start > threshold_sec:
            print(f"[FIX] Corrected chunk time for agent {agent_id}, chunk {chunk_id}: {start_time:.3f} → {start_time - jump_sec:.3f}")
            sub = sub.copy()
            sub["time_pc_sec_abs"] -= jump_sec
        corrected_chunks.append(sub)

    return pd.concat(corrected_chunks, ignore_index=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[INFO] Usage: python Plotter.py <csv-file> [<csv-file> ...]")
    else:
        plot_chunks(sys.argv[1:])
