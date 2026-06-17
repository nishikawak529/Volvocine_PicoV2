import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
import os
import numpy as np
import matplotlib.ticker as ticker  # 目盛りのフォーマット用
from datetime import datetime
import matplotlib.gridspec as gridspec  # 追加

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
    detect_time_anomalies(df_all)

    # agent_id==99のデータを分離
    df_99 = df_all[df_all["agent_id"] == 99].copy()
    df_main = df_all[df_all["agent_id"] != 99]

    # a1が170以上の時は-255する
    if not df_99.empty:
        df_99.loc[df_99["a1"] >= 170, "a1"] -= 255

    # サブプロットを4段に
    fig, axs = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
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
    axs[2].set_xlabel("PC time (sec)")
    for ax in axs[:3]:
        ax.grid(True)

    handles = [plt.Line2D([0], [0], color=color, lw=2, label=f"Agent {ag_id}")
               for ag_id, color in colors.items()]
    axs[0].legend(handles=handles, title="Agents")

    # agent_id==99 の a0, a1 を下段に重ねてプロット
    if not df_99.empty:
        axs[3].plot(df_99["time_pc_sec_abs"], df_99["a0"], label="Agent 99 a0", color="tab:blue")
        axs[3].plot(df_99["time_pc_sec_abs"], df_99["a1"], label="Agent 99 a1", color="tab:orange")
        axs[3].set_ylabel("Agent99\na0/a1")
        axs[3].legend()
        axs[3].grid(True)
    else:
        axs[3].set_visible(False)

    axs[3].set_xlabel("PC time (sec)")

    # agent_id==99 の a0, a1 を下段に重ねてプロット
    if not df_99.empty:
        axs[3].plot(df_99["time_pc_sec_abs"], df_99["a0"], label="Agent 99 a0", color="tab:blue")
        axs[3].plot(df_99["time_pc_sec_abs"], df_99["a1"], label="Agent 99 a1", color="tab:orange")
        axs[3].set_ylabel("Agent99\na0/a1")
        axs[3].legend()
        axs[3].grid(True)
    else:
        axs[3].set_visible(False)

    axs[3].set_xlabel("PC time (sec)")

    plt.tight_layout()
    plt.show()

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
        plt.show()
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
        plt.show()


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