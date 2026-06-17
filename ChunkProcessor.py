import os
import pandas as pd
from datetime import datetime

SAVE_FOLDER = "saved_chunks"  # 保存用フォルダ名

# 保存用フォルダを作成（存在しない場合のみ）
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

def build_dataframe_for_chunk(agent_id, chunk_data, chunk_send_micros, chunk_recv_times):
    if not chunk_data:
        return None, None  # データがない場合は None を返す

    # 平均オフセットを送信時刻の下位24ビット再構成で算出
    wrapped_send_secs = [(((s >> 8) % 16777216) << 8) / 1e6 for s in chunk_send_micros]
    offsets = [recv - send for send, recv in zip(wrapped_send_secs, chunk_recv_times)]
    offset = sum(offsets) / len(offsets)
    print(len(offsets), f"Average offset for agent {agent_id}: {offset:.6f} seconds")

    df = pd.DataFrame(chunk_data, columns=["micros24", "a0", "a1", "a2"])
    
    micros_list = df["micros24"].tolist()
    extended = [0] * len(micros_list)
    wrap_offset = 0
    prev = micros_list[0]
    extended[0] = prev
    for i in range(1, len(micros_list)):
        curr = micros_list[i]
        if curr < prev:
            wrap_offset += 16777216  # 24ビットのオーバーフローを補正
        extended[i] = curr + wrap_offset
        prev = curr

    df["micros32"] = extended
    df["micros32_raw"] = [val << 8 for val in extended]  # 24ビットを32ビットに拡張
    df["time_local_sec"] = [val / 1e6 for val in df["micros32_raw"]]
    df["time_pc_sec_abs"] = df["time_local_sec"] + offset

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chunk_id = timestamp
    df["agent_id"] = agent_id
    df["chunk_id"] = chunk_id

    # 保存先を保存用フォルダに変更
    filename = os.path.join(SAVE_FOLDER, f"chunk_agent_{agent_id}_{timestamp}.csv")
    save_columns = [
        "time_pc_sec_abs", "micros32", "micros32_raw", "time_local_sec",
        "a0", "a1", "a2", "agent_id", "chunk_id"
    ]
    df.to_csv(filename, index=False, columns=save_columns)
    print(f"[INFO] Agent={agent_id}, chunk size={len(df)} -> Saved to {filename}")

    # ファイルパスも返す
    return df[["agent_id", "chunk_id", "time_pc_sec_abs", "a0", "a1", "a2"]], filename

def merge_and_save_chunks(chunk_files):
    # チャンクファイルが存在しない場合は処理をスキップ
    if not chunk_files:
        print("[INFO] No chunk files provided. Skipping merge process.")
        return None

    # マージデータ専用フォルダの作成
    merged_folder = os.path.join(".", "merged_chunks")
    if not os.path.exists(merged_folder):
        os.makedirs(merged_folder)

    # マージ処理
    merged_data = pd.DataFrame()
    for file in chunk_files:
        df = pd.read_csv(file)
        merged_data = pd.concat([merged_data, df], ignore_index=True)

    # 保存先ファイル名の生成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_file = os.path.join(merged_folder, f"merged_{timestamp}.csv")

    # マージデータを保存
    merged_data.to_csv(merged_file, index=False)
    print(f"[INFO] Merged data saved to {merged_file}")

    # チャンクファイルを削除
    for file in chunk_files:
        try:
            os.remove(file)
            print(f"[INFO] Deleted chunk file: {file}")
        except OSError as e:
            print(f"[ERROR] Could not delete file {file}: {e}")

    return merged_file