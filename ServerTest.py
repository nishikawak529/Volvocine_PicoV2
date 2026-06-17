import socket
import struct
import time
import threading
import queue
from keyinput import check_key
import os  # フォルダ作成用にosモジュールをインポート
import subprocess
import sys
from ServerResponse import handle_handshake, handle_parameter_request  # 新しいモジュールをインポート
from ChunkProcessor import build_dataframe_for_chunk, merge_and_save_chunks  # 新しいモジュールをインポート


# ---------------------------
# 設定
# ---------------------------
UDP_PORT = 5000
BUFFER_SIZE = 1024
SOCKET_TIMEOUT = 1.0
CHUNK_TIMEOUT = 5.0


STRUCT_FORMAT = "<6B"  # micros24 (3バイト), a0, a1, a2
RECORD_SIZE = struct.calcsize(STRUCT_FORMAT)  # 6 bytes

SAVE_FOLDER = "saved_chunks"  # 保存用フォルダ名
MERGED_BASE_FOLDER = "merged_chunks_organized"  # マージファイル保存用ベースフォルダ

# ログレベル設定
DEBUG_LOGS = False  # Trueにするとデバッグログが表示される
TIMEOUT_LOGS = False  # Trueにするとタイムアウトログが詳細表示される
# 必要に応じて上記の設定をTrueに変更してください

# 保存用フォルダを作成（存在しない場合のみ）
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

if not os.path.exists(MERGED_BASE_FOLDER):
    os.makedirs(MERGED_BASE_FOLDER)

agent_buffers = {}  # agent_id -> (chunk_data, send_micros_list, recv_time_list)
agent_lastrecv_time = {}
current_chunk_files = []

# グローバル変数として追加
agent_addrs = {}  # agent_id -> addr

# スレッド関連の変数
agent_queues = {}  # agent_id -> queue.Queue
agent_threads = {}  # agent_id -> threading.Thread
agent_locks = {}  # agent_id -> threading.Lock
agent_sockets = {}  # agent_id -> socket object
shutdown_event = threading.Event()  # シャットダウン用イベント
chunk_files_lock = threading.Lock()  # current_chunk_files用のロック
main_socket = None  # メインソケット（グローバルで管理）


def launch_plotter(file_path):
    """Tk/matplotlibをサーバープロセスから分離してプロットする。"""
    if not file_path:
        return

    plotter_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Plotter.py")
    try:
        subprocess.Popen([sys.executable, plotter_path, file_path])
        print(f"[INFO] Plotter launched for: {file_path}")
    except Exception as e:
        print(f"[WARN] Failed to launch plotter for {file_path}: {e}")


# ---------------------------
# チャンク処理
# ---------------------------
def is_valid_log_packet(data):
    # ダミー: agent_id==0 かつ payloadがない（最低1レコード＝5バイト未満）
    return len(data) >= 10 and data[0] != 0

def send_control_command(sock, addr, cmd):
    sock.sendto(cmd.encode(), addr)

def create_agent_socket(agent_id):
    """エージェント専用のソケットを作成"""
    try:
        agent_port = UDP_PORT + agent_id
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", agent_port))
        sock.settimeout(SOCKET_TIMEOUT)
        agent_sockets[agent_id] = sock
        if DEBUG_LOGS:
            print(f"[DEBUG] Created socket for agent {agent_id} on port {agent_port}")
        return sock
    except Exception as e:
        print(f"[ERROR] Failed to create socket for agent {agent_id}: {e}")
        return None

def agent_communication_worker(agent_id):
    """エージェント専用の通信ワーカー"""
    sock = create_agent_socket(agent_id)
    if sock is None:
        return
        
    if DEBUG_LOGS:
        print(f"[DEBUG] Started communication worker for agent {agent_id}")
    
    while not shutdown_event.is_set():
        try:
            data, addr = sock.recvfrom(BUFFER_SIZE)
            recv_time = time.time()
            
            # パラメータリクエストの処理
            if data.startswith(b"REQUEST_PARAMS"):
                returned_agent_id = handle_parameter_request(sock, data, addr)
                if isinstance(returned_agent_id, int):
                    agent_addrs[returned_agent_id] = addr
                else:
                    print(f"[WARN] Invalid agent_id from parameter request: {returned_agent_id}")
                continue

            # ハンドシェイクメッセージの処理
            if data.startswith(b"HELLO"):
                handle_handshake(sock, data, addr)
                continue

            if not is_valid_log_packet(data):
                if DEBUG_LOGS:
                    print(f"[DEBUG] Ignored dummy or malformed packet from {addr}, length={len(data)}")
                continue
            elif len(data) < 5:
                print(f"[WARN] Short packet from {addr}")
                continue

            # データをキューに追加
            packet_data = (data, recv_time, addr, sock)
            agent_queues[agent_id].put(packet_data)
            
        except socket.timeout:
            continue
        except Exception as e:
            if not shutdown_event.is_set():
                print(f"[ERROR] Communication error for agent {agent_id}: {e}")
            continue
    
    # ソケットを閉じる
    try:
        sock.close()
        if DEBUG_LOGS:
            print(f"[DEBUG] Closed socket for agent {agent_id}")
    except:
        pass

# ---------------------------
# メイン受信ループ
# ---------------------------
def main():
    print(f"[INFO] Start listening UDP base port: {UDP_PORT}")
    global main_socket
    
    # メインソケット（ポート5000）も作成して、初期接続やブロードキャスト用に使用
    main_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    main_socket.bind(("0.0.0.0", UDP_PORT))
    main_socket.settimeout(SOCKET_TIMEOUT)

    try:
        while True:
            try:
                # メインソケットでの通信を確認（初期接続用）
                try:
                    data, addr = main_socket.recvfrom(BUFFER_SIZE)
                    recv_time = time.time()

                    # パラメータリクエストの処理
                    if data.startswith(b"REQUEST_PARAMS"):
                        agent_id = handle_parameter_request(main_socket, data, addr)
                        if isinstance(agent_id, int):
                            agent_addrs[agent_id] = addr
                            # エージェント専用スレッドを開始
                            ensure_agent_thread(agent_id)
                        else:
                            print(f"[WARN] Invalid agent_id from parameter request on main socket: {agent_id}")
                        continue

                    # ハンドシェイクメッセージの処理
                    if data.startswith(b"HELLO"):
                        handle_handshake(main_socket, data, addr)
                        continue

                    # 通常のデータパケットの場合、該当エージェントのスレッドを確保
                    if is_valid_log_packet(data) and len(data) >= 5:
                        agent_id = data[0]
                        ensure_agent_thread(agent_id)
                        
                except socket.timeout:
                    pass

            except socket.timeout:
                pass

            key = check_key()
            if key in ('\r', '\n'):
                print("[INFO] Manual chunk flush.")
                # すべてのエージェントのロックを取得してフラッシュ
                for ag_id in list(agent_buffers.keys()):
                    if ag_id in agent_locks:
                        with agent_locks[ag_id]:
                            data, send_list, recv_list = agent_buffers[ag_id]
                            if data:
                                _, saved_file = build_dataframe_for_chunk(ag_id, data, send_list, recv_list)
                                if saved_file:
                                    with chunk_files_lock:
                                        current_chunk_files.append(saved_file)
                            agent_buffers[ag_id] = ([], [], [])
                with chunk_files_lock:
                    merged_path = merge_and_save_chunks_by_date(current_chunk_files.copy())
                    current_chunk_files.clear()
                print("[INFO] Merged and saved chunks by date.")
                if merged_path:
                    launch_plotter(merged_path)
                print("[DEBUG] current_chunk_files cleared.")
            elif key == 's':
                print("[INFO] Sending START command.")
                for i in range(1, 5):
                    for agent_id in agent_addrs:
                        if agent_id in agent_sockets:
                            send_control_command(agent_sockets[agent_id], agent_addrs[agent_id], "START")
                        else:
                            # メインソケットを使用
                            send_control_command(main_socket, agent_addrs[agent_id], "START")

            elif key == 't':
                print("[INFO] Sending STOP command.")
                for i in range(1, 5):
                    for agent_id in agent_addrs:
                        if agent_id in agent_sockets:
                            send_control_command(agent_sockets[agent_id], agent_addrs[agent_id], "STOP")
                        else:
                            # メインソケットを使用
                            send_control_command(main_socket, agent_addrs[agent_id], "STOP")

            elif key == 'c':
                print("[INFO] Sending CALIBRATE command to IMU agent_id=99.")
                if 99 in agent_addrs:
                    if 99 in agent_sockets:
                        send_control_command(agent_sockets[99], agent_addrs[99], "CALIBRATE")
                    else:
                        send_control_command(main_socket, agent_addrs[99], "CALIBRATE")
                else:
                    print("[WARN] IMU agent_id=99 not found.")

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")

    finally:
        print("[INFO] Shutting down server...")
        
        # すべてのエージェントスレッドを終了
        shutdown_all_threads()
        
        # すべてのソケットを閉じる
        if main_socket:
            main_socket.close()
            print("[INFO] Main socket closed.")
        
        for agent_id, sock in agent_sockets.items():
            try:
                sock.close()
                if DEBUG_LOGS:
                    print(f"[DEBUG] Agent {agent_id} socket closed.")
            except:
                pass
        
        print("[INFO] All sockets closed.")

        # 残りのデータを処理
        for ag_id, (data, send_list, recv_list) in agent_buffers.items():
            if data:
                _, saved_file = build_dataframe_for_chunk(ag_id, data, send_list, recv_list)
                if saved_file:
                    with chunk_files_lock:
                        current_chunk_files.append(saved_file)

        with chunk_files_lock:
            if current_chunk_files:
                merged_path = merge_and_save_chunks_by_date(current_chunk_files.copy())
                if merged_path:
                    launch_plotter(merged_path)
            current_chunk_files.clear()
        print("[DEBUG] current_chunk_files cleared.")
        print("[INFO] Exit complete.")

def get_date_folder():
    """現在の日付に基づいてフォルダパスを取得・作成"""
    from datetime import datetime
    
    today = datetime.now().strftime("%Y-%m-%d")
    date_folder = os.path.join(MERGED_BASE_FOLDER, today)
    
    # 日付フォルダを作成（存在しない場合のみ）
    if not os.path.exists(date_folder):
        os.makedirs(date_folder)
        print(f"[INFO] Created date folder: {date_folder}")
    
    return date_folder

def merge_and_save_chunks_by_date(chunk_files):
    """日付ごとのフォルダにマージファイルを保存"""
    if not chunk_files:
        print("[WARN] No chunk files to merge.")
        return None
    
    try:
        # 日付フォルダを取得
        date_folder = get_date_folder()
        
        # ChunkProcessorのmerge_and_save_chunksを呼び出し、
        # 保存先を日付フォルダに変更
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        merged_filename = f"merged_{timestamp}.csv"
        merged_path = os.path.join(date_folder, merged_filename)
        
        # 元のmerge_and_save_chunks関数を使用し、結果を日付フォルダに移動
        temp_merged_path = merge_and_save_chunks(chunk_files)
        
        if temp_merged_path and os.path.exists(temp_merged_path):
            # ファイルを日付フォルダに移動
            import shutil
            shutil.move(temp_merged_path, merged_path)
            print(f"[INFO] Moved merged file to: {merged_path}")
            return merged_path
        else:
            print("[WARN] merge_and_save_chunks returned no valid file.")
            return None
            
    except Exception as e:
        print(f"[ERROR] Failed to merge and save chunks by date: {e}")
        return None

# ---------------------------
# エージェント専用ワーカー関数
# ---------------------------
def agent_worker(agent_id):
    """エージェント専用のワーカースレッド"""
    if DEBUG_LOGS:
        print(f"[DEBUG] Started worker thread for agent {agent_id}")
    
    while not shutdown_event.is_set():
        try:
            # キューからデータを取得（タイムアウト付き）
            data_item = agent_queues[agent_id].get(timeout=1.0)
            
            if data_item is None:  # シャットダウンシグナル
                break
                
            # データ処理
            process_agent_packet(agent_id, data_item)
            
            agent_queues[agent_id].task_done()
            
        except queue.Empty:
            # タイムアウト - チャンクタイムアウトをチェック
            check_chunk_timeout(agent_id)
            continue
        except Exception as e:
            print(f"[ERROR] Error in agent {agent_id} worker: {e}")
            continue
    
    if DEBUG_LOGS:
        print(f"[DEBUG] Worker thread for agent {agent_id} stopped")

def process_agent_packet(agent_id, data_item):
    """エージェントのパケットを処理"""
    data, recv_time, addr, sock = data_item
    
    # データ検証
    if len(data) < 5:
        print(f"[WARN] Short packet from agent {agent_id}")
        return
        
    agent_id_from_packet = data[0]
    send_micros = struct.unpack("<I", data[1:5])[0]
    raw = data[5:]
    
    if len(raw) % RECORD_SIZE != 0:
        print(f"[WARN] Invalid record size from agent {agent_id}")
        return
    
    with agent_locks[agent_id]:
        if agent_id not in agent_buffers:
            agent_buffers[agent_id] = ([], [], [])
            
        chunk_data, send_list, recv_list = agent_buffers[agent_id]
        
        offset_sec = recv_time - ((((send_micros >> 8) % 16777216) << 8) / 1e6)
        if DEBUG_LOGS:
            print(f"[DEBUG] Agent={agent_id}, send_micros={send_micros}, "
                  f"recv_time={recv_time:.6f}, offset_sec={offset_sec:.6f}")
        
        for i in range(len(raw) // RECORD_SIZE):
            record = raw[i*RECORD_SIZE:(i+1)*RECORD_SIZE]
            b0, b1, b2, a0, a1, a2 = struct.unpack(STRUCT_FORMAT, record)
            micros24 = b0 | (b1 << 8) | (b2 << 16)
            micros32 = micros24 & 0xFFFFFF
            chunk_data.append((micros32, a0, a1, a2))
        
        send_list.append(send_micros)
        recv_list.append(recv_time)
        agent_buffers[agent_id] = (chunk_data, send_list, recv_list)
        
        # ACK送信
        if len(raw) >= RECORD_SIZE:
            last_record = raw[-RECORD_SIZE:]
            b0, b1, b2, *_ = struct.unpack(STRUCT_FORMAT, last_record)
            last_micros24 = b0 | (b1 << 8) | (b2 << 16)
            
            ack = bytearray()
            ack.append(agent_id)
            ack += last_micros24.to_bytes(3, 'little')
            sock.sendto(ack, addr)
        
        agent_lastrecv_time[agent_id] = recv_time

def check_chunk_timeout(agent_id):
    """チャンクタイムアウトをチェック"""
    current_time = time.time()
    
    with agent_locks[agent_id]:
        if agent_id in agent_lastrecv_time:
            if (current_time - agent_lastrecv_time[agent_id]) > CHUNK_TIMEOUT:
                # タイムアウト時の処理を実行
                chunk_data, send_list, recv_list = agent_buffers[agent_id]
                if chunk_data:
                    # データがある場合のみログ出力（設定に応じて）
                    if TIMEOUT_LOGS:
                        print(f"[DEBUG] Agent {agent_id} chunk timeout - saving {len(chunk_data)} records.")
                    _, saved_file = build_dataframe_for_chunk(agent_id, chunk_data, send_list, recv_list)
                    if saved_file:
                        with chunk_files_lock:
                            current_chunk_files.append(saved_file)
                agent_buffers[agent_id] = ([], [], [])

def ensure_agent_thread(agent_id):
    """エージェント用のスレッドとキューを確保"""
    if agent_id not in agent_queues:
        agent_queues[agent_id] = queue.Queue()
        agent_locks[agent_id] = threading.Lock()
        
        # データ処理ワーカースレッドを開始
        data_thread = threading.Thread(target=agent_worker, args=(agent_id,), daemon=True)
        data_thread.start()
        
        # 通信ワーカースレッドを開始
        comm_thread = threading.Thread(target=agent_communication_worker, args=(agent_id,), daemon=True)
        comm_thread.start()
        
        # スレッドを辞書に保存（タプルで管理）
        agent_threads[agent_id] = (data_thread, comm_thread)
        
        if DEBUG_LOGS:
            print(f"[DEBUG] Created worker threads for agent {agent_id}")

def shutdown_all_threads():
    """すべてのエージェントスレッドを終了"""
    print("[INFO] Shutting down all agent threads...")
    shutdown_event.set()
    
    # すべてのキューにシャットダウンシグナルを送信
    for agent_id in agent_queues:
        agent_queues[agent_id].put(None)
    
    # すべてのスレッドの終了を待機
    for agent_id, threads in agent_threads.items():
        if isinstance(threads, tuple):
            data_thread, comm_thread = threads
            data_thread.join(timeout=2.0)
            comm_thread.join(timeout=2.0)
            if data_thread.is_alive() or comm_thread.is_alive():
                print(f"[WARN] Agent {agent_id} threads did not stop gracefully")
        else:
            # 後方互換性のため
            threads.join(timeout=2.0)
            if threads.is_alive():
                print(f"[WARN] Agent {agent_id} thread did not stop gracefully")

# ---------------------------

if __name__ == "__main__":
    main()
