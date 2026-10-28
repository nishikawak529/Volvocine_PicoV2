#!/usr/bin/env python3
"""
clean_robot_photo.py

RobotPhotoに格納されたロボットの写真から、背景およびロボットを持っている手をきれいに切り出し、
透明背景（アルファチャンネル付き透過PNG）として保存するスクリプト。

主な処理手順:
1. 白背景領域（ピュアホワイト [255, 255, 255] およびその連結成分）の特定
2. 人間の肌色スコアリング（YCrCb色空間 + BGR色差）による手の領域の特定
3. ロボット部品（黒い球体・フレーム、白いセンターバー、青いサーボ、配線等）の確実な保護
4. OpenCVのGrabCut（グラフカット法）を用いたエッジ適合型高精度境界抽出
5. アルファ境界のアンチエイリアス処理（滑らかな透過境界）
6. 透過PNG（RGBA）および確認用プレビュー（市松模様合成）の出力
"""

import os
import sys
import argparse
import cv2
import numpy as np


def compute_skin_score(img_bgr: np.ndarray) -> np.ndarray:
    """
    YCrCb 色空間と BGR 色差を組み合わせた肌色スコアを計算する。
    人間の肌色で高い正の値をとり、ロボットの無彩色（黒・白・グレー）や青サーボでは負値をとる。
    """
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1].astype(float)
    cb = ycrcb[:, :, 2].astype(float)
    b = img_bgr[:, :, 0].astype(float)
    g = img_bgr[:, :, 1].astype(float)
    r = img_bgr[:, :, 2].astype(float)

    # 肌色は R > G > B, Cr ≈ 135~170, Cb ≈ 80~125
    skin_score = (r - b) + (r - g) + 2.0 * (cr - 128.0) - 2.0 * np.abs(cb - 110.0)
    return skin_score


def segment_robot_7071(img: np.ndarray) -> np.ndarray:
    """
    IMG_7071（正面対称アングル）専用のセグメンテーション処理。
    """
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b = img[:, :, 0].astype(float)
    g = img[:, :, 1].astype(float)
    r = img[:, :, 2].astype(float)
    skin_score = compute_skin_score(img)

    # 1. 白背景（外周または大きな白連結成分）
    is_pure_white = np.all(img == 255, axis=-1).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(is_pure_white, connectivity=8)
    border_labels = (
        set(np.unique(labels[0, :]))
        | set(np.unique(labels[-1, :]))
        | set(np.unique(labels[:, 0]))
        | set(np.unique(labels[:, -1]))
    )
    border_labels.discard(0)

    mask = np.full((h, w), cv2.GC_PR_FGD, dtype=np.uint8)
    for lbl in range(1, num_labels):
        if lbl in border_labels or stats[lbl, cv2.CC_STAT_AREA] > 100:
            mask[labels == lbl] = cv2.GC_BGD

    # 2. ロボット本体のシード（確実な前景）
    # 黒い球体・樹脂フレーム（暗く、肌色スコアが低い）
    safe_dark = (gray < 70) & (skin_score < 5) & (mask != cv2.GC_BGD)
    mask[safe_dark] = cv2.GC_FGD

    # 白いセンターバー内部
    safe_white_bar = (
        (np.arange(h)[:, None] > 300)
        & (np.arange(h)[:, None] < 1690)
        & (np.arange(w)[None, :] > 1100)
        & (np.arange(w)[None, :] < 1380)
        & (mask != cv2.GC_BGD)
    )
    mask[safe_white_bar] = cv2.GC_FGD

    # 青いサーボモーター
    is_blue = (b > 100) & (b > r * 1.3) & (mask != cv2.GC_BGD)
    mask[is_blue] = cv2.GC_FGD

    # 赤い配線
    is_red_wire = (r > 120) & (r / (g + 1e-5) > 1.8) & (mask != cv2.GC_BGD)
    mask[is_red_wire] = cv2.GC_FGD

    # 3. 手のシード（確実な背景）
    hand_seed = (skin_score > 80) & (np.arange(h)[:, None] > 1730) & (mask != cv2.GC_BGD)
    mask[hand_seed] = cv2.GC_BGD

    # センターバー下端（y > 1705）の左右球体の間の隙間
    bottom_gap = (
        (np.arange(h)[:, None] > 1705)
        & (np.arange(w)[None, :] > 1100)
        & (np.arange(w)[None, :] < 1400)
    )
    mask[bottom_gap & (mask == cv2.GC_PR_FGD)] = cv2.GC_PR_BGD

    # 4. GrabCut 最適化
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)

    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # 微小ノイズ除去
    num_l, l_map, s_map, _ = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
    for i in range(1, num_l):
        if s_map[i, cv2.CC_STAT_AREA] < 100:
            fg_mask[l_map == i] = 0

    return fg_mask


def segment_robot_7073(img: np.ndarray) -> np.ndarray:
    """
    IMG_7073（斜めアングル）専用のセグメンテーション処理。
    """
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b = img[:, :, 0].astype(float)
    g = img[:, :, 1].astype(float)
    r = img[:, :, 2].astype(float)
    skin_score = compute_skin_score(img)

    # 1. 白背景
    is_pure_white = np.all(img == 255, axis=-1).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(is_pure_white, connectivity=8)
    border_labels = (
        set(np.unique(labels[0, :]))
        | set(np.unique(labels[-1, :]))
        | set(np.unique(labels[:, 0]))
        | set(np.unique(labels[:, -1]))
    )
    border_labels.discard(0)

    mask = np.full((h, w), cv2.GC_PR_FGD, dtype=np.uint8)
    for lbl in range(1, num_labels):
        if lbl in border_labels or stats[lbl, cv2.CC_STAT_AREA] > 50:
            mask[labels == lbl] = cv2.GC_BGD

    # 2. ロボット本体のシード（確実な前景）
    safe_dark = (gray < 70) & (skin_score < 0) & (mask != cv2.GC_BGD)
    mask[safe_dark] = cv2.GC_FGD

    # 白い三角柱バー内部
    safe_white_bar = (
        (np.arange(h)[:, None] > 150)
        & (np.arange(h)[:, None] < 450)
        & (np.arange(w)[None, :] > 920)
        & (np.arange(w)[None, :] < 1180)
        & (mask != cv2.GC_BGD)
    )
    mask[safe_white_bar] = cv2.GC_FGD

    # 青いサーボ
    is_blue = (b > 90) & (b > r * 1.3) & (mask != cv2.GC_BGD)
    mask[is_blue] = cv2.GC_FGD

    # 赤・黄の配線
    is_red_wire = (r > 120) & (r / (g + 1e-5) > 1.9) & (mask != cv2.GC_BGD)
    mask[is_red_wire] = cv2.GC_FGD
    is_yellow_wire = (r > 140) & (g > 120) & (b < 80) & (mask != cv2.GC_BGD)
    mask[is_yellow_wire] = cv2.GC_FGD

    # 球体表面の配線引き込み部（グルー接着部）の保護
    glue_area = (
        (np.arange(h)[:, None] >= 800)
        & (np.arange(h)[:, None] <= 920)
        & (np.arange(w)[None, :] >= 1200)
        & (np.arange(w)[None, :] <= 1450)
        & (mask != cv2.GC_BGD)
    )
    mask[glue_area] = cv2.GC_FGD

    # 3. 手・腕のシード（確実な背景）
    hand_core = (
        (skin_score > 50)
        & (r / (g + 1e-5) < 1.85)
        & (r / (b + 1e-5) < 2.8)
        & (np.arange(h)[:, None] > 850)
        & (np.arange(w)[None, :] > 920)
        & (np.arange(w)[None, :] < 1650)
        & (~glue_area)
        & (mask != cv2.GC_BGD)
    )
    mask[hand_core] = cv2.GC_BGD

    # 球体間の微小な隙間に見える指先
    gap_seed = (
        (skin_score > 20)
        & (np.arange(h)[:, None] >= 720)
        & (np.arange(h)[:, None] <= 780)
        & (np.arange(w)[None, :] >= 1000)
        & (np.arange(w)[None, :] <= 1030)
    )
    mask[gap_seed] = cv2.GC_BGD

    # 腕が下方に伸びる領域（おそらく背景）
    hand_zone = (
        (np.arange(h)[:, None] > 700)
        & (np.arange(w)[None, :] > 880)
        & (np.arange(w)[None, :] < 1650)
        & (gray > 50)
        & (mask == cv2.GC_PR_FGD)
        & (skin_score > 10)
        & (r / (g + 1e-5) < 1.85)
        & (~glue_area)
    )
    mask[hand_zone] = cv2.GC_PR_BGD

    # 4. GrabCut 最適化
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)

    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # 微小ノイズ除去
    num_l, l_map, s_map, _ = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
    for i in range(1, num_l):
        if s_map[i, cv2.CC_STAT_AREA] < 100:
            fg_mask[l_map == i] = 0

    return fg_mask


def process_image(img_path: str, output_path: str, preview_path: str = None) -> None:
    """
    1枚の画像を読み込み、手と白背景を除去して透過PNGとして保存する。
    """
    print(f"Reading image: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image at: {img_path}")

    # 画像アスペクト比またはファイル名により適切な処理を選択
    filename = os.path.basename(img_path).upper()
    h, w, _ = img.shape
    aspect = w / h

    if "7071" in filename or aspect < 1.4:
        print("  -> Applying IMG_7071 pipeline (front view)")
        fg_mask = segment_robot_7071(img)
    else:
        print("  -> Applying IMG_7073 pipeline (angled view)")
        fg_mask = segment_robot_7073(img)

    # アンチエイリアス処理（滑らかな透過境界）
    alpha = cv2.GaussianBlur(fg_mask.astype(float), (3, 3), 0.5)
    alpha = np.clip(alpha, 0, 255).astype(np.uint8)

    # RGBA画像の作成
    out_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    out_rgba[:, :, 3] = alpha

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, out_rgba)
    print(f"  -> Saved clean transparent PNG: {output_path}")

    # 確認用プレビュー（市松模様合成）の出力
    if preview_path:
        square_size = max(20, min(h, w) // 50)
        grid_y, grid_x = np.indices((h, w))
        checker = ((grid_y // square_size) + (grid_x // square_size)) % 2
        bg_checker = np.where(checker[:, :, None] == 0, 200, 255).astype(np.uint8)
        bg_checker = np.repeat(bg_checker, 3, axis=2)

        alpha_norm = alpha.astype(float)[:, :, None] / 255.0
        blended = (img.astype(float) * alpha_norm + bg_checker.astype(float) * (1.0 - alpha_norm)).astype(np.uint8)
        cv2.imwrite(preview_path, blended)
        print(f"  -> Saved checkerboard preview: {preview_path}")


def main():
    parser = argparse.ArgumentParser(description="RobotPhotoから手と背景を除去して透過PNGを作成するツール")
    parser.add_argument("--input", "-i", type=str, help="入力画像パス（省略時はRobotPhoto内のIMG_7071と7073を処理）")
    parser.add_argument("--output", "-o", type=str, help="出力先透過PNGパス")
    parser.add_argument("--preview", "-p", action="store_true", help="市松模様背景の確認用プレビュー画像を生成する")
    args = parser.parse_args()

    # スクリプトのディレクトリ
    script_dir = os.path.dirname(os.path.abspath(__file__))
    robot_photo_dir = os.path.join(script_dir, "RobotPhoto")

    if args.input:
        out_path = args.output or os.path.splitext(args.input)[0] + "_clean.png"
        prev_path = os.path.splitext(args.input)[0] + "_preview.png" if args.preview else None
        process_image(args.input, out_path, prev_path)
    else:
        # デフォルト: RobotPhoto 内の対象画像を処理
        targets = [
            ("IMG_7071.PNG", "IMG_7071_clean.png", "IMG_7071_preview.png"),
            ("IMG_7073.PNG", "IMG_7073_clean.png", "IMG_7073_preview.png"),
        ]
        for src_name, dst_name, prev_name in targets:
            src_path = os.path.join(robot_photo_dir, src_name)
            dst_path = os.path.join(robot_photo_dir, dst_name)
            prev_path = os.path.join(robot_photo_dir, prev_name)

            if os.path.exists(src_path):
                process_image(src_path, dst_path, prev_path)
            else:
                print(f"Warning: {src_path} not found.")


if __name__ == "__main__":
    main()
