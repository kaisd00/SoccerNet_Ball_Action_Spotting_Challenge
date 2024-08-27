import cv2
import numpy as np
import time

# 実行前の時間を記録
start_time = time.time()

# 既に二値化されているロゴマスクを読み込む
logo_mask = cv2.imread(r'', cv2.IMREAD_GRAYSCALE)

# 動画ファイルを読み込む
cap = cv2.VideoCapture(r'')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 出力用の動画ファイル
out = cv2.VideoWriter(r'', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# 閾値とリプレイ判定後に無視するフレーム数
threshold = 4000000  # 画素値の総和の閾値
ignore_frames = 50  # ロゴ検出後50フレームを無視

# 監督や観客席検出用の閾値
green_threshold = 100000  # 緑色成分の総和の閾値

# リプレイ判定状態とタイマーの初期化
in_replay = False
in_coach = False  # 監督映像検出状態の初期化

frames_since_last_detection = ignore_frames  # 初期状態でリプレイ検出が可能な状態に設定

current_frame_number = 0  # 現在のフレーム番号

# リプレイ開始と終了フレームを記録するリスト
replay_start_end_frames = []
coach_start_end_frames = []  # 監督や観客席が映っている区間を記録するリスト

# リプレイ表示の最大フレーム数 (30秒 * 25fps = 750フレーム)
max_replay_frames = 750

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 現在のフレーム番号を更新
    current_frame_number += 1

    # フレーム全体をグレースケールに変換
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 二値化処理
    _, binary_frame = cv2.threshold(gray_frame, 245, 255, cv2.THRESH_BINARY)

    # マスク画像とフレームの乗算
    masked_frame = cv2.multiply(binary_frame, logo_mask)

    # 乗算結果の画素値の総和を調べる
    masked_sum = np.sum(masked_frame)

    # HSI色空間への変換
    hsi_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, i = cv2.split(hsi_frame)

    # 緑色成分の検出 (H成分で緑色範囲を抽出)
    green_mask = (h >= 35) & (h <= 85)  # 35°から85°までが緑色の範囲
    green_sum = np.sum(green_mask)

    # 監督や観客席が映っていると推定された場合にのみ表示
    if green_sum < green_threshold:
        if not in_coach:
            in_coach = True
            coach_start_end_frames.append((current_frame_number, None))  # 開始フレームを記録
            print(f"監督や観客席の映像開始: フレーム {current_frame_number}")
        cv2.putText(frame, "COACH/SEATS", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 3, cv2.LINE_AA)
    else:
        if in_coach:
            in_coach = False
            coach_start_end_frames[-1] = (coach_start_end_frames[-1][0], current_frame_number)  # 終了フレームを記録
            print(f"監督や観客席の映像終了: フレーム {current_frame_number}")

    if masked_sum > threshold and frames_since_last_detection >= ignore_frames:
        if not in_replay:
            in_replay = True
            print(f"リプレイ開始: フレーム {current_frame_number}")
            replay_start_end_frames.append((current_frame_number, None))  # 開始フレームを記録
            frames_since_last_detection = 0  # フレームカウントをリセット

    if in_replay:
        # リプレイ中のフレームに「REPLAY」の文字を追加
        cv2.putText(frame, "REPLAY", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 3, cv2.LINE_AA)
        frames_since_last_detection += 1
        if masked_sum > threshold and frames_since_last_detection > ignore_frames or frames_since_last_detection > max_replay_frames:
            print(f"リプレイ終了: フレーム {current_frame_number}")
            replay_start_end_frames[-1] = (replay_start_end_frames[-1][0], current_frame_number)  # 終了フレームを記録
            in_replay = False
            frames_since_last_detection = 0  # フレームカウントをリセット

    # 動画にフレームを書き込む（リプレイ中も含む）
    out.write(frame)

    if not in_replay:
        # フレームカウントを進める
        frames_since_last_detection += 1

cap.release()
out.release()

# 実行後の時間を記録
end_time = time.time()

# リプレイ開始と終了フレームを出力
for start, end in replay_start_end_frames:
    print(f"リプレイ区間: 開始フレーム {start}, 終了フレーム {end}")

# 監督や観客席が映っている区間を出力
for start, end in coach_start_end_frames:
    print(f"監督や観客席区間: 開始フレーム {start}, 終了フレーム {end}")

print("動画の処理が完了しました。")

# 実行時間を計算
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
