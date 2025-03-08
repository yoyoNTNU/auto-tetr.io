import random

import cv2
import numpy as np
import pyautogui
import time

TETROMINO_COLORS = {
    "L": ([10, 0, 0], [19, 255, 255]),  # 橙色
    "O": ([20, 0, 0], [35, 255, 255]),  # 黃色
    "S": ([36, 0, 0], [60, 255, 255]),  # 綠色
    "I": ([70, 0, 0], [100, 255, 255]),  # 青色
    "J": ([105, 0, 0], [125, 255, 255]),  # 藍色
    "T": ([140, 0, 0], [160, 255, 255]),  # 紫色
    "Z": ([170, 0, 0], [200, 255, 255]),  # 紅色
}

TETROMINOS = {
    'S': np.array([[0, 1, 1], [1, 1, 0]]),  # S型
    'Z': np.array([[1, 1, 0], [0, 1, 1]]),  # Z型
    'I': np.array([[1, 1, 1, 1]]),  # I型
    'O': np.array([[1, 1], [1, 1]]),  # O型
    'J': np.array([[1, 0, 0], [1, 1, 1]]),  # J型
    'L': np.array([[0, 0, 1], [1, 1, 1]]),  # L型
    'T': np.array([[0, 1, 0], [1, 1, 1]])   # T型
}


rows, cols = 20, 10
block_size = 34.5


def capture_game_region():
    # 取得遊戲畫面 (座標需要手動調整)
    x, y, width, height = 788, 180, 345, 690  # 這裡的座標需根據你的螢幕調整
    screenshot = pyautogui.screenshot(region=(x, y, width, height))

    # 轉換為 OpenCV 格式
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # cv2.imshow("Captured Region", frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return frame


def capture_current_block_region():
    # 取得遊戲畫面 (座標需要手動調整)
    x, y, width, height = 926, 145, 35, 35  # 這裡的座標需根據你的螢幕調整
    screenshot = pyautogui.screenshot(region=(x, y, width, height))

    # 轉換為 OpenCV 格式
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # cv2.imshow("Captured Region", frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return frame


def capture_next_block_region():
    # 取得遊戲畫面 (座標需要手動調整)
    x, y, width, height = 1240, 260, 20, 20  # 這裡的座標需根據你的螢幕調整
    screenshot = pyautogui.screenshot(region=(x, y, width, height))

    # 轉換為 OpenCV 格式
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # cv2.imshow("Captured Region", frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return frame


def detect_blocks(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)  # 調整閾值以適應 TETR.IO 配色

    # TETR.IO 的 10x20 格子

    board = np.zeros((rows, cols), dtype=int)

    for row in range(0,rows):
        for col in range(cols):
            x, y = int(col * block_size), int(row * block_size)
            cell = binary[y:y + int(block_size), x:x + int(block_size)]  # 確保索引是整數

            if np.mean(cell) >= 200:  # 判斷是否有方塊
                board[row, col] = 1

    return board


def check_color(frame):
    # 計算格子的中心點
    center_x = int(block_size // 2)
    center_y = int(block_size // 2)

    # 擷取該點的顏色
    center_color = frame[center_y, center_x]

    # 轉換為 HSV 顏色空間
    hsv = cv2.cvtColor(np.uint8([[center_color]]), cv2.COLOR_BGR2HSV)
    h, s, v = hsv[0][0]

    # 根據顏色範圍判斷方塊是什麼顏色
    for piece, (lower, upper) in TETROMINO_COLORS.items():
        lower_bound = np.array(lower, dtype=np.uint8)
        upper_bound = np.array(upper, dtype=np.uint8)

        # 判斷顏色是否符合
        if all(lower_bound <= [h, s, v]) and all([h, s, v] <= upper_bound):
            return piece

    return None  # 沒有匹配的顏色


def get_tetrominos(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cell = gray[0:int(block_size), 0:int(block_size)]
    if np.mean(cell) < 200:  # 判斷有方塊（這裡根據像素亮度設定閾值）
        piece = check_color(frame)
        if piece:
            return piece
    return None


def get_tetromino_shapes(tetromino):
    """取得該方塊的所有旋轉形狀"""
    shapes = []
    current_shape = tetromino
    for _ in range(4):
        if current_shape.tolist() not in [s.tolist() for s in shapes]:  # 避免重複形狀
            shapes.append(current_shape)
        current_shape = np.rot90(current_shape)  # 旋轉90度
    return shapes


def get_drop_y(board, tetromino, x):
    """計算方塊從 x 位置開始自由落體的最終 y 位置"""
    max_y = rows- tetromino.shape[0]
    for y in range(max_y + 1):
        if check_collision(board, tetromino, (x, y)):
            return y - 1  # 回到上一個合法的位置
    return max_y  # 到達底部


def check_collision(board, tetromino, position):
    """檢查方塊是否與遊戲區發生碰撞"""
    x, y = position
    r, c = tetromino.shape

    if x < 0 or x + c > cols:  # 超出左右邊界
        return True
    if y + r > rows:  # 超出下邊界
        return True

    # 檢查是否碰撞已存在的方塊
    for i in range(r):
        for j in range(c):
            if tetromino[i, j] == 1 and board[y + i, x + j] == 1:
                return True
    return False


def generate_all_positions(board, tetromino):
    """列出所有可能的 (旋轉狀態, 橫向位置, 落地高度)"""
    s = get_tetromino_shapes(tetromino)
    positions = []

    for i, shape in enumerate(s):
        for x in range(cols - shape.shape[1] + 1):  # 確保不超出邊界
            y = get_drop_y(board, shape, x)
            if y != -1:
                positions.append((shape, x, y, i))  # 存儲 (旋轉狀態, x 座標, y 座標)

    return positions


def calculate_score(board, is_need_pos=True, position=(), is_print_board=False):
    if is_need_pos:
        # 克隆當前的遊戲板
        temp_board = board.copy()
        s, x, y, _ = position

        # 將方塊放置到模擬板上
        for i in range(s.shape[0]):
            for j in range(s.shape[1]):
                if s[i, j] == 1:  # 如果這個位置有方塊
                    # 計算放置位置
                    new_x = x + j
                    new_y = y + i
                    temp_board[new_y, new_x] = 1
    else:
        temp_board = board.copy()

    # 1. 高度懲罰
    height_penalty = 0
    for row in range(0, rows):
        for col in range(0, cols):
            if temp_board[row, col] == 1:
                height_penalty += (rows - row)
                break

    # 3. 計算下方空洞
    hole_area = 0
    for col in range(cols):
        for row in range(rows - 1):
            if temp_board[row, col] == 1:
                k = 1
                while row + k < rows and temp_board[row + k, col] == 0:
                    hole_area += 1
                    k += 1
    hole_penalty = hole_area * 8  # 空洞越多懲罰越大

    base_area = 0
    for col in range(cols):
        for row in range(rows):
            if temp_board[row, col] == 1 and (row == 19 or temp_board[row + 1, col] == 1):
                base_area += 1
    base_bonus = base_area  # 有地基要獎勵

    # 4. 計算即將形成的消行數量
    completed_lines = 0
    for row in range(rows):
        if np.all(temp_board[row, :] == 1):
            completed_lines += 1
    line_clear_bonus = completed_lines * 15 - 5  # 清除行的獎勳

    # 結合所有標準：得分 = 高度懲罰 + 缺口懲罰 + 表面懲罰 - 消行獎勳
    score = - height_penalty - hole_penalty + line_clear_bonus + base_bonus
    if is_print_board:
        print(temp_board)

    return score, temp_board


def find_best_position(board, tetromino, next_tetromino=[]):
    max_score = -100000
    pos = generate_all_positions(board, tetromino)
    if len(pos) == 0:
        return None
    best_pos = pos[0]
    for p in pos:
        s, temp = calculate_score(board, True, p)
        if len(next_tetromino) != 0:
            next_pos = generate_all_positions(temp, next_tetromino)
            for next_p in next_pos:
                ss, _ = calculate_score(temp, True, next_p)
                if ss > max_score:
                    max_score = ss
                    best_pos = p
                elif ss == max_score:
                    x = int(random.random() * 1523)
                    if x % 2 == 1:
                        best_pos = p
        else:
            if s > max_score:
                max_score = s
                best_pos = p
            elif s == max_score:
                x = int(random.random() * 1523)
                if x % 2 == 1:
                    best_pos = p
    # print("choose score:", calculate_score(board,True, best_pos, True))
    return best_pos


def simulate_move(position):

    _, target_x, target_y,  rotate_count = position
    # 旋轉
    c = 0
    while c < rotate_count:
        pyautogui.press('z')
        time.sleep(0.1)
        c += 1
    c = 0
    while c < 10:
        pyautogui.press('left')
        c += 1
    current_x = 0

    # 右移，直到達到目標位置
    while current_x < target_x:
        pyautogui.press('right')  # 持續按下右箭頭
        time.sleep(0.1)  # 等待一會
        current_x += 1  # 模擬右移一格

    time.sleep(0.1)
    # 使用空白鍵直接下落到底部
    pyautogui.press('space')  # 模擬空白鍵快速下落

see = 2
while True:
    # 捕捉當前遊戲畫面
    f = capture_game_region()

    # 將畫面轉換為佔用矩陣
    b = detect_blocks(f)
    fb = capture_current_block_region()

    tetromino_name = get_tetrominos(fb)
    if not tetromino_name:
        continue
    t = TETROMINOS[tetromino_name]
    t2 = []

    if see == 2:
        fb2 = capture_next_block_region()
        next_tetromino_name = get_tetrominos(fb2)
        if not next_tetromino_name:
            continue
        t2 = TETROMINOS[next_tetromino_name]

    # 計算最佳放置位置
    best_position = find_best_position(b, t, t2)
    # print(best_position)
    if not best_position:
        continue
    simulate_move(best_position)
    # print("current score:", calculate_score(b, False))
