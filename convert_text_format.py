"""
修改text列格式：
1. user部分添加指令
2. assistant部分改为 move scores格式（使用special tokens）

原格式:
<|im_start|>user
<chess_position>...</chess_position><|im_end|>
<|im_start|>assistant
<think>explanation</think>
<uci_move>move</uci_move><|im_end|>

新格式:
<|im_start|>user
<chess_position>...</chess_position>
Evaluate all legal moves and select the best one.<|im_end|>
<|im_start|>assistant
<c6><e8> 87 <c6><c8> 66 ... (按legal_moves顺序，使用special tokens)
<think></think>
<uci_move>c6e8</uci_move><|im_end|>

Usage:
    nohup python convert_text_format.py \
    --input /home/jcyang/global-chess-challenge-2025-starter-kit/data/ChessExplained_2500k_qwen3.parquet \
    --output ./data/ChessExplained_scored_obs_adaptk_dp12.parquet \
    --depth 12 \
    --workers 150 \
    --limit 2500000 \
    --chunksize 10 > speed_adaptk_dp12.log 2>&1 &

优化说明：
1. 使用 MultiPV 一次性获取所有走法评分（而非逐个分析）
2. 每个 worker 进程复用同一个 Stockfish 引擎实例（避免重复启动开销）
3. 使用 imap_unordered + chunksize 提高并行效率
"""

import re
import argparse
import chess
import chess.engine
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import atexit

STOCKFISH_PATH = "/usr/games/stockfish"
STOCKFISH_DEPTH = 10


# ============================================================
# 全局变量：每个 worker 进程持有一个 Stockfish 引擎实例
# ============================================================
_worker_engine = None
_worker_depth = None
_worker_k = None  # 衰减系数（仅在 adaptive_k=False 时使用）
_worker_adaptive_k = True  # 是否使用自适应 k


def _init_worker(stockfish_path, depth, k, adaptive_k=True):
    """Worker 进程初始化：启动并复用 Stockfish 引擎"""
    global _worker_engine, _worker_depth, _worker_k, _worker_adaptive_k
    _worker_depth = depth
    _worker_k = k
    _worker_adaptive_k = adaptive_k
    _worker_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    _worker_engine.configure({"Threads": 1})
    # 注册退出时关闭引擎
    atexit.register(_cleanup_worker)


def _cleanup_worker():
    """清理 worker 的引擎"""
    global _worker_engine
    if _worker_engine is not None:
        try:
            _worker_engine.quit()
        except:
            pass
        _worker_engine = None


import math

def cp_loss_to_score(cp_loss, k=25):
    """
    将 cp 损失（相对于最佳走法）转换为 0-100 分数

    cp_loss = best_cp - this_cp (总是 >= 0)

    使用指数衰减：score = 100 * exp(-loss/k)
    - loss=0   -> score=100 (最佳走法)
    - loss=k   -> score≈37
    - loss=2k  -> score≈14
    - loss=3k  -> score≈5
    """
    return int(100 * math.exp(-cp_loss / k))


def calculate_adaptive_k(cp_losses, min_k=15, max_k=500, target_percentile=0.25, target_score=75):
    """
    根据 cp_loss 分布动态计算 k 值

    核心思路：让处于 target_percentile 位置的走法得分约为 target_score
    这样无论局面走法差异大小，都能保持分数的区分度

    参数：
    - cp_losses: 所有走法的 cp_loss 列表（best=0）
    - min_k: k 的最小值，防止分数下降太快
    - max_k: k 的最大值，防止差走法得分过高
    - target_percentile: 目标位置（0.25 = 第25%的走法）
    - target_score: 目标位置走法的期望分数

    示例：
    - 局面A：cp_losses = [0, 10, 20, 50, 100, 200]
      target_idx=1, target_loss=10, k=10/ln(2)≈14.4 -> 使用min_k=15
    - 局面B：cp_losses = [0, 100, 200, 500, 800]
      target_idx=1, target_loss=100, k=100/ln(2)≈144 -> k≈144
    """
    sorted_losses = sorted(cp_losses)
    n = len(sorted_losses)

    if n < 2:
        return 25  # 只有一个走法，返回默认值

    # 找到目标位置的 cp_loss
    target_idx = max(1, int(n * target_percentile))
    target_loss = sorted_losses[target_idx]

    if target_loss <= 0:
        # 有多个最佳走法（cp_loss=0），使用较小的 k 区分后面的走法
        return min_k

    # 计算 k：使 target_loss 位置的走法得分约为 target_score
    # target_score = 100 * exp(-target_loss / k)
    # k = -target_loss / ln(target_score / 100) = target_loss / ln(100 / target_score)
    k = target_loss / math.log(100 / target_score)

    # 限制 k 的范围
    k = max(min_k, min(k, max_k))

    return k


def uci_to_special_token(uci_move, color):
    """
    将UCI格式的move转换为special token格式
    例如: e2e4 -> <e2><e4>
         e7e8q -> <e7><e8><White_Queen> (如果color是White)
    """
    from_sq = uci_move[0:2]
    to_sq = uci_move[2:4]
    promotion = uci_move[4:5] if len(uci_move) > 4 else ''

    token = f"<{from_sq}><{to_sq}>"

    if promotion:
        promo_map = {
            'q': f'<{color}_Queen>',
            'r': f'<{color}_Rook>',
            'b': f'<{color}_Bishop>',
            'n': f'<{color}_Knight>'
        }
        token += promo_map.get(promotion, '')

    return token


def get_scores_for_fen_multipv(board, engine, depth, k=25, adaptive_k=True):
    """
    使用 MultiPV 一次性获取所有合法走法的评分（相对分数方案）

    核心改进：
    - 使用相对分数而非绝对分数
    - best_move 得分 = 100
    - 其他走法根据与 best_move 的 cp 差距递减
    - 这样即使在平衡局面，模型也能区分最佳走法

    参数:
    - k: 衰减系数，控制分数下降速度（默认25，仅在 adaptive_k=False 时使用）
    - adaptive_k: 是否根据局面动态计算 k（默认True）

    返回: [(uci_move, score), ...] 按 legal_moves 顺序
    """
    legal_moves = list(board.legal_moves)
    num_moves = len(legal_moves)

    if num_moves == 0:
        return []

    try:
        # MultiPV = num_moves，一次性获取所有走法的评分
        infos = engine.analyse(
            board,
            chess.engine.Limit(depth=depth),
            multipv=num_moves
        )

        # 收集所有走法的原始 cp 值
        move_to_cp = {}
        for pv_info in infos:
            if "pv" in pv_info and len(pv_info["pv"]) > 0:
                move = pv_info["pv"][0]
                score = pv_info["score"].relative

                if score.is_mate():
                    # 正数表示我方将杀，负数表示被杀
                    cp = 10000 if score.mate() > 0 else -10000
                else:
                    cp = score.score()

                move_to_cp[move.uci()] = cp

        if not move_to_cp:
            return None

        # 找到最佳 cp 值
        best_cp = max(move_to_cp.values())

        # 计算所有走法的 cp_loss
        cp_losses = [best_cp - cp for cp in move_to_cp.values()]

        # 动态计算 k 或使用固定 k
        if adaptive_k:
            actual_k = calculate_adaptive_k(cp_losses)
        else:
            actual_k = k

        # 按 legal_moves 顺序返回结果
        # 使用相对分数：cp_loss = best_cp - this_cp
        result = []
        for m in legal_moves:
            uci = m.uci()
            if uci in move_to_cp:
                cp_loss = best_cp - move_to_cp[uci]
                score = cp_loss_to_score(cp_loss, actual_k)
            else:
                # 缺失的走法给一个较低的默认分数
                score = 5
            result.append((uci, score))

        return result

    except Exception as e:
        # 引擎出错时返回 None
        return None


USER_INSTRUCTION = "Evaluate all legal moves and select the best one."
# USER_INSTRUCTION = ""

def convert_text_fast(args):
    """
    转换单行 text，使用全局复用的引擎实例

    参数: (idx, text, fen, best_move)  -- 不再需要 stockfish_path 和 depth
    返回: (idx, new_text, disagreed)
    """
    global _worker_engine, _worker_depth, _worker_k, _worker_adaptive_k

    idx, text, fen, best_move = args

    # 使用复用的引擎实例 + MultiPV（相对分数方案）
    board = chess.Board(fen)
    move_scores = get_scores_for_fen_multipv(
        board, _worker_engine, _worker_depth, _worker_k, _worker_adaptive_k
    )

    if move_scores is None:
        return idx, text, False  # 失败则保留原文

    # 检查best_move是否是最高分
    max_score = max(s for _, s in move_scores)
    best_move_score = None
    best_move_idx = None

    for i, (m, s) in enumerate(move_scores):
        if m == best_move:
            best_move_score = s
            best_move_idx = i
            break

    # 统计是否Stockfish不同意
    disagreed = False
    if best_move_score is not None and best_move_score < max_score:
        disagreed = True

    # 如果best_move不在legal moves中（异常情况）
    if best_move_idx is None:
        disagreed = True

    # 如果disagreed，返回None表示丢弃该数据
    if disagreed:
        return idx, None, True

    # 获取当前走棋方颜色
    turn = fen.split()[1]
    color = "White" if turn == 'w' else "Black"

    # 构建scores行（使用special tokens）
    # scores_parts = [f"{uci_to_special_token(m, color)} {s}" for m, s in move_scores]
    # scores_line = " ".join(scores_parts)

    # scores_line = " ".join([f"{m} {s}" for m, s in move_scores])
    scores_line = " ".join([f"{s}" for _, s in move_scores])

    # 新的assistant内容
    new_assistant = f"<think>{scores_line}</think>\n<uci_move>{best_move}</uci_move>"

    # 1. 修改user部分：在</chess_position>后添加指令
    new_text = text.replace(
        "</chess_position><|im_end|>",
        f"</chess_position>\n{USER_INSTRUCTION}<|im_end|>"
    )

    # 2. 替换assistant部分
    pattern = r'<\|im_start\|>assistant\n.*?<\|im_end\|>'
    replacement = f'<|im_start|>assistant\n{new_assistant}<|im_end|>'
    new_text = re.sub(pattern, replacement, new_text, flags=re.DOTALL)

    return idx, new_text, disagreed


def main():
    parser = argparse.ArgumentParser(
        description="Convert chess training data format with Stockfish scoring (optimized with MultiPV)"
    )
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument("--output", required=True, help="Output parquet file")
    parser.add_argument("--stockfish", default=STOCKFISH_PATH, help="Path to Stockfish binary")
    parser.add_argument("--depth", type=int, default=STOCKFISH_DEPTH, help="Stockfish search depth")
    parser.add_argument("--k", type=int, default=25,
                        help="Score decay factor (default=25). Lower=steeper decay. "
                             "Only used when --no-adaptive-k is set.")
    parser.add_argument("--no-adaptive-k", action="store_true",
                        help="Disable adaptive k. Use fixed k value for all positions.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows to process")
    parser.add_argument("--chunksize", type=int, default=None, help="Chunksize for imap (auto if not set)")
    args = parser.parse_args()

    print(f"Loading {args.input}")
    df = pd.read_parquet(args.input)
    if args.limit:
        df = df.head(args.limit)
    print(f"Rows: {len(df)}")
    adaptive_k = not args.no_adaptive_k
    if adaptive_k:
        print(f"Workers: {args.workers}, Depth: {args.depth}, K: adaptive (auto-calculated per position)")
    else:
        print(f"Workers: {args.workers}, Depth: {args.depth}, K: {args.k} (fixed)")
    print(f"Using MultiPV optimization + relative scoring (best=100, decay by cp loss)")

    # 准备任务 - 不再需要传递 stockfish_path 和 depth（通过 initializer 传递）
    tasks = [
        (idx, row['text'], row['fen'], row['move'])
        for idx, row in df.iterrows()
    ]

    # 自动计算 chunksize：让每个 worker 处理多个任务再返回，减少 IPC 开销
    if args.chunksize is None:
        chunksize = max(1, len(tasks) // (args.workers * 10))
    else:
        chunksize = args.chunksize
    print(f"Chunksize: {chunksize}")

    # 并行处理
    results = {}  # idx -> new_text (只保存agree的)
    agree_indices = []  # 保留的索引
    disagree_count = 0
    total_processed = 0

    # 使用 initializer 让每个 worker 启动时创建并复用 Stockfish 引擎
    with Pool(
        args.workers,
        initializer=_init_worker,
        initargs=(args.stockfish, args.depth, args.k, adaptive_k)
    ) as pool:
        # 使用 imap_unordered：不保证顺序，但更快（任务完成即返回）
        iterator = pool.imap_unordered(convert_text_fast, tasks, chunksize=chunksize)

        for idx, new_text, disagreed in tqdm(iterator, total=len(tasks)):
            total_processed += 1
            if disagreed:
                disagree_count += 1
            else:
                results[idx] = new_text
                agree_indices.append(idx)

    # 只保留agree的数据（需要排序以保持原始顺序）
    agree_indices.sort()
    df_filtered = df.loc[agree_indices].copy()
    df_filtered['text'] = [results[idx] for idx in agree_indices]

    # 统计信息
    print("\n" + "="*60)
    print("Statistics:")
    print(f"  Total processed: {total_processed}")
    print(f"  Stockfish disagreed (dropped): {disagree_count} ({100*disagree_count/total_processed:.2f}%)")
    print(f"  Stockfish agreed (kept): {len(agree_indices)} ({100*len(agree_indices)/total_processed:.2f}%)")
    print("="*60)

    print(f"\nSaving {len(df_filtered)} rows to {args.output}")
    df_filtered.to_parquet(args.output, index=False)

    # 显示样例
    if len(df_filtered) > 0:
        print("\n" + "="*60)
        print("Sample:")
        print(df_filtered['text'].iloc[20])


if __name__ == "__main__":
    main()
