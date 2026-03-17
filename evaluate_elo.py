#!/usr/bin/env python3
"""
evaluate_elo.py — 国际象棋 LLM Elo 等级分评测工具

================================================================================
一、功能说明
================================================================================

让训练好的国际象棋 LLM 与不同强度的 Stockfish 对弈，按照 Elo 等级分系统逐盘
计算模型的 Elo 分数。

  - 对手强度：通过 Stockfish Skill Level (0-20) 控制
  - Elo 公式：Rn = Ro + K * (W - We), We = 1/(1+10^((Rs-Rm)/400))
  - K 值：20（论文标准，参考 Zhang et al. 2501.17186v2 Section 5.1）
  - Skill Level → Elo 映射：论文公式(3)(4)
      SK = 37.247*e³ - 40.852*e² + 22.294*e - 0.311, e = (Elo-1320)/1870
  - 输出：JSON 结果 + PGN 棋谱（可供 BayesElo/Ordo 进一步分析）

================================================================================
二、前置条件
================================================================================

  1. vLLM 推理服务已启动（提供 OpenAI 兼容 API）
  2. Stockfish 引擎已安装（apt install stockfish 或指定路径）
  3. Python 依赖：chess, requests（pip install python-chess requests）

================================================================================
三、使用方法
================================================================================

步骤 1：启动 vLLM 推理服务

CUDA_VISIBLE_DEVICES=0 nohup vllm serve ./qwen3_4b_new_format_val_adaptk_dp12/checkpoint-90000 --served-model-name aicrowd-chess-model --tokenizer ./qwen3_4b_new_format_val_adaptk_dp12/checkpoint-90000 --port 8000 --dtype bfloat16 --max-model-len 1200 --enforce-eager > vllm_serve.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup vllm serve /home/jcyang/verl/examples/grpo_trainer/checkpoints/verl_grpo_chess/qwen3_4b_grpo_new_format/global_step_1550/merged_model --served-model-name aicrowd-chess-model --tokenizer /home/jcyang/verl/examples/grpo_trainer/checkpoints/verl_grpo_chess/qwen3_4b_grpo_new_format/global_step_1550/merged_model --port 8000 --dtype bfloat16 --max-model-len 1200 --enforce-eager > vllm_serve.log 2>&1 &

步骤 2：运行评测

    # 基本用法（默认 Skill Level 0,1,2，每级 50 局）
    python evaluate_elo.py

    # 论文设置（每级 100 局）
    nohup python -u evaluate_elo.py --games-per-level 100 --depth 5 > elo_eval_tpt_grpo_dp5.log 2>&1 &

    # 扩展到更多等级
    python evaluate_elo.py --games-per-level 100 --skill-levels 0,1,2,3,4,5

    # 指定 Stockfish 路径和端口
    python evaluate_elo.py --stockfish-path /usr/games/stockfish --port 8000

nohup python -m http.server 8080 > viewer.log 2>&1 &
================================================================================
四、参数说明
================================================================================

    --games-per-level   每个 Skill Level 的对局数（默认 50，建议 >=100）
    --skill-levels      Stockfish Skill Level 列表（默认 "0,1,2"，范围 0-20）
    --stockfish-path    Stockfish 可执行文件路径（默认 "stockfish"）
    --port              vLLM 服务端口（默认 8000）
    --time-limit        Stockfish 每步思考时间/秒（默认 2.0）
    --max-moves         单局最大步数（默认 200）
    --output            JSON 结果文件路径（默认自动生成带时间戳的文件名）
    --pgn               PGN 棋谱文件路径（默认自动生成带时间戳的文件名）

================================================================================
五、Elo 计算流程
================================================================================

  1. 按 Skill Level 从低到高依次对弈，每级 N 局（奇数局 LLM 执白，偶数局执黑）
  2. 每完成一局，立即按公式更新 Elo（逐盘迭代，非批量计算）：
       - 计算期望得分 RE = 1 / (1 + 10^((对手Elo - 模型Elo) / 400))
       - 实际得分 RA = 胜1.0 / 平0.5 / 负0.0
       - 更新 Elo_new = Elo_old + K * (RA - RE)
  3. 对手 Elo 由公式(3)(4)从 Skill Level 反解得到
  4. 初始 Elo = 1500，K = 20（跨级连续累积，与论文一致）

================================================================================
六、输出文件
================================================================================

  1. JSON 文件（elo_results_<timestamp>.json）
     - 每级的 W/D/L、得分率、对手 Elo
     - 每级结束后的 Elo 快照
     - 最终 Elo 及标准差

  2. PGN 文件（elo_games_<timestamp>.pgn）
     - 所有对局的标准 PGN 格式棋谱
     - 可用于 BayesElo / Ordo 进行锦标赛 Elo 计算

================================================================================
七、Skill Level 与 Elo 对照表（公式 3-4）
================================================================================

    SK  0: 1347 - 1444      SK 10: 2786 - 2851
    SK  1: 1444 - 1566      SK 11: 2851 - 2910
    SK  2: 1566 - 1729      SK 12: 2910 - 2963
    SK  3: 1729 - 1953      SK 13: 2963 - 3012
    SK  4: 1953 - 2197      SK 14: 3012 - 3057
    SK  5: 2197 - 2383      SK 15: 3057 - 3099
    SK  6: 2383 - 2518      SK 16: 3099 - 3139
    SK  7: 2518 - 2624      SK 17: 3139 - 3176
    SK  8: 2624 - 2711      SK 18: 3176 - 3190
    SK  9: 2711 - 2786      SK 19: 3190 - 3190
                             SK 20: 3190+
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import math
import chess
import chess.engine
import time
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chess_llm import ChessLLM
from evaluation_helpers.eval_config import EvalConfig


# ---------------------------------------------------------------------------
# 实时可视化 — 每步棋/每局结束后刷新并保存 PNG
# ---------------------------------------------------------------------------

class LiveVisualizer:
    """
    每步棋后更新棋盘面板，每局结束后更新 Elo 轨迹、W/D/L 和对局色块流。

    面板布局（GridSpec 2×6）:
      ┌──────────────────────┬──────────┐
      │  Elo 轨迹 (4/6 宽)   │  棋盘    │
      ├────────┬─────────────┤ (2/6 宽) │
      │ W/D/L  │ 对局色块流  │          │
      └────────┴─────────────┴──────────┘
    """

    _RESULT_COLOR = {"win": "#4CAF50", "draw": "#9E9E9E", "loss": "#F44336"}
    _BAND_COLORS  = ["#FFF9C4", "#E8F5E9", "#E3F2FD", "#FCE4EC",
                     "#F3E5F5", "#FBE9E7", "#E0F2F1", "#F9FBE7"]


    def __init__(self, skill_levels: list, games_per_level: int,
                 opponent_elos: dict, output_path: str = "elo_live.png"):
        self.skill_levels    = skill_levels
        self.games_per_level = games_per_level
        self.opponent_elos   = opponent_elos   # {sk: rounded_elo}
        self.output_path     = output_path

        self.elo_history: list = [1500.0]
        self.game_log:    list = []            # (sk, result, reason, moves)
        self.wdl: dict = {sk: [0, 0, 0] for sk in skill_levels}

        # 记住最后一步，供全量重绘时复原棋盘
        self._last_board = chess.Board()
        self._last_move:  Optional[chess.Move] = None
        self._last_label: str = "Waiting..."

        self.enabled = False
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            from matplotlib.gridspec import GridSpec
            self._plt      = plt
            self._mpatches = mpatches
            self._GridSpec  = GridSpec
            self._setup()
            self.enabled = True
            print(f"  [LivePlot] 实时图像保存至: {output_path}")
        except ImportError:
            print("  [LivePlot] 未找到 matplotlib，跳过实时可视化 (pip install matplotlib)")

    def _setup(self):
        plt = self._plt
        self.fig = plt.figure(figsize=(16, 12))
        self.fig.patch.set_facecolor("#FAFAFA")
        gs = self._GridSpec(2, 6, figure=self.fig,
                            height_ratios=[3, 2], hspace=0.48, wspace=0.35)
        self.ax_elo   = self.fig.add_subplot(gs[0, :4])   # Elo 轨迹（宽）
        self.ax_board = self.fig.add_subplot(gs[0, 4:])   # 棋盘（右上）
        self.ax_wdl   = self.fig.add_subplot(gs[1, :2])   # W/D/L 条形
        self.ax_tiles = self.fig.add_subplot(gs[1, 2:])   # 对局色块流

    # ── 棋盘渲染（chess.svg + cairosvg）─────────────────────────────────

    def _draw_board(self, ax, board: chess.Board,
                    last_move: Optional[chess.Move] = None,
                    title: str = "Current Position"):
        """用 chess.svg + cairosvg 渲染矢量级棋盘并显示在 Axes 上。"""
        import chess.svg as _csvg
        import cairosvg as _cairo
        import io as _io
        import matplotlib.image as _mpimg

        ax.cla()
        ax.axis("off")

        arrows = ([_csvg.Arrow(last_move.from_square, last_move.to_square,
                               color="#FF6F00")]
                  if last_move else [])

        svg_str = _csvg.board(
            board,
            lastmove=last_move,
            arrows=arrows,
            size=400,
            coordinates=True,
        )
        png_bytes = _cairo.svg2png(bytestring=svg_str.encode())
        img = _mpimg.imread(_io.BytesIO(png_bytes))
        ax.imshow(img)
        ax.set_title(title, fontsize=9, fontweight="bold", pad=4)

    def on_move(self, board: chess.Board,
                last_move: Optional[chess.Move],
                label: str = ""):
        """每步棋后调用：只更新棋盘面板并保存 PNG。"""
        if not self.enabled:
            return
        self._last_board = board.copy()
        self._last_move  = last_move
        self._last_label = label
        self._draw_board(self.ax_board, board, last_move, label)
        self._save()

    def update(self, skill_level: int, result: str, reason: str,
               moves: int, new_elo: float):
        """每局结束后调用：更新 Elo/WDL/色块数据并全量重绘保存。"""
        if not self.enabled:
            return
        self.elo_history.append(new_elo)
        self.game_log.append((skill_level, result, reason, moves))
        wi, dr, lo = self.wdl[skill_level]
        if result == "win":    self.wdl[skill_level] = [wi+1, dr,   lo  ]
        elif result == "draw": self.wdl[skill_level] = [wi,   dr+1, lo  ]
        else:                  self.wdl[skill_level] = [wi,   dr,   lo+1]
        self._redraw()
        self._save()

    def _save(self):
        try:
            self.fig.savefig(self.output_path, dpi=120, bbox_inches="tight")
        except Exception as e:
            print(f"  [LivePlot] 保存失败: {e}")

    def _redraw(self):
        plt  = self._plt
        mpa  = self._mpatches
        n_games  = len(self.game_log)
        cur_elo  = self.elo_history[-1]
        eh       = self.elo_history

        # ── 面板 1: Elo 轨迹 ──────────────────────────────────────────────
        ax = self.ax_elo
        ax.cla()
        x = list(range(len(eh)))

        # 原始曲线
        ax.plot(x, eh, color="#1565C0", linewidth=1.0, alpha=0.55, zorder=3)
        # 滚动平均（窗口 = min(10, n)）
        w = max(1, min(10, len(eh)))
        smooth = [
            sum(eh[max(0, i - w): i + 1]) / len(eh[max(0, i - w): i + 1])
            for i in range(len(eh))
        ]
        ax.plot(x, smooth, color="#FF6F00", linewidth=2.2, zorder=4, label="Smoothed")

        # 各 Skill Level 背景色带 + 对手 Elo 参考线
        elo_vals = eh if len(eh) > 1 else [1500.0, 1500.0]
        y_min = min(elo_vals) - 30
        y_max = max(elo_vals) + 30
        ax.set_ylim(y_min, y_max)

        offset = 1
        for i, sk in enumerate(self.skill_levels):
            bc = self._BAND_COLORS[i % len(self._BAND_COLORS)]
            x0 = offset - 0.5
            x1 = offset + self.games_per_level - 0.5
            vis_x1 = min(x1, len(eh) - 0.5)
            if vis_x1 > x0:
                ax.axvspan(x0, vis_x1, alpha=0.45, color=bc, zorder=1)
                mid = (x0 + vis_x1) / 2
                ax.text(mid, y_min + (y_max - y_min) * 0.03,
                        f"SK{sk}", ha="center", va="bottom",
                        fontsize=8, color="#555", zorder=5)
            opp = self.opponent_elos.get(sk, 0)
            ax.axhline(opp, color="#888", linewidth=0.8,
                       linestyle="--", alpha=0.6, zorder=2)
            ax.text(len(eh) - 0.5, opp + 2,
                    f"SF SK{sk}≈{opp}", fontsize=7, color="#777",
                    ha="right", va="bottom", zorder=5)
            offset += self.games_per_level

        ax.set_xlim(0, max(10, len(eh) - 1))
        ax.set_title(
            f"Elo Trajectory   Current: {cur_elo:.0f}   Games: {n_games}",
            fontweight="bold", fontsize=12)
        ax.set_xlabel("Games")
        ax.set_ylabel("Elo")
        ax.legend(fontsize=9, loc="upper left", labels=["Smoothed"])
        ax.grid(True, alpha=0.22, zorder=0)

        # ── 面板 2: W/D/L 堆积条形 ───────────────────────────────────────
        ax = self.ax_wdl
        ax.cla()
        active = [sk for sk in self.skill_levels if sum(self.wdl[sk]) > 0]
        if active:
            xlabels = [f"SK{sk}\n~{self.opponent_elos[sk]:.0f}" for sk in active]
            wi_l = [self.wdl[sk][0] for sk in active]
            dr_l = [self.wdl[sk][1] for sk in active]
            lo_l = [self.wdl[sk][2] for sk in active]
            bw   = 0.55
            xp   = list(range(len(active)))
            ax.bar(xp, wi_l, bw, color=self._RESULT_COLOR["win"],  label="Win")
            ax.bar(xp, dr_l, bw, bottom=wi_l,
                   color=self._RESULT_COLOR["draw"], label="Draw")
            ax.bar(xp, lo_l, bw,
                   bottom=[a + b for a, b in zip(wi_l, dr_l)],
                   color=self._RESULT_COLOR["loss"], label="Loss")
            for i, sk in enumerate(active):
                total = sum(self.wdl[sk])
                score = (self.wdl[sk][0] + 0.5 * self.wdl[sk][1]) / total if total else 0
                ax.text(i, total + 0.35, f"{score:.1%}",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
            ax.set_xticks(xp)
            ax.set_xticklabels(xlabels, fontsize=9)
            ax.legend(fontsize=9, labels=["Win", "Draw", "Loss"])
        ax.set_title("Win / Draw / Loss by Level", fontweight="bold")
        ax.set_ylabel("Games")
        ax.grid(axis="y", alpha=0.25)

        # ── 面板 3: 对局色块流（最近 80 局）──────────────────────────────
        COLS, ROWS = 10, 8
        ax = self.ax_tiles
        ax.cla()
        ax.set_facecolor("#F0F0F0")
        recent = self.game_log[-(COLS * ROWS):]
        n_r    = len(recent)
        # 从右下角填入最新对局，逆序排列（最新→右下，最旧→左上）
        for slot in range(COLS * ROWS):
            r_ = slot // COLS
            c_ = slot % COLS
            # slot 0 = 最旧的那局（位于左上）
            # 对应 recent 中的 recent[slot - (COLS*ROWS - n_r)]
            game_pos = slot - (COLS * ROWS - n_r)
            if 0 <= game_pos < n_r:
                sk, res, rea, mov = recent[game_pos]
                fc    = self._RESULT_COLOR.get(res, "#DDDDDD")
                alpha = 0.90
            else:
                fc    = "#DDDDDD"
                alpha = 0.25
                mov   = None
            rect = mpa.Rectangle(
                (c_, ROWS - r_ - 1), 0.88, 0.88,
                facecolor=fc, edgecolor="white", linewidth=0.8, alpha=alpha)
            ax.add_patch(rect)
            if mov is not None:
                ax.text(c_ + 0.44, ROWS - r_ - 0.5, str(mov),
                        ha="center", va="center",
                        fontsize=5.5, color="white", fontweight="bold")

        ax.set_xlim(-0.05, COLS + 0.05)
        ax.set_ylim(-0.05, ROWS + 0.05)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        # 图例
        leg = [mpa.Patch(facecolor=self._RESULT_COLOR[k], label=lb)
               for k, lb in [("win", "Win"), ("draw", "Draw"), ("loss", "Loss")]]
        ax.legend(handles=leg, loc="lower right",
                  fontsize=8, bbox_to_anchor=(1.02, -0.02))

        w_tot = sum(1 for _, r, _, _ in self.game_log if r == "win")
        d_tot = sum(1 for _, r, _, _ in self.game_log if r == "draw")
        l_tot = sum(1 for _, r, _, _ in self.game_log if r == "loss")
        ax.set_title(
            f"Recent Games (last {min(COLS*ROWS, n_games)})  "
            f"W{w_tot} D{d_tot} L{l_tot}",
            fontweight="bold")

        total_plan = len(self.skill_levels) * self.games_per_level
        self.fig.suptitle(
            f"Chess LLM Elo Evaluation  |  Elo: {cur_elo:.0f}  |  "
            f"Progress: {n_games}/{total_plan}",
            fontsize=13, fontweight="bold", y=0.995)

        # ── 面板 4: 棋盘（复原最后记录的局面）────────────────────────────
        self._draw_board(self.ax_board, self._last_board,
                         self._last_move, self._last_label)

    # ── 逐步走法可视化 ────────────────────────────────────────────────────

    def save_move_detail(self, board_before: chess.Board,
                         move: chess.Move,
                         thinking: Optional[str],
                         raw_response: Optional[str],
                         prompt: Optional[str],
                         filepath: str):
        """
        为 LLM 的单步走法保存详细可视化 PNG：
          左：走法后棋盘（高亮本步）
          右：全部合法走法的分数条形图（所选走法橙色标注）
        """
        if not self.enabled:
            return

        plt = self._plt
        mpa = self._mpatches

        # ── 解析 thinking 分数 ────────────────────────────────────────────
        legal_moves = list(board_before.legal_moves)
        scores: list = []
        if thinking:
            try:
                scores = [int(s) for s in thinking.split()]
            except ValueError:
                scores = []

        scores_valid = (len(scores) == len(legal_moves)) and len(scores) > 0

        # 本步走法的 SAN
        try:
            chosen_san = board_before.san(move)
        except Exception:
            chosen_san = move.uci()

        # ── 建图（独立 Figure，用后关闭避免内存泄漏）────────────────────
        fig = plt.figure(figsize=(14, max(6, len(legal_moves) * 0.28 + 2)))
        fig.patch.set_facecolor("#FAFAFA")

        from matplotlib.gridspec import GridSpec as _GS
        gs = _GS(1, 2, figure=fig,
                 width_ratios=[1, 1.8], wspace=0.08, left=0.04, right=0.97,
                 top=0.92, bottom=0.08)
        ax_board = fig.add_subplot(gs[0, 0])
        ax_bar   = fig.add_subplot(gs[0, 1])

        # ── 左：棋盘（走棋后局面，chess.svg 渲染）────────────────────────
        board_after = board_before.copy()
        board_after.push(move)
        self._draw_board(ax_board, board_after, move,
                         title=f"LLM: {chosen_san}  ({move.uci()})")
        ax_board.set_aspect("equal", adjustable="box")

        # ── 右：分数条形图 ────────────────────────────────────────────────
        if scores_valid:
            # 构建 (san, score, is_chosen) 三元组并按分数降序排列
            pairs = []
            chosen_uci = move.uci()
            for lm, sc in zip(legal_moves, scores):
                try:
                    san = board_before.san(lm)
                except Exception:
                    san = lm.uci()
                pairs.append((san, sc, lm.uci() == chosen_uci))

            pairs.sort(key=lambda x: x[1], reverse=True)

            # 找选中走法的排名
            chosen_rank = next(
                (i + 1 for i, (_, _, ic) in enumerate(pairs) if ic), None)
            chosen_score = next((s for _, s, ic in pairs if ic), None)
            max_score    = pairs[0][1]

            # 最多显示 40 步（超过时截断末尾低分步）
            MAX_SHOW = 40
            pairs_show = pairs[:MAX_SHOW]
            n_show     = len(pairs_show)

            labels    = [p[0] for p in pairs_show]
            vals      = [p[1] for p in pairs_show]
            is_chosen = [p[2] for p in pairs_show]

            # y 坐标：排名 1 在最上面
            ypos   = list(range(n_show - 1, -1, -1))
            colors = ["#FF6F00" if ic else "#1565C0" for ic in is_chosen]

            ax_bar.barh(ypos, vals, color=colors,
                        edgecolor="white", linewidth=0.4, height=0.78)

            # 条形末端标注分数
            x_max = max(vals) if vals else 1
            for y, val, ic in zip(ypos, vals, is_chosen):
                ax_bar.text(val + x_max * 0.008, y, str(val),
                            va="center",
                            fontsize=6.5 if n_show <= 20 else 5.5,
                            color="#FF6F00" if ic else "#555",
                            fontweight="bold" if ic else "normal")

            ax_bar.set_yticks(ypos)
            ax_bar.set_yticklabels(
                labels, fontsize=7 if n_show <= 20 else 5.5)

            # 选中走法纵向参考线
            if chosen_score is not None:
                ax_bar.axvline(chosen_score, color="#FF6F00",
                               linewidth=1.2, linestyle="--", alpha=0.6)

            is_best  = (chosen_rank == 1)
            best_tag = "  ✓ BEST" if is_best else f"  (best={max_score})"
            title_str = (
                f"Move Scores  —  Rank {chosen_rank}/{len(legal_moves)}"
                f"{best_tag}\n"
                f"Score: {chosen_score}   "
                f"Showing top {n_show}/{len(legal_moves)} legal moves"
            )
            ax_bar.set_title(title_str, fontsize=9, fontweight="bold")
            ax_bar.set_xlabel("Score", fontsize=8)
            ax_bar.set_xlim(0, x_max * 1.14)
            ax_bar.grid(axis="x", alpha=0.25)

            # 图例
            from matplotlib.patches import Patch as _P
            ax_bar.legend(
                handles=[_P(facecolor="#FF6F00", label="Chosen"),
                         _P(facecolor="#1565C0", label="Other")],
                fontsize=8, loc="lower right")
        else:
            msg = ("No score data" if not thinking
                   else f"Score count mismatch\n"
                        f"({len(scores)} scores vs "
                        f"{len(legal_moves)} legal moves)")
            ax_bar.text(0.5, 0.5, msg,
                        ha="center", va="center",
                        transform=ax_bar.transAxes, fontsize=11)
            ax_bar.set_title("Move Scores", fontweight="bold")
            ax_bar.axis("off")

        # FEN 脚注
        fen_str = board_before.fen()
        fig.text(0.5, 0.005, f"FEN: {fen_str}",
                 ha="center", fontsize=6.5, color="#999", family="monospace")

        # ── 保存 PNG 并关闭（防内存泄漏）────────────────────────────────
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        try:
            fig.savefig(filepath, dpi=100, bbox_inches="tight")
        except Exception as e:
            print(f"  [MoveViz] PNG 保存失败 {filepath}: {e}")
        finally:
            plt.close(fig)

        # ── 保存原始模型输出文本（同名 .txt）────────────────────────────
        if raw_response is not None:
            txt_path = filepath.replace(".png", ".txt")
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(raw_response)
            except Exception as e:
                print(f"  [MoveViz] TXT 保存失败 {txt_path}: {e}")

        # ── 保存模型输入 prompt（同名 .prompt.txt）────────────────────
        if prompt is not None:
            prompt_path = filepath.replace(".png", ".prompt.txt")
            try:
                with open(prompt_path, "w", encoding="utf-8") as f:
                    f.write(prompt)
            except Exception as e:
                print(f"  [MoveViz] PROMPT 保存失败 {prompt_path}: {e}")


# ---------------------------------------------------------------------------
# 实时 Viewer 写入 — 每局结束后写 viewer_data.json
# ---------------------------------------------------------------------------

class LiveViewerWriter:
    """每局结束后立即写入 viewer_data.json，供 viewer.html 实时轮询。"""

    def __init__(self, run_dir: str, moves_dir: str, config: dict,
                 stockfish_path: str = "stockfish",
                 analysis_depth: int = 20,
                 analyze: bool = True):
        self.run_dir = run_dir
        self.moves_dir = moves_dir
        self.config = config
        self.stockfish_path = stockfish_path
        self.analysis_depth = analysis_depth
        self.analyze = analyze
        self.matches: list = []
        self.elo_history: list = [1500.0]
        self.final_elo: int = 1500
        self.game_counter: int = 0
        self.viewer_path = os.path.join(run_dir, "viewer_data.json")

    def on_game(self, skill_level: int, game_detail: dict, new_elo: float):
        """每局结束后调用：处理对局数据并写入 JSON。"""
        import re as _re
        self.game_counter += 1
        self.elo_history.append(new_elo)
        self.final_elo = round(new_elo)

        opp_elo = round(stockfish_skill_to_elo(skill_level))

        # 重放走法，得到 FEN 和 SAN
        board = chess.Board()
        fens = [board.fen()]
        sans = []
        for uci_str in game_detail["pgn_moves"]:
            move = chess.Move.from_uci(uci_str)
            try:
                sans.append(board.san(move))
            except Exception:
                sans.append(uci_str)
            board.push(move)
            fens.append(board.fen())

        # 读取 .txt 获取 thinking / raw / prompt
        sk_dir = os.path.join(self.moves_dir, f"sk{skill_level}")
        thinking = {}
        raw_responses = {}
        prompts = {}
        for ply in range(len(game_detail["pgn_moves"])):
            is_white = (ply % 2 == 0)
            if is_white == game_detail["llm_is_white"]:
                half_move = ply + 1
                base_name = f"g{self.game_counter:04d}_m{half_move:03d}"
                txt_path = os.path.join(sk_dir, base_name + ".txt")
                prompt_path = os.path.join(sk_dir, base_name + ".prompt.txt")
                if os.path.exists(txt_path):
                    try:
                        with open(txt_path, "r") as f:
                            content = f.read()
                        m = _re.search(r'<think>(.*?)</think>', content, _re.DOTALL)
                        if m:
                            thinking[str(ply)] = m.group(1).strip()
                        raw_responses[str(ply)] = content
                    except Exception:
                        pass
                if os.path.exists(prompt_path):
                    try:
                        with open(prompt_path, "r") as f:
                            prompts[str(ply)] = f.read()
                    except Exception:
                        pass

        # 将 thinking 分数与合法走法配对
        move_scores = {}
        for ply_str, think_text in thinking.items():
            ply_idx = int(ply_str)
            try:
                scores = [int(s) for s in think_text.split()]
                board_at = chess.Board(fens[ply_idx])
                legal = list(board_at.legal_moves)
                if len(scores) == len(legal):
                    chosen_uci = game_detail["pgn_moves"][ply_idx]
                    entries = []
                    for lm, sc in zip(legal, scores):
                        try:
                            sn = board_at.san(lm)
                        except Exception:
                            sn = lm.uci()
                        entries.append({
                            "u": lm.uci(), "s": sn,
                            "v": sc, "c": lm.uci() == chosen_uci,
                        })
                    entries.sort(key=lambda x: x["v"], reverse=True)
                    move_scores[ply_str] = entries
            except Exception:
                pass

        # Stockfish 实时分析（计算 ACPL）
        evals_list = None
        cp_losses = None
        acpl = None
        worst_cp = None
        if self.analyze:
            try:
                evals_list = _analyze_game_positions(
                    fens, self.stockfish_path, self.analysis_depth)
                cp_losses = []
                llm_losses = []
                for pi in range(len(game_detail["pgn_moves"])):
                    is_w = (pi % 2 == 0)
                    eb, ea = evals_list[pi], evals_list[pi + 1]
                    loss = max(0, eb - ea) if is_w else max(0, ea - eb)
                    cp_losses.append(round(loss))
                    if (is_w == game_detail["llm_is_white"]):
                        llm_losses.append(loss)
                acpl = round(sum(llm_losses) / len(llm_losses), 1) \
                    if llm_losses else 1000
                worst_cp = max(llm_losses) if llm_losses else 0
                print(f"    [ACPL] game {self.game_counter}: {acpl}")
            except Exception as e:
                print(f"    [ACPL] 分析失败 game {self.game_counter}: {e}")

        self.matches.append({
            "id": self.game_counter - 1,
            "skill_level": skill_level,
            "opponent": f"stockfish-skill{skill_level}",
            "opponent_elo": opp_elo,
            "llm_is_white": game_detail["llm_is_white"],
            "result": game_detail["result_str"],
            "reason": game_detail["reason"],
            "total_moves": game_detail["moves"],
            "fens": fens,
            "sans": sans,
            "ucis": game_detail["pgn_moves"],
            "thinking": thinking,
            "move_scores": move_scores,
            "raw": raw_responses,
            "prompts": prompts,
            "evals": evals_list,
            "cp_losses": cp_losses,
            "acpl": acpl,
            "worst_cp": worst_cp,
        })

        self._write()

    def _write(self):
        viewer_data = {
            "matches": self.matches,
            "elo_history": [round(e, 1) for e in self.elo_history],
            "final_elo": self.final_elo,
            "config": self.config,
            "live": True,
        }
        try:
            with open(self.viewer_path, "w") as f:
                json.dump(viewer_data, f, ensure_ascii=False)
        except Exception as e:
            print(f"  [LiveViewer] JSON 写入失败: {e}")


# ---------------------------------------------------------------------------
# Stockfish 对手 — 使用 Skill Level (0-20) 控制强度
# ---------------------------------------------------------------------------


class StockfishOpponent:
    """通过 Skill Level (0-20) 控制 Stockfish 强度。

    注意：当同时设置 depth 和 Skill Level 时，depth 会在 Skill Level 的内部
    弱化机制之上再叠加截断，对高 SK 级（SK5+）会使实际强度明显低于
    stockfish_skill_to_elo 公式的预测值，导致 Elo 标定偏差。
    低 SK 级（SK0-SK2）影响较小（随机噪声已主导）。
    如需精确 Elo 估计，建议使用时间制（不传 depth）。
    """

    def __init__(self, skill_level: int, stockfish_path: str = "stockfish",
                 time_limit: float = 2.0, depth: Optional[int] = None):
        if not (0 <= skill_level <= 20):
            raise ValueError(f"Skill level must be 0-20, got {skill_level}")
        self.skill_level = skill_level
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.time_limit = time_limit
        self.depth = depth
        self.engine.configure({
            "Skill Level": skill_level,
            "Threads": 1,
            "Hash": 64,
        })

    def choose_move(self, board: chess.Board) -> Optional[chess.Move]:
        try:
            if self.depth is not None:
                limit = chess.engine.Limit(depth=self.depth, time=self.time_limit)
            else:
                limit = chess.engine.Limit(time=self.time_limit)
            result = self.engine.play(board, limit)
            return result.move
        except Exception as e:
            print(f"  [Stockfish error] {e}")
            return None

    def close(self):
        try:
            self.engine.quit()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 对局
# ---------------------------------------------------------------------------

def play_one_game(
    llm: ChessLLM,
    stockfish: StockfishOpponent,
    llm_is_white: bool,
    max_moves: int = 200,
    move_callback=None,       # move_callback(board, last_move, move_count) 每步后调用
    llm_move_callback=None,   # llm_move_callback(board_before, move, thinking, raw_response, prompt, half_move_num)
                              #   在 LLM 走棋后、board.push 前调用（board_before 状态）
) -> dict:
    """完成一局对弈，返回结构化结果。"""
    board = chess.Board()
    move_count = 0
    game_moves = []

    while not board.is_game_over() and move_count < max_moves:
        is_llm_turn = (board.turn == chess.WHITE) == llm_is_white

        if is_llm_turn:
            move, thinking, illegal, raw_response, prompt = llm.try_move(board)
            if illegal or move is None:
                if move_callback:
                    move_callback(board, None, move_count)
                return {
                    "result": "loss",
                    "reason": "illegal_move",
                    "moves": move_count,
                    "pgn_moves": game_moves,
                    "final_fen": board.fen(),
                }
            # board 此时是走棋前状态，传给可视化回调
            if llm_move_callback:
                llm_move_callback(board, move, thinking, raw_response, prompt, move_count + 1)
            board.push(move)
            game_moves.append(move.uci())
        else:
            move = stockfish.choose_move(board)
            if move is None:
                if move_callback:
                    move_callback(board, None, move_count)
                return {
                    "result": "win",
                    "reason": "stockfish_error",
                    "moves": move_count,
                    "pgn_moves": game_moves,
                    "final_fen": board.fen(),
                }
            board.push(move)
            game_moves.append(move.uci())

        move_count += 1
        if move_callback:
            move_callback(board, move, move_count)

    # 从 LLM 视角判定结果
    if board.is_game_over():
        outcome = board.outcome()
        if outcome is None or outcome.winner is None:
            result, reason = "draw", _termination_reason(board)
        elif (outcome.winner == chess.WHITE) == llm_is_white:
            result, reason = "win", _termination_reason(board)
        else:
            result, reason = "loss", _termination_reason(board)
    else:
        result, reason = "draw", "max_moves"

    return {
        "result": result,
        "reason": reason,
        "moves": move_count,
        "pgn_moves": game_moves,
        "final_fen": board.fen(),
    }


def _termination_reason(board: chess.Board) -> str:
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material():
        return "insufficient_material"
    if board.is_fifty_moves():
        return "fifty_moves"
    if board.is_repetition():
        return "repetition"
    return "other"


# ---------------------------------------------------------------------------
# Elo 计算 — 公式参考 Zhang et al. 2501.17186v2, Section 5.1
#   K 值参考 Zhang et al. 2501.17186v2, Section 5.1 (K=20)
#
#   (1) EloN = EloO + K * (RA - RE)
#   (2) RE   = 1 / (1 + 10^((EloS - EloM) / 400))
#   (3) SK   = 37.247*e³ - 40.852*e² + 22.294*e - 0.311
#   (4) e    = (Elo - 1320) / 1870
# ---------------------------------------------------------------------------

def stockfish_skill_to_elo(skill_level: int) -> float:
    """公式(3)(4)反解：由 Skill Level 求 Stockfish Elo（取中值）。

    Stockfish 在某个 Skill Level 下的 Elo 是一个范围 [SK_lower, SK+1_lower]。
    取中值作为该级别的代表 Elo，比取下界更合理。
    例：SK2 范围 1566-1729，中值 1647。
    """
    def _sk_to_elo_lower(sk_target: float) -> float:
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2.0
            sk = 37.247 * mid**3 - 40.852 * mid**2 + 22.294 * mid - 0.311
            if sk < sk_target:
                lo = mid
            else:
                hi = mid
        e = (lo + hi) / 2.0
        return e * 1870.0 + 1320.0

    elo_lower = _sk_to_elo_lower(float(skill_level))
    elo_upper = _sk_to_elo_lower(float(skill_level + 1))
    return (elo_lower + elo_upper) / 2.0


def iterative_elo_update(game_scores, opponent_elo: float,
                         initial_elo: float = 1500.0, K: float = 20.0) -> float:
    """公式(1)(2)：逐盘更新 Elo。"""
    elo = initial_elo
    for RA in game_scores:
        RE = 1.0 / (1.0 + 10.0 ** ((opponent_elo - elo) / 400.0))
        elo = elo + K * (RA - RE)
    return elo


# ---------------------------------------------------------------------------
# Viewer Data — 生成 viewer_data.json 供 Web 可视化使用
# ---------------------------------------------------------------------------

def _analyze_game_positions(fens: list, stockfish_path: str,
                            depth: int = 20,
                            movetime_ms: int = 1000) -> list:
    """用 Stockfish 分析每个局面，返回白方视角的 centipawn 评估列表。

    与 eval_vs_stockfish.py (_StockfishAnalyzer) 保持一致：
      - mate_score=1000, 评分 cap 到 [-1000, 1000]
      - 同时使用 depth + movetime 限制
    """
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Threads": 1, "Hash": 64})
    limit = chess.engine.Limit(
        depth=depth,
        time=movetime_ms / 1000.0,
    )
    evals = []
    for fen in fens:
        try:
            board = chess.Board(fen)
            if board.is_game_over():
                outcome = board.outcome()
                if outcome and outcome.winner == chess.WHITE:
                    evals.append(1000)
                elif outcome and outcome.winner == chess.BLACK:
                    evals.append(-1000)
                else:
                    evals.append(0)
            else:
                info = engine.analyse(board, limit)
                score = info["score"].white()
                cp = score.score(mate_score=1000)
                cp = max(-1000, min(1000, cp)) if cp is not None else 0
                evals.append(cp)
        except Exception:
            evals.append(0)
    engine.quit()
    return evals


def save_viewer_data(all_results: list, run_dir: str, stockfish_path: str,
                     elo_history: list, final_elo: int, config: dict,
                     analyze: bool = True, depth: int = 20):
    """从对局数据生成 viewer_data.json 供 Web 可视化。"""
    import re as _re

    matches = []
    global_game_num = 0

    for level_result in all_results:
        sk = level_result["skill_level"]
        opp_elo = level_result["opponent_elo"]
        sk_moves_dir = os.path.join(run_dir, "moves", f"sk{sk}")

        for i, game in enumerate(level_result["games_detail"]):
            global_game_num += 1

            # 重放走法，得到 FEN 和 SAN
            board = chess.Board()
            fens = [board.fen()]
            sans = []
            for uci_str in game["pgn_moves"]:
                move = chess.Move.from_uci(uci_str)
                try:
                    sans.append(board.san(move))
                except Exception:
                    sans.append(uci_str)
                board.push(move)
                fens.append(board.fen())

            # 读取 .txt / .prompt.txt 文件获取 thinking / raw / prompt 数据
            thinking = {}
            raw_responses = {}
            prompts = {}
            for ply in range(len(game["pgn_moves"])):
                is_white_move = (ply % 2 == 0)
                is_llm = (is_white_move == game["llm_is_white"])
                if is_llm:
                    half_move = ply + 1
                    base_name = f"g{global_game_num:04d}_m{half_move:03d}"
                    txt_path = os.path.join(sk_moves_dir, base_name + ".txt")
                    prompt_path = os.path.join(sk_moves_dir, base_name + ".prompt.txt")
                    if os.path.exists(txt_path):
                        try:
                            with open(txt_path, "r") as f:
                                content = f.read()
                            think_match = _re.search(
                                r'<think>(.*?)</think>', content, _re.DOTALL)
                            if think_match:
                                thinking[str(ply)] = think_match.group(1).strip()
                            raw_responses[str(ply)] = content
                        except Exception:
                            pass
                    if os.path.exists(prompt_path):
                        try:
                            with open(prompt_path, "r") as f:
                                prompts[str(ply)] = f.read()
                        except Exception:
                            pass

            # 将 thinking 分数与合法走法配对（供 viewer 分数条形图使用）
            move_scores = {}
            for ply_str, think_text in thinking.items():
                ply_idx = int(ply_str)
                try:
                    scores = [int(s) for s in think_text.split()]
                    board_at = chess.Board(fens[ply_idx])
                    legal = list(board_at.legal_moves)
                    if len(scores) == len(legal):
                        chosen_uci = game["pgn_moves"][ply_idx]
                        entries = []
                        for lm, sc in zip(legal, scores):
                            try:
                                sn = board_at.san(lm)
                            except Exception:
                                sn = lm.uci()
                            entries.append({
                                "u": lm.uci(), "s": sn,
                                "v": sc, "c": lm.uci() == chosen_uci,
                            })
                        entries.sort(key=lambda x: x["v"], reverse=True)
                        move_scores[ply_str] = entries
                except Exception:
                    pass

            # Stockfish 后分析
            evals_list = None
            cp_losses = None
            acpl = None
            worst_cp = None
            if analyze:
                try:
                    evals_list = _analyze_game_positions(
                        fens, stockfish_path, depth)
                    # 逐步计算 centipawn loss
                    cp_losses = []
                    llm_losses = []
                    for pi in range(len(game["pgn_moves"])):
                        is_w = (pi % 2 == 0)
                        eb, ea = evals_list[pi], evals_list[pi + 1]
                        loss = max(0, eb - ea) if is_w else max(0, ea - eb)
                        cp_losses.append(round(loss))
                        if (is_w == game["llm_is_white"]):
                            llm_losses.append(loss)
                    acpl = round(sum(llm_losses) / len(llm_losses), 1) \
                        if llm_losses else 1000
                    worst_cp = max(llm_losses) if llm_losses else 0
                except Exception as e:
                    print(f"  [Viewer] 分析失败 game {global_game_num}: {e}")

            matches.append({
                "id": global_game_num - 1,
                "skill_level": sk,
                "opponent": f"stockfish-skill{sk}",
                "opponent_elo": opp_elo,
                "llm_is_white": game["llm_is_white"],
                "result": game["result_str"],
                "reason": game["reason"],
                "total_moves": game["moves"],
                "fens": fens,
                "sans": sans,
                "ucis": game["pgn_moves"],
                "thinking": thinking,
                "move_scores": move_scores,
                "raw": raw_responses,
                "prompts": prompts,
                "evals": evals_list,
                "cp_losses": cp_losses,
                "acpl": acpl,
                "worst_cp": worst_cp,
            })

            if global_game_num % 10 == 0:
                print(f"  [Viewer] 已处理 {global_game_num} 局...")

    viewer_data = {
        "matches": matches,
        "elo_history": elo_history,
        "final_elo": final_elo,
        "config": config,
    }

    viewer_path = os.path.join(run_dir, "viewer_data.json")
    with open(viewer_path, "w") as f:
        json.dump(viewer_data, f, ensure_ascii=False)
    print(f"  Viewer data saved to: {viewer_path}")
    return viewer_path


# ---------------------------------------------------------------------------
# PGN 导出（供 BayesElo / Ordo 使用）
# ---------------------------------------------------------------------------

def export_pgn(all_games: list, output_path: str):
    """导出 PGN 文件，供 BayesElo/Ordo 计算 Elo。"""
    with open(output_path, "w") as f:
        for g in all_games:
            white_name = g["white"]
            black_name = g["black"]
            if g["result_str"] == "win":
                pgn_result = "1-0" if g["llm_is_white"] else "0-1"
            elif g["result_str"] == "loss":
                pgn_result = "0-1" if g["llm_is_white"] else "1-0"
            else:
                pgn_result = "1/2-1/2"

            f.write(f'[Event "ELO Evaluation"]\n')
            f.write(f'[White "{white_name}"]\n')
            f.write(f'[Black "{black_name}"]\n')
            f.write(f'[Result "{pgn_result}"]\n')
            f.write(f'\n')

            # 按编号格式写入着法
            board = chess.Board()
            move_strs = []
            for i, uci in enumerate(g["pgn_moves"]):
                move = chess.Move.from_uci(uci)
                try:
                    san = board.san(move)
                except Exception:
                    san = uci
                if board.turn == chess.WHITE:
                    move_strs.append(f"{board.fullmove_number}. {san}")
                else:
                    move_strs.append(san)
                board.push(move)

            f.write(" ".join(move_strs))
            f.write(f" {pgn_result}\n\n")


# ---------------------------------------------------------------------------
# 评测主流程
# ---------------------------------------------------------------------------

def evaluate_at_level(llm, stockfish_path, skill_level, n_games, time_limit=2.0,
                      depth: Optional[int] = None,
                      initial_elo: float = 1500.0, K: float = 20.0,
                      on_game_end=None, move_callback=None, llm_move_callback=None):
    """在指定 Skill Level 下进行 n_games 局对弈（支持实时 Elo 更新回调）。

    on_game_end(result, reason, moves, new_elo)               每局结束后调用。
    move_callback(board, last_move, label)                     每步棋后调用。
    llm_move_callback(board_before, move, thinking, raw_response,
                      prompt, half_move_num, game_num_in_level)  每步 LLM 走法后调用。
    返回 final_elo_after_level 供主循环传递给下一级。
    """
    estimated_elo = stockfish_skill_to_elo(skill_level)
    wins, draws, losses = 0, 0, 0
    illegal_count = 0
    games_detail = []
    elo = initial_elo       # 本级迭代 Elo，从主循环传入

    for i in range(n_games):
        llm_is_white = (i % 2 == 0)
        color = "W" if llm_is_white else "B"
        llm_color_str = "白" if llm_is_white else "黑"

        # 构造每步棋回调（携带当前对局信息）
        game_move_cb = None
        if move_callback is not None:
            def _make_move_cb(game_idx, llm_white, sk, opp_elo):
                def cb(board, last_move, move_cnt):
                    color_tag = "White" if llm_white else "Black"
                    lbl = (f"SK{sk} (~{opp_elo:.0f})  "
                           f"Game {game_idx}/{n_games}  LLM={color_tag}  "
                           f"Move {move_cnt}")
                    move_callback(board, last_move, lbl)
                return cb
            game_move_cb = _make_move_cb(i + 1, llm_is_white,
                                         skill_level, estimated_elo)

        # 构造本局 LLM 走法回调（携带 game_num_in_level）
        game_llm_move_cb = None
        if llm_move_callback is not None:
            def _make_llm_cb(game_idx):
                def cb(board_before, move, thinking, raw_response, prompt, half_move_num):
                    llm_move_callback(board_before, move, thinking,
                                      raw_response, prompt, half_move_num, game_idx)
                return cb
            game_llm_move_cb = _make_llm_cb(i + 1)

        sf = StockfishOpponent(skill_level, stockfish_path, time_limit, depth)
        try:
            t0 = time.time()
            result = play_one_game(llm, sf, llm_is_white,
                                   move_callback=game_move_cb,
                                   llm_move_callback=game_llm_move_cb)
            elapsed = time.time() - t0
        finally:
            sf.close()

        r = result["result"]
        if r == "win":
            wins += 1
            symbol = "+"
            RA = 1.0
        elif r == "draw":
            draws += 1
            symbol = "="
            RA = 0.5
        else:
            losses += 1
            symbol = "-"
            RA = 0.0
            if result["reason"] == "illegal_move":
                illegal_count += 1

        # 逐盘更新 Elo（与 compute_final_elo 公式相同）
        RE  = 1.0 / (1.0 + 10.0 ** ((estimated_elo - elo) / 400.0))
        elo = elo + K * (RA - RE)

        print(
            f"  [{i+1:3d}/{n_games}] {color} {symbol}  "
            f"{result['reason']:<22s} {result['moves']:3d} moves  "
            f"{elapsed:5.1f}s  Elo≈{elo:.0f}"
        )

        sf_name = f"Stockfish_SK{skill_level}"
        game_detail = {
            "llm_is_white": llm_is_white,
            "white": "ChessLLM" if llm_is_white else sf_name,
            "black": sf_name if llm_is_white else "ChessLLM",
            "result_str": r,
            "reason": result["reason"],
            "moves": result["moves"],
            "pgn_moves": result["pgn_moves"],
        }
        games_detail.append(game_detail)

        # 触发实时可视化回调（传入完整 game_detail 供 LiveViewer 使用）
        if on_game_end is not None:
            on_game_end(
                result=r,
                reason=result["reason"],
                moves=result["moves"],
                new_elo=elo,
                game_detail=game_detail,
            )

    total = wins + draws + losses
    score = (wins + 0.5 * draws) / total if total > 0 else 0.0

    return {
        "skill_level": skill_level,
        "opponent_elo": round(estimated_elo),
        "games": total,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "illegal_moves": illegal_count,
        "score": round(score, 4),
        "games_detail": games_detail,
        "final_elo_after_level": round(elo, 1),
    }


def compute_final_elo(results: list, initial_elo: float = 1500.0,
                      K: float = 20.0) -> dict:
    """跨级连续迭代 Elo（与论文 Zhang et al. 2501.17186v2 一致）。

    从 initial_elo 开始，按照对弈顺序逐盘更新 Elo，上一级的终点 Elo
    直接作为下一级的起点（跨级连续累积）。
    final_elo = 最后一盘结束后的 Elo（连续迭代终点）。
    """
    elo = initial_elo
    elo_history = [elo]
    per_level_elo = []
    total_wins = total_draws = total_losses = 0

    for r in results:
        opp_elo = stockfish_skill_to_elo(r["skill_level"])
        for g in r["games_detail"]:
            RA = {"win": 1.0, "draw": 0.5, "loss": 0.0}[g["result_str"]]
            RE = 1.0 / (1.0 + 10.0 ** ((opp_elo - elo) / 400.0))
            elo = elo + K * (RA - RE)
            elo_history.append(elo)
            if g["result_str"] == "win":
                total_wins += 1
            elif g["result_str"] == "draw":
                total_draws += 1
            else:
                total_losses += 1
        per_level_elo.append(round(elo))

    # 最终 Elo = 连续迭代终点（与论文一致）
    final_elo = elo

    # 用胜率的二项分布标准误差推导 Elo 不确定性（1-sigma）
    # σ_Elo = 400/ln(10) * sqrt(p*(1-p)/N)
    N = total_wins + total_draws + total_losses
    if N > 0:
        p = (total_wins + 0.5 * total_draws) / N
        p = max(1e-6, min(1 - 1e-6, p))   # 避免 p=0 或 p=1 时退化
        sd = (400.0 / math.log(10)) * math.sqrt(p * (1 - p) / N)
    else:
        sd = 0.0

    return {
        "final_elo": round(final_elo),
        "initial_elo": round(initial_elo),
        "K": K,
        "per_level_elo": per_level_elo,
        "std_dev": round(sd, 1),
        "elo_history": [round(e, 1) for e in elo_history],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Chess LLM ELO by playing against Stockfish at multiple levels"
    )
    parser.add_argument(
        "--games-per-level", type=int, default=50,
        help="Number of games per ELO level (default: 50, recommend >=100 for accuracy)"
    )
    parser.add_argument(
        "--stockfish-path", type=str, default="/usr/games/stockfish",
        help="Path to Stockfish binary"
    )
    parser.add_argument(
        "--skill-levels", type=str, default="0,1,2",
        help="Comma-separated Stockfish Skill Levels (0-20) to test against. "
             "Paper uses 0,1,2. (default: 0,1,2)"
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="vLLM server port (default: 8000)"
    )
    parser.add_argument(
        "--time-limit", type=float, default=2.0,
        help="Stockfish time limit per move in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--depth", type=int, default=None,
        help="Stockfish search depth per move for gameplay (e.g. 1, 5). "
             "If set, overrides --time-limit for opponent moves."
    )
    parser.add_argument(
        "--max-moves", type=int, default=200,
        help="Maximum moves per game (default: 200)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file path (default: elo_results_<timestamp>.json)"
    )
    parser.add_argument(
        "--pgn", type=str, default=None,
        help="Output PGN file for all games (for use with bayeselo/ordo)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save all outputs (default: elo_run_<timestamp>/)"
    )
    parser.add_argument(
        "--no-analyze", action="store_true",
        help="Skip Stockfish post-analysis for viewer (faster, no ACPL/eval bar)"
    )
    parser.add_argument(
        "--analysis-depth", type=int, default=20,
        help="Stockfish analysis depth for viewer data (default: 20)"
    )
    args = parser.parse_args()

    skill_levels = sorted(int(x) for x in args.skill_levels.split(","))

    # 校验 Skill Level
    for sk in skill_levels:
        if not (0 <= sk <= 20):
            print(f"ERROR: Skill level {sk} not in range 0-20.")
            sys.exit(1)

    # 初始化 LLM
    config = EvalConfig()
    config.port = args.port
    llm = ChessLLM(config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 创建本次运行的输出目录
    run_dir = Path(args.output_dir or f"elo_run_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Skill Level → Elo 映射
    print("=" * 70)
    print("  CHESS LLM ELO EVALUATION")
    print("=" * 70)
    print(f"  Time:            {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Games/level:     {args.games_per_level}")
    print(f"  Skill levels:    {skill_levels}")
    print(f"  Stockfish:       {args.stockfish_path}")
    print(f"  vLLM port:       {args.port}")
    if args.depth is not None:
        print(f"  Depth:           {args.depth} (overrides time limit; "
              f"SK5+ actual Elo may be lower than formula estimate)")
    else:
        print(f"  Time limit:      {args.time_limit}s per move")
    print(f"  Output dir:      {run_dir}/")
    print()
    print("  Skill Level → Estimated Elo (equations 3-4):")
    for sk in skill_levels:
        elo_est = stockfish_skill_to_elo(sk)
        elo_next = stockfish_skill_to_elo(sk + 1) if sk < 20 else elo_est
        print(f"    SK {sk:2d} → {elo_est:.0f} - {elo_next:.0f}")
    print("=" * 70)

    # 初始化实时可视化
    opponent_elos  = {sk: round(stockfish_skill_to_elo(sk)) for sk in skill_levels}
    live_plot_path = str(run_dir / "elo_live.png")
    viz            = LiveVisualizer(skill_levels, args.games_per_level,
                                    opponent_elos, live_plot_path)

    # 每步走法可视化的输出目录（按 skill level 分子目录）
    moves_dir = run_dir / "moves"
    moves_dir.mkdir(exist_ok=True)

    K            = 20.0   # 论文标准 (Zhang et al. 2501.17186v2, Section 5.1)
    current_elo  = 1500.0  # 跨级连续累积，从 1500 起步
    all_results  = []
    all_games_for_pgn = []
    game_offset  = [0]   # 跨等级的全局局号偏移（用闭包引用）
    t_total_start = time.time()

    # 实时 Viewer 写入器（每局结束写 viewer_data.json）
    viewer_cfg = {
        "games_per_level": args.games_per_level,
        "skill_levels": skill_levels,
        "time_limit": args.time_limit,
    }
    viewer_writer = LiveViewerWriter(
        str(run_dir), str(moves_dir), viewer_cfg,
        stockfish_path=args.stockfish_path,
        analysis_depth=args.analysis_depth,
        analyze=not args.no_analyze,
    )
    # 复制 viewer.html 到输出目录（方便一站式启动 HTTP server）
    _viewer_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer.html")
    _viewer_dst = str(run_dir / "viewer.html")
    if os.path.exists(_viewer_src) and not os.path.exists(_viewer_dst):
        import shutil
        shutil.copy2(_viewer_src, _viewer_dst)
        print(f"  [LiveViewer] viewer.html copied to {run_dir}/")
    print(f"  [LiveViewer] Start HTTP server to watch live:")
    print(f"    cd {run_dir} && python -m http.server 8080")

    for sk in skill_levels:
        elo_est = stockfish_skill_to_elo(sk)
        print(f"\n{'='*60}")
        print(f"  VS Stockfish Skill Level {sk}  (est. Elo ~{elo_est:.0f})")
        print(f"{'='*60}")

        # 本等级走法图存放目录
        sk_moves_dir = moves_dir / f"sk{sk}"
        sk_moves_dir.mkdir(exist_ok=True)

        # 工厂函数捕获当前 sk，避免闭包引用循环变量
        def _make_cb(skill_lv):
            def cb(result, reason, moves, new_elo, game_detail=None):
                viz.update(skill_lv, result, reason, moves, new_elo)
                if game_detail is not None:
                    viewer_writer.on_game(skill_lv, game_detail, new_elo)
            return cb

        def _make_llm_move_cb(skill_lv, sk_dir, offset):
            """为每步 LLM 走法生成可视化并保存到 sk_dir。"""
            def cb(board_before, move, thinking, raw_response, prompt,
                   half_move_num, game_num_in_level):
                g = offset[0] + game_num_in_level
                base = str(sk_dir / f"g{g:04d}_m{half_move_num:03d}")
                viz.save_move_detail(board_before, move, thinking,
                                     raw_response, prompt, base + ".png")
            return cb

        result = evaluate_at_level(
            llm, args.stockfish_path, sk,
            args.games_per_level, args.time_limit,
            depth=args.depth,
            initial_elo=current_elo, K=K,  # 跨级连续累积
            on_game_end=_make_cb(sk),
            move_callback=viz.on_move if viz.enabled else None,
            llm_move_callback=(_make_llm_move_cb(sk, sk_moves_dir, game_offset)
                                if viz.enabled else None),
        )
        current_elo = result['final_elo_after_level']  # 带入下一级
        game_offset[0] += args.games_per_level
        all_results.append(result)
        all_games_for_pgn.extend(result["games_detail"])

        print(f"\n  W/D/L:          {result['wins']}/{result['draws']}/{result['losses']}")
        print(f"  Score:          {result['score']:.3f}")
        print(f"  Illegal moves:  {result['illegal_moves']}")
        print(f"  Elo after level: {result['final_elo_after_level']:.0f}  (连续累积)")

    total_time = time.time() - t_total_start

    # 逐盘迭代计算 Elo（与实时更新结果一致）
    final = compute_final_elo(all_results, K=K)

    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY  (Paper: Zhang et al. 2501.17186v2)")
    print("  EloN = EloO + K*(RA - RE),  RE = 1/(1+10^((EloS-EloM)/400))")
    print(f"  Initial Elo = {final['initial_elo']},  K = {final['K']}")
    print("=" * 70)
    print(f"\n  {'SK':<4} {'Opp Elo':<10} {'W/D/L':<14} {'Score':<8} {'Elo after level'}")
    print("  " + "-" * 55)
    for r, elo_snap in zip(all_results, final["per_level_elo"]):
        wdl = f"{r['wins']}/{r['draws']}/{r['losses']}"
        print(f"  {r['skill_level']:<4} {r['opponent_elo']:<10} {wdl:<14} {r['score']:<8.3f} {elo_snap}")

    print(f"\n  Final Elo:   {final['final_elo']}  (± {final['std_dev']:.0f})")
    print(f"  Total time:  {total_time:.0f}s ({total_time/60:.1f}min)")
    print("=" * 70)

    # 保存 JSON 结果
    output_file = args.output or str(run_dir / "results.json")
    results_for_json = []
    for r in all_results:
        rj = {k: v for k, v in r.items() if k != "games_detail"}
        # 保存紧凑的逐局记录（供可视化/分析使用）
        rj["game_results"] = [g["result_str"] for g in r["games_detail"]]
        rj["game_reasons"] = [g["reason"]     for g in r["games_detail"]]
        rj["game_moves"]   = [g["moves"]      for g in r["games_detail"]]
        results_for_json.append(rj)

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "method": "Continuous iterative Elo update (Zhang et al. 2501.17186v2, Eq.1-2, K=20)",
        "note": "Cross-level continuous accumulation, consistent with paper",
        "config": {
            "games_per_level": args.games_per_level,
            "skill_levels": skill_levels,
            "stockfish_path": args.stockfish_path,
            "port": args.port,
            "time_limit": args.time_limit,
            "initial_elo": final["initial_elo"],
            "K": final["K"],
        },
        "per_level": results_for_json,
        "per_level_elo": final["per_level_elo"],
        "elo_history": final["elo_history"],
        "final_elo": final["final_elo"],
        "final_std": final["std_dev"],
        "total_time_sec": round(total_time, 1),
        "live_plot": live_plot_path if viz.enabled else None,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n  JSON results saved to: {output_file}")
    if viz.enabled:
        print(f"  Live plot saved to:   {live_plot_path}")

    # 导出 PGN
    pgn_file = args.pgn or str(run_dir / "games.pgn")
    export_pgn(all_games_for_pgn, pgn_file)
    print(f"  PGN games saved to:   {pgn_file}")

    # 生成最终 viewer_data.json（含 Stockfish 后分析，覆盖 live 版本）
    print(f"\n  Generating final viewer data (with Stockfish analysis)...")
    viewer_path = save_viewer_data(
        all_results, str(run_dir), args.stockfish_path,
        final["elo_history"], final["final_elo"],
        output_data["config"],
        analyze=not args.no_analyze,
        depth=args.analysis_depth,
    )

    print(f"\n  Output directory: {run_dir}/")
    print(f"  View: cd {run_dir} && python -m http.server 8080")
    print(f"\n  To compute ELO with BayesElo:")
    print(f"    bayeselo < commands.txt")
    print(f"  Or with Ordo:")
    print(f"    ordo -p {pgn_file} -a 0 -A 'ChessLLM'")


if __name__ == "__main__":
    main()
