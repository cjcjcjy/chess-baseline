#!/usr/bin/env python3
"""
evaluate_elo_h2h.py — Head-to-head matches: Our model vs ChessGPT-Base/Chat.

Reports win rates and computes Elo from direct confrontation,
comparable to ChessLLM paper Table 3.

================================================================================
Prerequisites
================================================================================

  1. Start vLLM for our model (port 8000):
     CUDA_VISIBLE_DEVICES=0 vllm serve ./qwen3_4b_new_format_val_adaptk_dp12/checkpoint-90000 \
         --served-model-name aicrowd-chess-model --port 8000 \
         --dtype bfloat16 --max-model-len 1200 --enforce-eager &

  2. Start vLLM for ChessGPT-Base (port 8001):
     CUDA_VISIBLE_DEVICES=1 vllm serve ./chessgpt-base-v1 \
         --served-model-name chessgpt-base --port 8001 \
         --dtype float16 --max-model-len 2048 --enforce-eager &

  3. (Optional) Start vLLM for ChessGPT-Chat (port 8002):
     CUDA_VISIBLE_DEVICES=2 vllm serve ./chessgpt-chat-v1 \
         --served-model-name chessgpt-chat --port 8002 \
         --dtype float16 --max-model-len 2048 --enforce-eager &

================================================================================
Usage
================================================================================

    # Our model vs ChessGPT-Base, 100 games
    python evaluate_elo_h2h.py --games 100 \
        --our-port 8000 --opp-port 8001 --opp-type base --opp-name chessgpt-base

    # Our model vs ChessGPT-Chat
    python evaluate_elo_h2h.py --games 100 \
        --our-port 8000 --opp-port 8002 --opp-type chat --opp-name chessgpt-chat
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys
import math
import time
import json
import argparse
import chess
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chess_llm import ChessLLM
from chessgpt_player import ChessGPTPlayer
from evaluation_helpers.eval_config import EvalConfig


# ---------------------------------------------------------------------------
# LLM vs LLM game
# ---------------------------------------------------------------------------

def play_llm_vs_llm(white_player, black_player, max_moves: int = 200) -> dict:
    """Play one game between two LLM players.

    Both players must implement try_move(board) → (move, thinking, illegal, raw, prompt).

    Returns result from WHITE's perspective:
        "white_win", "black_win", or "draw"
    """
    board = chess.Board()
    move_count = 0
    game_moves = []

    while not board.is_game_over() and move_count < max_moves:
        player = white_player if board.turn == chess.WHITE else black_player
        side = "white" if board.turn == chess.WHITE else "black"

        move, thinking, illegal, raw_response, prompt = player.try_move(board)

        if illegal or move is None:
            # Player failed to produce legal move → loses
            winner = "black_win" if board.turn == chess.WHITE else "white_win"
            return {
                "result": winner,
                "reason": f"{side}_illegal_move",
                "moves": move_count,
                "pgn_moves": game_moves,
                "final_fen": board.fen(),
            }

        board.push(move)
        game_moves.append(move.uci())
        move_count += 1

    # Determine result
    if board.is_game_over():
        outcome = board.outcome()
        if outcome is None or outcome.winner is None:
            result = "draw"
        elif outcome.winner == chess.WHITE:
            result = "white_win"
        else:
            result = "black_win"
        reason = _termination_reason(board)
    else:
        result = "draw"
        reason = "max_moves"

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
# Elo from head-to-head (two-player iterative update)
# ---------------------------------------------------------------------------

def compute_h2h_elo(game_results, K=20.0, initial_elo_a=1500.0, initial_elo_b=1500.0):
    """Compute Elo for two players from head-to-head results.

    game_results: list of ("a_win", "b_win", "draw")
    Returns (final_elo_a, final_elo_b, history_a, history_b).
    """
    elo_a = initial_elo_a
    elo_b = initial_elo_b
    hist_a = [elo_a]
    hist_b = [elo_b]

    for result in game_results:
        # Expected scores
        ea = 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))
        eb = 1.0 - ea

        # Actual scores
        if result == "a_win":
            sa, sb = 1.0, 0.0
        elif result == "b_win":
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5

        elo_a += K * (sa - ea)
        elo_b += K * (sb - eb)
        hist_a.append(elo_a)
        hist_b.append(elo_b)

    return elo_a, elo_b, hist_a, hist_b


# ---------------------------------------------------------------------------
# PGN export
# ---------------------------------------------------------------------------

def export_h2h_pgn(games, output_path, white_name, black_name):
    """Export head-to-head games to PGN."""
    with open(output_path, "w") as f:
        for i, g in enumerate(games):
            # Determine PGN result
            if g["result"] == "white_win":
                pgn_result = "1-0"
            elif g["result"] == "black_win":
                pgn_result = "0-1"
            else:
                pgn_result = "1/2-1/2"

            # White/black names depend on who played which side
            w = g.get("white_name", white_name)
            b = g.get("black_name", black_name)

            f.write(f'[Event "H2H Match Game {i+1}"]\n')
            f.write(f'[White "{w}"]\n')
            f.write(f'[Black "{b}"]\n')
            f.write(f'[Result "{pgn_result}"]\n\n')

            board = chess.Board()
            parts = []
            for uci in g["pgn_moves"]:
                move = chess.Move.from_uci(uci)
                try:
                    san = board.san(move)
                except Exception:
                    san = uci
                if board.turn == chess.WHITE:
                    parts.append(f"{board.fullmove_number}. {san}")
                else:
                    parts.append(san)
                board.push(move)
            f.write(" ".join(parts) + f" {pgn_result}\n\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Head-to-head: Our model vs ChessGPT"
    )
    # Our model
    parser.add_argument("--our-port", type=int, default=8000,
                        help="vLLM port for our model (default: 8000)")
    parser.add_argument("--our-name", type=str, default="aicrowd-chess-model",
                        help="vLLM model name for our model")
    # Opponent
    parser.add_argument("--opp-port", type=int, default=8001,
                        help="vLLM port for ChessGPT (default: 8001)")
    parser.add_argument("--opp-name", type=str, default="chessgpt-base",
                        help="vLLM model name for ChessGPT")
    parser.add_argument("--opp-type", type=str, default="base",
                        choices=["base", "chat"],
                        help="Opponent model type: base=PGN completion, chat=conversational")
    parser.add_argument("--our-label", type=str, default=None,
                        help="Display label for player A (default: 'Ours')")
    parser.add_argument("--our-type", type=str, default=None,
                        choices=["base", "chat"],
                        help="If set, player A is a ChessGPTPlayer (for baseline vs baseline)")
    parser.add_argument("--our-template", type=str, default=None,
                        help="Jinja template path for our model (overrides EvalConfig default)")
    parser.add_argument("--opp-label", type=str, default=None,
                        help="Display label for opponent (default: auto from opp-name)")
    parser.add_argument("--opp-template", type=str, default=None,
                        help="Jinja template path for opponent ChessLLM (when set, use ChessLLM instead of ChessGPTPlayer)")
    # Match settings
    parser.add_argument("--games", type=int, default=100,
                        help="Total number of games (default: 100)")
    parser.add_argument("--max-moves", type=int, default=200,
                        help="Max moves per game (default: 200)")
    parser.add_argument("--max-retries", type=int, default=50,
                        help="ChessGPT max retries for illegal moves (default: 50)")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    # Initialize player A
    if args.our_type:
        our_player = ChessGPTPlayer(
            port=args.our_port,
            model_name=args.our_name,
            model_type=args.our_type,
            max_retries=args.max_retries,
        )
    elif args.our_template:
        from jinja2 import Environment, BaseLoader
        env = Environment(loader=BaseLoader(), autoescape=False,
                          trim_blocks=False, lstrip_blocks=False)
        config = EvalConfig()
        config.port = args.our_port
        config.model_name = args.our_name
        config.chess_template = env.from_string(open(args.our_template).read())
        our_player = ChessLLM(config)
    else:
        config = EvalConfig()
        config.port = args.our_port
        config.model_name = args.our_name
        our_player = ChessLLM(config)

    # Initialize player B
    if args.opp_template:
        from jinja2 import Environment, BaseLoader
        env = Environment(loader=BaseLoader(), autoescape=False,
                          trim_blocks=False, lstrip_blocks=False)
        opp_config = EvalConfig()
        opp_config.port = args.opp_port
        opp_config.model_name = args.opp_name
        opp_config.chess_template = env.from_string(
            open(args.opp_template).read())
        opp_player = ChessLLM(opp_config)
    else:
        opp_player = ChessGPTPlayer(
            port=args.opp_port,
            model_name=args.opp_name,
            model_type=args.opp_type,
            max_retries=args.max_retries,
        )

    our_label = args.our_label or "Ours"
    opp_label = args.opp_label or args.opp_name

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir or f"h2h_{args.opp_type}_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  HEAD-TO-HEAD: {our_label} vs {opp_label}")
    print("=" * 70)
    print(f"  Our model:    {args.our_name} (port {args.our_port})")
    print(f"  Opponent:     {args.opp_name} (port {args.opp_port}, type={args.opp_type})")
    print(f"  Games:        {args.games} (alternating colors)")
    print(f"  Output:       {run_dir}/")
    print("=" * 70)

    games = []
    elo_results = []  # "a_win" / "b_win" / "draw" where a=ours, b=opponent
    our_wins, opp_wins, draws = 0, 0, 0
    our_illegal, opp_illegal = 0, 0
    t_start = time.time()

    for i in range(args.games):
        # Alternate colors: even games → our model is white
        our_is_white = (i % 2 == 0)

        if our_is_white:
            white_p, black_p = our_player, opp_player
            white_name, black_name = our_label, opp_label
        else:
            white_p, black_p = opp_player, our_player
            white_name, black_name = opp_label, our_label

        t0 = time.time()
        result = play_llm_vs_llm(white_p, black_p, args.max_moves)

        # Retry on repetition draws (up to 2 retries to get a decisive result)
        retry_count = 0
        while result["reason"] == "repetition" and retry_count < 2:
            retry_count += 1
            result = play_llm_vs_llm(white_p, black_p, args.max_moves)

        elapsed = time.time() - t0

        # Determine outcome from our model's perspective
        if our_is_white:
            if result["result"] == "white_win":
                our_result = "win"
                elo_results.append("a_win")
                our_wins += 1
            elif result["result"] == "black_win":
                our_result = "loss"
                elo_results.append("b_win")
                opp_wins += 1
            else:
                our_result = "draw"
                elo_results.append("draw")
                draws += 1
        else:
            if result["result"] == "black_win":
                our_result = "win"
                elo_results.append("a_win")
                our_wins += 1
            elif result["result"] == "white_win":
                our_result = "loss"
                elo_results.append("b_win")
                opp_wins += 1
            else:
                our_result = "draw"
                elo_results.append("draw")
                draws += 1

        # Track illegal moves
        if "illegal" in result.get("reason", ""):
            if "white" in result["reason"] and our_is_white:
                our_illegal += 1
            elif "black" in result["reason"] and not our_is_white:
                our_illegal += 1
            else:
                opp_illegal += 1

        color = "W" if our_is_white else "B"
        symbol = {"win": "+", "loss": "-", "draw": "="}[our_result]
        retry_tag = f" (retry {retry_count})" if retry_count > 0 else ""
        print(
            f"  [{i+1:3d}/{args.games}] {color} {symbol}  "
            f"{result['reason']:<22s} {result['moves']:3d} moves  "
            f"{elapsed:5.1f}s  "
            f"W{our_wins} D{draws} L{opp_wins}{retry_tag}"
        )

        game_record = {
            **result,
            "our_is_white": our_is_white,
            "white_name": white_name,
            "black_name": black_name,
            "our_result": our_result,
        }
        games.append(game_record)

    total_time = time.time() - t_start

    # Compute Elo
    elo_a, elo_b, hist_a, hist_b = compute_h2h_elo(elo_results)

    # Win rate (from our perspective)
    total = our_wins + opp_wins + draws
    our_score = (our_wins + 0.5 * draws) / total if total > 0 else 0.5
    opp_score = 1.0 - our_score

    # Standard error of win rate
    if total > 0:
        se = math.sqrt(our_score * (1 - our_score) / total)
    else:
        se = 0

    # Elo difference from win rate
    if 0 < our_score < 1:
        elo_diff = -400 * math.log10(1 / our_score - 1)
    elif our_score >= 1:
        elo_diff = 800  # cap
    else:
        elo_diff = -800

    # Summary
    print(f"\n{'='*70}")
    print(f"  RESULTS: {our_label} vs {opp_label}  ({total} games)")
    print(f"{'='*70}")
    print(f"  {our_label:>20s}:  W {our_wins}  D {draws}  L {opp_wins}  "
          f"Score: {our_score:.3f} ± {se:.3f}")
    print(f"  {opp_label:>20s}:  W {opp_wins}  D {draws}  L {our_wins}  "
          f"Score: {opp_score:.3f}")
    print(f"  Win rate ({our_label}):  {our_score*100:.1f}%  ± {se*100:.1f}%")
    print(f"  Elo difference:       {elo_diff:+.0f} ({our_label} - {opp_label})")
    print(f"  Iterative Elo:        {our_label}={elo_a:.0f}  {opp_label}={elo_b:.0f}")
    print(f"  Illegal moves:        {our_label}={our_illegal}  {opp_label}={opp_illegal}")
    print(f"  Total time:           {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"{'='*70}")

    # Save results
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "matchup": f"{our_label} vs {opp_label}",
        "config": {
            "our_model": args.our_name,
            "our_port": args.our_port,
            "opp_model": args.opp_name,
            "opp_port": args.opp_port,
            "opp_type": args.opp_type,
            "games": args.games,
            "max_moves": args.max_moves,
        },
        "results": {
            "our_wins": our_wins,
            "opp_wins": opp_wins,
            "draws": draws,
            "our_score": round(our_score, 4),
            "our_score_se": round(se, 4),
            "elo_diff": round(elo_diff, 1),
            "our_elo_iterative": round(elo_a, 1),
            "opp_elo_iterative": round(elo_b, 1),
            "our_illegal": our_illegal,
            "opp_illegal": opp_illegal,
        },
        "elo_history_ours": [round(e, 1) for e in hist_a],
        "elo_history_opp": [round(e, 1) for e in hist_b],
        "total_time_sec": round(total_time, 1),
    }

    json_path = str(run_dir / "results.json")
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n  JSON saved to: {json_path}")

    # Save PGN
    pgn_path = str(run_dir / "games.pgn")
    export_h2h_pgn(games, pgn_path, our_label, opp_label)
    print(f"  PGN saved to:  {pgn_path}")


if __name__ == "__main__":
    main()
