#!/usr/bin/env python3
"""
evaluate_elo_chessgpt.py — Evaluate ChessGPT-Base/Chat Elo against Stockfish.

Reuses the Elo evaluation pipeline from evaluate_elo.py, but uses ChessGPTPlayer
(PGN input + SAN output) instead of ChessLLM (special token encoding).

================================================================================
Usage
================================================================================

Step 1: Download models from HuggingFace

    pip install huggingface_hub
    huggingface-cli download Waterhorse/chessgpt-base-v1 --local-dir ./chessgpt-base-v1
    huggingface-cli download Waterhorse/chessgpt-chat-v1 --local-dir ./chessgpt-chat-v1

Step 2: Start vLLM server (GPT-NeoX, completions API)

    # ChessGPT-Base (port 8001)
    CUDA_VISIBLE_DEVICES=1 vllm serve ./chessgpt-base-v1 \
        --served-model-name chessgpt-base \
        --port 8001 --dtype float16 --max-model-len 2048 \
        --enforce-eager > vllm_chessgpt_base.log 2>&1 &

    # ChessGPT-Chat (port 8002)
    CUDA_VISIBLE_DEVICES=1 vllm serve ./chessgpt-chat-v1 \
        --served-model-name chessgpt-chat \
        --port 8002 --dtype float16 --max-model-len 2048 \
        --enforce-eager > vllm_chessgpt_chat.log 2>&1 &

Step 3: Run evaluation

    # Evaluate ChessGPT-Base
    python evaluate_elo_chessgpt.py --model-type base --port 8001 \
        --model-name chessgpt-base --games-per-level 100

    # Evaluate ChessGPT-Chat
    python evaluate_elo_chessgpt.py --model-type chat --port 8002 \
        --model-name chessgpt-chat --games-per-level 100
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import sys
import time
import json
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chessgpt_player import ChessGPTPlayer
from evaluate_elo import (
    play_one_game,
    StockfishOpponent,
    stockfish_skill_to_elo,
    compute_final_elo,
    export_pgn,
)


def evaluate_at_level(player, stockfish_path, skill_level, n_games,
                      time_limit=2.0, depth=None,
                      initial_elo=1500.0, K=20.0):
    """Run n_games at given Stockfish Skill Level.
    Simplified version of evaluate_elo.evaluate_at_level (no visualization)."""
    estimated_elo = stockfish_skill_to_elo(skill_level)
    wins, draws, losses = 0, 0, 0
    illegal_count = 0
    games_detail = []
    elo = initial_elo

    for i in range(n_games):
        llm_is_white = (i % 2 == 0)
        color = "W" if llm_is_white else "B"

        sf = StockfishOpponent(skill_level, stockfish_path, time_limit, depth)
        try:
            t0 = time.time()
            result = play_one_game(player, sf, llm_is_white)
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

        RE = 1.0 / (1.0 + 10.0 ** ((estimated_elo - elo) / 400.0))
        elo = elo + K * (RA - RE)

        print(
            f"  [{i+1:3d}/{n_games}] {color} {symbol}  "
            f"{result['reason']:<22s} {result['moves']:3d} moves  "
            f"{elapsed:5.1f}s  Elo≈{elo:.0f}"
        )

        sf_name = f"Stockfish_SK{skill_level}"
        games_detail.append({
            "llm_is_white": llm_is_white,
            "white": "ChessGPT" if llm_is_white else sf_name,
            "black": sf_name if llm_is_white else "ChessGPT",
            "result_str": r,
            "reason": result["reason"],
            "moves": result["moves"],
            "pgn_moves": result["pgn_moves"],
        })

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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ChessGPT Elo by playing against Stockfish"
    )
    parser.add_argument(
        "--model-type", type=str, default="base", choices=["base", "chat"],
        help="ChessGPT model type: 'base' or 'chat' (default: base)"
    )
    parser.add_argument(
        "--model-name", type=str, default="chessgpt-base",
        help="vLLM served model name (default: chessgpt-base)"
    )
    parser.add_argument(
        "--port", type=int, default=8001,
        help="vLLM server port (default: 8001)"
    )
    parser.add_argument(
        "--games-per-level", type=int, default=50,
        help="Number of games per Skill Level (default: 50)"
    )
    parser.add_argument(
        "--skill-levels", type=str, default="0,1,2",
        help="Comma-separated Stockfish Skill Levels (default: 0,1,2)"
    )
    parser.add_argument(
        "--stockfish-path", type=str, default="/usr/games/stockfish",
        help="Path to Stockfish binary"
    )
    parser.add_argument(
        "--time-limit", type=float, default=2.0,
        help="Stockfish time limit per move in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--depth", type=int, default=None,
        help="Stockfish search depth (overrides --time-limit)"
    )
    parser.add_argument(
        "--max-retries", type=int, default=50,
        help="Max retries for illegal moves (default: 50)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.3,
        help="Sampling temperature (default: 0.3, consistent with our model)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: elo_chessgpt_<timestamp>/)"
    )
    args = parser.parse_args()

    skill_levels = sorted(int(x) for x in args.skill_levels.split(","))

    # Initialize ChessGPT player
    player = ChessGPTPlayer(
        port=args.port,
        model_name=args.model_name,
        model_type=args.model_type,
        max_retries=args.max_retries,
        temperature=args.temperature,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir or f"elo_chessgpt_{args.model_type}_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  CHESSGPT ELO EVALUATION ({args.model_type.upper()})")
    print("=" * 70)
    print(f"  Time:            {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Model:           {args.model_name} (type={args.model_type})")
    print(f"  Port:            {args.port}")
    print(f"  Games/level:     {args.games_per_level}")
    print(f"  Skill levels:    {skill_levels}")
    print(f"  Max retries:     {args.max_retries}")
    print(f"  Temperature:     {args.temperature}")
    if args.depth is not None:
        print(f"  SF Depth:        {args.depth}")
    else:
        print(f"  SF Time limit:   {args.time_limit}s")
    print(f"  Output dir:      {run_dir}/")
    print()
    print("  Skill Level → Estimated Elo:")
    for sk in skill_levels:
        elo_est = stockfish_skill_to_elo(sk)
        print(f"    SK {sk:2d} → ~{elo_est:.0f}")
    print("=" * 70)

    K = 20.0
    current_elo = 1500.0
    all_results = []
    all_games = []
    t_start = time.time()

    for sk in skill_levels:
        elo_est = stockfish_skill_to_elo(sk)
        print(f"\n{'='*60}")
        print(f"  VS Stockfish Skill Level {sk}  (est. Elo ~{elo_est:.0f})")
        print(f"{'='*60}")

        result = evaluate_at_level(
            player, args.stockfish_path, sk,
            args.games_per_level, args.time_limit,
            depth=args.depth,
            initial_elo=current_elo, K=K,
        )
        current_elo = result["final_elo_after_level"]
        all_results.append(result)
        all_games.extend(result["games_detail"])

        print(f"\n  W/D/L:          {result['wins']}/{result['draws']}/{result['losses']}")
        print(f"  Score:          {result['score']:.3f}")
        print(f"  Illegal moves:  {result['illegal_moves']}")
        print(f"  Elo after level: {result['final_elo_after_level']:.0f}")

    total_time = time.time() - t_start
    final = compute_final_elo(all_results, K=K)

    # Summary
    print("\n" + "=" * 70)
    print(f"  RESULTS SUMMARY — ChessGPT-{args.model_type.capitalize()}")
    print("=" * 70)
    print(f"\n  {'SK':<4} {'Opp Elo':<10} {'W/D/L':<14} {'Score':<8} {'Elo'}")
    print("  " + "-" * 55)
    for r, elo_snap in zip(all_results, final["per_level_elo"]):
        wdl = f"{r['wins']}/{r['draws']}/{r['losses']}"
        print(f"  {r['skill_level']:<4} {r['opponent_elo']:<10} {wdl:<14} {r['score']:<8.3f} {elo_snap}")

    print(f"\n  Final Elo:   {final['final_elo']}  (± {final['std_dev']:.0f})")
    print(f"  Total time:  {total_time:.0f}s ({total_time/60:.1f}min)")
    print("=" * 70)

    # Save JSON
    output_file = str(run_dir / "results.json")
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "model": f"ChessGPT-{args.model_type.capitalize()}",
        "config": {
            "model_name": args.model_name,
            "model_type": args.model_type,
            "port": args.port,
            "games_per_level": args.games_per_level,
            "skill_levels": skill_levels,
            "max_retries": args.max_retries,
            "temperature": args.temperature,
            "time_limit": args.time_limit,
            "depth": args.depth,
        },
        "per_level": [
            {k: v for k, v in r.items() if k != "games_detail"}
            for r in all_results
        ],
        "per_level_elo": final["per_level_elo"],
        "elo_history": final["elo_history"],
        "final_elo": final["final_elo"],
        "final_std": final["std_dev"],
        "total_time_sec": round(total_time, 1),
    }
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {output_file}")

    # Save PGN
    pgn_file = str(run_dir / "games.pgn")
    export_pgn(all_games, pgn_file)
    print(f"  PGN saved to:    {pgn_file}")


if __name__ == "__main__":
    main()
