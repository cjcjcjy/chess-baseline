"""Main runner for Chess LLM evaluation suite"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime

from evaluation_helpers.eval_config import EvalConfig, vprint
from evaluation_helpers.eval_puzzles import evaluate_puzzles
from evaluation_helpers.eval_vs_random import evaluate_vs_random
from evaluation_helpers.eval_vs_stockfish import evaluate_vs_stockfish
from chess_llm import ChessLLM


def run_full_evaluation(config):
    """Run all three evaluations and save results"""
    vprint(config, "\n" + "="*80)
    vprint(config, "CHESS LLM EVALUATION")
    vprint(config, f"Model: {config.model_name}")
    vprint(config, f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    vprint(config, "="*80)

    # Initialize LLM
    vprint(config, f"\nLoading model: {config.model_name}")
    llm = ChessLLM(config)

    all_results = {}
    t0 = time.time()

    # Run evaluations
    try:
        results = evaluate_puzzles(llm, config)
        all_results.update(results)
    except Exception as e:
        print(f"\nPuzzle evaluation failed: {e}")

    try:   
        results = evaluate_vs_random(llm, config)
        all_results.update(results)
    except Exception as e:
        print(f"\nRandom evaluation failed: {e}")

    try:   
        results = evaluate_vs_stockfish(llm, config)
        all_results.update(results)
    except Exception as e:
        print(f"\nStockfish evaluation failed: {e}")

    # Save results
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)

    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f"eval_results_{timestamp_str}.json"

    total_seconds = time.time() - t0
    all_results["total_eval_time"] = total_seconds

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    vprint(config, "\n" + "="*80)
    vprint(config, f"Results saved to: {output_file}")
    vprint(config, f"Total evaluation time: {total_seconds:.2f} seconds")
    vprint(config, "="*80)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Chess LLM Evaluation Suite")
    parser.add_argument("--eval",
                        choices=["all", "puzzles", "random", "stockfish"],
                        default="all", help="Which evaluation to run")
    parser.add_argument("--model", type=str, help="Override model path")
    parser.add_argument("--n-puzzles", type=int, help="Number of puzzles to evaluate")
    parser.add_argument("--n-random", type=int, help="Number of games vs random")
    parser.add_argument("--n-stockfish", type=int, help="Number of games vs Stockfish")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Create config
    config = EvalConfig()
    
    # Override config if specified
    if args.verbose:
        config.verbose = True
    if args.model:
        config.model_name = args.model
    if args.n_puzzles:
        config.n_puzzles = args.n_puzzles
    if args.n_random:
        config.n_random_games = args.n_random
    if args.n_stockfish:
        config.n_stockfish_games = args.n_stockfish
    
    # Initialize LLM
    llm = ChessLLM(config)
    
    # Run requested evaluation
    if args.eval == "all":
        run_full_evaluation(config)
    elif args.eval == "puzzles":
        t0 = time.time()
        results = evaluate_puzzles(llm, config)
        print(json.dumps(results, indent=2))
        print(f"Puzzle evaluation time: {results.get('time_taken_sec', time.time()-t0):.1f}s")
    elif args.eval == "random":
        t0 = time.time()
        results = evaluate_vs_random(llm, config)
        print(json.dumps(results, indent=2))
        print(f"Random evaluation time: {results.get('total_time_sec', time.time()-t0):.1f}s")
    elif args.eval == "stockfish":
        t0 = time.time()
        results = evaluate_vs_stockfish(llm, config)
        print(json.dumps(results, indent=2))
        print(f"Stockfish evaluation time: {results.get('total_time_sec', time.time()-t0):.1f}s")


if __name__ == "__main__":
    main()

