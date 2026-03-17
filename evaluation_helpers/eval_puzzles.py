"""Evaluation 1: Puzzle Solving (<600 ELO)"""

import chess
import csv
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from evaluation_helpers.eval_config import vprint


def solve_puzzle(llm, puzzle_data, config):
    """Solve a single puzzle. Returns (solved, puzzle_info)"""
    puzzle_id, rating, fen, moves_str = puzzle_data
    
    try:
        # Parse the moves
        moves = moves_str.strip().split()
        if len(moves) < 2:
            raise ValueError("Not enough moves")
        
        # Create board from FEN
        board = chess.Board(fen)
        
        # Play the puzzle: opponent move, llm move, opponent move, llm move, ...
        for i, expected_move in enumerate(moves):
            if i % 2 == 0:
                # Opponent's move - play it directly
                try:
                    move = chess.Move.from_uci(expected_move)
                    if move not in board.legal_moves:
                        raise ValueError(f"Invalid opponent move at step {i}")
                    board.push(move)
                except ValueError:
                    raise ValueError(f"Cannot parse opponent move at step {i}")
            else:
                # LLM's move - get prediction and check if correct
                llm_move_uci, thinking = llm.get_move(board, temperature=config.temperature)
                
                if llm_move_uci != expected_move:
                    # Wrong move - puzzle failed
                    return False, {
                        "puzzle_id": puzzle_id, 
                        "rating": rating,
                        "failed_at_move": i + 1,
                        "expected": expected_move,
                        "got": llm_move_uci
                    }
                
                # Correct move - play it
                try:
                    move = chess.Move.from_uci(llm_move_uci)
                    board.push(move)
                except ValueError:
                    return False, {"puzzle_id": puzzle_id, "error": "Invalid LLM move format"}
        
        # All moves played correctly - puzzle solved!
        return True, {"puzzle_id": puzzle_id, "rating": rating}
        
    except Exception as e:
        return False, {"puzzle_id": puzzle_id, "error": str(e)}


def evaluate_puzzles(llm, config):
    """Evaluate LLM on chess puzzles under 600 ELO"""
    t_start = time.time()
    vprint(config, "\n" + "="*80)
    vprint(config, "EVALUATION 1: PUZZLE SOLVING (<600 ELO)")
    vprint(config, "="*80)
    
    vprint(config, f"Loading puzzles from {config.puzzle_file}...")
    
    # Load and filter puzzles
    puzzles = []
    try:
        df = pd.read_csv(config.puzzle_file)
        df = df[df['Rating'] < config.puzzle_max_elo]
        puzzles = [
            (row['PuzzleId'], int(row['Rating']), row['FEN'], row['Moves'])
            for _, row in df.head(config.n_puzzles).iterrows()
        ]
    except FileNotFoundError:
        vprint(config, f"Error: Puzzle file '{config.puzzle_file}' not found")
        return {
            "total_puzzles": 0,
            "solved": 0,
            "failed": 0,
            "success_rate": 0.0,
            "time_taken_sec": time.time() - t_start
        }
    
    vprint(config, f"Loaded {len(puzzles)} puzzles with rating < {config.puzzle_max_elo}")
    
    if len(puzzles) == 0:
        vprint(config, "No puzzles to evaluate")
        return {
            "total_puzzles": 0,
            "solved": 0,
            "failed": 0,
            "success_rate": 0.0,
            "time_taken_sec": time.time() - t_start
        }
    
    # Solve puzzles in parallel
    vprint(config, f"Solving {len(puzzles)} puzzles...")
    
    def solve_puzzle_wrapper(idx_puzzle):
        """Wrapper for parallel execution"""
        idx, puzzle = idx_puzzle
        vprint(config, f"Puzzle {idx+1}/{len(puzzles)} (ID: {puzzle[0]}, Rating: {puzzle[1]})...", end=" ", flush=True)
        solved, info = solve_puzzle(llm, puzzle, config)
        if solved:
            vprint(config, "✓ Solved")
        else:
            vprint(config, f"✗ Failed - {info.get('error', info.get('expected', 'unknown error'))}")
        return solved, info
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        puzzle_results = list(executor.map(solve_puzzle_wrapper, enumerate(puzzles)))
    
    # Aggregate results
    solved_count = sum(1 for solved, _ in puzzle_results if solved)
    failed_count = len(puzzles) - solved_count
    success_rate = (solved_count / len(puzzles)) * 100
    
    vprint(config, "\n" + "-"*80)
    vprint(config, "RESULTS:")
    vprint(config, f"  Total puzzles: {len(puzzles)}")
    vprint(config, f"  Solved: {solved_count}/{len(puzzles)} ({success_rate:.1f}%)")
    vprint(config, f"  Failed: {failed_count}/{len(puzzles)} ({(failed_count/len(puzzles))*100:.1f}%)")
    
    t_end = time.time()
    results = {
        "total_puzzles": len(puzzles),
        "success_rate_puzzles": success_rate,
        "time_puzzles": t_end - t_start
    }
    
    return results

