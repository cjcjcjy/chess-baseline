"""Evaluation 2: VS Random Legal Moves"""

import os
import sys
import chess
import time
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from evaluation_helpers.eval_config import vprint

# Add chess-env to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../chess-env"))

from agents import ChessAgent, RandomAgent
from env import ChessEnvironment
from run_game import _StockfishAnalyzer


class LLMChessAgent(ChessAgent):
    """Wrapper to adapt LLM to ChessAgent interface"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def choose_move(
        self,
        board: chess.Board,
        legal_moves: List[chess.Move],
        move_history: List[str],
        side_to_move: str,
    ) -> Tuple[Optional[chess.Move], Optional[str]]:
        """Choose a move using the LLM"""
        if not legal_moves:
            return None, "No legal moves available"
        
        move, thinking, illegal = self.llm.try_move(board)
        if illegal:
            return None, "Illegal move"
        return move, thinking


def play_vs_random(llm, llm_is_white, config):
    """Play one game vs random legal moves using chess-env"""
    t_start = time.time()
    
    # Create agents
    llm_agent = LLMChessAgent(llm)
    random_agent = RandomAgent()
    
    # Set up agents based on colors
    if llm_is_white:
        white_agent = llm_agent
        black_agent = random_agent
    else:
        white_agent = random_agent
        black_agent = llm_agent
    
    # Create environment
    env = ChessEnvironment(white_agent, black_agent, max_moves=config.max_moves_random, time_limit=30.0)
    
    # Play the game
    result = env.play_game(verbose=False)
    
    t_end = time.time()
    duration = t_end - t_start
    
    # Determine outcome
    game_result = result["result"]
    moves_played = result["moves_played"]
    move_history = result["move_history"]
    
    # Analyze ACPL
    white_acpl = 0.0
    black_acpl = 0.0
    if len(move_history) > 0:
        try:
            analyzer = _StockfishAnalyzer(depth=20, movetime_ms=1000)
            acpl_result = analyzer.analyze_game(move_history)
            white_acpl = acpl_result["white_acpl"]
            black_acpl = acpl_result["black_acpl"]
        except Exception as e:
            print(f"Warning: ACPL analysis failed: {e}")
    
    # Determine player ACPL based on color
    player_acpl = white_acpl if llm_is_white else black_acpl
    
    # Check for resignation or illegal move
    illegal_move = "resign" in game_result.lower() or "illegal" in game_result.lower()
    
    # Determine win/loss from LLM's perspective
    if illegal_move:
        # If LLM resigned or made illegal move, LLM loses
        if llm_is_white and ("White resigns" in game_result or "White illegal" in game_result):
            outcome = "loss"
        elif not llm_is_white and ("Black resigns" in game_result or "Black illegal" in game_result):
            outcome = "loss"
        else:
            # Random agent resigned or made illegal move, LLM wins
            outcome = "win"
    elif "White wins" in game_result or "Black wins" in game_result:
        # Someone won (by checkmate or other means)
        if (llm_is_white and "White wins" in game_result) or (not llm_is_white and "Black wins" in game_result):
            outcome = "win"
        else:
            outcome = "loss"
    else:
        # Draw or other outcome
        outcome = "draw"
    
    return outcome, moves_played, illegal_move, duration, player_acpl


def evaluate_vs_random(llm, config):
    """Evaluate LLM vs random legal moves"""
    t_start = time.time()
    vprint(config, "\n" + "="*80)
    vprint(config, "EVALUATION 2: VS RANDOM LEGAL MOVES")
    vprint(config, "="*80)
    vprint(config, f"Games: {config.n_random_games} ({config.n_random_games//2} White, {config.n_random_games - config.n_random_games//2} Black)")
    vprint(config, f"Max moves per game: {config.max_moves_random}")
    
    wins = 0
    draws = 0
    losses = 0
    illegal_count = 0
    move_counts = []
    game_times = []
    acpl_values = []
    
    def play_game_wrapper(game_idx):
        """Wrapper to play a single game and return results with game info"""
        llm_is_white = (game_idx < config.n_random_games // 2)
        color_name = "White" if llm_is_white else "Black"

        vprint(config, f"\nGame {game_idx+1}/{config.n_random_games} (LLM as {color_name})...", end=" ", flush=True)
        
        result, moves, illegal, duration, player_acpl = play_vs_random(llm, llm_is_white, config)
        
        if illegal:
            vprint(config, f"Loss (illegal move) - {moves} moves, ACPL: {player_acpl:.1f} ({duration:.1f}s)")
        elif result == "win":
            vprint(config, f"Win - {moves} moves, ACPL: {player_acpl:.1f} ({duration:.1f}s)")
        elif result == "draw":
            vprint(config, f"Draw - {moves} moves, ACPL: {player_acpl:.1f} ({duration:.1f}s)")
        else:
            vprint(config, f"Loss - {moves} moves, ACPL: {player_acpl:.1f} ({duration:.1f}s)")
        
        return {
            "result": result,
            "moves": moves,
            "illegal": illegal,
            "game_idx": game_idx,
            "time_sec": duration,
            "acpl": player_acpl
        }
    
    # Run games in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=10) as executor:
        game_results = list(executor.map(play_game_wrapper, range(config.n_random_games)))
    
    # Aggregate results
    for game_result in game_results:
        move_counts.append(game_result["moves"])
        game_times.append(game_result["time_sec"])
        acpl_values.append(game_result["acpl"])
        if game_result["illegal"]:
            illegal_count += 1
            losses += 1
        elif game_result["result"] == "win":
            wins += 1
        elif game_result["result"] == "draw":
            draws += 1
        else:
            losses += 1
    
    win_rate = (wins / config.n_random_games) * 100 if config.n_random_games > 0 else 0
    draw_rate = (draws / config.n_random_games) * 100 if config.n_random_games > 0 else 0
    loss_rate = (losses / config.n_random_games) * 100 if config.n_random_games > 0 else 0
    avg_moves = sum(move_counts) / len(move_counts) if move_counts else 0
    avg_duration = sum(game_times) / len(game_times) if game_times else 0
    avg_acpl = sum(acpl_values) / len(acpl_values) if acpl_values else 0
    
    vprint(config, "\n" + "-"*80)
    vprint(config, "RESULTS:")
    vprint(config, f"  Wins: {wins}/{config.n_random_games} ({win_rate:.1f}%)")
    vprint(config, f"  Draws: {draws}/{config.n_random_games} ({draw_rate:.1f}%)")
    vprint(config, f"  Losses: {losses}/{config.n_random_games} ({loss_rate:.1f}%)")
    vprint(config, f"  Illegal moves: {illegal_count}/{config.n_random_games} ({(illegal_count/config.n_random_games)*100:.1f}%)")
    vprint(config, f"  Avg moves per game: {avg_moves:.1f}")
    vprint(config, f"  Average ACPL: {avg_acpl:.1f}")
    vprint(config, f"  Avg game duration: {avg_duration:.1f}s")
    
    t_end = time.time()
    results = {
        "win_rate_vs_random": win_rate,
        "draw_rate_vs_random": draw_rate,
        "loss_rate_vs_random": loss_rate,
        "avg_moves_vs_random": avg_moves,
        "avg_acpl_vs_random": avg_acpl,
        "illegal_moves_vs_random": illegal_count,
        "total_time_sec": t_end - t_start
    }
    
    return results

