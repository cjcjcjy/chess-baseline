"""Evaluation 3: VS Stockfish"""

import os
import sys
import chess
import chess.engine
import time
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from evaluation_helpers.eval_config import vprint

# Add chess-env to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../chess-env"))

from agents import ChessAgent
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


class StockfishAgent(ChessAgent):
    """Simple Stockfish agent using python-chess engine"""
    
    def __init__(self, depth: int = 1, skill_level: int = 0, time_limit_ms: int = 100):
        """
        Initialize Stockfish agent.
        
        Args:
            depth: Search depth for Stockfish
            skill_level: Skill level (0-20, lower is weaker)
            time_limit_ms: Time limit in milliseconds
        """
        self.depth = depth
        self.skill_level = skill_level
        self.time_limit_ms = time_limit_ms
        self.engine = chess.engine.SimpleEngine.popen_uci("stockfish")
        if skill_level is not None:
            self.engine.configure({"Skill Level": skill_level})
    
    def choose_move(
        self,
        board: chess.Board,
        legal_moves: List[chess.Move],
        move_history: List[str],
        side_to_move: str,
    ) -> Tuple[Optional[chess.Move], Optional[str]]:
        """Choose a move using Stockfish"""
        if not legal_moves:
            return None, "No legal moves available"
        
        try:
            result = self.engine.play(
                board,
                chess.engine.Limit(depth=self.depth, time=self.time_limit_ms / 1000.0)
            )
            return result.move, "Stockfish move"
        except Exception as e:
            print(f"Stockfish error: {e}")
            return None, f"Stockfish error: {e}"
    
    def close(self):
        """Close the Stockfish engine"""
        if hasattr(self, 'engine'):
            self.engine.quit()


def play_vs_stockfish(llm, llm_is_white, config):
    """Play one game vs Stockfish using chess-env"""
    t_start = time.time()
    
    # Create agents
    llm_agent = LLMChessAgent(llm)
    stockfish_agent = StockfishAgent(
        depth=config.stockfish_depth,
        skill_level=config.stockfish_skill_level,
        time_limit_ms=int(config.stockfish_time * 1000)
    )
    
    try:
        # Set up agents based on colors
        if llm_is_white:
            white_agent = llm_agent
            black_agent = stockfish_agent
        else:
            white_agent = stockfish_agent
            black_agent = llm_agent
        
        # Create environment
        env = ChessEnvironment(white_agent, black_agent, max_moves=200, time_limit=30.0)
        
        # Play the game
        result = env.play_game(verbose=False)
        
        t_end = time.time()
        total_time = t_end - t_start
        
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
        
        # Determine winner from LLM's perspective
        if illegal_move:
            # If LLM resigned or made illegal move, Stockfish wins
            if llm_is_white and ("White resigns" in game_result or "White illegal" in game_result):
                winner = "stockfish"
            elif not llm_is_white and ("Black resigns" in game_result or "Black illegal" in game_result):
                winner = "stockfish"
            else:
                # Stockfish resigned or made illegal move, LLM wins
                winner = "llm"
        elif "White wins" in game_result or "Black wins" in game_result:
            # Someone won (by checkmate or other means)
            if (llm_is_white and "White wins" in game_result) or (not llm_is_white and "Black wins" in game_result):
                winner = "llm"
            else:
                winner = "stockfish"
        else:
            # Draw or other outcome
            winner = "draw"
        
        return {
            "winner": winner,
            "illegal_move": illegal_move,
            "total_moves": moves_played,
            "time_taken_sec": total_time,
            "acpl": player_acpl
        }
    finally:
        # Clean up Stockfish engine
        stockfish_agent.close()


def evaluate_vs_stockfish(llm, config):
    """Evaluate LLM vs Stockfish"""
    t_start = time.time()
    vprint(config, "\n" + "="*80)
    vprint(config, "EVALUATION 3: VS STOCKFISH")
    vprint(config, "="*80)
    vprint(config, f"Games: {config.n_stockfish_games} ({config.n_stockfish_games//2} White, {config.n_stockfish_games - config.n_stockfish_games//2} Black)")
    vprint(config, f"Stockfish: Skill Level {config.stockfish_skill_level}, "
          f"Depth {config.stockfish_depth}, Time {config.stockfish_time}s")
    
    wins = 0
    losses = 0
    draws = 0
    illegal_count = 0
    per_game_times = []
    move_counts = []
    acpl_values = []
    
    def play_game_wrapper(game_idx):
        """Wrapper to play a single game and return results with game info"""
        llm_is_white = (game_idx < config.n_stockfish_games // 2)
        color_name = "White" if llm_is_white else "Black"
        vprint(config, f"\nGame {game_idx+1}/{config.n_stockfish_games} (LLM as {color_name})...", end=" ", flush=True)
        
        result = play_vs_stockfish(llm, llm_is_white, config)
        per_game_times.append(result.get("time_taken_sec", 0.0))
        
        if result["illegal_move"]:
            outcome = "Loss (illegal)"
        elif result["winner"] == "llm":
            outcome = "Win"
        elif result["winner"] == "draw":
            outcome = "Draw"
        else:
            outcome = "Loss"
        
        vprint(config, f"{outcome} - {result['total_moves']} moves, "
              f"ACPL: {result['acpl']:.1f} ({result['time_taken_sec']:.1f}s)")
        
        return result
    
    # Run games in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=10) as executor:
        game_results = list(executor.map(play_game_wrapper, range(config.n_stockfish_games)))
    
    # Aggregate results
    for result in game_results:
        move_counts.append(result["total_moves"])
        acpl_values.append(result["acpl"])
        if result["illegal_move"]:
            illegal_count += 1
            losses += 1
        elif result["winner"] == "llm":
            wins += 1
        elif result["winner"] == "draw":
            draws += 1
        else:
            losses += 1
    
    # Aggregate metrics
    total_games = config.n_stockfish_games
    win_rate = (wins / total_games) * 100 if total_games > 0 else 0
    draw_rate = (draws / total_games) * 100 if total_games > 0 else 0
    loss_rate = (losses / total_games) * 100 if total_games > 0 else 100
    avg_moves = sum(move_counts) / len(move_counts) if move_counts else 0
    avg_acpl = sum(acpl_values) / len(acpl_values) if acpl_values else 0

    t_end = time.time()
    total_time = t_end - t_start

    vprint(config, "\n" + "-"*80)
    vprint(config, "RESULTS:")
    vprint(config, f"  Win rate: {wins}/{total_games} ({win_rate:.1f}%)")
    vprint(config, f"  Draw rate: {draws}/{total_games} ({draw_rate:.1f}%)")
    vprint(config, f"  Loss rate: {losses}/{total_games} ({loss_rate:.1f}%)")
    vprint(config, f"  Illegal moves: {illegal_count}/{total_games} ({(illegal_count/total_games)*100:.1f}%)")
    vprint(config, f"  Average moves per game: {avg_moves:.1f}")
    vprint(config, f"  Average ACPL: {avg_acpl:.1f} centipawns")
    vprint(config, f"  Total evaluation time: {total_time:.1f}s")

    results = {
        "illegal_moves_vs_stockfish": illegal_count,
        "win_rate_vs_stockfish": win_rate,
        "draw_rate_vs_stockfish": draw_rate,
        "loss_rate_vs_stockfish": loss_rate,
        "avg_moves_vs_stockfish": avg_moves,
        "avg_acpl_vs_stockfish": avg_acpl,
        "total_time_sec": total_time,
    }
    
    return results

