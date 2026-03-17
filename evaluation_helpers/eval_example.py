"""Example runner showing how to use the evaluation system"""

from eval_config import EvalConfig
from eval_llm import ChessLLM
from eval_puzzles import evaluate_puzzles
from eval_vs_random import evaluate_vs_random
from eval_vs_stockfish import evaluate_vs_stockfish
from run_evaluation import run_full_evaluation


def example_quick_eval():
    """Example: Quick evaluation with fewer games"""
    print("=== Quick Evaluation Example ===\n")
    
    # Create custom config
    config = EvalConfig()
    config.n_puzzles = 50  # Only 50 puzzles
    config.n_random_games = 5  # Only 5 games vs random
    config.n_stockfish_games = 5  # Only 5 games vs stockfish
    config.verbose = True  # Show progress
    
    # Run full evaluation
    results = run_full_evaluation(config)
    return results


def example_single_evaluation():
    """Example: Run only puzzle evaluation"""
    print("=== Single Evaluation Example (Puzzles Only) ===\n")
    
    # Create config
    config = EvalConfig()
    config.n_puzzles = 100
    config.verbose = True
    
    # Initialize LLM
    llm = ChessLLM(config)
    
    # Run only puzzle evaluation
    results = evaluate_puzzles(llm, config)
    
    print(f"\nPuzzle Success Rate: {results['success_rate']:.1f}%")
    print(f"Solved: {results['solved']}/{results['total_puzzles']}")
    return results


def example_custom_model():
    """Example: Evaluate a different model checkpoint"""
    print("=== Custom Model Example ===\n")
    
    # Create config with custom model
    config = EvalConfig()
    config.model_name = "/path/to/your/custom/model"  # Change this!
    config.n_puzzles = 20
    config.n_random_games = 3
    config.n_stockfish_games = 3
    config.verbose = True
    
    # Run evaluation
    results = run_full_evaluation(config)
    return results


def example_vs_random_only():
    """Example: Evaluate only vs random player"""
    print("=== VS Random Player Only ===\n")
    
    # Create config
    config = EvalConfig()
    config.n_random_games = 20
    config.max_moves_random = 150  # Allow longer games
    config.verbose = True
    
    # Initialize LLM
    llm = ChessLLM(config)
    
    # Run evaluation
    results = evaluate_vs_random(llm, config)
    
    print(f"\nWin Rate: {results['win_rate']:.1f}%")
    print(f"Avg Moves: {results['avg_moves']:.1f}")
    print(f"Illegal Moves: {results['illegal_moves']}")
    return results


def example_vs_stockfish_only():
    """Example: Evaluate only vs Stockfish"""
    print("=== VS Stockfish Only ===\n")
    
    # Create config
    config = EvalConfig()
    config.n_stockfish_games = 10
    config.stockfish_skill_level = 1  # Slightly harder
    config.verbose = True
    
    # Initialize LLM
    llm = ChessLLM(config)
    
    # Run evaluation
    results = evaluate_vs_stockfish(llm, config)
    
    print(f"\nWin Rate: {results['win_rate']:.1f}%")
    print(f"Draw Rate: {results['draw_rate']:.1f}%")
    print(f"Avg Blunder Rate: {results['avg_blunder_rate']:.1f}%")
    print(f"Avg CP Advantage: {results['avg_cp_advantage']:.1f}")
    return results


if __name__ == "__main__":
    # Choose which example to run
    print("Choose an example:")
    print("1. Quick evaluation (fewer games)")
    print("2. Puzzle evaluation only")
    print("3. VS Random player only")
    print("4. VS Stockfish only")
    print("5. Custom model (edit code first!)")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        example_quick_eval()
    elif choice == "2":
        example_single_evaluation()
    elif choice == "3":
        example_vs_random_only()
    elif choice == "4":
        example_vs_stockfish_only()
    elif choice == "5":
        example_custom_model()
    else:
        print("Invalid choice. Running quick eval by default.")
        example_quick_eval()

