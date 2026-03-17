"""Configuration for chess LLM evaluation"""
from jinja2 import Environment, BaseLoader


class EvalConfig:
    """Configuration for evaluation runs"""
    def __init__(self):
        # Model settings
        self.model_name = "aicrowd-chess-model"
        # self.model_name = "aicrowd-chess-model-novp"
        self.port = 8000

        env = Environment(
            loader=BaseLoader(),
            autoescape=False,
            trim_blocks=False,
            lstrip_blocks=False,
        )

        self.chess_template = env.from_string(open('data_preparation/chess_encode_special_tokens.jinja').read())
        # self.chess_template = env.from_string(open('/home/jcyang/global-chess-challenge-2025-starter-kit/player_agents/sft_agent_prompt_template.jinja').read())
        
        # Puzzle evaluation settings
        self.n_puzzles = 200
        self.puzzle_max_elo = 600
        self.puzzle_file = "data/puzzles.csv"
        
        # Random player evaluation settings
        self.n_random_games = 5
        self.max_moves_random = 200  # counting both players' moves
        
        # Stockfish evaluation settings
        self.n_stockfish_games = 50
        self.stockfish_depth = 1
        self.stockfish_time = 0.05
        self.stockfish_skill_level = 0
        
        # LLM settings
        self.max_length = 1200
        self.temperature = 0.3
        self.retry_temperature = 0.3
        self.max_retries = 3
        self.max_new_tokens = 10000
        
        # Output settings
        self.output_dir = "eval_results"
        self.verbose = False
        

        
    def to_dict(self):
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


def vprint(config, *args, **kwargs):
    """Print only if verbose mode is enabled"""
    if config.verbose:
        print(*args, **kwargs)