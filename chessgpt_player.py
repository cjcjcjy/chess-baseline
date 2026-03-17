"""ChessGPT player adapter for Elo evaluation.

Wraps ChessGPT-Base / ChessGPT-Chat (GPT-NeoX 2.8B) models served via vLLM
to implement the same try_move(board) interface used by evaluate_elo.py.

Key differences from ChessLLM:
  - Input: PGN game history (not special token encoding)
  - Output: SAN move (not <uci_move> tag)
  - API: /v1/completions (not /v1/chat/completions) — GPT-NeoX is a completion model
  - High illegal move rate — uses up to max_retries sampling attempts
"""

import re
import chess
import chess.pgn
from openai import OpenAI


def board_to_pgn_string(board: chess.Board) -> str:
    """Reconstruct PGN move list from board's move stack.

    Returns e.g. '1. e4 e5 2. Nf3 Nc6 3. Bb5'
    """
    temp = chess.Board()
    parts = []
    for move in board.move_stack:
        if temp.turn == chess.WHITE:
            parts.append(f"{temp.fullmove_number}.")
        parts.append(temp.san(move))
        temp.push(move)
    return " ".join(parts)


def build_chessgpt_prompt(board: chess.Board, model_type: str = "base") -> str:
    """Build the prompt for ChessGPT move generation.

    For 'base': raw PGN completion (model continues the move sequence).
    For 'chat': conversational format with <|endoftext|> delimiters.

    The PGN headers follow ChessGPT's General Policy format.
    """
    pgn_moves = board_to_pgn_string(board)

    # Determine whose turn and build partial move number if needed
    if board.turn == chess.WHITE:
        pgn_with_prompt = f"{pgn_moves} {board.fullmove_number}." if pgn_moves else "1."
    else:
        pgn_with_prompt = pgn_moves

    # PGN headers (simplified — ChessGPT attends to Elo weakly)
    color = "black" if board.turn == chess.BLACK else "white"
    headers = (
        '[Event "Rated Classical game"]\n'
        '[Date "2024.01.01"]\n'
        '[White "???"]\n'
        '[Black "???"]\n'
        '[Result "*"]\n'
        '[WhiteElo "2000"]\n'
        '[BlackElo "2000"]\n'
    )

    if model_type == "chat":
        # ChessGPT-Chat conversational format
        inner = (
            f"In the following chess game, you play {color}. "
            f"What is the best next move?\n\n"
            f"{headers}\n{pgn_with_prompt}"
        )
        return (
            f"A chess game between two players.\n<|endoftext|>"
            f"Human 0: {inner}\n<|endoftext|>"
            f"Human 1:"
        )
    else:
        # ChessGPT-Base: raw PGN completion
        return f"{headers}\n{pgn_with_prompt}"


def parse_san_from_output(text: str, board: chess.Board):
    """Try to extract a legal SAN move from ChessGPT's output text.

    Attempts multiple strategies:
      1. First whitespace-delimited token (most common)
      2. Strip move numbers (e.g. '23. Nf3' → 'Nf3', '23... Bg7' → 'Bg7')
      3. Regex for SAN-like patterns anywhere in text
    """
    text = text.strip()
    if not text:
        return None

    # Strategy 1: first token
    first_token = text.split()[0].rstrip(".,;!?")

    # Strip move number prefix like "23." or "23..."
    first_token = re.sub(r'^\d+\.{1,3}\s*', '', first_token)

    if first_token:
        try:
            move = board.parse_san(first_token)
            if move in board.legal_moves:
                return move
        except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
            pass

    # Strategy 2: try each token in the output
    for token in text.split()[:10]:
        token = token.rstrip(".,;!?")
        token = re.sub(r'^\d+\.{1,3}\s*', '', token)
        if not token:
            continue
        try:
            move = board.parse_san(token)
            if move in board.legal_moves:
                return move
        except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
            pass

    # Strategy 3: regex for SAN patterns (e.g. Nf3, exd5, O-O, Qxh7+)
    san_pattern = r'\b(O-O-O|O-O|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)\b'
    for match in re.finditer(san_pattern, text):
        try:
            move = board.parse_san(match.group(1))
            if move in board.legal_moves:
                return move
        except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
            pass

    return None


class ChessGPTPlayer:
    """ChessGPT-Base/Chat adapter with the same interface as ChessLLM.

    Usage:
        player = ChessGPTPlayer(port=8001, model_name="chessgpt-base")
        move, thinking, illegal, raw_response, prompt = player.try_move(board)
    """

    def __init__(self, port: int = 8001, model_name: str = "chessgpt-base",
                 model_type: str = "base", max_retries: int = 50,
                 temperature: float = 0.3, retry_temperature: float = 0.3):
        self.client = OpenAI(
            base_url=f"http://localhost:{port}/v1",
            api_key="EMPTY",
        )
        self.model_name = model_name
        self.model_type = model_type  # "base" or "chat"
        self.max_retries = max_retries
        self.temperature = temperature
        self.retry_temperature = retry_temperature
        self._logged = False

    def get_move(self, board: chess.Board, temperature: float = None):
        """Generate a single move attempt from ChessGPT.

        Returns (move_or_None, raw_response, prompt).
        """
        if temperature is None:
            temperature = self.temperature

        prompt = build_chessgpt_prompt(board, self.model_type)

        response = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            temperature=temperature,
            max_tokens=20,
            stop=["\n", "<|endoftext|>"],
        )
        raw_text = response.choices[0].text.strip()

        if not self._logged:
            print(f"  [ChessGPTPlayer] model={self.model_name} type={self.model_type} "
                  f"temp={self.temperature} retries={self.max_retries}")
            print(f"  [ChessGPTPlayer] sample prompt (first 200): {prompt[:200]}...")
            print(f"  [ChessGPTPlayer] sample output: {repr(raw_text)}")
            self._logged = True

        move = parse_san_from_output(raw_text, board)
        return move, raw_text, prompt

    def try_move(self, board: chess.Board):
        """Try to get a legal move with retries. Same interface as ChessLLM.try_move.

        Returns (move, thinking, illegal, raw_response, prompt).
        thinking is always None for ChessGPT (no value prediction).
        """
        last_raw = None
        last_prompt = None

        for attempt in range(self.max_retries):
            temp = self.temperature if attempt == 0 else self.retry_temperature
            move, raw_text, prompt = self.get_move(board, temperature=temp)
            last_raw = raw_text
            last_prompt = prompt

            if move is not None:
                return move, None, False, raw_text, prompt

        # All retries exhausted — pick a random legal move as fallback
        # (consistent with ChessLLM paper's approach for models that can't generate legal moves)
        legal_moves = list(board.legal_moves)
        if legal_moves:
            import random
            fallback = random.choice(legal_moves)
            return fallback, None, False, f"[FALLBACK random] {last_raw}", last_prompt

        return None, None, True, last_raw, last_prompt
