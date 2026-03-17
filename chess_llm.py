"""LLM utilities for chess evaluation"""

import chess
import re
from openai import OpenAI


class ChessLLM:
    """Chess LLM interface for move generation"""
    
    def __init__(self, config):
        self.config = config
        self.chess_template = config.chess_template
    
    # Unicode piece mapping for board rendering
    UNICODE_PIECES = {
        'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙',
        'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟',
    }

    @staticmethod
    def _render_board_unicode(board: chess.Board) -> str:
        """Render board with coordinates and borders, matching local_evaluation.py format."""
        files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        ranks = ['8', '7', '6', '5', '4', '3', '2', '1']
        lines = []
        coord_parts = [f" {f} " for f in files]
        coord_line = "   " + "".join(coord_parts) + "  "
        lines.append(coord_line)
        lines.append("   +" + "-" * (len(files) * 3) + "+")
        for rank in ranks:
            parts = [f"{rank} |"]
            for file in files:
                sq = chess.parse_square(file + rank)
                piece = board.piece_at(sq)
                ch = ChessLLM.UNICODE_PIECES[piece.symbol()] if piece else "·"
                parts.append(f" {ch} ")
            parts.append(f"| {rank}")
            lines.append("".join(parts))
        lines.append("   +" + "-" * (len(files) * 3) + "+")
        lines.append(coord_line)
        return "\n".join(lines)

    def encode_board_position_jinja(self, fen):
        """Encode FEN using Jinja template.

        Supports both special-token templates (value prediction) and
        plain-text templates (no value prediction / SFT agent).
        Extra variables are passed to cover both template types.
        """
        board = chess.Board(fen)

        # Variables for special-token template
        legal_moves_uci = " ".join([move.uci() for move in board.legal_moves])

        # Variables for plain-text SFT template
        legal_moves_list = [move.uci() for move in board.legal_moves]
        side_to_move = "white" if board.turn == chess.WHITE else "black"
        board_utf = self._render_board_unicode(board)
        first_legal_move = legal_moves_list[0] if legal_moves_list else "e2e4"

        # Render template — each template uses only the variables it needs
        encoded = self.chess_template.render(
            FEN=fen,
            legal_moves_uci=legal_moves_uci,
            board_utf=board_utf,
            side_to_move=side_to_move,
            legal_moves_uci_list=legal_moves_list,
            first_legal_move=first_legal_move,
        )

        return encoded.strip()
    
    def extract_uci_move(self, response):
        """Extract UCI move from model response"""
        try:
            match = re.search(r'<uci_move>([a-h][1-8][a-h][1-8][qrbn]?)</uci_move>', response)
            if match:
                return match.group(1)
            
            # Try to find any UCI-like move pattern
            match = re.search(r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b', response)
            if match:
                return match.group(1)
            
            return "a1a1"

        except Exception:
            return "a1a1"
    
    def generate(self, messages, temperature=None):
        """Generate completion from vLLM server using OpenAI Python client

        Args:
            messages: List of message dicts, e.g. [{"role": "user", "content": "..."}]
            temperature: Sampling temperature (0.0 = greedy)
        """
        if temperature is None:
            temperature = self.config.temperature
        if not hasattr(self, '_logged_temp'):
            print(f"  [ChessLLM] temperature={temperature}")
            self._logged_temp = True
        client = OpenAI(
            base_url=f"http://localhost:{self.config.port}/v1",
            api_key="EMPTY"  # vLLM doesn't require a real API key
        )
        response = client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content
    
    def get_move(self, board, temperature=None):
        """Get a move from the LLM for the current board position"""
        if temperature is None:
            temperature = self.config.temperature
            
        fen = board.fen()
        prompt = self.encode_board_position_jinja(fen)
        messages = [{"role": "user", "content": prompt}]
        response = self.generate(messages, temperature=temperature)
        
        # Extract thinking
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        thinking = think_match.group(1).strip() if think_match else None

        # Extract move
        move = self.extract_uci_move(response)

        return move, thinking, response, prompt   # response=完整原始模型输出, prompt=模型输入

    def try_move(self, board):
        """Try to get a legal move from LLM with retries"""
        for attempt in range(self.config.max_retries):
            temperature = self.config.temperature if attempt == 0 else self.config.retry_temperature
            move_uci, thinking, raw_response, prompt = self.get_move(board, temperature=temperature)

            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    return move, thinking, False, raw_response, prompt
            except (ValueError, AttributeError):
                pass
            except:
                import pdb; pdb.set_trace()

        return None, None, True, None, None  # illegal_move = True

