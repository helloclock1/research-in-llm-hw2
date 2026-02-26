from abc import ABC, abstractmethod
from base.verifier import Verifier
from base.data import Data
from typing import Optional
import polars as pl
import random
import chess


class Env(ABC):
    """
    Base class for game
    @param name: name of the game
    @param verifier: class of the verifier
    """

    def __init__(self, name: str, verifier: type[Verifier]):
        self.name = name
        self.verifier = verifier()

    @abstractmethod
    def generate(
        self,
        num_of_questions: int = 100,
        max_attempts: int = 100,
        difficulty: Optional[int] = 1,
    ):
        """
        Generate game questions and answers
        @param num_of_questions: int
        @param max_attempts: int
        @return: list of Data
        """
        raise NotImplementedError("Game.generate() is not implemented")

    def verify(self, data: Data, test_solution: str):
        """
        Verify whether the test solution is consistent with the
        answer of the game data
        @param data: Data
        @param test_solution: str
        @return: bool
        """
        return self.verifier.verify(data, test_solution)

    @abstractmethod
    def extract_answer(self, test_solution: str):
        """
        Extract the answer from the test solution
        @param test_solution: str
        @return: str
        """
        raise NotImplementedError("Game.extract_answer() is not implemented")


def analyze_position(fen: str, opponent_move_uci: str = None) -> str:
    board = chess.Board(fen)
    if opponent_move_uci:
        move = chess.Move.from_uci(opponent_move_uci)
        board.push(move)
    lines = []
    lines.append(f"Board representation in FEN format: `{fen}`")
    lines.append(
        "Board representation in visual format. Capital letters represent white pieces, lowercase letters represent black pieces."
    )
    board_str = str(board)
    ranks = board_str.split("\n")
    lines.append("  a b c d e f g h")
    for i, rank in enumerate(ranks):
        lines.append(f"{8 - i} {rank}")
    lines.append("")
    white_pieces = []
    black_pieces = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            name = chess.piece_name(piece.piece_type).capitalize()
            sq_name = chess.square_name(square)
            if piece.color == chess.WHITE:
                white_pieces.append(f"{name} on {sq_name}")
            else:
                black_pieces.append(f"{name} on {sq_name}")
    lines.append(f"White pieces: {', '.join(white_pieces)}")
    lines.append(f"Black pieces: {', '.join(black_pieces)}")
    lines.append("")
    defending_color = not board.turn
    attacking_color = board.turn
    king_sq = board.king(defending_color)
    king_name = chess.square_name(king_sq)
    color_name = "White" if defending_color == chess.WHITE else "Black"
    escape_info = []
    for sq in board.attacks(king_sq):
        sq_name = chess.square_name(sq)
        piece_on_sq = board.piece_at(sq)
        if piece_on_sq and piece_on_sq.color == defending_color:
            escape_info.append(f"{sq_name}(blocked by own piece)")
            continue
        attackers = board.attackers(attacking_color, sq)
        if attackers:
            attacker_names = []
            for att_sq in attackers:
                att_piece = board.piece_at(att_sq)
                attacker_names.append(
                    f"{chess.piece_name(att_piece.piece_type).capitalize()} on {chess.square_name(att_sq)}"
                )
            escape_info.append(f"{sq_name}(attacked by {', '.join(attacker_names)})")
        else:
            escape_info.append(f"{sq_name}(free)")
    legal = [m.uci() for m in board.legal_moves]
    random.shuffle(legal)
    lines.append(f"Legal moves in no particular order: {', '.join(legal)}")
    lines.append(f"\nSide to move: {'White' if board.turn == chess.WHITE else 'Black'}")
    return "\n".join(lines)


class ChessPuzzlesEnv(Env):
    """
    Base class for game
    @param name: name of the game
    @param verifier: class of the verifier
    """

    def __init__(self, name: str, verifier: type[Verifier], games_df: pl.DataFrame):
        self.name = name
        self.verifier = verifier()
        """
        games_df schema:
        PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags
        """
        self.games_df = games_df

    def generate(
        self,
        num_of_questions: int = 100,
        max_attempts: int = 100,
        difficulty: Optional[int] = 1,
    ):
        """
        Generate game questions and answers
        @param num_of_questions: int
        @param max_attempts: int
        @return: list of Data
        """
        if difficulty:
            games = (
                self.games_df.filter(pl.col("Difficulty") == difficulty)
                .sample(num_of_questions)
                .to_dicts()
            )
        else:
            games = self.games_df.sample(num_of_questions).to_dicts()
        questions = []
        for game in games:
            setup = game["FEN"]
            moves = game["Moves"].split(" ")
            opp_move = moves[0]
            best_move = moves[1]
            game_difficulty = game["Difficulty"]
            board = chess.Board(setup)
            board.push(chess.Move.from_uci(opp_move))
            post_move_fen = board.fen()
            metadata = {
                "answer": best_move,
                "board_desc": analyze_position(setup, opp_move),
                "fen": post_move_fen,
            }
            data = Data(setup, best_move, game_difficulty, metadata)
            questions.append(data)
        return questions

    def extract_answer(self, test_solution: str):
        """
        Extract the answer from the test solution
        @param test_solution: str
        @return: str
        """
        return test_solution
