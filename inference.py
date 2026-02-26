from unsloth import FastLanguageModel
import polars as pl
import chess
import random

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./output_phase_2/checkpoint-3000",
    # model_name="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    max_seq_length=1024,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)


def extract_xml_answer(text: str) -> str:
    if "<answer>" not in text:
        return ""
    answer = text.split("<answer>")[-1]
    if "</answer>" in answer:
        answer = answer.split("</answer>")[0]
    return answer.strip()


tasks = pl.read_csv("tasks.csv")


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

    defending_color = not board.turn  # the side that might get mated
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


SYSTEM_PROMPT = """\
Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>

"""
USER_PROMPT = """\
You are a chess engine. Your task is to find the single move that delivers checkmate.

{board_info}

Briefly identify which of your pieces can give check and which checking move leaves the king with no escape in block surrounded with <think> tags. Then provide the move in UCI format (e.g. e2e4) in <answer> tags.
"""
model_evals = []
model_moves = []
correct_moves = []
for task in tasks.iter_rows(named=True):
    moves = task["Moves"].split(" ")
    opp_move = moves[0]
    best_move = moves[1]
    fen = task["FEN"]
    prompt = USER_PROMPT.format(board_info=analyze_position(fen, opp_move))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(input_ids=inputs, max_new_tokens=224)
    generated_tokens = outputs[0][inputs.shape[1] :]
    result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    move = extract_xml_answer(result)

    board = chess.Board(fen)
    board.push_uci(opp_move)
    try:
        board.push_uci(move)
        if board.is_checkmate():
            print("Mate")
            model_evals.append("Mate")
        elif board.is_check():
            print("Check")
            model_evals.append("Check")
        else:
            print("Wrong")
            model_evals.append("Wrong")
    except Exception:
        print(f"Invalid")
        model_evals.append("Invalid")
    model_moves.append(move)
    correct_moves.append(best_move)

print(model_evals)
print(model_moves)
print(correct_moves)
