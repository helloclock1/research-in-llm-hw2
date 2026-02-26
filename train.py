from unsloth import FastLanguageModel
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from environment import ChessPuzzlesEnv
from base.verifier import ChessPuzzlesVerifier
import chess
import re
import polars as pl

SEED = 1489
max_seq_length = 1024
lora_rank = 32

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",
)


def format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    return [0.0 if re.match(pattern, r, re.DOTALL) else -2.0 for r in responses]


def extract_xml_answer(text: str) -> str:
    if "<answer>" not in text:
        return ""
    answer = text.split("<answer>")[-1]
    if "</answer>" in answer:
        answer = answer.split("</answer>")[0]
    return answer.strip()


def correctness_reward_func(prompts, completions, answer, fen, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]

    rewards = []
    for r, correct_move, f_str in zip(extracted_responses, answer, fen):
        r = r.strip().lower()

        if not r or len(r) < 4:
            rewards.append(-1.0)
            continue

        if r == correct_move.lower():
            rewards.append(4.0)
            continue

        try:
            board = chess.Board(f_str)
            move = chess.Move.from_uci(r)

            if move in board.legal_moves:
                board.push(move)
                if board.is_checkmate():
                    rewards.append(4.0)
                elif board.is_check():
                    rewards.append(1.5)
                else:
                    rewards.append(-1.0)
            else:
                rewards.append(-1.0)
        except Exception:
            rewards.append(-2.0)

    return rewards


df = pl.read_csv("mateIn1.csv")
env = ChessPuzzlesEnv("chess", ChessPuzzlesVerifier, df)
DATASET_SIZE = 10000
data_list = env.generate(num_of_questions=DATASET_SIZE, difficulty=None)

SYSTEM_PROMPT = """\
Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>

"""

PROMPT_TEMPLATE = """\
You are a chess engine. Your task is to find the single move that delivers checkmate.

{board_info}

Briefly identify which of your pieces can give check and which checking move leaves the king with no escape in block surrounded with <think> tags. Then provide the move in UCI format (e.g. e2e4) in <answer> tags.
"""

dataset = Dataset.from_list(
    [
        {
            "prompt": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": PROMPT_TEMPLATE.format(
                        board_info=d.metadata["board_desc"]
                    ),
                },
            ],
            "answer": d.answer,
            "fen": d.metadata["fen"],
        }
        for d in data_list
    ]
)

training_args = GRPOConfig(
    use_vllm=True,
    vllm_gpu_memory_utilization=0.2,
    output_dir="output_phase_2",
    optim="paged_adamw_8bit",
    learning_rate=4e-5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    num_generations=8,
    max_prompt_length=800,
    max_completion_length=224,
    max_steps=5000,
    bf16=True,
    logging_steps=1,
    report_to="tensorboard",
    logging_dir="./logs/phase_2",
    beta=0.1,
    log_completions=True,
    num_completions_to_print=4,
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[correctness_reward_func, format_reward_func],
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
