# HW №2, Research in LLM

## Training

To train the model in a way that it was trained for the report, run `train.py` twice:
1. First epoch is exactly the script
2. For the second epoch, shorten the training length to 3000 steps

## Inference

To achieve the results from the report, run `inference.py` making sure that `tasks.csv` is in the same folder as the script. The vanilla model used in the report is `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit`.

The finetuned model itself is available at https://huggingface.co/helloclock/research-in-llm-hw2 and is retrievable just as any other HF model. Make sure to use PEFT when loading the model.
