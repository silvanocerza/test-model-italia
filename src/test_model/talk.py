import argparse
from typing import Optional

import torch
import transformers as tr


def talk(system_prompt: Optional[str] = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = tr.AutoTokenizer.from_pretrained("sapienzanlp/modello-italia-9b")
    model = tr.AutoModelForCausalLM.from_pretrained(
        "sapienzanlp/modello-italia-9b", device_map=device, torch_dtype=torch.bfloat16
    )

    messages = []

    if system_prompt:
        # system_prompt = "Tu sei Modello Italia, un modello di linguaggio naturale addestrato da iGenius."
        messages.append(
            {"role": "system", "content": system_prompt}
        )

    while True:
        prompt = input(">>> ")
        messages.append(
            {"role": "user", "content": prompt}
        )
        tokenized_chat = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        out = model.generate(tokenized_chat, max_new_tokens=200, do_sample=False)
        print(out)
        messages.append(out)







if __name__ == "__main__":
    parser = argparse.ArgumentParser("model-testing")

    parser.add_argument("-s", "--system-prompt")

    args = parser.parse_args()

    talk(args.system_prompt)
