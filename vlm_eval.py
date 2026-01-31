# coding: utf-8

# Author: Mingzhe Du
# Date: 2026-01-05
# Description: Evaluate the performance of VLMs.

import argparse
from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive chat with VLM using vLLM")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model or HuggingFace model name",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32", "auto"],
        help="Data type for model weights",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (0.0 to 1.0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="System prompt for the conversation",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    tokenizer = llm.get_tokenizer()
    print("Model loaded successfully!\n")

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print("Interactive Chat Interface")
    print("=" * 40)
    print("Type 'quit' or 'exit' to end the session.")
    print("Type 'clear' to clear conversation history.")
    print("=" * 40 + "\n")

    conversation_history = []
    if args.system_prompt:
        conversation_history.append({"role": "system", "content": args.system_prompt})

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        if user_input.lower() == "clear":
            conversation_history = []
            if args.system_prompt:
                conversation_history.append({"role": "system", "content": args.system_prompt})
            print("Conversation history cleared.\n")
            continue

        conversation_history.append({"role": "user", "content": user_input})

        prompt = build_prompt(conversation_history, tokenizer)
        outputs = llm.generate([prompt], sampling_params)
        response = outputs[0].outputs[0].text.strip()

        conversation_history.append({"role": "assistant", "content": response})

        print(f"\nAssistant: {response}\n")


def build_prompt(conversation_history, tokenizer):
    """Build a prompt string from conversation history using tokenizer's chat template."""
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            conversation_history,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt_parts = []
        for message in conversation_history:
            role = message["role"]
            content = message["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)


if __name__ == "__main__":
    main()
