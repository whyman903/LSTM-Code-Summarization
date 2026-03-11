import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="embedding using CodeT5+ for LSTM training."
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to input text file (one sample per line)"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to output .pt file"
    )
    parser.add_argument(
        "--max_length", type=int, default=512,
        help="Max token length"
    )
    args = parser.parse_args()

    checkpoint = "Salesforce/codet5p-220m"
    print(f"Loading tokenizer and model: {checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    print("Model loaded.")

    embedding_matrix = model.encoder.embed_tokens.weight.detach().clone()
    print(f"Embedding matrix shape: {embedding_matrix.shape}")
    print(f"  Vocab size:     {embedding_matrix.shape[0]}")
    print(f"  Embedding dim:  {embedding_matrix.shape[1]}")

    with open(args.input, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(lines)} samples from {args.input}")

    token_ids = []
    for line in tqdm(lines, desc="Tokenizing"):
        ids = tokenizer.encode(
            line,
            truncation=True,
            max_length=args.max_length,
        )
        token_ids.append(ids)

    lengths = [len(ids) for ids in token_ids]
    print(f"\nToken length stats:")
    print(f"  Mean: {sum(lengths)/len(lengths):.1f}")
    print(f"  Max:  {max(lengths)}")
    print(f"  Min:  {min(lengths)}")


    output = {
        "token_ids": token_ids,
        "embedding_matrix": embedding_matrix,
        "tokenizer_name": checkpoint,
        "vocab_size": embedding_matrix.shape[0],
        "embedding_dim": embedding_matrix.shape[1],
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    torch.save(output, args.output)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
