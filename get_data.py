"""
Download CodeSearchNet Java subset from CodeXGLUE code-to-text,
extract ~50K train and 1K validation code-summary pairs,
and write them as .txt files (one sample per line).

Code: whitespace-normalized flat Java method.
Summary: lowercased natural language docstring.

Usage (run in Colab or locally):
    pip install datasets
    python get_data.py
"""

from datasets import load_dataset

def flatten_code(code: str) -> str:
    """Collapse a Java method to a single whitespace-normalized line."""
    return " ".join(code.split())

def clean_summary(summary: str) -> str:
    """Lowercase, strip, normalize whitespace."""
    s = summary.strip().lower()
    s = " ".join(s.split())
    return s

def is_valid(code: str, summary: str) -> bool:
    if not code or not summary:
        return False
    # skip trivially short methods or summaries
    if len(code.split()) < 5 or len(summary.split()) < 3:
        return False
    # skip auto-generated / getter-setter boilerplate summaries
    if summary.startswith("{") or summary.startswith("/*"):
        return False
    return True

def main():
    print("Loading CodeXGLUE code-to-text Java dataset...")
    ds = load_dataset("google/code_x_glue_ct_code_to_text", "java")

    # --- Training set: ~50K pairs ---
    print("Processing training data...")
    train_codes, train_sums = [], []
    for sample in ds["train"]:
        code = flatten_code(sample["code"])
        summary = clean_summary(sample["docstring"])
        if is_valid(code, summary):
            train_codes.append(code)
            train_sums.append(summary)
            if len(train_codes) >= 50000:
                break

    # --- Validation set: 1000 pairs ---
    print("Processing validation data...")
    val_codes, val_sums = [], []
    for sample in ds["validation"]:
        code = flatten_code(sample["code"])
        summary = clean_summary(sample["docstring"])
        if is_valid(code, summary):
            val_codes.append(code)
            val_sums.append(summary)
            if len(val_codes) >= 1000:
                break

    # Write files
    for name, lines in [
        ("train_code.txt", train_codes),
        ("train_summary.txt", train_sums),
        ("val_code.txt", val_codes),
        ("val_summary.txt", val_sums),
    ]:
        with open(name, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
        print(f"  {name}: {len(lines)} samples")

    print("Done!")

if __name__ == "__main__":
    main()
