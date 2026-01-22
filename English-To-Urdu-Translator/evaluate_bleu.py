import os
import sys
from sacrebleu import corpus_bleu

def read_refs(path):
    refs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                ur = parts[1].strip()
                if ur:
                    refs.append(ur)
    return refs

def read_hyps(path):
    hyps = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            hyps.append(line.rstrip("\n"))
    return hyps

def main():
    test_path = os.path.join("datasets", "bleu_test.tsv")
    hyp_path = os.path.join("datasets", "bleu_results.txt")
    model_info_path = os.path.join("datasets", "bleu_model.txt")
    if not os.path.exists(test_path) or not os.path.exists(hyp_path):
        print("Missing input files. Ensure bleu_test.tsv and bleu_results.txt exist in datasets/.")
        sys.exit(1)
    refs = read_refs(test_path)
    hyps = read_hyps(hyp_path)
    if len(refs) != len(hyps):
        print("Mismatch between reference and hypothesis counts.")
        sys.exit(1)
    score = corpus_bleu(hyps, [refs]).score
    model_name = "unknown"
    if os.path.exists(model_info_path):
        try:
            with open(model_info_path, "r", encoding="utf-8") as f:
                model_name = f.read().strip() or "unknown"
        except Exception:
            pass
    print("Model evaluated:", model_name)
    print("BLEU score:", round(score, 2))
    print("Explanation: BLEU compares model translations to human references. Higher is better; around 20–30 means reasonable conversational accuracy, higher scores indicate closer matches.")
    try:
        with open(os.path.join("datasets", "bleu_score.txt"), "w", encoding="utf-8") as f:
            f.write(f"model_name\t{model_name}\n")
            f.write(f"bleu\t{score:.2f}\n")
    except Exception:
        pass

if __name__ == "__main__":
    main()
