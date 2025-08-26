# Labels used in the project
LABELS = ["Normal", "Hate", "Offensive"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# Label-specific max lengths 
MAXLEN = {"Normal": 88, "Hate": 62, "Offensive": 60}

# Tokenizer max length used in training (≈ 1.3 × longest)
DEFAULT_MAX_LENGTH = int(88 * 1.3)  # ~114
