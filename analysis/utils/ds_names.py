"""Mapping between short dataset abbreviations and HuggingFace config names."""

# Short abbrev → HF dataset config name used in load_dataset(data_dir, config_name)
DS_HF_NAME: dict[str, str] = {
    "cs":      "coralscapes",
    "cou":     "cou",
    "df":      "deepfish",
    "l4s":     "l4s",
    "liaci":   "liaci",
    "sc":      "seaclear",
    "suim":    "suim",
    "tc":      "trashcan",
    "uiis10k": "uiis10k",
    "usis16k": "usis16k",
}

# Canonical ordered list of short names (9 currently available in the HF store)
ALL_DATASETS = ["cs", "cou", "df", "l4s", "liaci", "sc", "suim", "tc", "uiis10k"]
