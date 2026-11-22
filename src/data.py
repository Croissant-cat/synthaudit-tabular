"""Data layer.

Loads two public tabular datasets (UCI Adult, UCI Bank Marketing), cleans them,
builds leakage-free train/val/test splits, constructs controlled label-imbalance
settings and encodes mixed-type features for generative models.

Design notes
------------
* Numeric features are quantile-transformed to an approximately standard normal
  space, fitted on the training split only.
* Categorical features are one-hot encoded with an explicit "Unknown" category
  for missing values, so that missingness itself is modelled rather than imputed.
* The generative models operate on the concatenated vector
  ``[numeric | one-hot categorical]``; categorical dimensions are decoded by a
  per-feature softmax over that feature's slice.
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

from .utils import RAW_DIR

# --------------------------------------------------------------------------- #
# Dataset definitions
# --------------------------------------------------------------------------- #

ADULT_NUM = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
ADULT_CAT = [
    "workclass",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]
ADULT_COLS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

BANK_NUM = ["age", "balance", "campaign", "pdays", "previous"]
BANK_CAT = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]
BANK_COLS = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "y",
]

DATASET_SPECS = {
    "adult": {
        "num_cols": ADULT_NUM,
        "cat_cols": ADULT_CAT,
        "target": "income",
        "positive": ">50K",
        "int_cols": ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"],
        "pretty": "UCI Adult (census income >50K)",
    },
    "bank": {
        "num_cols": BANK_NUM,
        "cat_cols": BANK_CAT,
        "target": "y",
        "positive": "yes",
        "int_cols": ["age", "balance", "campaign", "pdays", "previous"],
        "pretty": "UCI Bank Marketing (term-deposit subscription)",
    },
}


def load_adult() -> pd.DataFrame:
    """Load and concatenate the official Adult train/test files (48,842 rows)."""
    path = RAW_DIR / "adult.zip"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run scripts/download_data.py")
    with zipfile.ZipFile(path) as z:
        frames = []
        for name in ("adult.data", "adult.test"):
            text = z.read(name).decode("utf-8", errors="replace")
            lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("|")]
            df = pd.read_csv(
                io.StringIO("\n".join(lines)),
                header=None,
                names=ADULT_COLS,
                skipinitialspace=True,
                na_values=["?"],
            )
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["income"] = df["income"].astype(str).str.strip().str.rstrip(".")
    df = df[df["income"].isin([">50K", "<=50K"])].reset_index(drop=True)
    df["income"] = (df["income"] == ">50K").astype(int)
    # Redundant / non-predictive columns.
    df = df.drop(columns=["fnlwgt", "education"])
    for c in ADULT_CAT:
        df[c] = df[c].fillna("Unknown").astype(str).str.strip()
    for c in ADULT_NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_bank() -> pd.DataFrame:
    """Load bank-full.csv (45,211 rows) from the nested UCI archive."""
    path = RAW_DIR / "bank.zip"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run scripts/download_data.py")
    with zipfile.ZipFile(path) as outer:
        inner = zipfile.ZipFile(io.BytesIO(outer.read("bank.zip")))
        raw = inner.read("bank-full.csv")
    df = pd.read_csv(io.BytesIO(raw), sep=";")
    df = df.rename(columns={"day": "day_of_week"})
    df["y"] = (df["y"].astype(str).str.strip() == "yes").astype(int)
    # `duration` is the call length and is only known after the call -> leakage.
    df = df.drop(columns=["duration"])
    for c in BANK_CAT:
        df[c] = df[c].fillna("Unknown").astype(str).str.strip()
    for c in BANK_NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # 999 encodes "client was not previously contacted".
    df["pdays"] = df["pdays"].replace(999, -1)
    return df


LOADERS = {"adult": load_adult, "bank": load_bank}


def load_dataset(name: str) -> pd.DataFrame:
    if name not in LOADERS:
        raise KeyError(f"Unknown dataset {name!r}; choose from {list(LOADERS)}")
    return LOADERS[name]()


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #

@dataclass
class Encoding:
    num: np.ndarray  # (n, n_num)  float32
    cat: np.ndarray  # (n, n_cat)  float32 one-hot
    y: np.ndarray    # (n,)        int64


class TabularPreprocessor:
    """Fit-on-train encoder for mixed numeric/categorical tabular data."""

    def __init__(self, num_cols, cat_cols, int_cols, target_name: str = "y"):
        self.num_cols = list(num_cols)
        self.cat_cols = list(cat_cols)
        self.int_cols = list(int_cols)
        self.target_name = target_name
        self.qt: QuantileTransformer | None = None
        self.ohe: OneHotEncoder | None = None
        self.num_min: np.ndarray | None = None
        self.num_max: np.ndarray | None = None
        self.num_fill: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        Xn = df[self.num_cols].to_numpy(dtype=np.float64)
        self.num_fill = np.nanmedian(Xn, axis=0)
        Xn = np.where(np.isnan(Xn), self.num_fill, Xn)
        self.num_min = np.nanmin(Xn, axis=0)
        self.num_max = np.nanmax(Xn, axis=0)
        self.qt = QuantileTransformer(
            n_quantiles=min(1000, len(df)),
            output_distribution="normal",
            subsample=min(len(df), 100_000),
            random_state=0,
        ).fit(Xn)
        self.ohe = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, dtype=np.float32
        ).fit(df[self.cat_cols].astype(str))
        return self

    @property
    def n_num(self) -> int:
        return len(self.num_cols)

    @property
    def cat_sizes(self) -> list[int]:
        return [len(c) for c in self.ohe.categories_]

    @property
    def cat_slices(self) -> list[slice]:
        out, start = [], 0
        for k in self.cat_sizes:
            out.append(slice(start, start + k))
            start += k
        return out

    @property
    def n_cat(self) -> int:
        return int(sum(self.cat_sizes))

    def transform(self, df: pd.DataFrame) -> Encoding:
        Xn = df[self.num_cols].to_numpy(dtype=np.float64)
        Xn = np.where(np.isnan(Xn), self.num_fill, Xn)
        num = self.qt.transform(Xn).astype(np.float32)
        num = np.clip(num, -5.0, 5.0)
        cat = self.ohe.transform(df[self.cat_cols].astype(str)).astype(np.float32)
        if self.target_name in df.columns:
            y = df[self.target_name].to_numpy(dtype=np.int64)
        else:
            y = np.zeros(len(df), dtype=np.int64)
        return Encoding(num=num, cat=cat, y=y)

    # -- inverse ---------------------------------------------------------- #
    def decode_numeric(self, num: np.ndarray) -> np.ndarray:
        num = np.clip(np.asarray(num, dtype=np.float64), -5.0, 5.0)
        Xn = self.qt.inverse_transform(num)
        Xn = np.clip(Xn, self.num_min, self.num_max)
        return Xn

    def decode_categorical(self, cat: np.ndarray, mode: str = "logits",
                           temperature: float = 1.0, rng: np.random.Generator | None = None,
                           hard: bool = True) -> np.ndarray:
        """Decode concatenated categorical scores into category indices.

        mode='logits' treats the values as unnormalised logits (VAE/GAN);
        mode='prob'   treats them as non-negative one-hot estimates (diffusion).
        """
        cat = np.asarray(cat, dtype=np.float64)
        rng = rng or np.random.default_rng(0)
        out = np.zeros((len(cat), len(self.cat_slices)), dtype=np.int64)
        for j, sl in enumerate(self.cat_slices):
            v = cat[:, sl]
            if mode == "prob":
                v = np.clip(v, 0.0, None) + 1e-8
                p = v / v.sum(axis=1, keepdims=True)
            else:
                z = v / max(temperature, 1e-6)
                z = z - z.max(axis=1, keepdims=True)
                p = np.exp(z)
                p /= p.sum(axis=1, keepdims=True)
            if hard:
                out[:, j] = p.argmax(axis=1)
            else:
                out[:, j] = (p.cumsum(axis=1) > rng.random((len(p), 1))).argmax(axis=1)
        return out

    def to_frame(self, num: np.ndarray, cat: np.ndarray | None = None,
                 cat_idx: np.ndarray | None = None, mode: str = "logits",
                 temperature: float = 1.0, rng: np.random.Generator | None = None,
                 hard: bool = True) -> pd.DataFrame:
        """Decode a synthetic matrix back into the original feature space."""
        Xn = self.decode_numeric(num)
        if cat_idx is None:
            cat_idx = self.decode_categorical(cat, mode=mode, temperature=temperature,
                                              rng=rng, hard=hard)
        df = pd.DataFrame(Xn, columns=self.num_cols)
        for c in self.int_cols:
            if c in df.columns:
                df[c] = np.rint(df[c]).astype(np.int64)
        for j, c in enumerate(self.cat_cols):
            cats = self.ohe.categories_[j]
            df[c] = pd.Categorical.from_codes(cat_idx[:, j], categories=cats)
            df[c] = df[c].astype(str)
        return df


# --------------------------------------------------------------------------- #
# Splits and imbalance control
# --------------------------------------------------------------------------- #

@dataclass
class Split:
    name: str
    df: pd.DataFrame
    enc: Encoding
    meta: dict = field(default_factory=dict)


@dataclass
class PreparedData:
    name: str
    ratio: float
    preprocessor: TabularPreprocessor
    train: Split
    val: Split
    test: Split
    meta: dict = field(default_factory=dict)


def subsample_minority(df: pd.DataFrame, target: str, ratio: float, seed: int) -> pd.DataFrame:
    """Keep all negatives and a deterministic, nested subset of positives so that
    prevalence == ratio.  The permutation is seeded, so a 5% subset is contained
    in the 10% subset for the same seed."""
    y = df[target].to_numpy()
    n = len(df)
    n_pos = int(y.sum())
    current = n_pos / n
    if ratio >= current - 1e-12:
        return df
    neg = n - n_pos
    n_pos_target = int(round(ratio / (1.0 - ratio) * neg))
    pos_idx = np.flatnonzero(y == 1)
    rng = np.random.default_rng(seed + 17)
    keep = rng.permutation(pos_idx)[: min(n_pos_target, len(pos_idx))]
    keep_all = np.concatenate([np.flatnonzero(y == 0), keep])
    keep_all.sort()
    return df.iloc[keep_all].reset_index(drop=True)


def prepare_dataset(name: str, ratio: float | None = None, seed: int = 0,
                    test_frac: float = 0.2, val_frac: float = 0.2) -> PreparedData:
    spec = DATASET_SPECS[name]
    df = load_dataset(name)
    target = spec["target"]
    y = df[target].to_numpy()

    # Split first, then control imbalance inside each split.  This keeps the
    # test rows identical across imbalance settings (positives are nested).
    df_tr, df_tmp, y_tr, y_tmp = train_test_split(
        df, y, test_size=test_frac + val_frac, stratify=y, random_state=seed
    )
    val_share = val_frac / (test_frac + val_frac)
    df_va, df_te, _, _ = train_test_split(
        df_tmp, y_tmp, test_size=1 - val_share, stratify=y_tmp, random_state=seed
    )
    if ratio is not None:
        df_tr = subsample_minority(df_tr, target, ratio, seed)
        df_va = subsample_minority(df_va, target, ratio, seed)
        df_te = subsample_minority(df_te, target, ratio, seed)

    pre = TabularPreprocessor(spec["num_cols"], spec["cat_cols"], spec["int_cols"], target)
    pre.fit(df_tr)

    splits = {}
    for nm, d in (("train", df_tr), ("val", df_va), ("test", df_te)):
        enc = pre.transform(d.reset_index(drop=True))
        splits[nm] = Split(
            name=nm,
            df=d.reset_index(drop=True),
            enc=enc,
            meta={"n": len(d), "n_pos": int(enc.y.sum()), "prevalence": float(enc.y.mean())},
        )

    meta = {
        "dataset": name,
        "pretty": spec["pretty"],
        "ratio_target": ratio,
        "seed": seed,
        "n_total": int(len(df)),
        "prevalence": float(y.mean()),
        "n_num": pre.n_num,
        "n_cat": pre.n_cat,
        "cat_sizes": pre.cat_sizes,
        "num_cols": spec["num_cols"],
        "cat_cols": spec["cat_cols"],
        "n_features_raw": len(spec["num_cols"]) + len(spec["cat_cols"]),
        "splits": {k: v.meta for k, v in splits.items()},
    }
    return PreparedData(
        name=name, ratio=float(ratio if ratio is not None else y.mean()),
        preprocessor=pre, train=splits["train"], val=splits["val"], test=splits["test"],
        meta=meta,
    )


def augment_indices(n_real: int, n_syn: int, seed: int) -> np.ndarray:
    """Random oversampling indices (with replacement)."""
    rng = np.random.default_rng(seed)
    return rng.choice(n_real, size=n_syn, replace=True)
