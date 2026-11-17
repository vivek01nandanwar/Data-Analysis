"""Data-type fixes and memory compression. Nothing is changed here without the user's approval:
suggest_changes() only inspects; convert() is called by the app for the changes the user ticked."""
import re
import numpy as np
import pandas as pd

NUM_JUNK = r"[₹$€£,\s%]"
DATE_PAT = r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}"
INT_TYPES = ["int8", "int16", "int32"]


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1024


def mem_total(df: pd.DataFrame) -> int:
    return int(df.memory_usage(deep=True).sum())


def _mem(s: pd.Series) -> int:
    return int(s.memory_usage(deep=True, index=False))


def is_text(s: pd.Series) -> bool:
    return not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s)
                or pd.api.types.is_bool_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype))


def convert(s: pd.Series, code: str) -> pd.Series:
    if code == "date_ymd":      # 20240115 -> 2024-01-15
        return pd.to_datetime(s.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    if code == "date":
        return pd.to_datetime(s, errors="coerce", dayfirst=True)
    if code == "number":
        return pd.to_numeric(s.astype(str).str.replace(NUM_JUNK, "", regex=True), errors="coerce")
    return s.astype(code)


def suggest_changes(df: pd.DataFrame) -> list:
    """Return one suggestion per column that can be fixed or shrunk safely."""
    out = []
    for c in df.columns:
        s = df[c]
        nn = s.dropna()
        if nn.empty or pd.api.types.is_bool_dtype(s) or pd.api.types.is_datetime64_any_dtype(s) \
                or isinstance(s.dtype, pd.CategoricalDtype):
            continue
        cur, before = str(s.dtype), _mem(s)
        sug = None  # (kind, code, label, reason, default_apply)
        rng = ""

        if pd.api.types.is_numeric_dtype(s):
            lo, hi = nn.min(), nn.max()
            rng = f"{lo:,.4g} to {hi:,.4g}"
            whole = bool((nn % 1 == 0).all())
            is_int = pd.api.types.is_integer_dtype(s)

            if whole and 19000101 <= lo and hi <= 21001231:
                dt = pd.to_datetime(nn.astype("int64").astype(str), format="%Y%m%d", errors="coerce")
                if dt.notna().all():
                    sug = ("Wrong type", "date_ymd", "datetime",
                           f"Marked as {cur}, but values like {int(nn.iloc[0])} are dates (YYYYMMDD)", True)

            if sug is None and whole and (is_int or not s.isna().any()):
                for t in INT_TYPES:
                    info = np.iinfo(t)
                    if lo >= info.min and hi <= info.max:
                        if not is_int or np.dtype(t).itemsize < s.dtype.itemsize:
                            extra = "" if is_int else " (no decimals are used)"
                            sug = ("Smaller size", t, t,
                                   f"Values from {lo:,.0f} to {hi:,.0f} fit in {t}{extra}", True)
                        break

            if sug is None and not whole and s.dtype == np.float64:
                f32 = np.finfo("float32")
                if lo >= -f32.max and hi <= f32.max:
                    back = nn.astype("float32").astype("float64")
                    rel = (np.abs(back - nn) / np.maximum(np.abs(nn), 1e-30)).max()
                    if rel < 1e-6:
                        sug = ("Smaller size", "float32", "float32",
                               "Range fits; keeps about 7 significant digits (fine for prices, not for exact accounting)",
                               False)

        elif is_text(s):
            sv = nn.astype(str)
            id_like = sv.str.fullmatch(r"0\d+").any()      # IDs, phone numbers, zip codes: never turn into numbers
            if True:
                num = pd.to_numeric(sv.str.replace(NUM_JUNK, "", regex=True), errors="coerce")
                if not id_like and num.notna().mean() >= 0.95:
                    bad = int(num.isna().sum())
                    tail = f" {bad} value(s) cannot be read and will become blank." if bad else ""
                    sug = ("Wrong type", "number", "number",
                           f"Stored as text, but values are numbers (e.g. '{sv.iloc[0]}').{tail}", True)
                elif sv.head(50).str.match(DATE_PAT).mean() > 0.8 or re.search(r"date|time", c, re.I):
                    dt = pd.to_datetime(sv, errors="coerce", dayfirst=True)
                    if dt.notna().mean() >= 0.95:
                        bad = int(dt.isna().sum())
                        tail = f" {bad} value(s) cannot be read and will become blank." if bad else ""
                        sug = ("Wrong type", "date", "datetime",
                               f"Stored as text, but values are dates (e.g. '{sv.iloc[0]}').{tail}", True)
            if sug is None and nn.nunique() <= 0.5 * len(nn) and nn.nunique() <= 5000:
                if _mem(s.astype("category")) < before * 0.8:
                    sug = ("Smaller size", "category", "category",
                           f"Only {nn.nunique():,} different values repeat across {len(nn):,} rows", True)

        if sug:
            kind, code, label, reason, default = sug
            after = _mem(convert(s, code))
            if kind == "Smaller size" and after >= before:
                continue
            out.append(dict(col=c, kind=kind, code=code, cur=cur, new=label, reason=reason,
                            rng=rng, before=before, after=after, default=default))
    return out