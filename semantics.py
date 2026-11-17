"""Works out what each column means. Layers: (1) alias + fuzzy name matching, (2) value profiling,
(3) optional LLM. Layers 1-2 run offline. Layer 3 sends only column names, ranges and 5 sample values."""
import difflib
import json
import os
import re
import numpy as np
import pandas as pd

ALIASES = {
    "date": ["date", "order date", "sale date", "sales date", "invoice date", "transaction date",
             "bill date", "day", "timestamp", "dt"],
    "product": ["product", "product name", "item", "item name", "sku", "article", "goods",
                "description", "particulars"],
    "category": ["category", "cat", "product type", "segment", "group", "department", "class", "type"],
    "qty": ["qty", "quantity", "units", "unit", "units sold", "unit sold", "sales unit", "sales units",
            "sales qty", "quantity sold", "sold", "pcs", "pieces", "volume", "order quantity", "no of units"],
    "price": ["price", "unit price", "selling price", "sale price", "rate", "mrp", "price per unit", "sp"],
    "discount": ["discount", "disc", "discount percent", "discount pct", "discount %", "discount amount",
                 "offer", "rebate", "markdown", "promo"],
    "revenue": ["revenue", "sales amount", "sales value", "total sales", "net sales", "gross sales",
                "sale amount", "amount", "total amount", "turnover", "income", "total", "sales",
                "line total", "net amount"],
}
LABELS = {"date": "Date", "product": "Product", "category": "Category", "qty": "Quantity sold",
          "price": "Price", "discount": "Discount", "revenue": "Revenue"}
KINDS = {"percent": "Percentage (10 = 10%)", "fraction": "Fraction (0.10 = 10%)",
         "amount_unit": "Amount off per unit", "amount_total": "Amount off the whole sale"}


def norm(name) -> str:
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(name))
    return re.sub(r"[^A-Za-z0-9%]+", " ", s).lower().strip()


def _score(col: str, alias: str) -> float:
    if col == alias:
        return 1.0
    ct, at = set(col.split()), set(alias.split())
    if at and at <= ct:
        return 0.8 + 0.1 * len(at) / len(ct)
    r = difflib.SequenceMatcher(None, col.replace(" ", ""), alias.replace(" ", "")).ratio()
    return r * 0.75 if r >= 0.85 else 0.0


def _fits(role: str, s: pd.Series) -> bool:
    if role == "date":
        return pd.api.types.is_datetime64_any_dtype(s)
    if role in ("product", "category"):
        return not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s))
    return pd.api.types.is_numeric_dtype(s)


def detect_roles(df: pd.DataFrame, ai: dict | None = None):
    """Return (roles {role: column}, reasons {role: text}, warnings [text])."""
    cands, warns = [], []
    for c in df.columns:
        n = norm(c)
        for role, aliases in ALIASES.items():
            sc = max(_score(n, a) for a in aliases)
            if sc < 0.7:
                continue
            if _fits(role, df[c]):
                cands.append((sc, role, c))
            elif role == "date" and sc >= 0.9:
                warns.append(f"'{c}' looks like a date but is stored as {df[c].dtype}. "
                             "Apply the type fix in the step above to use it.")
    roles, why, used = {}, {}, set()
    for sc, role, c in sorted(cands, key=lambda x: -x[0]):
        if role in roles or c in used:
            continue
        roles[role], why[role] = c, f"name matches '{role}'"
        used.add(c)

    if "qty" not in roles and "price" in roles and norm(roles.get("revenue", "")) == "sales":
        warns.append("'Sales' could mean units sold or money earned. Please confirm it below.")

    if ai:                                           # the LLM's reading wins where it names a real column
        for c, info in ai.get("columns", {}).items():
            role = info.get("role")
            if c in df.columns and role in ALIASES and _fits(role, df[c]):
                for r in [r for r, v in roles.items() if v == c or r == role]:
                    roles.pop(r, None)
                roles[role], why[role] = c, "AI"
    return roles, why, warns


def discount_kind(d: pd.Series, price: pd.Series | None = None, name: str = ""):
    """Guess how a discount is expressed, from its values. Returns (kind, reason)."""
    nn = d.dropna()
    if nn.empty:
        return "percent", "no values to inspect"
    n, lo, hi = norm(name), nn.min(), nn.max()
    if re.search(r"pct|percent|%", n):
        return "percent", "column name says percent"
    if hi <= 1:
        return "fraction", f"values only range from {lo:g} to {hi:g}"
    if hi <= 100:
        if re.search(r"amount|amt|value|off", n):
            return "amount_unit", "column name says amount"
        if price is not None:
            both = pd.concat([nn, price], axis=1, join="inner").dropna()
            if len(both) and (both.iloc[:, 0] > both.iloc[:, 1]).mean() > 0.2:
                return "percent", "discount is often larger than the price, so it can't be an amount per unit"
        return "percent", f"values from {lo:g} to {hi:g}, typical of percentages"
    if price is not None:
        both = pd.concat([nn, price], axis=1, join="inner").dropna()
        if len(both) and (both.iloc[:, 0] <= both.iloc[:, 1]).all():
            return "amount_unit", "never above the unit price"
    return "amount_total", "values too large to be percentages or per-unit amounts"


# ---------------------------------------------------------------- optional LLM layer
def _ask(system, prompt, api_key, max_tokens=900, as_json=False):
    from groq import Groq
    client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))
    kw = {"response_format": {"type": "json_object"}} if as_json else {}
    r = client.chat.completions.create(
        model=os.getenv("STOCKSENSE_MODEL", "llama-3.3-70b-versatile"), max_tokens=max_tokens, temperature=0.2,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}], **kw)
    return r.choices[0].message.content


def profile_columns(df: pd.DataFrame) -> list:
    out = []
    for c in df.columns:
        s = df[c].dropna()
        p = {"name": str(c), "dtype": str(df[c].dtype), "unique": int(s.nunique()),
             "samples": [str(v) for v in s.sample(min(5, len(s)), random_state=0)] if len(s) else []}
        if pd.api.types.is_numeric_dtype(df[c]) and len(s):
            p.update(min=float(s.min()), max=float(s.max()), mean=round(float(s.mean()), 3))
        out.append(p)
    return out


def llm_column_roles(df: pd.DataFrame, api_key=None) -> dict:
    system = "You label columns of retail sales tables. Reply with a single JSON object and nothing else."
    prompt = (f"Roles allowed: {list(ALIASES)} (use null if none fits). For each column give a role and a "
              "one-sentence plain-English meaning. Also say how the discount column (if any) is expressed: "
              "one of percent, fraction, amount_unit, amount_total, or null.\n"
              'Format: {"columns": {"<name>": {"role": "...", "meaning": "..."}}, "discount_kind": "..."}\n\n'
              f"Columns:\n{json.dumps(profile_columns(df), ensure_ascii=False)}")
    text = _ask(system, prompt, api_key, 1200, as_json=True)
    return json.loads(text[text.index("{"): text.rindex("}") + 1])


def llm_insights(summary: str, api_key=None) -> str:
    system = ("You are a retail analyst advising a small shop owner. Use only the numbers provided, "
              "never invent figures. Plain language, short sentences.")
    prompt = ("Write 4 to 5 specific insights, then 2 concrete actions the owner can take this week. "
              "Mention discounts if present. Use '-' bullets, no headings.\n\n" + summary)
    return _ask(system, prompt, api_key, 700)