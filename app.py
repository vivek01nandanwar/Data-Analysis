"""StockSense - upload sales files, clean them, get sales insights.
Run:  streamlit run app.py
"""
import io, os, re, sqlite3, tempfile, warnings
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from semantics import detect_roles, discount_kind, llm_column_roles, llm_insights, KINDS
from optimizer import suggest_changes, convert, fmt_bytes, mem_total, is_text

warnings.filterwarnings("ignore")
st.set_page_config(page_title="StockSense", page_icon="📦", layout="wide")

TEAL, INK, MARIGOLD, CORAL, MIST = "#0F3D3E", "#12222A", "#F2A900", "#D1495B", "#EEF3F2"

# ------------------------------------------------------------------ styling
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;700;800&family=Source+Sans+3:wght@400;600&display=swap');
html, body, [class*="css"], .stMarkdown, label {{ font-family:'Source Sans 3',sans-serif; color:{INK}; }}
h1,h2,h3,.display {{ font-family:'Bricolage Grotesque',sans-serif !important; letter-spacing:-0.02em; color:{TEAL}; }}
.block-container {{ max-width:1100px; padding-top:2.5rem; }}
#MainMenu, footer, header {{ visibility:hidden; }}
.hero {{ padding:1.5rem 0 1rem; }}
.hero .display {{ font-size:3.4rem; line-height:1.05; font-weight:800; margin:0 0 .8rem; max-width:760px; }}
.hero p {{ font-size:1.2rem; max-width:620px; color:#41565c; margin:0; }}
.brand {{ font-family:'Bricolage Grotesque'; font-weight:800; font-size:1.2rem; color:{TEAL}; margin-bottom:1.2rem; }}
.brand span {{ display:inline-block; width:14px; height:14px; background:{MARIGOLD}; border-radius:4px; margin-right:8px; }}
.chips span {{ display:inline-block; background:{MIST}; color:{TEAL}; font-weight:600; font-size:.9rem;
  padding:.25rem .7rem; border-radius:999px; margin:0 .35rem .35rem 0; }}
[data-testid="stFileUploaderDropzone"] {{ border:2px dashed {TEAL}; background:{MIST}; border-radius:16px; padding:2.2rem 1rem; }}
[data-testid="stFileUploaderDropzone"]:hover {{ border-color:{MARIGOLD}; }}
.steps {{ display:flex; gap:.6rem; margin:.5rem 0 1.5rem; flex-wrap:wrap; }}
.step {{ padding:.35rem .9rem; border-radius:999px; background:{MIST}; color:#6a7f84; font-weight:600; font-size:.95rem; }}
.step.on {{ background:{TEAL}; color:#fff; }}
.step.done {{ background:#CFE5E0; color:{TEAL}; }}
.how {{ display:grid; grid-template-columns:repeat(3,1fr); gap:1.5rem; margin-top:2rem; }}
.how div {{ border-top:3px solid {MARIGOLD}; padding-top:.7rem; }}
.how b {{ font-family:'Bricolage Grotesque'; font-size:1.1rem; color:{TEAL}; }}
.how p {{ margin:.25rem 0 0; color:#41565c; }}
.kpi {{ background:{MIST}; border-radius:14px; padding:1rem 1.2rem; }}
.kpi .v {{ font-family:'Bricolage Grotesque'; font-size:1.9rem; font-weight:800; color:{TEAL}; line-height:1.1; }}
.kpi .l {{ color:#52686d; font-size:.95rem; }}
.note {{ border-left:4px solid {MARIGOLD}; background:#FFF8E6; padding:.7rem 1rem; border-radius:6px; margin:.4rem 0; }}
.stButton>button, .stDownloadButton>button {{ background:{TEAL}; color:#fff; border:0; border-radius:10px; font-weight:600; padding:.55rem 1.3rem; }}
.stButton>button:hover, .stDownloadButton>button:hover {{ background:{MARIGOLD}; color:{INK}; }}
@media (max-width:700px) {{ .hero .display {{ font-size:2.3rem; }} .how {{ grid-template-columns:1fr; }} }}
</style>
""", unsafe_allow_html=True)


def steps(active: int):
    names = ["Upload", "Clean", "Insights"]
    html = "".join(
        f'<div class="step {"on" if i == active else "done" if i < active else ""}">{n}</div>'
        for i, n in enumerate(names))
    st.markdown(f'<div class="steps">{html}</div>', unsafe_allow_html=True)


def kpi(col, value, label):
    col.markdown(f'<div class="kpi"><div class="v">{value}</div><div class="l">{label}</div></div>',
                 unsafe_allow_html=True)


# ------------------------------------------------------------------ loading
@st.cache_data(show_spinner=False)
def load_file(name: str, data: bytes) -> dict:
    """Return {table_name: DataFrame} for any supported file."""
    ext = name.rsplit(".", 1)[-1].lower()
    buf, out = io.BytesIO(data), {}
    if ext in ("csv", "tsv", "txt"):
        out[name] = pd.read_csv(buf, sep=None, engine="python")
    elif ext in ("xlsx", "xlsm", "xls"):
        for sheet, d in pd.read_excel(buf, sheet_name=None).items():
            out[f"{name} / {sheet}"] = d
    elif ext == "json":
        out[name] = pd.read_json(buf)
    elif ext in ("db", "sqlite", "sqlite3"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            tmp.write(data)
        con = sqlite3.connect(tmp.name)
        for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
            out[f"{name} / {t}"] = pd.read_sql_query(f'SELECT * FROM "{t}"', con)
        con.close(); os.unlink(tmp.name)
    elif ext == "pdf":
        import pdfplumber
        with pdfplumber.open(buf) as pdf:
            for p, page in enumerate(pdf.pages, 1):
                for k, tb in enumerate(page.extract_tables(), 1):
                    if len(tb) > 1:
                        out[f"{name} / page {p} table {k}"] = pd.DataFrame(tb[1:], columns=tb[0])
    else:
        raise ValueError(f"Unsupported file type: .{ext}")
    return {k: v for k, v in out.items() if not v.empty}


# ------------------------------------------------------------------ cleaning
BLANKS = {"", "nan", "none", "null", "n/a", "na", "-", "--", "#n/a"}


def auto_clean(df: pd.DataFrame):
    """Safe, automatic fixes. Returns (clean_df, report dict)."""
    rep = {}
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    n0 = len(df)
    df = df.dropna(how="all").dropna(axis=1, how="all")
    rep["empty rows/columns removed"] = n0 - len(df)

    for c in [c for c in df.columns if is_text(df[c])]:        # trim text, blanks -> NaN
        s = df[c].astype(str).str.strip()
        df[c] = s.where(df[c].notna() & ~s.str.lower().isin(BLANKS), np.nan)


    dups = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    rep["duplicate rows removed"] = dups
    return df, rep


def fill_options(series: pd.Series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return ["Previous date (forward fill)", "Next date (backward fill)", "Most common date", "Custom date", "Drop rows"]
    if pd.api.types.is_numeric_dtype(series):
        return ["Median", "Mean", "Most common", "Zero", "Custom value", "Previous value (forward fill)", "Drop rows"]
    return ["Most common", "Unknown", "Custom value", "Previous value (forward fill)", "Drop rows"]


def apply_fill(df, plan):
    df = df.copy()
    for c, (method, custom) in plan.items():
        if isinstance(df[c].dtype, pd.CategoricalDtype):
            df[c] = df[c].astype(object)
        s = df[c]
        if method == "Drop rows":
            df = df.dropna(subset=[c]); continue
        if method == "Median": v = s.median()
        elif method == "Mean": v = s.mean()
        elif method in ("Most common", "Most common date"): v = s.mode().iloc[0]
        elif method == "Zero": v = 0
        elif method == "Unknown": v = "Unknown"
        elif method in ("Custom value", "Custom date"): v = pd.Timestamp(custom) if method == "Custom date" else custom
        elif method.startswith(("Previous")): df[c] = s.ffill().bfill(); continue
        elif method.startswith("Next"): df[c] = s.bfill().ffill(); continue
        df[c] = s.fillna(v)
    return df.reset_index(drop=True)


# ------------------------------------------------------------------ insights
def pick(label, df, default, cols):
    opts = ["(none)"] + list(cols)
    v = st.selectbox(label, opts, index=opts.index(default) if default in opts else 0)
    return None if v == "(none)" else v


def revenue_math(d, qty, price, disc, kind):
    """Return (gross, discount_value or None, net, formula text). Done in float64 to avoid overflow."""
    q, p = d[qty].astype("float64"), d[price].astype("float64")
    gross = q * p
    if not disc:
        return gross, None, gross, f"Revenue = {qty} × {price}"
    x = d[disc].astype("float64").fillna(0)
    if kind == "percent":
        dv, f = gross * x / 100, f"Revenue = {qty} × {price} × (1 − {disc} ÷ 100)"
    elif kind == "fraction":
        dv, f = gross * x, f"Revenue = {qty} × {price} × (1 − {disc})"
    elif kind == "amount_unit":
        dv, f = q * x, f"Revenue = {qty} × ({price} − {disc})"
    else:
        dv, f = x, f"Revenue = {qty} × {price} − {disc}"
    return gross, dv, gross - dv, f


def show_insights(df: pd.DataFrame):
    steps(2)
    st.header("Sales insights")
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    txt_cols = [c for c in df.columns if c not in num_cols + date_cols]
    ai = st.session_state.get("ai_roles")
    roles, why, warns = detect_roles(df, ai)

    with st.expander("Check which columns mean what", expanded=bool(warns)):
        for w in warns:
            st.warning(w)
        st.checkbox("Use AI to explain my columns and write a summary (optional)", key="ai_on")
        if st.session_state.get("ai_on"):
            st.text_input("Groq API key (leave blank to use the GROQ_API_KEY environment variable)",
                          type="password", key="ai_key")
            st.caption("Only column names, value ranges and 5 sample values per column are sent, never your full file. "
                       "The summary uses totals only.")
            if st.button("Ask AI about my columns"):
                try:
                    with st.spinner("Reading your columns..."):
                        st.session_state["ai_roles"] = llm_column_roles(df, st.session_state.get("ai_key") or None)
                    st.rerun()
                except Exception as e:
                    st.error(f"AI request failed: {e}")
        if ai:
            st.markdown("**What the AI thinks each column is**")
            for c, info in ai.get("columns", {}).items():
                st.markdown(f"- `{c}`: {info.get('meaning', '')}")

        a, b, c3 = st.columns(3)
        with a:
            date = pick("Date", df, roles.get("date"), date_cols)
            product = pick("Product", df, roles.get("product"), txt_cols)
        with b:
            category = pick("Category", df, roles.get("category"), txt_cols)
            qty = pick("Quantity sold", df, roles.get("qty"), num_cols)
        with c3:
            price = pick("Price", df, roles.get("price"), num_cols)
            rev = pick("Revenue (leave empty to calculate it)", df, roles.get("revenue"), num_cols)
        disc, kind = pick("Discount", df, roles.get("discount"), num_cols), None
        if disc:
            dk, dwhy = discount_kind(df[disc], df[price] if price else None, disc)
            if ai and ai.get("discount_kind") in KINDS:
                dk, dwhy = ai["discount_kind"], "AI reading"
            kind = st.radio("How is the discount given?", list(KINDS), index=list(KINDS).index(dk),
                            format_func=KINDS.get, horizontal=True)
            st.caption(f"Detected: {KINDS[dk]} ({dwhy}). Change it if this is wrong.")

    d, dv, gross, formula = df.copy(), None, None, None
    if qty and price:
        gross, dv, net, f = revenue_math(d, qty, price, disc, kind)
        if rev is None:
            d["Revenue (calculated)"], rev, formula = net, "Revenue (calculated)", f
        if dv is not None:
            d["Discount value"] = dv
    if rev and formula is None:
        formula = f"Revenue taken from the '{rev}' column"
    measure = rev or qty
    if measure is None:
        st.warning("No sales column found. Pick Quantity or Revenue above to see sales charts. "
                   "Here is a general summary of your data instead.")
        st.dataframe(d.describe(include="all").T, use_container_width=True)
        return d, {}

    d[measure] = d[measure].astype("float64")
    label = "Revenue" if measure == rev else "Units sold"
    k = st.columns(4)
    kpi(k[0], f"{d[measure].sum():,.0f}", f"Total {label.lower()}")
    kpi(k[1], f"{len(d):,}", "Transactions")
    kpi(k[2], f"{d[measure].mean():,.1f}", "Average per transaction")
    kpi(k[3], f"{d[product].nunique():,}" if product else "-", "Products")
    if formula and measure == rev:
        st.caption(formula)
    st.write("")

    notes, summary = [], {}
    if dv is not None and gross.sum() > 0:
        notes.append(f"Discounts given: **{dv.sum():,.0f}**, which is {dv.sum() / gross.sum():.1%} of sales before discount.")
    if date:
        d["_month"] = d[date].dt.to_period("M").dt.to_timestamp()
        m = d.groupby("_month")[measure].sum().reset_index()
        fig = px.area(m, x="_month", y=measure, title=f"{label} by month", color_discrete_sequence=[TEAL])
        fig.update_layout(xaxis_title=None, yaxis_title=label, margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
        summary["Monthly"] = m
        if len(m) >= 2:
            best = m.loc[m[measure].idxmax()]
            notes.append(f"Best month: **{best['_month']:%b %Y}** with {best[measure]:,.0f}.")
            last, prev = m[measure].iloc[-1], m[measure].iloc[-2]
            if prev:
                notes.append(f"Latest month is **{(last - prev) / prev:+.0%}** compared with the one before.")
        d["_dow"] = d[date].dt.day_name()

    c1, c2 = st.columns(2)
    if product:
        top = d.groupby(product, observed=True)[measure].sum().nlargest(10).reset_index()
        fig = px.bar(top.iloc[::-1], x=measure, y=product, orientation="h", title="Top 10 products",
                     color_discrete_sequence=[TEAL])
        fig.update_layout(yaxis_title=None, xaxis_title=label, margin=dict(t=50, b=10))
        c1.plotly_chart(fig, use_container_width=True)
        summary["Top products"] = top
        share = top[measure].iloc[0] / d[measure].sum()
        notes.append(f"**{top[product].iloc[0]}** is your top product, {share:.0%} of all {label.lower()}.")
    if category:
        cat = d.groupby(category, observed=True)[measure].sum().reset_index()
        fig = px.pie(cat, names=category, values=measure, hole=.55, title=f"{label} by category",
                     color_discrete_sequence=px.colors.qualitative.Safe)
        fig.update_layout(margin=dict(t=50, b=10))
        c2.plotly_chart(fig, use_container_width=True)
        summary["Categories"] = cat
    elif date:
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        w = d.groupby("_dow")[measure].sum().reindex(order).reset_index()
        fig = px.bar(w, x="_dow", y=measure, title=f"{label} by weekday", color_discrete_sequence=[MARIGOLD])
        fig.update_layout(xaxis_title=None, margin=dict(t=50, b=10))
        c2.plotly_chart(fig, use_container_width=True)

    if disc and qty and d[disc].nunique() >= 2:          # does a bigger discount sell more?
        bins = pd.qcut(d[disc], q=min(4, d[disc].nunique()), duplicates="drop")
        if bins.nunique() > 1:
            g = d.groupby(bins, observed=True)[qty].mean().reset_index()
            g[disc] = g[disc].astype(str)
            fig = px.bar(g, x=disc, y=qty, title="Average units per sale by discount level",
                         color_discrete_sequence=[MARIGOLD])
            fig.update_layout(xaxis_title=disc, yaxis_title="Avg units", margin=dict(t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
            summary["Discount vs units"] = g
            lo_u, hi_u = g[qty].iloc[0], g[qty].iloc[-1]
            if lo_u > 0:
                notes.append(f"Sales with the highest discounts sell **{(hi_u - lo_u) / lo_u:+.0%}** units per sale "
                             "compared with the lowest discounts.")

    if notes:
        st.subheader("What stands out")
        for n in notes:
            st.markdown(f'<div class="note">{n}</div>', unsafe_allow_html=True)

    if st.session_state.get("ai_on"):
        if st.button("Write an AI summary with actions"):
            facts = [f"Metric: {label}. {formula or ''}", f"Total {label.lower()}: {d[measure].sum():,.0f}",
                     f"Transactions: {len(d):,}"] + [re.sub(r"\*\*", "", n) for n in notes]
            for name, t in summary.items():
                facts.append(f"{name}:\n{t.head(10).to_string(index=False)}")
            try:
                with st.spinner("Writing summary..."):
                    st.session_state["ai_text"] = llm_insights("\n".join(facts), st.session_state.get("ai_key") or None)
            except Exception as e:
                st.error(f"AI request failed: {e}")
        if st.session_state.get("ai_text"):
            st.subheader("AI summary")
            st.markdown(st.session_state["ai_text"])
    return d, summary


def downloads(df, summary):
    st.subheader("Download")
    export = df.drop(columns=[c for c in df.columns if str(c).startswith("_")])
    c1, c2 = st.columns(2)
    c1.download_button("Cleaned data (CSV)", export.to_csv(index=False), "clean_sales.csv", "text/csv")
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        export.to_excel(xl, sheet_name="Clean data", index=False)
        for name, t in summary.items():
            t.to_excel(xl, sheet_name=name[:30], index=False)
    c2.download_button("Report (Excel)", out.getvalue(), "sales_report.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ------------------------------------------------------------------ page flow
st.markdown('<div class="brand"><span></span>StockSense</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Drop your files here", accept_multiple_files=True, label_visibility="collapsed",
    type=["csv", "tsv", "txt", "xlsx", "xls", "xlsm", "pdf", "db", "sqlite", "sqlite3", "json"])

if not uploaded:
    st.markdown("""
    <div class="hero">
      <div class="display">Upload your sales files. Know what to reorder.</div>
      <p>Drop in your sales records and StockSense cleans them, finds the patterns, and builds your report in minutes.</p>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="chips"><span>CSV</span><span>Excel</span><span>PDF tables</span>'
                '<span>SQLite / DB</span><span>JSON</span><span>Multiple files at once</span></div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="how">
      <div><b>We tidy it up</b><p>Duplicates and stray spaces are fixed automatically. Mixed-up numbers and dates are corrected.</p></div>
      <div><b>You decide on gaps</b><p>If values are missing, you choose how to fill them: average, median, a number you type, or skip those rows.</p></div>
      <div><b>You get the story</b><p>Totals, trends, best sellers and a downloadable Excel report.</p></div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ---- load every file
tables, failed = {}, []
for f in uploaded:
    try:
        tables.update(load_file(f.name, f.getvalue()))
    except Exception as e:
        failed.append(f"{f.name}: {e}")
for msg in failed:
    st.error(f"Could not read {msg}")
if not tables:
    st.info("No data tables found. For PDFs, the table must be real text, not a scanned image.")
    st.stop()

steps(1)
st.header("Prepare your data")

names = list(tables)
mode = "One table"
if len(names) > 1:
    mode = st.radio("You uploaded several tables. What should we analyze?",
                    ["One table", "Combine tables that share the same columns"], horizontal=True)
if mode == "One table":
    choice = st.selectbox("Table", names)
    raw = tables[choice]
else:
    choice = st.multiselect("Tables to combine", names, default=names) or names[:1]
    raw = pd.concat([tables[n] for n in choice], ignore_index=True)

sig = "|".join(sorted([f.name + str(f.size) for f in uploaded]) + [mode] + list(map(str, [choice])))
ck, ok = f"clean::{sig}", f"opt::{sig}"
if ck not in st.session_state:
    with st.spinner("Cleaning your data..."):
        st.session_state[ck] = auto_clean(raw)
df, rep = st.session_state[ck]
if ok in st.session_state:
    df = st.session_state[ok]

c = st.columns(4)
kpi(c[0], f"{len(df):,}", "Rows after cleaning")
kpi(c[1], f"{df.shape[1]}", "Columns")
kpi(c[2], f"{rep['duplicate rows removed']:,}", "Duplicates removed")
kpi(c[3], fmt_bytes(mem_total(df)), "Size in memory")
if f"{ok}::msg" in st.session_state:
    st.success(st.session_state[f"{ok}::msg"])
with st.expander("Preview data"):
    st.dataframe(df.head(50), use_container_width=True)

# ---- data types and size: only change what the user approves
if ok not in st.session_state:
    sk = f"sugg::{sig}"
    if sk not in st.session_state:
        with st.spinner("Checking data types..."):
            st.session_state[sk] = suggest_changes(df)
    sugg = st.session_state[sk]
    if not sugg:
        st.session_state[ok] = df
    else:
        total = mem_total(df)
        saving = sum(s["before"] - s["after"] for s in sugg if s["kind"] == "Smaller size")
        st.subheader("Fix data types and shrink the file")
        if total >= 50 * 1024 ** 2:
            st.markdown(f'<div class="note"><b>Large file ({fmt_bytes(total)} in memory).</b> '
                        f'Compressing can free about {fmt_bytes(saving)} and make everything faster.</div>',
                        unsafe_allow_html=True)
        else:
            st.caption("We checked each column's real values. Tick the changes you want, then apply. Nothing changes until you do.")
        table = pd.DataFrame([{
            "Apply": s["default"], "Column": s["col"], "Type of change": s["kind"],
            "Now": s["cur"], "Change to": s["new"], "Value range": s["rng"],
            "Why": s["reason"], "Memory saved": fmt_bytes(s["before"] - s["after"]) if s["after"] < s["before"] else "-",
        } for s in sugg])
        with st.form("opt_form"):
            edited = st.data_editor(table, hide_index=True, use_container_width=True,
                                    disabled=[x for x in table.columns if x != "Apply"],
                                    column_config={"Apply": st.column_config.CheckboxColumn("Apply")})
            b1, b2, _ = st.columns([2, 1, 3])
            go = b1.form_submit_button("Compress and fix selected")
            skip = b2.form_submit_button("Skip")
        if go or skip:
            new, n = df.copy(), 0
            if go:
                for s, do in zip(sugg, edited["Apply"]):
                    if do:
                        new[s["col"]] = convert(new[s["col"]], s["code"]); n += 1
            st.session_state[ok] = new
            if n:
                st.session_state[f"{ok}::msg"] = (f"Changed {n} column(s). Memory: {fmt_bytes(mem_total(df))} "
                                                  f"to {fmt_bytes(mem_total(new))}.")
            st.rerun()
        st.stop()

# ---- missing values: ask the user
missing = df.isna().sum()
missing = missing[missing > 0]
if len(missing):
    key = f"filled::{sig}"
    if key not in st.session_state:
        st.markdown(f'<div class="note"><b>{int(missing.sum()):,} blank values</b> in {len(missing)} column(s). '
                    'Tell us how to fill each one.</div>', unsafe_allow_html=True)
        with st.form("fill_form"):
            plan = {}
            for col, n in missing.items():
                a, b = st.columns([2, 2])
                a.markdown(f"**{col}**  \n{n:,} blank ({n / len(df):.0%})")
                method = b.selectbox("Fill with", fill_options(df[col]), key=f"m_{col}", label_visibility="collapsed")
                custom = None
                if method == "Custom value":
                    custom = (b.number_input("Value", key=f"c_{col}") if pd.api.types.is_numeric_dtype(df[col])
                              else b.text_input("Value", key=f"c_{col}"))
                elif method == "Custom date":
                    custom = b.date_input("Date", key=f"c_{col}")
                plan[col] = (method, custom)
            if st.form_submit_button("Apply and continue"):
                st.session_state[key] = apply_fill(df, plan)
                st.rerun()
        st.stop()
    df = st.session_state[key]
    st.success("Blank values filled. Your data is ready.")
else:
    st.success("Your data is clean. No blank values found.")

enriched, summary = show_insights(df)
downloads(enriched, summary)