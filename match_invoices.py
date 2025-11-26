import io
import re
from itertools import combinations
from math import isfinite

import numpy as np
import pandas as pd
import streamlit as st
from difflib import SequenceMatcher

# ============================================================
# إعدادات الأعمدة (غيّرها لو أسماء الأعمدة عندك مختلفة)
# ============================================================
COL_INV = "فواتير"              # رقم الفاتورة في ملف المبيعات
COL_DATE = "التاريخ"             # تاريخ الفاتورة
COL_NAME = "اسم الشركة"          # اسم العميل في المبيعات
COL_AMOUNT = "صافى المبيعات"     # صافي المبيعات

COL_TAX_NAME = "اسم الجهة"         # اسم الجهة في كشف الخصم
COL_TAX_AMOUNT = "القيمة الصافية للتعامل"
COL_TAX_TAXED = "محصل لحساب الضريبه"   # المبلغ المخصوم (الضريبة)
COL_TAX_RATE = "نسبة الخصم"        # 0.5% أو 1% أو 2%
COL_TAX_DATE = "تاريخ التعامل"

# الأعمدة الجديدة اللي هتتضاف
NEW_COLS = [
    "المطلوب رقم الفاتورة من ملف المبيعات",
    "سنة الفاتورة من ملف المبيعات",
    "تاريخ الفاتورة من ملف المبيعات",
    "مبلغ الفواتير المستخدمة للتحقق",
    "ملاحظات عن المرتجع",
]

# ============================================================
# دوال مساعدة لتنظيف الأسماء (ممتازة للعربي)
# ============================================================
STOPWORDS = {
    "شركة", "الشركة", "شركه", "الشركه", "وال", "لل", "ل", "مصر", "القاهرة",
    "العالمية", "الدولية", "الجديدة", "مصنع", "الصناعات", "للتجارة", "تجارية"
}

WORD_MAP = {
    "الصرف": "صرف", "والصرف": "صرف", "صرف الصحي": "صرف صحي", "الصرف الصحي": "صرف صحي",
    "الشرب": "شرب", "المياه": "مياه", "المياة": "مياه", "مياة": "مياه",
    "بسوهج": "بسوهاج", "بسوهـاج": "بسوهاج", "بسوهاج": "بسوهاج",
}

def normalize_letters(text):
    if pd.isna(text): return ""
    s = str(text)
    s = re.sub(r"[أإآا]", "ا", s)
    s = re.sub(r"[ة]", "ه", s)
    s = re.sub(r"[ىيئ]", "ي", s)
    s = re.sub(r"[ؤ]", "و", s)
    s = re.sub(r"[ًٌٍَُِّْـ]", "", s)
    return s

def remove_al_prefix(word):
    return word[2:] if word.startswith("ال") else word

def normalize_name(s):
    if pd.isna(s): return ""
    s = normalize_letters(s).lower()
    s = re.sub(r"[^ء-ي\s]", " ", s)
    words = [w for w in s.split() if w.strip()]
    words = [remove_al_prefix(w) for w in words]
    normalized = []
    for w in words:
        normalized.append(WORD_MAP.get(w, w))
    final = " ".join(normalized)
    for k, v in WORD_MAP.items():
        final = re.sub(rf"\b{k}\b", v, final)
    final = re.sub(r"\s+", " ", final).strip()
    return final

def tokenize(s):
    norm = normalize_name(s)
    return set(w for w in norm.split() if w and w not in STOPWORDS)

def fuzzy(a, b):
    return SequenceMatcher(None, a, b).ratio()

def to_num(v):
    try:
        return float(str(v).replace(",", "").strip())
    except:
        return np.nan

def parse_dates(series, dayfirst):
    dt = pd.to_datetime(series, errors="coerce", dayfirst=dayfirst)
    return dt, dt.dt.year.fillna(0).astype(int), dt.dt.month.fillna(0).astype(int)

# ============================================================
# تجهيز ملف المبيعات
# ============================================================
def prepare_sales(df_raw):
    df = df_raw.copy()
    df["amt"] = df[COL_AMOUNT].apply(to_num)
    df_pos = df[df["amt"] > 0]
    df_neg = df[df["amt"] < 0]

    grouped = df.groupby(COL_INV).agg(
        net_amount=("amt", "sum"),
        pos_date=(COL_DATE, lambda x: x[df.loc[x.index, "amt"] > 0].iloc[0] if any(df.loc[x.index, "amt"] > 0) else np.nan),
        has_return=("amt", lambda s: any(s < 0)),
        name=(COL_NAME, "first"),
    ).reset_index()

    grouped = grouped[grouped["net_amount"] > 0]
    grouped["date_parsed"], grouped["year"], grouped["month"] = parse_dates(grouped["pos_date"], dayfirst=False)
    grouped["name_norm"] = grouped["name"].apply(normalize_name)
    grouped["tokens"] = grouped["name"].apply(tokenize)
    return grouped

# ============================================================
# تجهيز كشف الخصم
# ============================================================
def prepare_tax(df_raw):
    df = df_raw.copy()
    df["v_file"] = df[COL_TAX_AMOUNT].apply(to_num)
    df["v_tax_paid"] = df[COL_TAX_TAXED].apply(to_num)

    def rate_to_float(x):
        try:
            return float(str(x).replace("%", "").strip()) / 100.0
        except:
            return np.nan
    df["rate"] = df[COL_TAX_RATE].apply(rate_to_float)

    df["v_tax"] = df.apply(lambda r: r["v_tax_paid"] / r["rate"] if pd.notna(r["v_tax_paid"]) and pd.notna(r["rate"]) and r["rate"] > 0 else np.nan, axis=1)
    df["v_mix"] = df[["v_file", "v_tax"]].mean(axis=1, skipna=True)
    df["date_parsed"], df["year"], df["month"] = parse_dates(df[COL_TAX_DATE], dayfirst=True)
    df["name_norm"] = df[COL_TAX_NAME].apply(normalize_name)
    df["tokens"] = df[COL_TAX_NAME].apply(tokenize)
    return df

# ============================================================
# فلتر السنة والتاريخ
# ============================================================
def filter_year_and_date(sales_df, tax_date, tax_year, tax_month):
    if tax_year == 0 or pd.isna(tax_date):
        return sales_df.iloc[0:0]
    if tax_month in [1, 2, 3]:
        mask_year = (sales_df["year"] == tax_year) | ((sales_df["year"] == tax_year - 1) & sales_df["month"].isin([10, 11, 12]))
    else:
        mask_year = (sales_df["year"] == tax_year)
    mask_date = (sales_df["date_parsed"] <= tax_date)
    return sales_df[mask_year & mask_date]

# ============================================================
# البحث الموسع (للمبالغ الكبيرة)
# ============================================================
def extended_subset_search(cand, v_file, v_tax, v_mix, max_invoices=50, max_nodes=200000):
    targets = [t for t in (v_file, v_tax, v_mix) if pd.notna(t) and t > 0]
    if not targets: return None
    max_t, min_t = max(targets), min(targets)
    cand = cand.head(max_invoices).sort_values("net_amount", ascending=False)
    rows = list(cand.itertuples(index=False))
    n = len(rows)
    if n == 0: return None
    amounts = [r.net_amount for r in rows]
    suffix = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + amounts[i]

    best = None
    best_diff = float("inf")
    nodes = 0

    def dfs(i, cur_sum, chosen):
        nonlocal best, best_diff, nodes
        nodes += 1
        if nodes > max_nodes: return
        if cur_sum > max_t * 1.05: return
        if cur_sum + suffix[i] < min_t * 0.95: return
        if i == n:
            diff = min(abs(cur_sum - t) for t in targets)
            if diff <= 0.05 * max_t and diff < best_diff:
                best_diff = diff
                best = chosen[:]
            return
        # نأخذ
        chosen.append(i)
        dfs(i + 1, cur_sum + amounts[i], chosen)
        chosen.pop()
        # نترك
        dfs(i + 1, cur_sum, chosen)

    dfs(0, 0.0, [])
    return [rows[i] for i in best] if best else None

# ============================================================
# المطابقة النهائية (الدالة الرئيسية)
# ============================================================
def find_best_match(tax_row, sales_df, used_invoices):
    tax_date = tax_row["date_parsed"]
    if pd.isna(tax_date): return None

    v_file = tax_row["v_file"]
    v_tax = tax_row["v_tax"]
    v_mix = tax_row["v_mix"]

    cand = filter_year_and_date(sales_df, tax_date, tax_row["year"], tax_row["month"])
    if cand.empty: return None

    cand = cand[~cand[COL_INV].astype(str).isin(used_invoices)]
    if cand.empty: return None

    cand = cand.copy()
    cand["token_score"] = cand["tokens"].apply(lambda t: len(t & tax_row["tokens"]))
    cand["fuzzy"] = cand["name_norm"].apply(lambda s: fuzzy(s, tax_row["name_norm"]))
    cand = cand[(cand["token_score"] >= 1) | (cand["fuzzy"] >= 0.85)]
    if cand.empty: return None

    cand["value_dist"] = cand["net_amount"].apply(lambda x: min(abs(x - t) for t in (v_file, v_tax, v_mix) if pd.notna(t)))
    cand = cand.sort_values(by=["value_dist", "fuzzy", "token_score"], ascending=[True, False, False])

    def within_5pct(val):
        for t in (v_file, v_tax, v_mix):
            if pd.notna(t) and t > 0 and abs(val - t) <= 0.05 * t:
                return True
        return False

    # 1. فاتورة واحدة
    for _, r in cand.head(40).iterrows():
        if within_5pct(r["net_amount"]):
            return [str(r[COL_INV])], [str(r["year"])], [str(r["pos_date"])], float(r["net_amount"]), r["has_return"]

    # 2. مجموع 2 أو 3 فواتير
    for n in [2, 3]:
        for combo in combinations(cand.head(80).itertuples(index=False), n):
            total = sum(r.net_amount for r in combo)
            if not within_5pct(total): continue
            invs = [str(r._asdict()[COL_INV]) for r in combo]
            if len(set(invs)) != len(invs): continue
            years = [str(r.year) for r in combo]
            dates = [str(r.pos_date) for r in combo]
            ret = any(r.has_return for r in combo)
            return invs, years, dates, float(total), ret

    # 3. مجموع كبير (أكتر من 100 ألف)
    target = v_mix if pd.notna(v_mix) else (v_tax if pd.notna(v_tax) else v_file)
    if pd.notna(target) and target >= 100000:
        ext = extended_subset_search(cand, v_file, v_tax, v_mix)
        if ext:
            total = sum(r.net_amount for r in ext)
            if within_5pct(total):
                invs = [str(r._asdict()[COL_INV]) for r in ext]
                years = [str(r.year) for r in ext]
                dates = [str(r.pos_date) for r in ext]
                ret = any(r.has_return for r in ext)
                return invs, years, dates, float(total), ret

    return None

# ============================================================
# تشغيل المطابقة على الكل
# ============================================================
def match_all(sales_df, tax_df):
    used = set()
    result = tax_df.copy()
    for col in NEW_COLS:
        result[col] = ""

    matched = 0
    for idx, row in result.iterrows():
        res = find_best_match(row, sales_df, used)
        if res:
            invs, years, dates, amt, has_ret = res
            result.at[idx, NEW_COLS[0]] = " + ".join(invs)
            result.at[idx, NEW_COLS[1]] = " + ".join(years)
            result.at[idx, NEW_COLS[2]] = " + ".join(dates)
            result.at[idx, NEW_COLS[3]] = amt
            result.at[idx, NEW_COLS[4]] = "له مرتجع" if has_ret else ""
            used.update(invs)
            matched += 1

    return result, matched, len(result) - matched

# ============================================================
# واجهة Streamlit
# ============================================================
st.set_page_config(page_title="مطابقة خصم المنبع 2025 - النهائي", layout="wide")
st.title("مطابقة خصم المنبع - الإصدار الذهبي 2025")
st.markdown("**يدعم كل حاجة: مرتجعات، حساب من الضريبة، مبالغ كبيرة، عربي 100%**")

c1, c2 = st.columns(2)
with c1:
    sales_file = st.file_uploader("ملف المبيعات (CSV)", type="csv")
with c2:
    tax_file = st.file_uploader("كشف خصم المنبع (CSV)", type="csv")

if st.button("ابدأ المطابقة الآن", type="primary"):
    if not sales_file or not tax_file:
        st.error("ارفع الملفين أولاً!")
    else:
        with st.spinner("جاري المطابقة... (ممكن ياخد دقايق لو الملف كبير)"):
            sales_raw = pd.read_csv(sales_file, encoding="utf-8-sig", dtype=str)
            tax_raw = pd.read_csv(tax_file, encoding="utf-8-sig", dtype=str)

            sales_prepared = prepare_sales(sales_raw)
            tax_prepared = prepare_tax(tax_raw)

            final_df, ok, bad = match_all(sales_prepared, tax_prepared)

            st.success(f"تمت المطابقة: {ok:,} صف | غير مطابق: {bad:,} صف | نسبة النجاح: {(ok/(ok+bad)*100):.2f}%")

            # ملف كامل
            output = io.BytesIO()
            final_df.to_csv(output, index=False, encoding="utf-8-sig")
            st.download_button(
                label="تحميل الكشف الكامل بعد المطابقة",
                data=output.getvalue(),
                file_name="كشف_خصم_المنبع_مطابق_نهائي.csv",
                mime="text/csv"
            )

            # غير المطابق فقط
            unmatched = final_df[final_df[NEW_COLS[0]] == ""]
            if not unmatched.empty:
                out2 = io.BytesIO()
                unmatched.to_csv(out2, index=False, encoding="utf-8-sig")
                st.download_button(
                    label="تحميل غير المطابق فقط (للمراجعة)",
                    data=out2.getvalue(),
                    file_name="غير_مطابق_للمراجعة.csv",
                    mime="text/csv"
                )

st.markdown("---")
st.caption("تم التطوير بواسطة المحاسب القانوني: مايكل نبيل © 2025 - النسخة السريعة اللي مش بتهنج أبدًا")
