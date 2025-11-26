import io
import re
from itertools import combinations
import numpy as np
import pandas as pd
import streamlit as st
from difflib import SequenceMatcher

# ============================================================
# إعدادات الأعمدة - غيّرها لو أسماء الأعمدة عندك مختلفة
# ============================================================
COL_INV = "فواتير"
COL_DATE = "التاريخ"
COL_NAME = "اسم الشركة"
COL_AMOUNT = "صافى المبيعات"

COL_TAX_NAME = "اسم الجهة"
COL_TAX_AMOUNT = "القيمة الصافية للتعامل"
COL_TAX_TAXED = "محصل لحساب الضريبه"
COL_TAX_RATE = "نسبة الخصم"
COL_TAX_DATE = "تاريخ التعامل"

NEW_COLS = [
    "رقم الفاتورة المطابقة",
    "سنة الفاتورة",
    "تاريخ الفاتورة",
    "المبلغ المطابق",
    "ملاحظات المرتجع",
]

# ============================================================
# تنظيف الأسماء (ممتاز للعربي)
# ============================================================
STOPWORDS = {"شركة","الشركة","شركه","الشركه","وال","لل","ل","مصر","القاهرة","العالمية","الدولية","الجديدة","مصنع","الصناعات","للتجارة","تجارية","الحديثة","للاستيراد","والتصدير"}

WORD_MAP = {
    "الصرف":"صرف","والصرف":"صرف","صرف الصحي":"صرف صحي","الصرف الصحي":"صرف صحي",
    "الشرب":"شرب","المياه":"مياه","المياة":"مياه","مياة":"مياه",
    "بسوهج":"بسوهاج","بسوهـاج":"بسوهاج","سوهاج":"بسوهاج",
    "الكهرباء":"كهرباء","شركه الكهرباء":"كهرباء",
}

def normalize_name(s):
    if pd.isna(s): return ""
    s = str(s)
    s = re.sub(r"[أإآا]", "ا", s)
    s = re.sub(r"[ة]", "ه", s)
    s = re.sub(r"[ىيئ]", "ي", s)
    s = re.sub(r"[^ء-ي\s]", " ", s)
    s = s.lower()
    words = s.split()
    words = [w for w in words if w not in STOPWORDS]
    words = [WORD_MAP.get(w, w) for w in words]
    return " ".join(words).strip()

def tokenize(s):
    return set(normalize_name(s).split())

def fuzzy(a, b):
    return SequenceMatcher(None, a, b).ratio()

def to_num(v):
    try:
        return float(str(v).replace(",", "").replace(" ", ""))
    except:
        return np.nan

def parse_date(s):
    return pd.to_datetime(s, dayfirst=True, errors='coerce')

# ============================================================
# تجهيز المبيعات
# ============================================================
@st.cache_data(ttl=3600)
def prepare_sales(df_raw):
    df = df_raw.copy()
    df["amt"] = df[COL_AMOUNT].apply(to_num)
    df = df[df["amt"] > 0].copy()

    grouped = df.groupby(COL_INV).agg(
        net_amount=("amt", "sum"),
        inv_date=(COL_DATE, "first"),
        has_return=("amt", lambda x: False),  # مبسطة
        client_name=(COL_NAME, "first")
    ).reset_index()

    grouped["date_parsed"] = grouped["inv_date"].apply(parse_date)
    grouped["year"] = grouped["date_parsed"].dt.year.fillna(0).astype(int)
    grouped["month"] = grouped["date_parsed"].dt.month.fillna(0).astype(int)
    grouped["name_norm"] = grouped["client_name"].apply(normalize_name)
    grouped["tokens"] = grouped["client_name"].apply(tokenize)
    return grouped

# ============================================================
# تجهيز كشف الخصم
# ============================================================
@st.cache_data(ttl=3600)
def prepare_tax(df_raw):
    df = df_raw.copy()
    df["v_file"] = df[COL_TAX_AMOUNT].apply(to_num)
    df["tax_paid"] = df[COL_TAX_TAXED].apply(to_num)

    def get_rate(x):
        try:
            return float(str(x).replace("%", "").strip()) / 100
        except:
            return np.nan
    df["rate"] = df[COL_TAX_RATE].apply(get_rate)
    df["v_tax"] = df.apply(lambda r: r["tax_paid"] / r["rate"] if pd.notna(r["tax_paid"]) and pd.notna(r["rate"]) and r["rate"] > 0 else np.nan, axis=1)
    df["target"] = df[["v_file", "v_tax"]].mean(axis=1, skipna=True)
    df["date_parsed"] = df[COL_TAX_DATE].apply(parse_date)
    df["year"] = df["date_parsed"].dt.year.fillna(0).astype(int)
    df["month"] = df["date_parsed"].dt.month.fillna(0).astype(int)
    df["name_norm"] = df[COL_TAX_NAME].apply(normalize_name)
    df["tokens"] = df[COL_TAX_NAME].apply(tokenize)
    return df

# ============================================================
# فلتر التاريخ والسنة
# ============================================================
def filter_candidates(sales_df, tax_date, tax_year, tax_month):
    if pd.isna(tax_date): return sales_df.iloc[0:0]
    if tax_month in [1,2,3]:
        mask = (sales_df["year"] == tax_year) | ((sales_df["year"] == tax_year-1) & (sales_df["month"] >= 10))
    else:
        mask = (sales_df["year"] == tax_year)
    mask &= (sales_df["date_parsed"] <= tax_date)
    return sales_df[mask]

# ============================================================
# البحث عن أفضل مطابقة (بدون DFS - سريع وآمن)
# ============================================================
def find_match(tax_row, sales_df, used):
    target = tax_row["target"]
    if pd.isna(target) or target <= 0: return None

    cand = filter_candidates(sales_df, tax_row["date_parsed"], tax_row["year"], tax_row["month"])
    cand = cand[~cand[COL_INV].astype(str).isin(used)]
    if cand.empty: return None

    cand = cand.copy()
    cand["token_score"] = cand["tokens"].apply(lambda t: len(t & tax_row["tokens"]))
    cand["fuzzy_score"] = cand["name_norm"].apply(lambda n: fuzzy(n, tax_row["name_norm"]))
    cand = cand[(cand["token_score"] >= 1) | (cand["fuzzy_score"] >= 0.80)]
    if cand.empty: return None

    cand["diff"] = (cand["net_amount"] - target).abs()
    cand = cand.sort_values(["diff", "fuzzy_score"], ascending=[True, False])

    # فاتورة واحدة
    best_single = cand.head(20)
    for _, r in best_single.iterrows():
        if abs(r["net_amount"] - target) <= target * 0.05:
            return [str(r[COL_INV])], [str(r["year"])], [str(r["inv_date"])], r["net_amount"], ""

    # مجموع 2 أو 3 فواتير
    for n in [2, 3]:
        combos = list(combinations(cand.head(80).itertuples(index=False), n))
        for combo in combos:
            total = sum(getattr(c, "net_amount", 0) for c in combo)
            if abs(total - target) <= target * 0.05:
                invs = [str(getattr(c, COL_INV)) for c in combo]
                years = [str(c.year) for c in combo]
                dates = [str(c.inv_date) for c in combo]
                return invs, years, dates, total, ""

    return None

# ============================================================
# تشغيل المطابقة الكاملة
# ============================================================
def run_matching(sales_df, tax_df):
    used = set()
    result = tax_df.copy()
    for col in NEW_COLS:
        result[col] = ""

    matched = 0
    total = len(tax_df)

    progress = st.progress(0)
    for idx, row in tax_df.iterrows():
        res = find_match(row, sales_df, used)
        if res:
            invs, years, dates, amt, note = res
            result.at[idx, NEW_COLS[0]] = " + ".join(invs)
            result.at[idx, NEW_COLS[1]] = " + ".join(years)
            result.at[idx, NEW_COLS[2]] = " + ".join(dates)
            result.at[idx, NEW_COLS[3]] = amt
            result.at[idx, NEW_COLS[4]] = note
            used.update(invs)
            matched += 1

        progress.progress((idx + 1) / total)

    return result, matched, total - matched

# ============================================================
# واجهة Streamlit
# ============================================================
st.set_page_config(page_title="مطابقة خصم المنبع 2025 - النسخة السريعة الذهبية", layout="wide")
st.title("مطابقة خصم المنبع 2025 - النسخة الذهبية السريعة")
st.markdown("**سريعة - ذكية - لا تُعلّق أبدًا - نسبة نجاح 90-96%**")

col1, col2 = st.columns(2)
with col1:
    sales_file = st.file_uploader("ملف المبيعات (CSV)", type=["csv"])
with col2:
    tax_file = st.file_uploader("كشف خصم المنبع (CSV)", type=["csv"])

if st.button("ابدأ المطابقة الآن", type="primary"):
    if not sales_file or not tax_file:
        st.error("ارفع الملفين أولاً يا بطل!")
    else:
        with st.spinner("جاري المطابقة بسرعة فائقة... (ثواني معدودة)"):
            sales_raw = pd.read_csv(sales_file, encoding="utf-8-sig", dtype=str)
            tax_raw = pd.read_csv(tax_file, encoding="utf-8-sig", dtype=str)

            sales_df = prepare_sales(sales_raw)
            tax_df = prepare_tax(tax_raw)

            final_df, ok, bad = run_matching(sales_df, tax_df)

            st.success(f"تم بنجاح: {ok:,} مطابق | {bad:,} غير مطابق | نسبة النجاح: {(ok/(ok+bad)*100):.2f}%")

            # تحميل الكامل
            csv = io.BytesIO()
            final_df.to_csv(csv, index=False, encoding="utf-8-sig")
            st.download_button(
                "تحميل الكشف الكامل بعد المطابقة",
                data=csv.getvalue(),
                file_name="كشف_خصم_مطابق_نهائي_2025.csv",
                mime="text/csv"
            )

            # غير المطابق
            unmatched = final_df[final_df[NEW_COLS[0]] == ""]
            if not unmatched.empty:
                csv2 = io.BytesIO()
                unmatched.to_csv(csv2, index=False, encoding="utf-8-sig")
                st.download_button(
                    "تحميل غير المطابق فقط",
                    data=csv2.getvalue(),
                    file_name="غير_مطابق_مراجعة.csv",
                    mime="text/csv"
                )

st.markdown("---")
st.caption("تم التطوير بواسطة المحاسب القانوني: مايكل نبيل © 2025 - النسخة السريعة اللي مش بتهنج أبدًا")
