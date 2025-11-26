import io
import re
from itertools import combinations
import numpy as np
import pandas as pd
import streamlit as st
from difflib import SequenceMatcher

# ============================================================
# إعدادات الأعمدة
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
    "المطلوب رقم الفاتورة من ملف المبيعات",
    "سنة الفاتورة من ملف المبيعات",
    "تاريخ الفاتورة من ملف المبيعات",
    "مبلغ الفواتير المستخدمة للتحقق",
    "ملاحظات عن المرتجع",
]

# ============================================================
# دوال تنظيف الأسماء
# ============================================================
STOPWORDS = {
    "شركة", "الشركة", "شركه", "الشركه", "وال", "لل", "ل", "مصر", "القاهرة",
    "العالمية", "الدولية", "الجديدة", "مصنع", "الصناعات", "للتجارة", "تجارية",
    "جروب", "للصناعات", "الغذائية", "اجرواندس", "والصناعات"
}

WORD_MAP = {
    "الصرف": "صرف", "والصرف": "صرف", "صرف الصحي": "صرف صحي",
    "الشرب": "شرب", "المياه": "مياه", "المياة": "مياه", "مياة": "مياه",
    "بسوهج": "بسوهاج", "بسوهـاج": "بسوهاج",
    "الزراعى": "زراعي", "الزراعي": "زراعي", "للاستثمار": "استثمار",
    "كليوباترا": "كليوباترا",  # نثبت الكلمات المهمة
}

def normalize_letters(text):
    if pd.isna(text): return ""
    s = str(text)
    s = re.sub(r"[أإآا]", "ا", s)
    s = re.sub(r"[ة]", "ه", s)
    s = re.sub(r"[ىيئ]", "ي", s)
    s = re.sub(r"[ؤ]", "و", s)
    s = re.sub(r"[ًٌٍَُِّْـ]", "", s)
    return s

def remove_al_prefix(word):
    for pref in ("وال", "بال", "لل", "ال", "ل"):
        if word.startswith(pref) and len(word) > len(pref) + 1:
            return word[len(pref):]
    return word

def normalize_name(s):
    if pd.isna(s): return ""
    s = normalize_letters(s).lower()
    s = re.sub(r"[^ء-ي\s]", " ", s)
    words = [remove_al_prefix(w) for w in s.split() if w.strip()]
    normalized = [WORD_MAP.get(w, w) for w in words]
    final = " ".join(normalized)
    for k, v in WORD_MAP.items():
        final = re.sub(rf"\b{k}\b", v, final)
    return re.sub(r"\s+", " ", final).strip()

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
# تجهيز الملفات
# ============================================================
def prepare_sales(df_raw):
    df = df_raw.copy()
    df["amt"] = df[COL_AMOUNT].apply(to_num)
    
    grouped = df.groupby(COL_INV).agg(
        net_amount=("amt", "sum"),
        pos_date=(COL_DATE, lambda x: x[df.loc[x.index, "amt"] > 0].iloc[0] if any(df.loc[x.index, "amt"] > 0) else np.nan),
        has_return=("amt", lambda s: any(s < 0)),
        name=(COL_NAME, "first"),
    ).reset_index()
    
    grouped = grouped[grouped["net_amount"] > 0]
    grouped["date_parsed"], grouped["year"], grouped["month"] = parse_dates(grouped["pos_date"], dayfirst=True)
    grouped["name_norm"] = grouped["name"].apply(normalize_name)
    grouped["tokens"] = grouped["name"].apply(tokenize)
    return grouped

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
# فلاتر البحث - محسّن
# ============================================================
def filter_year_and_date(sales_df, tax_date, tax_year, tax_month):
    """
    فلتر موسّع: يبحث في السنة الحالية + السنتين اللي قبلها
    لتغطية حالات التأخير في الإقرار
    """
    if tax_year == 0 or pd.isna(tax_date):
        return sales_df.iloc[0:0]
    
    # نبحث في 3 سنوات: السنة الحالية والسنتين اللي قبلها
    allowed_years = [tax_year, tax_year - 1, tax_year - 2]
    mask_year = sales_df["year"].isin(allowed_years)
    
    # التاريخ لازم يكون قبل أو يساوي تاريخ كشف الخصم
    mask_date = (sales_df["date_parsed"] <= tax_date)
    
    return sales_df[mask_year & mask_date]

def extended_subset_search(cand, targets, max_invoices=50, max_nodes=200000):
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
        if nodes > max_nodes or cur_sum > max_t * 1.05: return
        if cur_sum + suffix[i] < min_t * 0.95: return
        if i == n:
            diff = min(abs(cur_sum - t) for t in targets)
            if diff <= 0.05 * max_t and diff < best_diff:
                best_diff = diff
                best = chosen[:]
            return
        chosen.append(i)
        dfs(i + 1, cur_sum + amounts[i], chosen)
        chosen.pop()
        dfs(i + 1, cur_sum, chosen)
    
    dfs(0, 0.0, [])
    return [rows[i] for i in best] if best else None

# ============================================================
# المطابقة الرئيسية - محسّنة جداً
# ============================================================
def find_best_match(tax_row, sales_df, used_invoices):
    tax_date = tax_row["date_parsed"]
    if pd.isna(tax_date): return None
    
    v_file, v_tax, v_mix = tax_row["v_file"], tax_row["v_tax"], tax_row["v_mix"]
    targets = [t for t in (v_file, v_tax, v_mix) if pd.notna(t) and t > 0]
    if not targets: return None
    
    # فلتر موسّع للسنوات
    cand = filter_year_and_date(sales_df, tax_date, tax_row["year"], tax_row["month"])
    if cand.empty: return None
    
    cand = cand[~cand[COL_INV].astype(str).isin(used_invoices)].copy()
    if cand.empty: return None
    
    # حساب التشابه
    cand["token_score"] = cand["tokens"].apply(lambda t: len(t & tax_row["tokens"]))
    cand["fuzzy"] = cand["name_norm"].apply(lambda s: fuzzy(s, tax_row["name_norm"]))
    
    # فلتر متوازن
    cand = cand[(cand["token_score"] >= 1) | (cand["fuzzy"] >= 0.75)]
    if cand.empty: return None
    
    def within_pct(val, pct=0.05):
        return any(abs(val - t) <= pct * t for t in targets)
    
    def within_absolute(val, max_diff=1.0):
        """للمبالغ المتطابقة تماماً (فرق ≤ 1 جنيه)"""
        return any(abs(val - t) <= max_diff for t in targets)
    
    cand["value_dist"] = cand["net_amount"].apply(lambda x: min(abs(x - t) for t in targets))
    
    # ترتيب أذكى: الأقرب في المبلغ أولاً، ثم التشابه اللفظي
    cand = cand.sort_values(
        by=["value_dist", "token_score", "fuzzy"], 
        ascending=[True, False, False]
    )
    
    # 🔥 الأولوية الأولى: فاتورة واحدة متطابقة تماماً (فرق ≤ 1 جنيه)
    for _, r in cand.head(100).iterrows():
        if within_absolute(r["net_amount"], max_diff=1.0):
            return [str(r[COL_INV])], [str(r["year"])], [str(r["pos_date"])], float(r["net_amount"]), r["has_return"]
    
    # 🎯 الأولوية الثانية: فاتورة واحدة في حدود 5%
    for _, r in cand.head(50).iterrows():
        if within_pct(r["net_amount"]):
            return [str(r[COL_INV])], [str(r["year"])], [str(r["pos_date"])], float(r["net_amount"]), r["has_return"]
    
    # ⚡ الأولوية الثالثة: مجموع 2 فواتير متطابق تماماً
    for combo in combinations(cand.head(80).itertuples(index=False), 2):
        total = sum(r.net_amount for r in combo)
        if not within_absolute(total, max_diff=1.0): continue
        invs = [str(r._asdict()[COL_INV]) for r in combo]
        if len(set(invs)) != len(invs): continue
        years = [str(r.year) for r in combo]
        dates = [str(r.pos_date) for r in combo]
        ret = any(r.has_return for r in combo)
        return invs, years, dates, float(total), ret
    
    # 📊 الأولوية الرابعة: مجموع 2-3 فواتير في حدود 5%
    for n in [2, 3]:
        for combo in combinations(cand.head(80).itertuples(index=False), n):
            total = sum(r.net_amount for r in combo)
            if not within_pct(total): continue
            invs = [str(r._asdict()[COL_INV]) for r in combo]
            if len(set(invs)) != len(invs): continue
            years = [str(r.year) for r in combo]
            dates = [str(r.pos_date) for r in combo]
            ret = any(r.has_return for r in combo)
            return invs, years, dates, float(total), ret
    
    # 🚀 الأولوية الخامسة: بحث موسع للمبالغ الكبيرة
    if max(targets) >= 100000:
        ext = extended_subset_search(cand, targets)
        if ext:
            total = sum(r.net_amount for r in ext)
            if within_pct(total):
                invs = [str(r._asdict()[COL_INV]) for r in ext]
                years = [str(r.year) for r in ext]
                dates = [str(r.pos_date) for r in ext]
                ret = any(r.has_return for r in ext)
                return invs, years, dates, float(total), ret
    
    return None

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
st.set_page_config(page_title="مطابقة خصم المنبع 2025", layout="wide")
st.title("🎯 مطابقة خصم المنبع - النسخة الذهبية المحسنة")
st.markdown("""
**✨ التحسينات الجديدة:**
- 🔍 بحث في 3 سنوات (بدلاً من سنة واحدة)
- 🎯 أولوية للفواتير المتطابقة تماماً (فرق ≤ 1 جنيه)
- ⚡ ترتيب أذكى حسب المبلغ أولاً
- 📊 معالجة أفضل للأسماء المتشابهة
""")

c1, c2 = st.columns(2)
with c1:
    sales_file = st.file_uploader("📊 ملف المبيعات (CSV)", type="csv")
with c2:
    tax_file = st.file_uploader("📑 كشف خصم المنبع (CSV)", type="csv")

if st.button("🚀 ابدأ المطابقة الآن", type="primary"):
    if not sales_file or not tax_file:
        st.error("⚠️ ارفع الملفين أولاً!")
    else:
        with st.spinner("⏳ جاري المطابقة..."):
            sales_raw = pd.read_csv(sales_file, encoding="utf-8-sig", dtype=str)
            tax_raw = pd.read_csv(tax_file, encoding="utf-8-sig", dtype=str)
            
            sales_prepared = prepare_sales(sales_raw)
            tax_prepared = prepare_tax(tax_raw)
            
            final_df, ok, bad = match_all(sales_prepared, tax_prepared)
            
            success_rate = (ok/(ok+bad)*100) if (ok+bad) > 0 else 0
            
            # عرض النتائج بطريقة جذابة
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("✅ المطابق", f"{ok:,}", delta="نجح")
            with col2:
                st.metric("❌ غير المطابق", f"{bad:,}", delta="للمراجعة")
            with col3:
                st.metric("📈 نسبة النجاح", f"{success_rate:.2f}%")
            
            st.divider()
            
            output = io.BytesIO()
            final_df.to_csv(output, index=False, encoding="utf-8-sig")
            st.download_button(
                label="📥 تحميل الكشف الكامل بعد المطابقة",
                data=output.getvalue(),
                file_name="كشف_مطابق_ذهبي.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            unmatched = final_df[final_df[NEW_COLS[0]] == ""]
            if not unmatched.empty:
                out2 = io.BytesIO()
                unmatched.to_csv(out2, index=False, encoding="utf-8-sig")
                st.download_button(
                    label="📥 تحميل غير المطابق فقط (للمراجعة اليدوية)",
                    data=out2.getvalue(),
                    file_name="غير_مطابق_للمراجعة.csv",
                    mime="text/csv",
                    use_container_width=True
                )

st.markdown("---")
st.caption("💼 تطوير: محاسب قانوني مايكل نبيل | 🚀 النسخة الذهبية 2025")
