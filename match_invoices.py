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
# دوال تنظيف الأسماء / WORD_MAP + STOPWORDS
# ============================================================
WORD_MAP = {

    # ====================
    #  حروف وتبديلات إملائية
    # ====================
    "المياه": "مياه", "المياة": "مياه", "مياة": "مياه",
    "الماء": "مياه", "مائ": "مياه",

    "الصرف": "صرف", "الصرق": "صرف",
    "الصرف الصحي": "صرف صحي", "الصرف الصحى": "صرف صحي",
    "صرف الصحي": "صرف صحي", "صرف الصحى": "صرف صحي",

    "الشرب": "شرب", "الشراب": "شرب", "شراب": "شرب",

    "الوحدات": "وحدات", "الوحده": "وحدات",

    "بسوهج": "بسوهاج", "بسوهـاج": "بسوهاج",
    "سوهاج": "بسوهاج", "سهاج": "بسوهاج",

    "القاهره": "القاهرة", "قاهره": "القاهرة",

    "الاسكندريه": "اسكندرية", "اسكندريه": "اسكندرية",
    "الإسكندرية": "اسكندرية",

    "الجيزة": "جيزه", "الجيزه": "جيزه", "جيزة": "جيزه",

    # ====================
    #  كلمات تجارية موحدة
    # ====================
    "للمقاولات": "مقاولات", "المقاولات": "مقاولات", "مقاولون": "مقاولات",
    "مقاولات عامة": "مقاولات", "المقاولات العامة": "مقاولات",

    "للصناعات": "صناعات", "الصناعات": "صناعات",
    "صناعيه": "صناعية",

    "للخدمات": "خدمات", "الخدمات": "خدمات",
    "الخدمية": "خدمات",

    "للتجارة": "تجارة", "التجارة": "تجارة",
    "تجارية": "تجارة",

    "للتنمية": "تنمية", "التنمية": "تنمية",

    "للاستثمار": "استثمار", "استثمارية": "استثمار",

    "للتطوير": "تطوير", "التطوير": "تطوير",

    "للتوريدات": "توريدات", "التوريدات": "توريدات",
    "توريد": "توريدات",

    "للانتاج": "انتاج", "الانتاج": "انتاج",

    "للتوزيع": "توزيع", "التوزيع": "توزيع",

    "للمعالجات": "معالجة", "معالجه": "معالجة",

    # ====================
    #  كلمات صناعية / فنية
    # ====================
    "الكيماويات": "كيماويات", "كيماوي": "كيماويات",
    "كيمياويات": "كيماويات",

    "البلاستيك": "بلاستيك", "البلاستيكية": "بلاستيك",

    "الزجاج": "زجاج", "الزجاجيه": "زجاج",

    "الاخشاب": "اخشاب", "خشب": "اخشاب",

    "البويات": "بويا", "دهانات": "بويا",

    "الورق": "ورق", "اوراق": "ورق", "ورقيه": "ورق",

    "الحديد": "حديد", "حديدية": "حديد",

    "الاسمنت": "اسمنت", "اسمنتية": "اسمنت",

    # ====================
    #  أغذية ومشروبات
    # ====================
    "اغذية": "اغذيه", "الغذائية": "اغذيه", "الغذائيه": "اغذيه",
    "للاغذية": "اغذيه",

    "مخبوزات": "مخبوزات", "بسكويت": "مخبوزات",

    "البان": "البان", "الالبان": "البان",

    "العجائن": "مكرونة", "معجنات": "مكرونة",

    "اللحوم": "لحوم", "لحوم": "لحوم",

    # ====================
    #  كهرباء / طاقة
    # ====================
    "كهرباء": "كهرباء", "كهربية": "كهرباء",
    "الكترونيات": "الكترونيات",

    "محولات": "محولات", "محول": "محولات",

    # ====================
    #  نقل / سيارات
    # ====================
    "للنقل": "نقل", "النقل": "نقل", "النقليات": "نقل",
    "الشحن": "نقل",

    "للسيارات": "سيارات", "السيارات": "سيارات",
    "سيارة": "سيارات",

    # ====================
    #  زراعة / أراضي
    # ====================
    "الزراعى": "زراعي", "الزراعي": "زراعي",
    "زراعية": "زراعي",

    "أرضي": "اراضي", "الاراضي": "اراضي",

    # ====================
    #  كلمات عامة
    # ====================
    "جروب": "مجموعة", "مجموعة": "مجموعة",

    "القابضة": "قابضة",

    "المصرية": "مصرية", "المصريه": "مصرية",

    "العرب": "عربي", "العربية": "عربي",
}

STOPWORDS = {
    "شركة", "الشركة", "شركه", "الشركه",
    "وال", "بال", "لل", "ل",
    "مصر", "القاهرة",
    "العالمية", "الدولية", "الجديدة",
    "مصنع", "الصناعات", "صناعية",
    "للتجارة", "تجارية",
    "جروب", "مجموعة", "للصناعات",
    "الغذائية", "الاغذية", "اغذية",
    "والصناعات",
}

# ============================================================
# دوال التطبيع للأسماء
# ============================================================
def normalize_letters(text):
    if pd.isna(text): 
        return ""
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
    if pd.isna(s): 
        return ""
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

# ====== نسخ مبسطة للتعلم التلقائي (بدون WORD_MAP) ======
def basic_normalize_name(s):
    if pd.isna(s):
        return ""
    s = normalize_letters(s).lower()
    s = re.sub(r"[^ء-ي\s]", " ", s)
    words = [remove_al_prefix(w) for w in s.split() if w.strip()]
    words = [w for w in words if w not in STOPWORDS]
    return " ".join(words)

def tokenize_basic(s):
    return [w for w in basic_normalize_name(s).split() if w.strip()]

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
# Auto-Learn: بناء WORD_MAP مقترح من الأسماء (اختياري داخل UI)
# ============================================================
def build_auto_word_map(names_series, min_freq=2, sim_threshold=0.9):
    all_tokens = []
    for name in names_series.dropna():
        toks = tokenize_basic(name)
        all_tokens.extend(toks)

    if not all_tokens:
        return [], {}

    freq = {}
    for t in all_tokens:
        freq[t] = freq.get(t, 0) + 1

    vocab = [w for w, c in freq.items() if c >= min_freq]
    vocab_sorted = sorted(vocab, key=lambda w: freq[w], reverse=True)

    suggestions = []
    auto_map = {}

    for i, base in enumerate(vocab_sorted):
        for other in vocab_sorted[i+1:]:
            b, o = base, other
            if freq[o] > freq[b]:
                b, o = o, b

            sim = SequenceMatcher(None, b, o).ratio()
            if sim >= sim_threshold and b != o:
                if o not in auto_map:
                    auto_map[o] = b
                    suggestions.append({
                        "الكلمة_الأقل_تكرارًا": o,
                        "الكلمة_الأكثر_شيوعًا": b,
                        "تكرار_الأقل": freq[o],
                        "تكرار_الأكثر": freq[b],
                        "نسبة_التشابه": round(sim, 3),
                    })

    return suggestions, auto_map

# ============================================================
# تجهيز الملفات
# ============================================================
def prepare_sales(df_raw):
    df = df_raw.copy()
    df["amt"] = df[COL_AMOUNT].apply(to_num)
    
    grouped = df.groupby(COL_INV).agg(
        net_amount=("amt", "sum"),
        pos_date=(
            COL_DATE,
            lambda x: x[df.loc[x.index, "amt"] > 0].iloc[0]
            if any(df.loc[x.index, "amt"] > 0) else np.nan
        ),
        has_return=("amt", lambda s: any(s < 0)),
        name=(COL_NAME, "first"),
    ).reset_index()
    
    grouped = grouped[grouped["net_amount"] > 0]
    grouped["date_parsed"], grouped["year"], grouped["month"] = parse_dates(
        grouped["pos_date"], dayfirst=False
    )
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
    df["v_tax"] = df.apply(
        lambda r: r["v_tax_paid"] / r["rate"]
        if pd.notna(r["v_tax_paid"]) and pd.notna(r["rate"]) and r["rate"] > 0
        else np.nan,
        axis=1
    )
    df["v_mix"] = df[["v_file", "v_tax"]].mean(axis=1, skipna=True)
    df["date_parsed"], df["year"], df["month"] = parse_dates(
        df[COL_TAX_DATE], dayfirst=True
    )
    df["name_norm"] = df[COL_TAX_NAME].apply(normalize_name)
    df["tokens"] = df[COL_TAX_NAME].apply(tokenize)
    return df

# ============================================================
# فلاتر البحث
# ============================================================
def filter_year_and_date(sales_df, tax_date, tax_year, tax_month):
    if tax_year == 0 or pd.isna(tax_date):
        return sales_df.iloc[0:0]
    
    allowed_years = [tax_year, tax_year - 1, tax_year - 2]
    mask_year = sales_df["year"].isin(allowed_years)
    mask_date = (sales_df["date_parsed"] <= tax_date)
    
    return sales_df[mask_year & mask_date]

def extended_subset_search(cand, targets, max_invoices=50, max_nodes=200000):
    if not targets: 
        return None
    max_t, min_t = max(targets), min(targets)
    
    cand = cand.head(max_invoices).sort_values("net_amount", ascending=False)
    rows = list(cand.itertuples(index=False))
    n = len(rows)
    if n == 0: 
        return None
    
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
        if nodes > max_nodes or cur_sum > max_t * 1.05: 
            return
        if cur_sum + suffix[i] < min_t * 0.95: 
            return
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
# المطابقة الرئيسية (نفسها، تستخدم لاحقًا في المرحلتين)
# ============================================================
def find_best_match(tax_row, sales_df, used_invoices):
    tax_date = tax_row["date_parsed"]
    if pd.isna(tax_date): 
        return None
    
    v_file, v_tax, v_mix = tax_row["v_file"], tax_row["v_tax"], tax_row["v_mix"]
    targets = [t for t in (v_file, v_tax, v_mix) if pd.notna(t) and t > 0]
    if not targets: 
        return None
    
    cand = filter_year_and_date(sales_df, tax_date, tax_row["year"], tax_row["month"])
    if cand.empty: 
        return None
    
    cand = cand[~cand[COL_INV].astype(str).isin(used_invoices)].copy()
    if cand.empty: 
        return None
    
    cand["token_score"] = cand["tokens"].apply(lambda t: len(t & tax_row["tokens"]))
    cand["fuzzy"] = cand["name_norm"].apply(lambda s: fuzzy(s, tax_row["name_norm"]))
    cand = cand[(cand["token_score"] >= 1) | (cand["fuzzy"] >= 0.75)]
    
    if cand.empty: 
        return None
    
    def within_absolute(val, max_diff=1.0):
        return any(abs(val - t) <= max_diff for t in targets)
    
    def within_pct(val, pct=0.05):
        return any(abs(val - t) <= pct * t for t in targets)
    
    cand["value_dist"] = cand["net_amount"].apply(
        lambda x: min(abs(x - t) for t in targets)
    )
    cand = cand.sort_values(
        by=["value_dist", "token_score", "fuzzy"], ascending=[True, False, False]
    )
    
    # 1. فاتورة واحدة متطابقة تماماً
    for _, r in cand.head(100).iterrows():
        if within_absolute(r["net_amount"], max_diff=1.0):
            return (
                [str(r[COL_INV])],
                [str(r["year"])],
                [str(r["pos_date"])],
                float(r["net_amount"]),
                r["has_return"],
            )
    
    # 2. فاتورة واحدة 5%
    for _, r in cand.head(50).iterrows():
        if within_pct(r["net_amount"]):
            return (
                [str(r[COL_INV])],
                [str(r["year"])],
                [str(r["pos_date"])],
                float(r["net_amount"]),
                r["has_return"],
            )
    
    # 3. مجموع 2 فواتير متطابق
    for combo in combinations(cand.head(80).itertuples(index=False), 2):
        total = sum(r.net_amount for r in combo)
        if not within_absolute(total, max_diff=1.0): 
            continue
        invs = [str(r._asdict()[COL_INV]) for r in combo]
        if len(set(invs)) != len(invs): 
            continue
        years = [str(r.year) for r in combo]
        dates = [str(r.pos_date) for r in combo]
        ret = any(r.has_return for r in combo)
        return invs, years, dates, float(total), ret
    
    # 4. مجموع 2-3 فواتير 5%
    for n in [2, 3]:
        for combo in combinations(cand.head(80).itertuples(index=False), n):
            total = sum(r.net_amount for r in combo)
            if not within_pct(total): 
                continue
            invs = [str(r._asdict()[COL_INV]) for r in combo]
            if len(set(invs)) != len(invs): 
                continue
            years = [str(r.year) for r in combo]
            dates = [str(r.pos_date) for r in combo]
            ret = any(r.has_return for r in combo)
            return invs, years, dates, float(total), ret
    
    # 5. بحث موسع للمبالغ الكبيرة
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

def match_all_basic(sales_df, tax_df):
    """
    مطابقة كاملة عادية (تُستخدم في المرحلة الأولى لبناء جدول المطابقة المبدئية)
    """
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
# مرحلة ثانية: مطابقة نهائية باستخدام جدول المطابقة + STOPWORDS
# ============================================================
def match_with_user_feedback(
    sales_df_original,
    tax_df_original,
    matches_edited: pd.DataFrame,
    stopwords_edited: pd.DataFrame
):
    """
    - يستخدم جدول التطابقات المبدئية بعد تعديل المستخدم
    - يستخدم قائمة STOPWORDS المعدلة
    - يعيد حساب tokens بالأسماء باستخدام STOPWORDS الجديدة
    - يثبت التطابقات التي وافق عليها المستخدم
    - يكمل المطابقة لباقي السطور
    """

    # 1) تحديث STOPWORDS من الجدول
    words = []
    if "كلمة" in stopwords_edited.columns:
        for v in stopwords_edited["كلمة"].astype(str).tolist():
            v = v.strip()
            if v:
                words.append(v)
    new_stopwords = set(words)

    global STOPWORDS
    STOPWORDS = new_stopwords

    # 2) إعادة تجهيز tokens بناءً على STOPWORDS الجديدة
    sales_df = sales_df_original.copy()
    tax_df = tax_df_original.copy()

    sales_df["name_norm"] = sales_df["name"].apply(normalize_name)
    sales_df["tokens"] = sales_df["name"].apply(tokenize)

    tax_df["name_norm"] = tax_df[COL_TAX_NAME].apply(normalize_name)
    tax_df["tokens"] = tax_df[COL_TAX_NAME].apply(tokenize)

    # 3) تجهيز result_df و used_invoices
    result = tax_df.copy()
    for col in NEW_COLS:
        result[col] = ""

    used = set()

    # 4) تثبيت التطابقات التي وافق عليها المستخدم
    if matches_edited is not None and not matches_edited.empty:
        if "row_id" in matches_edited.columns:
            for _, r in matches_edited.iterrows():
                # لو المستخدم حط عمود "اعتماد_التطابق" وخلّاه False → نستبعد الصف
                if "اعتماد_التطابق" in matches_edited.columns:
                    if not bool(r.get("اعتماد_التطابق", True)):
                        continue

                row_id = int(r["row_id"])
                inv_str = str(r[NEW_COLS[0]]).strip()
                if not inv_str:
                    continue

                invs = [x.strip() for x in inv_str.split("+") if x.strip()]
                years = str(r.get(NEW_COLS[1], "")).split("+")
                dates = str(r.get(NEW_COLS[2], "")).split("+")
                amt = r.get(NEW_COLS[3], np.nan)
                note = r.get(NEW_COLS[4], "")

                result.at[row_id, NEW_COLS[0]] = " + ".join(invs)
                result.at[row_id, NEW_COLS[1]] = " + ".join([y.strip() for y in years])
                result.at[row_id, NEW_COLS[2]] = " + ".join([d.strip() for d in dates])
                result.at[row_id, NEW_COLS[3]] = amt
                result.at[row_id, NEW_COLS[4]] = note

                for inv in invs:
                    used.add(inv)

    # 5) إكمال المطابقة لباقي الصفوف
    matched = 0
    for idx, row in result.iterrows():
        if str(result.at[idx, NEW_COLS[0]]).strip():
            matched += 1
            continue

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

    not_matched = len(result) - matched
    return result, matched, not_matched

# ============================================================
# واجهة Streamlit
# ============================================================
st.set_page_config(page_title="مطابقة خصم المنبع", layout="wide")

st.title("🎯 مطابقة خصم المنبع - الإصدار الذهبي (خطوتين)")
st.markdown("---")

with st.expander("📖 فكرة العمل باختصار", expanded=False):
    st.markdown("""
1️⃣ **الخطوة الأولى**:  
- تحميل الملفين  
- تجهيز البيانات  
- عمل مطابقة مبدئية كاملة  
- عرض جدول التطابقات المبدئية (تقدر تمسح/تعدل فيه)  
- عرض جدول STOPWORDS (تقدر تحذف/تضيف كلمات)

2️⃣ **الخطوة الثانية**:  
- تضغط زر **متابعة المطابقة النهائية**  
- البرنامج يعيد المطابقة من الأول باستخدام:
  - STOPWORDS المعدلة
  - التطابقات التي وافقت عليها في الجدول (ويثبتها)
  - ويكمل الباقي تلقائيًا
""")

col1, col2 = st.columns(2)
with col1:
    sales_file = st.file_uploader("📊 ملف المبيعات (CSV)", type="csv")
with col2:
    tax_file = st.file_uploader("📑 كشف خصم المنبع (CSV)", type="csv")

st.markdown("---")

# =======================================
# الخطوة 1: مطابقة مبدئية + بناء الجداول
# =======================================
if st.button("🚀 الخطوة 1: مطابقة مبدئية وبناء جداول المراجعة", use_container_width=True):
    if not sales_file or not tax_file:
        st.error("⚠️ من فضلك ارفع الملفين أولاً!")
        st.stop()
    try:
        with st.spinner("⏳ جاري قراءة الملفات..."):
            sales_raw = pd.read_csv(sales_file, encoding="utf-8-sig", dtype=str)
            tax_raw = pd.read_csv(tax_file, encoding="utf-8-sig", dtype=str)

        with st.spinner("🔄 جاري تجهيز البيانات..."):
            sales_prepared = prepare_sales(sales_raw)
            tax_prepared = prepare_tax(tax_raw)

        with st.spinner("🎯 جاري عمل المطابقة المبدئية..."):
            draft_df, ok, bad = match_all_basic(sales_prepared, tax_prepared)

        # حفظ في session_state لاستخدامها في الخطوة 2
        st.session_state["sales_prepared"] = sales_prepared
        st.session_state["tax_prepared"] = tax_prepared

        # نضيف row_id لربطه لاحقًا
        draft_df = draft_df.copy()
        draft_df.insert(0, "row_id", draft_df.index.astype(int))
        st.session_state["draft_df"] = draft_df

        # جدول التطابقات فقط
        matches_only = draft_df[draft_df[NEW_COLS[0]] != ""].copy()
        matches_only["اعتماد_التطابق"] = True  # افتراضيًا كل التطابقات مقبولة
        st.session_state["matches_table"] = matches_only

        # جدول STOPWORDS قابل للتعديل
        stopwords_df = pd.DataFrame({"كلمة": sorted(STOPWORDS)})
        st.session_state["stopwords_table"] = stopwords_df

        st.success(f"✅ تم تنفيذ المطابقة المبدئية: {ok:,} صف مطابق | {bad:,} غير مطابق.")
        st.info("⬇ انزل للأسفل لمراجعة جدول التطابقات وجدول STOPWORDS ثم اضغط على 'متابعة المطابقة النهائية'.")

    except Exception as e:
        st.error(f"❌ خطأ أثناء الخطوة 1: {str(e)}")
        st.exception(e)

st.markdown("---")

# =======================================
# عرض جداول المراجعة (لو موجودة)
# =======================================
if "draft_df" in st.session_state:

    st.subheader("🧾 جدول التطابقات المبدئية (يمكنك استبعاد الصفوف غير المنطقية)")
    matches_df = st.session_state.get("matches_table", pd.DataFrame())
    if matches_df.empty:
        st.info("لا توجد تطابقات مبدئية حتى الآن.")
    else:
        edited_matches = st.data_editor(
            matches_df,
            key="matches_editor",
            num_rows="dynamic",
            use_container_width=True
        )

    st.subheader("🧹 جدول الكلمات التي يتم تجاهلها في الاسم (STOPWORDS)")
    stopwords_df = st.session_state.get("stopwords_table", pd.DataFrame({"كلمة": sorted(STOPWORDS)}))
    edited_stopwords = st.data_editor(
        stopwords_df,
        key="stopwords_editor",
        num_rows="dynamic",
        use_container_width=True
    )

    st.markdown("---")

    # ========================
    # الخطوة 2: مطابقة نهائية
    # ========================
    if st.button("✅ متابعة المطابقة النهائية باستخدام التعديلات", use_container_width=True):
        try:
            sales_prepared = st.session_state["sales_prepared"]
            tax_prepared = st.session_state["tax_prepared"]

            # لو المستخدم مسح كل التطابقات، edited_matches ممكن تكون فاضية
            edited_matches = st.session_state.get("matches_editor", matches_df)
            edited_stopwords = st.session_state.get("matches_editor_stopwords", edited_stopwords) \
                if "matches_editor_stopwords" in st.session_state else edited_stopwords

            with st.spinner("🔁 جاري إعادة المطابقة بالاعتماد على التعديلات..."):
                final_df, ok2, bad2 = match_with_user_feedback(
                    sales_prepared,
                    tax_prepared,
                    edited_matches if isinstance(edited_matches, pd.DataFrame) else matches_df,
                    edited_stopwords if isinstance(edited_stopwords, pd.DataFrame) else stopwords_df,
                )

            st.success("🎉 تمت المطابقة النهائية بنجاح!")

            total_rows = len(final_df)
            success_rate = (ok2 / total_rows * 100) if total_rows > 0 else 0.0
            fail_rate = (bad2 / total_rows * 100) if total_rows > 0 else 0.0

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("✅ المطابق نهائيًا", f"{ok2:,}", delta=f"{success_rate:.1f}%")
            with c2:
                st.metric("❌ غير المطابق نهائيًا", f"{bad2:,}", delta=f"{fail_rate:.1f}%")
            with c3:
                st.metric("📈 نسبة النجاح النهائية", f"{success_rate:.2f}%")

            st.markdown("---")
            st.markdown("### 📥 تحميل النتائج النهائية")

            colD1, colD2 = st.columns(2)
            with colD1:
                out_all = io.BytesIO()
                final_df.to_csv(out_all, index=False, encoding="utf-8-sig")
                st.download_button(
                    "📥 تحميل الكشف الكامل (نهائي)",
                    data=out_all.getvalue(),
                    file_name="كشف_خصم_منبع_مطابق_نهائي.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with colD2:
                unmatched_final = final_df[final_df[NEW_COLS[0]] == ""]
                if not unmatched_final.empty:
                    out_un = io.BytesIO()
                    unmatched_final.to_csv(out_un, index=False, encoding="utf-8-sig")
                    st.download_button(
                        "📥 تحميل غير المطابق فقط (نهائي)",
                        data=out_un.getvalue(),
                        file_name="غير_مطابق_نهائي_للمراجعة.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.success("👏 كل الصفوف تم مطابقتها نهائيًا.")

            st.markdown("### 👀 معاينة أول 10 صفوف من النتيجة النهائية")
            st.dataframe(final_df.head(10), use_container_width=True)

        except Exception as e:
            st.error(f"❌ خطأ أثناء المطابقة النهائية: {str(e)}")
            st.exception(e)

st.markdown("---")
st.caption("💼 تطوير: محاسب قانوني مايكل نبيل | 🚀 النسخة الذهبية 2025")
