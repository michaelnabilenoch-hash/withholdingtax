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
COL_REG = "رقم التسجيل"  # رقم التسجيل الضريبي

COL_TAX_NAME = "اسم الجهة"
COL_TAX_AMOUNT = "القيمة الصافية للتعامل"
COL_TAX_TAXED = "محصل لحساب الضريبه"
COL_TAX_RATE = "نسبة الخصم"
COL_TAX_DATE = "تاريخ التعامل"
COL_TAX_REG = "رقم التسجيل"  # رقم التسجيل في كشف الخصم

NEW_COLS = [
    "المطلوب رقم الفاتورة من ملف المبيعات",
    "سنة الفاتورة من ملف المبيعات",
    "تاريخ الفاتورة من ملف المبيعات",
    "مبلغ الفواتير المستخدمة للتحقق",
    "ملاحظات عن المرتجع",
]

# ============================================================
# WORD_MAP & STOPWORDS
# ============================================================
WORD_MAP = {
    "المياه": "مياه", "المياة": "مياه", "مياة": "مياه", "الماء": "مياه",
    "الصرف": "صرف", "الصرف الصحي": "صرف صحي", "صرف الصحي": "صرف صحي",
    "الشرب": "شرب", "الشراب": "شرب",
    "بسوهج": "بسوهاج", "بسوهـاج": "بسوهاج", "سوهاج": "بسوهاج",
    "الزراعى": "زراعي", "الزراعي": "زراعي", "زراعية": "زراعي",
    "للاستثمار": "استثمار", "استثمارية": "استثمار",
    "للمقاولات": "مقاولات", "المقاولات": "مقاولات",
    "للصناعات": "صناعات", "الصناعات": "صناعات",
    "للتوريدات": "توريدات", "توريد": "توريدات",
    "الغذائية": "غذائية", "اغذية": "غذائية", "اغذيه": "غذائية",
}

STOPWORDS = {
    "شركة", "الشركة", "شركه", "الشركه",
    "وال", "بال", "لل", "ل", "و",
    "مصر", "القاهرة", "مصرية",
    "العالمية", "الدولية", "الجديدة",
    "مصنع", "صناعية", "تجارية",
    "جروب", "مجموعة",
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

def normalize_reg_number(reg):
    """تنظيف رقم التسجيل الضريبي"""
    if pd.isna(reg): return ""
    s = str(reg).strip()
    # إزالة المسافات والشرطات
    s = re.sub(r"[\s\-_]", "", s)
    return s

def fuzzy(a, b):
    return SequenceMatcher(None, a, b).ratio()

def to_num(v):
    if pd.isna(v): return np.nan
    s = str(v).strip()
    if not s or s.startswith("#"): return np.nan
    try:
        return float(s.replace(",", ""))
    except:
        return np.nan

def parse_dates(series, dayfirst):
    dt = pd.to_datetime(series, errors="coerce", dayfirst=dayfirst)
    return dt, dt.dt.year.fillna(0).astype(int), dt.dt.month.fillna(0).astype(int)

# ============================================================
# تجهيز الملفات - مع الإصلاح الحرج
# ============================================================
def prepare_sales(df_raw):
    df = df_raw.copy()
    df["amt"] = df[COL_AMOUNT].apply(to_num)
    
    # التعامل مع رقم التسجيل (اختياري)
    if COL_REG in df.columns:
        df["reg_clean"] = df[COL_REG].apply(normalize_reg_number)
    else:
        df["reg_clean"] = ""
    
    grouped = df.groupby(COL_INV).agg(
        net_amount=("amt", "sum"),
        pos_date=(
            COL_DATE,
            lambda x: x[df.loc[x.index, "amt"] > 0].iloc[0]
            if any(df.loc[x.index, "amt"] > 0) else np.nan
        ),
        has_return=("amt", lambda s: any(s < 0)),
        name=(COL_NAME, "first"),
        reg_clean=("reg_clean", "first"),  # إضافة رقم التسجيل
    ).reset_index()
    
    grouped = grouped[grouped["net_amount"] > 0]
    
    # 🔥 الإصلاح الحرج: التاريخ بصيغة dd/mm/yyyy
    grouped["date_parsed"], grouped["year"], grouped["month"] = parse_dates(
        grouped["pos_date"], dayfirst=True  # ✅ تم تصحيحه!
    )
    
    grouped["name_norm"] = grouped["name"].apply(normalize_name)
    grouped["tokens"] = grouped["name"].apply(tokenize)
    return grouped

def prepare_tax(df_raw):
    df = df_raw.copy()
    df["v_file"] = df[COL_TAX_AMOUNT].apply(to_num)
    df["v_tax_paid"] = df[COL_TAX_TAXED].apply(to_num)
    
    # التعامل مع رقم التسجيل (اختياري)
    if COL_TAX_REG in df.columns:
        df["reg_clean"] = df[COL_TAX_REG].apply(normalize_reg_number)
    else:
        df["reg_clean"] = ""
    
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
    
    # بحث موسع في 3 سنوات
    allowed_years = [tax_year, tax_year - 1, tax_year - 2]
    mask_year = sales_df["year"].isin(allowed_years)
    mask_date = (sales_df["date_parsed"] <= tax_date)
    
    return sales_df[mask_year & mask_date]

def extended_subset_search(cand, targets, max_invoices=25, max_nodes=200000):
    """
    يحاول إيجاد مجموعة من الفواتير (أي عدد) مجموعها قريب من أحد الـ targets
    مع حدود max_invoices (أقصى عدد فواتير نجربه) و max_nodes (أقصى عُقد بحث)
    """
    if not targets:
        return None

    max_t, min_t = max(targets), min(targets)

    # نقتصر على أول max_invoices فاتورة بعد الترتيب تنازلي بالقيمة
    cand = cand.sort_values("net_amount", ascending=False).head(max_invoices)

    rows = list(cand.itertuples(index=False))
    n = len(rows)
    if n == 0:
        return None

    amounts = [r.net_amount for r in rows]

    # suffix sums علشان نعرف أقصى ما يمكن إضافته في الفروع القادمة (للقص pruning)
    suffix = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + amounts[i]

    best = None
    best_diff = float("inf")
    nodes = 0

    def dfs(i, cur_sum, chosen):
        nonlocal best, best_diff, nodes
        nodes += 1

        # حد أقصى لعُقد البحث
        if nodes > max_nodes:
            return

        # لو تجاوزنا أعلى target بهامش 5% نوقف هذا الفرع
        if cur_sum > max_t * 1.05:
            return

        # لو حتى لو أخذنا كل الباقي مش هنوصل لـ 95% من أقل target → الفرع ده ملوش لازمة
        if cur_sum + suffix[i] < min_t * 0.95:
            return

        if i == n:
            diff = min(abs(cur_sum - t) for t in targets)
            if diff <= 0.05 * max_t and diff < best_diff:
                best_diff = diff
                best = chosen[:]
            return

        # 1) نجرب نأخذ الفاتورة الحالية
        chosen.append(i)
        dfs(i + 1, cur_sum + amounts[i], chosen)
        chosen.pop()

        # 2) نجرب نتجاهل الفاتورة الحالية
        dfs(i + 1, cur_sum, chosen)

    dfs(0, 0.0, [])
    return [rows[i] for i in best] if best else None

# ============================================================
# المطابقة الرئيسية
# ============================================================
def find_best_match(tax_row, sales_df, used_invoices):
    tax_date = tax_row["date_parsed"]
    if pd.isna(tax_date):
        return None

    v_file, v_tax, v_mix = tax_row["v_file"], tax_row["v_tax"], tax_row["v_mix"]
    targets = [t for t in (v_file, v_tax, v_mix) if pd.notna(t) and t > 0]
    if not targets:
        return None

    # فواتير نفس الفترة
    cand = filter_year_and_date(sales_df, tax_date, tax_row["year"], tax_row["month"])
    if cand.empty:
        return None

    # استبعاد الفواتير اللي اتستخدمت قبل كده
    cand = cand[~cand[COL_INV].astype(str).isin(used_invoices)].copy()
    if cand.empty:
        return None

    # تصفية برقم التسجيل
    tax_reg = str(tax_row.get("reg_clean", "")).strip()
    if tax_reg:
        cand_with_reg = cand[cand["reg_clean"] == tax_reg]
        if not cand_with_reg.empty:
            cand = cand_with_reg.copy()

    if cand.empty:
        return None

    # حساب تشابه الاسم
    cand["token_score"] = cand["tokens"].apply(lambda t: len(t & tax_row["tokens"]))
    cand["fuzzy"] = cand["name_norm"].apply(lambda s: fuzzy(s, tax_row["name_norm"]))
    cand = cand[(cand["token_score"] >= 1) | (cand["fuzzy"] >= 0.70)]
    if cand.empty:
        return None

    def within_absolute(val, max_diff=5.0):
        return any(abs(val - t) <= max_diff for t in targets)

    def within_pct(val, pct=0.05):
        return any(abs(val - t) <= pct * t for t in targets)

    cand["value_dist"] = cand["net_amount"].apply(
        lambda x: min(abs(x - t) for t in targets)
    )
    cand["reg_match"] = (cand["reg_clean"] == tax_reg) & (tax_reg != "")

    # ترتيب المرشحين
    cand = cand.sort_values(
        by=["reg_match", "value_dist", "token_score", "fuzzy"],
        ascending=[False, True, False, False]
    )

    # 🆕 (0) لو فيه رقم تسجيل: جرّب جمع كل الفواتير لنفس الرقم مرة واحدة
    if tax_reg and not cand.empty:
        total_reg = cand["net_amount"].sum()
        if within_absolute(total_reg, 5.0) or within_pct(total_reg):
            invs = cand[COL_INV].astype(str).tolist()
            years = cand["year"].astype(str).tolist()
            dates = cand["pos_date"].astype(str).tolist()
            has_ret = cand["has_return"].any()
            return invs, years, dates, float(total_reg), has_ret

    # 1️⃣ فاتورة واحدة
    for _, r in cand.head(100).iterrows():
        if within_absolute(r["net_amount"], max_diff=5.0) or within_pct(r["net_amount"]):
            return (
                [str(r[COL_INV])],
                [str(r["year"])],
                [str(r["pos_date"])],
                float(r["net_amount"]),
                r["has_return"],
            )

    # 2️⃣ مجموع 2 فواتير
    for combo in combinations(cand.head(60).itertuples(index=False), 2):
        total = sum(r.net_amount for r in combo)
        if not (within_absolute(total, 5.0) or within_pct(total)):
            continue
        invs = [str(r._asdict()[COL_INV]) for r in combo]
        if len(set(invs)) != len(invs):
            continue
        years = [str(r.year) for r in combo]
        dates = [str(r.pos_date) for r in combo]
        ret = any(r.has_return for r in combo)
        return invs, years, dates, float(total), ret

    # 3️⃣ مجموع 3 فواتير
    for combo in combinations(cand.head(60).itertuples(index=False), 3):
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

    # 4️⃣ بحث عام لأي عدد فواتير (4 فأكثر) مع حدود منطقية
    max_invoices = 25  # تقدر تزودها لـ 30 لو بياناتك مش ضخمة
    ext = extended_subset_search(
        cand,
        targets,
        max_invoices=max_invoices,
        max_nodes=200000
    )
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
# المطابقة النهائية مع التعديلات
# ============================================================
def match_with_user_feedback(sales_df_original, tax_df_original, matches_edited, stopwords_edited):
    global STOPWORDS
    
    # تحديث STOPWORDS
    if "كلمة" in stopwords_edited.columns:
        words = [str(v).strip() for v in stopwords_edited["كلمة"].tolist() if str(v).strip()]
        STOPWORDS = set(words)
    
    # إعادة تجهيز tokens
    sales_df = sales_df_original.copy()
    tax_df = tax_df_original.copy()
    
    sales_df["name_norm"] = sales_df["name"].apply(normalize_name)
    sales_df["tokens"] = sales_df["name"].apply(tokenize)
    
    tax_df["name_norm"] = tax_df[COL_TAX_NAME].apply(normalize_name)
    tax_df["tokens"] = tax_df[COL_TAX_NAME].apply(tokenize)
    
    result = tax_df.copy()
    for col in NEW_COLS:
        result[col] = ""
    
    used = set()
    
    # تثبيت التطابقات المعتمدة
    if matches_edited is not None and not matches_edited.empty and "row_id" in matches_edited.columns:
        for _, r in matches_edited.iterrows():
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
            
            used.update(invs)
    
    # إكمال الباقي
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
    
    return result, matched, len(result) - matched

# ============================================================
# واجهة Streamlit
# ============================================================
st.set_page_config(page_title="مطابقة خصم المنبع", layout="wide")

st.title("🎯 مطابقة خصم المنبع - نظام الخطوتين المحسّن")
st.markdown("---")

with st.expander("📖 كيفية الاستخدام", expanded=True):
    st.markdown("""
    ### الخطوة 1️⃣: المطابقة المبدئية
    - حمّل ملف المبيعات وكشف الخصم
    - اضغط "مطابقة مبدئية"
    - ستظهر لك جداول للمراجعة:
      * جدول التطابقات (يمكنك حذف الخاطئة أو إلغاء اعتمادها)
      * جدول STOPWORDS (الكلمات المستبعدة)
    
    ### الخطوة 2️⃣: المطابقة النهائية
    - راجع الجداول وعدّل فيها
    - اضغط "متابعة المطابقة النهائية"
    - البرنامج يثبت التطابقات المعتمدة ويكمل الباقي
    
    ### 🆕 ميزة جديدة: رقم التسجيل الضريبي
    - إذا كان ملف المبيعات وكشف الخصم يحتويان على عمود **"رقم التسجيل"**
    - البرنامج سيعطي **أولوية قصوى** للفواتير بنفس رقم التسجيل
    - كما سيجرب أولاً تجميع كل فواتير نفس الرقم في الفترة للوصول لقيمة كشف الخصم
    """)

col1, col2 = st.columns(2)
with col1:
    sales_file = st.file_uploader("📊 ملف المبيعات (CSV)", type="csv")
with col2:
    tax_file = st.file_uploader("📑 كشف خصم المنبع (CSV)", type="csv")

st.markdown("---")

# الخطوة 1
if st.button("🚀 الخطوة 1: مطابقة مبدئية", use_container_width=True, type="primary"):
    if not sales_file or not tax_file:
        st.error("⚠️ ارفع الملفين أولاً!")
        st.stop()
    
    try:
        with st.spinner("⏳ جاري المعالجة..."):
            sales_raw = pd.read_csv(sales_file, encoding="utf-8-sig", dtype=str)
            tax_raw = pd.read_csv(tax_file, encoding="utf-8-sig", dtype=str)
            
            sales_prepared = prepare_sales(sales_raw)
            tax_prepared = prepare_tax(tax_raw)
            
            draft_df, ok, bad = match_all_basic(sales_prepared, tax_prepared)
        
        st.session_state["sales_prepared"] = sales_prepared
        st.session_state["tax_prepared"] = tax_prepared
        
        draft_df = draft_df.copy()
        draft_df.insert(0, "row_id", draft_df.index.astype(int))
        st.session_state["draft_df"] = draft_df
        
        matches_only = draft_df[draft_df[NEW_COLS[0]] != ""].copy()
        matches_only["اعتماد_التطابق"] = True
        st.session_state["matches_table"] = matches_only
        
        stopwords_df = pd.DataFrame({"كلمة": sorted(STOPWORDS)})
        st.session_state["stopwords_table"] = stopwords_df
        
        success_rate = (ok/(ok+bad)*100) if (ok+bad) > 0 else 0
        st.success(f"✅ المطابقة المبدئية: {ok:,} مطابق ({success_rate:.1f}%) | {bad:,} غير مطابق")
        st.info("⬇ انزل للأسفل لمراجعة الجداول")
        
    except Exception as e:
        st.error(f"❌ خطأ: {str(e)}")
        st.exception(e)

st.markdown("---")

# عرض الجداول
if "draft_df" in st.session_state:
    st.subheader("🧾 جدول التطابقات المبدئية")
    matches_df = st.session_state.get("matches_table", pd.DataFrame())
    
    if not matches_df.empty:
        edited_matches = st.data_editor(
            matches_df,
            key="matches_editor",
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "اعتماد_التطابق": st.column_config.CheckboxColumn(
                    "اعتماد التطابق",
                    help="ألغِ الاختيار لرفض هذا التطابق",
                    default=True,
                )
            }
        )
    
    st.subheader("🧹 كلمات STOPWORDS المستبعدة")
    stopwords_df = st.session_state.get("stopwords_table", pd.DataFrame())
    edited_stopwords = st.data_editor(
        stopwords_df,
        key="stopwords_editor",
        num_rows="dynamic",
        use_container_width=True
    )
    
    st.markdown("---")
    
    # الخطوة 2
    if st.button("✅ الخطوة 2: المطابقة النهائية", use_container_width=True, type="primary"):
        try:
            sales_prepared = st.session_state["sales_prepared"]
            tax_prepared = st.session_state["tax_prepared"]
            
            with st.spinner("🔁 جاري المطابقة النهائية..."):
                final_df, ok2, bad2 = match_with_user_feedback(
                    sales_prepared,
                    tax_prepared,
                    edited_matches if 'edited_matches' in locals() else matches_df,
                    edited_stopwords if 'edited_stopwords' in locals() else stopwords_df,
                )
            
            success_rate = (ok2/(ok2+bad2)*100) if (ok2+bad2) > 0 else 0
            
            st.success("🎉 تمت المطابقة النهائية!")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("✅ المطابق", f"{ok2:,}", delta=f"{success_rate:.1f}%")
            with c2:
                st.metric("❌ غير المطابق", f"{bad2:,}")
            with c3:
                st.metric("📈 النجاح", f"{success_rate:.2f}%")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            with col1:
                out = io.BytesIO()
                final_df.to_csv(out, index=False, encoding="utf-8-sig")
                st.download_button(
                    "📥 تحميل الكشف الكامل",
                    data=out.getvalue(),
                    file_name="كشف_مطابق_نهائي.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                unmatched = final_df[final_df[NEW_COLS[0]] == ""]
                if not unmatched.empty:
                    out2 = io.BytesIO()
                    unmatched.to_csv(out2, index=False, encoding="utf-8-sig")
                    st.download_button(
                        "📥 تحميل غير المطابق",
                        data=out2.getvalue(),
                        file_name="غير_مطابق.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            st.markdown("### 👀 معاينة النتائج")
            st.dataframe(final_df.head(20), use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ خطأ: {str(e)}")
            st.exception(e)

st.markdown("---")
st.caption("💼 محاسب قانوني: مايكل نبيل | 🚀 2025")
