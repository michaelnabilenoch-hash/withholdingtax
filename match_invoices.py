
import streamlit as st
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
import re

st.set_page_config(page_title="مطابقة خصم المنبع - سريعة 2025", layout="wide")
st.title("مطابقة خصم المنبع - النسخة السريعة اللي مش بتهنج أبدًا")

# إعدادات الأعمدة
COL_INV = "فواتير"
COL_DATE = "التاريخ"
COL_NAME = "اسم الشركة"
COL_AMOUNT = "صافى المبيعات"

COL_TAX_NAME = "اسم الجهة"
COL_TAX_AMOUNT = "القيمة الصافية للتعامل"
COL_TAX_TAXED = "محصل لحساب الضريبه"
COL_TAX_RATE = "نسبة الخصم"
COL_TAX_DATE = "تاريخ التعامل"

# تنظيف الأسماء بسرعة
def normalize(s):
    if pd.isna(s): return ""
    s = str(s)
    s = re.sub(r"[أإآا]", "ا", s)
    s = re.sub(r"[ة]", "ه", s)
    s = re.sub(r"[ىي]", "ي", s)
    s = s.lower()
    s = re.sub(r"[^ا-ي\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def fuzzy(a, b):
    return SequenceMatcher(None, a, b).ratio()

def to_num(x):
    try: return float(str(x).replace(",", ""))
    except: return np.nan

# تحميل الملفات
sales_file = st.file_uploader("ملف المبيعات CSV", type="csv")
tax_file = st.file_uploader("كشف خصم المنبع CSV", type="csv")

if st.button("ابدأ المطابقة السريعة", type="primary"):
    if not sales_file or not tax_file:
        st.error("ارفع الملفين الأول")
    else:
        with st.spinner("جاري المطابقة بسرعة البرق..."):
            # قراءة الملفات
            sales = pd.read_csv(sales_file, encoding="utf-8-sig", dtype=str)
            tax = pd.read_csv(tax_file, encoding="utf-8-sig", dtype=str)

            # تجهيز المبيعات
            sales["amt"] = sales[COL_AMOUNT].apply(to_num)
            sales = sales[sales["amt"] > 0]
            sales["norm_name"] = sales[COL_NAME].apply(normalize)

            # تجهيز كشف الخصم
            tax["v_file"] = tax[COL_TAX_AMOUNT].apply(to_num)
            tax["tax_paid"] = tax[COL_TAX_TAXED].apply(to_num)
            tax["rate"] = tax[COL_TAX_RATE].astype(str).str.replace("%","").astype(float)/100
            tax["v_tax"] = tax["tax_paid"] / tax["rate"].replace(0, np.nan)
            tax["target"] = tax[["v_file", "v_tax"]].mean(axis=1, skipna=True)
            tax["norm_name"] = tax[COL_TAX_NAME].apply(normalize)

            result = tax.copy()
            result["رقم الفاتورة"] = ""
            result["مبلغ المطابقة"] = ""
            result["ملاحظات"] = ""

            used_invoices = set()

            for idx, row in tax.iterrows():
                target = row["target"]
                if pd.isna(target): continue

                # مرشحين بالاسم
                mask_name = sales["norm_name"].apply(lambda x: fuzzy(x, row["norm_name"]) > 0.78)
                cand = sales[mask_name].copy()

                if cand.empty: continue

                cand = cand[~cand[COL_INV].astype(str).isin(used_invoices)]
                cand["diff"] = (cand["amt"] - target).abs()
                cand = cand.sort_values("diff")

                # فاتورة واحدة أو مجموع بسيط (2-3)
                best = None
                for n in [1, 2, 3]:
                    for combo in cand.head(50).groupby(level=0).head(n).groupby(cand.index // 1000):
                        total = combo["amt"].sum()
                        if abs(total - target) <= target * 0.05:
                            best = combo
                            break
                    if best is not None: break

                if best is None:
                    best = cand.head(1)

                invs = " + ".join(best[COL_INV].astype(str).tolist())
                total_amt = best["amt"].sum()
                result.at[idx, "رقم الفاتورة"] = invs
                result.at[idx, "مبلغ المطابقة"] = total_amt
                result.at[idx, "ملاحظات"] = "مرتجع" if (best["amt"] < 0).any() else ""
                used_invoices.update(best[COL_INV].astype(str).tolist())

            st.success("تمت المطابقة بنجاح!")
            csv = result.to_csv(index=False, encoding="utf-8-sig")
            st.download_button("تحميل النتيجة", csv, "مطابقة_سريعة.csv", "text/csv")
