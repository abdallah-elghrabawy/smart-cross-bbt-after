
import io
import math
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Smart BBT Analysis (Fixed)", layout="wide")

st.title("🧠 Smart Cross-BBT — Fixed Package")
st.caption("Equal-coverage transfers with strict donor floor at Coverage target and leveling up to Phase step only. No apply(axis=1) used.")

with st.sidebar:
    st.header("⚙️ Settings")
    sales_days = st.number_input("Sales period (days) to compute AvgDaily", min_value=1, max_value=365, value=90, step=1)
    coverage_days = st.number_input("Coverage target (days) — donor never goes below this", min_value=1, max_value=365, value=60, step=1)
    phase_step = st.number_input("Leveling phase step (days) — raise receivers up to this cap", min_value=1, max_value=365, value=25, step=1)
    keep_boxes_zero_sales = st.number_input("Keep boxes if zero sales (donor floor when AvgDaily=0)", min_value=0, max_value=100, value=1, step=1)
    fair_rotation = st.checkbox("Fair rotation (optional: distribute from largest donors first)", value=True)

    st.caption('CSV expected columns (case-insensitive): **branch | item code | item name | quantity | sales**')

    if phase_step > coverage_days:
        st.error("Phase step لا بد أن يكون ≤ Coverage target — وإلا لن تُنفَّذ أي مرحلة.")
        st.stop()

uploaded = st.file_uploader("Upload CSV", type=["csv"])

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cmap = {}
    for c in df.columns:
        cl = c.strip().lower().replace(" ", "").replace("_", "")
        if cl in ["branch", "store", "outlet"]:
            cmap[c] = "Branch"
        elif cl in ["itemcode", "code", "sku"]:
            cmap[c] = "Item Code"
        elif cl in ["itemname", "name", "description"]:
            cmap[c] = "Item Name"
        elif cl in ["quantity", "stock", "stockunits", "qty"]:
            cmap[c] = "Stock_boxes"
        elif cl in ["sales", "sold", "unitsold", "qtysold"]:
            cmap[c] = "Sales_boxes"
    df = df.rename(columns=cmap)
    need = ["Branch","Item Code","Item Name","Stock_boxes","Sales_boxes"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")
    return df[need].copy()


def _read_csv_robust(file):
    """
    Robust CSV reader:
    - tries utf-8, utf-8-sig, cp1256 (Arabic Windows), cp1252, latin1
    - falls back to binary -> text decode with 'replace'
    - auto detects sep when possible (engine='python', sep=None)
    """
    import io, pandas as pd
    tried = []
    for enc in ["utf-8", "utf-8-sig", "cp1256", "cp1252", "latin1"]:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc, engine="python", sep=None, on_bad_lines="skip")
        except Exception as e:
            tried.append(f"{enc}: {e}")
    # Fallback: read bytes, decode with 'utf-8' replace, then parse
    try:
        file.seek(0)
        raw = file.read()
        if isinstance(raw, bytes):
            text = raw.decode("utf-8", errors="replace")
        else:
            text = raw
        return pd.read_csv(io.StringIO(text), engine="python", sep=None, on_bad_lines="skip")
    except Exception as e:
        raise RuntimeError("Failed to parse CSV with multiple encodings. Tried:\n" + "\n".join(tried) + f"\nFinal error: {e}")

def allocate_equal_coverage_collect_all(df: pd.DataFrame) -> pd.DataFrame:
    # df columns: Branch, Item Code, Item Name, Stock_boxes, Sales_boxes
    # Compute AvgDaily per branch-item
    df = df.copy()
    df["Sales_boxes"] = pd.to_numeric(df["Sales_boxes"], errors="coerce").fillna(0.0)
    df["Stock_boxes"] = pd.to_numeric(df["Stock_boxes"], errors="coerce").fillna(0.0)

    # group by (Branch, Item)
    keys = ["Item Code","Item Name","Branch"]
    g = df.groupby(keys, as_index=False).agg(Stock_boxes=("Stock_boxes","sum"),
                                            Sales_boxes=("Sales_boxes","sum"))
    g["AvgDaily"] = (g["Sales_boxes"] / float(sales_days)).astype(float).clip(lower=0.0)
    # MonthTarget per branch capped by coverage_days target
    g["MonthTarget"] = np.minimum(g["AvgDaily"] * float(coverage_days), np.inf)

    transfers = []  # list of dicts: Item Code, Item Name, From_Branch, To_Branch, Qty_Boxes

    for (icode, iname), item_df in g.groupby(["Item Code","Item Name"], as_index=False):
        item_df = item_df.copy()
        # Live stock per branch
        item_df["LiveStock"] = item_df["Stock_boxes"].astype(float).clip(lower=0.0)

        # Receivers: those with need up to phase_step (and AvgDaily>0)
        # Target for receivers this run: ceil(min(phase_step*AvgDaily, MonthTarget))
        target_recv = np.ceil(np.minimum(float(phase_step)*item_df["AvgDaily"].to_numpy(),
                                         item_df["MonthTarget"].to_numpy()))
        need_raw = target_recv - item_df["LiveStock"].to_numpy()
        need_d = np.maximum(need_raw, 0.0)

        receivers_mask = (item_df["AvgDaily"].to_numpy() > 0.0) & (need_d > 0)
        receivers = item_df.loc[receivers_mask, ["Branch","AvgDaily","MonthTarget","LiveStock"]].copy()
        receivers["Need_d"] = need_d[receivers_mask].astype(float)

        # Donors: floor always at coverage_days target (never below)
        floor_donor = np.where(
            item_df["AvgDaily"].to_numpy() <= 0.0,
            float(keep_boxes_zero_sales),
            np.ceil(np.minimum(float(coverage_days)*item_df["AvgDaily"].to_numpy(),
                               item_df["MonthTarget"].to_numpy()))
        )
        free_raw = item_df["LiveStock"].to_numpy() - floor_donor
        free_d = np.maximum(free_raw, 0.0)

        donors_mask = free_d > 0
        donors = item_df.loc[donors_mask, ["Branch","AvgDaily","MonthTarget","LiveStock"]].copy()
        donors["FreeThisPhase"] = free_d[donors_mask].astype(float)

        if receivers.empty or donors.empty:
            continue

        # Sorting rules
        # Receivers: lowest coverage first (LiveStock/AvgDaily)
        def cov_row(stock, avg):
            if avg <= 0: 
                return np.inf
            return stock/avg
        receivers["Coverage"] = receivers.apply(lambda r: cov_row(r["LiveStock"], r["AvgDaily"]), axis=1)
        receivers = receivers.sort_values(by=["Coverage","AvgDaily"], ascending=[True, False]).reset_index(drop=True)

        # Donors: largest free first
        donors = donors.sort_values(by=["FreeThisPhase","AvgDaily"], ascending=[False, False]).reset_index(drop=True)

        # Greedy allocation
        i = 0
        while i < len(receivers) and donors["FreeThisPhase"].sum() > 0:
            need_left = float(receivers.at[i, "Need_d"])
            if need_left <= 0:
                i += 1
                continue
            # consume from donors in order
            for j in donors.index:
                dav = float(donors.at[j, "FreeThisPhase"])
                if dav <= 0:
                    continue
                give = min(need_left, dav)
                if give <= 0:
                    continue
                # record transfer
                transfers.append({
                    "Item Code": icode,
                    "Item Name": iname,
                    "From_Branch": donors.at[j, "Branch"],
                    "To_Branch": receivers.at[i, "Branch"],
                    "Qty_Boxes": float(give)
                })
                # update states
                need_left -= give
                donors.at[j, "FreeThisPhase"] = dav - give
                donors.at[j, "LiveStock"] = float(donors.at[j, "LiveStock"]) - give
                receivers.at[i, "LiveStock"] = float(receivers.at[i, "LiveStock"]) + give
                receivers.at[i, "Need_d"] = need_left
                if need_left <= 0:
                    break
            if need_left > 0 and donors["FreeThisPhase"].sum() <= 0:
                break
            if need_left <= 0:
                i += 1

    if not len(transfers):
        return pd.DataFrame(columns=["Item Code","Item Name","From_Branch","To_Branch","Qty_Boxes"])

    raw = pd.DataFrame(transfers)
    # Final merge
    final_plan = (raw.groupby(["Item Code","Item Name","From_Branch","To_Branch"], as_index=False, dropna=False)
                      .agg(Qty_Boxes=("Qty_Boxes","sum")))
    # round to int where appropriate
    final_plan["Qty_Boxes"] = final_plan["Qty_Boxes"].round(0).astype(int)
    final_plan = final_plan[final_plan["Qty_Boxes"] > 0].copy()
    return final_plan

if uploaded is not None:
    df = _read_csv_robust(uploaded)
    try:
        df = _normalize_columns(df)
    except Exception as e:
        st.error(f"Column mapping error: {e}")
        st.stop()

    st.subheader("📦 Input (normalized preview)")
    st.dataframe(df.head(30))

    with st.spinner("Running equal-coverage allocation..."):
        final_plan = allocate_equal_coverage_collect_all(df)

    st.subheader("🚚 Final Transfers (merged)")
    if len(final_plan):
        st.dataframe(final_plan, use_container_width=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_bytes = final_plan.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ تحميل FINAL_TRANSFERS CSV", data=csv_bytes, file_name=f"FINAL_TRANSFERS_{ts}.csv", mime="text/csv")
    else:
        st.info("لا توجد تحويلات مطلوبة وفقًا للمدخلات والإعدادات الحالية.")

else:
    st.info("ارفع ملف CSV لبدء التحليل.")
