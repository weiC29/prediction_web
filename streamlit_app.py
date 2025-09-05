import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, timedelta
import random
import io

TAB_NAME = "Sheet1" 
CLAIM_TTL_MIN = 30
ADMIN_COLS = [
    "submission_status","claimed_by","claimed_at",
    "reviewer_name","reviewer_email",
    "pred_surgery_success","pred_confidence","pred_snot22_6mo",
    "submitted_at"
]

#Google Sheets helpers
@st.cache_resource
def get_ws():
    scope = ["https://spreadsheets.google.com/feeds",
             "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(
        st.secrets["gcp_service_account"], scope
    )
    gc = gspread.authorize(creds)
    ws = gc.open_by_key(st.secrets["SHEET_ID"]).worksheet(TAB_NAME)
    return ws

def read_all(ws):
    rows = ws.get_all_values()
    if not rows:
        return pd.DataFrame(), []
    header = rows[0]
    data = rows[1:]
    df = pd.DataFrame(data, columns=header)
    return df, header

def ensure_admin_columns(ws, header):
    """Append missing admin columns at the far right, fill blanks for existing rows."""
    missing = [c for c in ADMIN_COLS if c not in header]
    if not missing:
        return header

    start_col = len(header) + 1
    ws.update(range_name=gspread.utils.rowcol_to_a1(1, start_col),
              values=[missing])  # single header row to the right

    n_rows = ws.row_count
    values = ws.get_all_values()
    current_rows = len(values)
    if current_rows > 1:
        blanks = [[""] * len(missing) for _ in range(current_rows - 1)]
        ws.update(
            range_name=f"{gspread.utils.rowcol_to_a1(2, start_col)}:"
                       f"{gspread.utils.rowcol_to_a1(current_rows, start_col + len(missing) - 1)}",
            values=blanks
        )

    return header + missing

def write_row_cells(ws, row_num, header, updates: dict):
    """Update specific columns on a given 1-based sheet row (row 1 is header)."""
    for k, v in updates.items():
        col = header.index(k) + 1
        ws.update_cell(row_num, col, v)


def get_pending_mask(df):
    col = "submission_status"
    if col not in df.columns:
        return pd.Series([True] * len(df), index=df.index)
    s = df[col].astype(str).str.strip().str.lower()
    return (s == "") | (s == "pending")

def release_stale_claims(ws, header, df, ttl_min=CLAIM_TTL_MIN):
    """Convert stale 'Claimed' rows back to Pending if CLAIM_TTL expired."""
    if "submission_status" not in df.columns or "claimed_at" not in df.columns:
        return
    now = datetime.utcnow()
    for idx, row in df.iterrows():
        if str(row.get("submission_status","")).lower() == "claimed":
            try:
                t = datetime.fromisoformat(str(row.get("claimed_at","")))
                if now - t > timedelta(minutes=ttl_min):
                    sheet_row = idx + 2 
                    write_row_cells(ws, sheet_row, header, {
                        "submission_status": "Pending",
                        "claimed_by": "",
                        "claimed_at": ""
                    })
            except Exception:
                pass

def pick_and_claim_random(ws, header, df, reviewer_name):
    """Pick a random pending row, mark as Claimed."""
    release_stale_claims(ws, header, df)
    df_latest, header = read_all(ws)
    if df_latest.empty:
        return None, None, None

    pending_mask = get_pending_mask(df_latest)
    pending_idx = df_latest[pending_mask].index.tolist()
    if not pending_idx:
        return None, df_latest, header

    choice_idx = random.choice(pending_idx)
    sheet_row = choice_idx + 2
    df_check, _ = read_all(ws)
    status_now = str(df_check.iloc[choice_idx].get("submission_status","")).strip().lower()
    if status_now not in ["", "pending"]:
        return pick_and_claim_random(ws, header, df_check, reviewer_name)

    write_row_cells(ws, sheet_row, header, {
        "submission_status": "Claimed",
        "claimed_by": reviewer_name,
        "claimed_at": datetime.utcnow().isoformat()
    })
    return choice_idx, df_latest, header

def combined_csv_bytes(ws):
    df, _ = read_all(ws)
    return df.to_csv(index=False).encode("utf-8")


st.set_page_config(page_title="Expert Surgical Outcome Survey", page_icon="🩺", layout="centered")

ws = get_ws()
df, header = read_all(ws)
header = ensure_admin_columns(ws, header)
df, header = read_all(ws)

pending_count = get_pending_mask(df).sum()
submitted_count = (df.get("submission_status","").astype(str).str.lower() == "submitted").sum()

st.title("Expert Surgical Outcome Survey")
st.caption("Predict surgery outcome (0/1), confidence, and 6-month SNOT-22. Each patient can be submitted once.")
c1, c2 = st.columns(2)
c1.metric("Patients available", int(pending_count))
c2.metric("Submitted", int(submitted_count))
st.divider()

page = st.sidebar.radio("Navigate", ["I want to predict", "View dataset", "Download dataset"], index=0)

if page == "View dataset":
    st.subheader("Live dataset (same tab)")
    st.dataframe(df, use_container_width=True, height=520)

elif page == "Download dataset":
    st.subheader("Download CSV (same tab)")
    st.download_button("⬇️ Download", combined_csv_bytes(ws),
                       file_name="expert_predictions_live.csv", mime="text/csv")

else:
    st.subheader("Start a prediction")
    with st.form("gate"):
        name = st.text_input("Your name*", "")
        email = st.text_input("Your email*", "")
        ok = st.form_submit_button("Find me a patient")
    if ok:
        if not name or not email:
            st.error("Please enter name and email.")
            st.stop()

        claimed_idx, df0, header = pick_and_claim_random(ws, header, df, name)
        if claimed_idx is None:
            st.success("All patients have been completed. 🙌")
            st.stop()

        df_now, _ = read_all(ws)
        row = df_now.iloc[claimed_idx]
        sheet_row = claimed_idx + 2

        st.subheader(f"Patient (row {sheet_row})")
        admin_set = set(ADMIN_COLS)
        for col in df_now.columns:
            if col in admin_set:
                continue
            st.write(f"**{col}**: {row[col]}")

        st.markdown("---")
        st.subheader("Your prediction")
        with st.form("pred"):
            outcome = st.selectbox("Surgical outcome", options=[0,1],
                                   format_func=lambda x: f"{x} – {'Successful' if x==1 else 'Unsuccessful'}")
            confidence = st.radio("How confident are you?", [
                "Very confident","Somewhat confident","Neutral","Somewhat unsure","Not at all confident"
            ], index=1)
            snot22 = st.slider("Estimated postoperative SNOT-22 at 6 months", 0, 110, 24, 1)
            submit = st.form_submit_button("Submit")

        if submit:
            df_check, _ = read_all(ws)
            status_now = str(df_check.iloc[claimed_idx].get("submission_status","")).strip().lower()
            claimed_by = str(df_check.iloc[claimed_idx].get("claimed_by","")).strip()
            if status_now != "claimed" or claimed_by != name:
                st.error("This patient is no longer editable. Please start again.")
                st.stop()

            updates = {
                "submission_status": "Submitted",
                "reviewer_name": name,
                "reviewer_email": email,
                "pred_surgery_success": outcome,
                "pred_confidence": confidence,
                "pred_snot22_6mo": snot22,
                "submitted_at": datetime.utcnow().isoformat()
            }
            write_row_cells(ws, sheet_row, header, updates)
            st.success("Thank you! Your submission has been saved.")
            st.balloons()
