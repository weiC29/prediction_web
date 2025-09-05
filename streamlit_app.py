import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta
import random
import io


TAB_NAME = "Sheet1"         
CLAIM_TTL_MIN = 30
ADMIN_COLS = [
    "submission_status","claimed_by","claimed_at",
    "reviewer_name","reviewer_email","submitted_at"
]
PRED_COLS = {
    "expert_prediction": "outcome",               
    "expert_confidence": "confidence_text",       
    "expert_SNOT22score_prediction": "snot22_6mo" 
}
FRIENDLY = {
    "TREATMENT": "Treatment",
    "Age": "Age",
    "SEX": "Sex",
    "RACE": "Race",
    "ETHNICITY": "Ethnicity (NIH)",
    "EDUCATION": "Years of education",
    "HOUSEHOLD_INCOME": "Annual household income",
    "PREVIOUS_SURGERY": "Prior sinus surgery (#)",
    "INSURANCE": "Insurance Type",
    "AFS": "AFRS",
    "SEPT_DEV": "Septal Deviation",
    "CRS_POLYPS": "Polyps",
    "RAS": "Recurrent Acute Sinusitis",
    "HYPER_TURB": "Inferior Turb Hypertrophy",
    "MUCOCELE": "Mucocele",
    "ASTHMA": "Asthma",
    "ASA_INTOLERANCE": "AERD",
    "ALLERGY_TESTING": "Positive allergy skin testing",
    "COPD": "COPD",
    "DEPRESSION": "Depression",
    "FIBROMYALGIA": "Fibromyalgia",
    "OSA_HISTORY": "OSA History",
    "SMOKER": "Smoker (ppd)",
    "ALCOHOL": "Alcohol Use (drinks/wk)",
    "STEROID": "Steroid dependence",
    "DIABETES": "Diabetes",
    "GERD": "GERD",
    "BLN_CT_TOTAL": "CT score (LM 0–24)",
    "BLN_ENDOSCOPY_TOTAL": "Endoscopy Score (LK 0–20)",
    "SNOT22_BLN_TOTAL": "SNOT-22 total (0–110)",
    "expert_prediction": "Expert surgery outcome (0/1)",
    "expert_confidence": "Expert confidence",
    "expert_SNOT22score_prediction": "Expert postop SNOT-22 (6mo)"
}

DISPLAY_ORDER = [
    "Age","SEX","RACE","ETHNICITY","EDUCATION","HOUSEHOLD_INCOME",
    "PREVIOUS_SURGERY","INSURANCE","AFS","SEPT_DEV","CRS_POLYPS","RAS",
    "HYPER_TURB","MUCOCELE","ASTHMA","ASA_INTOLERANCE","ALLERGY_TESTING",
    "COPD","DEPRESSION","FIBROMYALGIA","OSA_HISTORY","SMOKER","ALCOHOL",
    "STEROID","DIABETES","GERD","BLN_CT_TOTAL","BLN_ENDOSCOPY_TOTAL","SNOT22_BLN_TOTAL"
]


@st.cache_resource
def get_ws():
    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=scope
    )
    gc = gspread.authorize(creds)
    return gc.open_by_key(st.secrets["SHEET_ID"]).worksheet(TAB_NAME)

def read_all(ws):
    rows = ws.get_all_values()
    if not rows:
        return pd.DataFrame(), []
    header = rows[0]
    data = rows[1:]
    df = pd.DataFrame(data, columns=header)
    return df, header

def append_missing_admin(ws, header):
    missing = [c for c in ADMIN_COLS if c not in header]
    if missing:
        start_col = len(header) + 1
        ws.update(gspread.utils.rowcol_to_a1(1, start_col), [missing])
        values = ws.get_all_values()
        n_rows = len(values)
        if n_rows > 1:
            blanks = [[""] * len(missing) for _ in range(n_rows - 1)]
            ws.update(
                f"{gspread.utils.rowcol_to_a1(2, start_col)}:{gspread.utils.rowcol_to_a1(n_rows, start_col+len(missing)-1)}",
                blanks
            )
        header += missing
    return header

def write_row(ws, row_num, header, updates: dict):
    for k, v in updates.items():
        if k not in header:
            continue
        ws.update_cell(row_num, header.index(k)+1, v)


def pending_mask(df, header):
    s_status = df["submission_status"].astype(str).str.strip().str.lower() if "submission_status" in df.columns else pd.Series([""]*len(df))
    s_pred = df["expert_prediction"].astype(str).str.strip() if "expert_prediction" in df.columns else pd.Series([""]*len(df))
    return ((s_status.eq("")) | (s_status.eq("pending"))) & (s_pred.eq(""))

def release_stale_claims(ws, header, df):
    if "submission_status" not in df.columns or "claimed_at" not in df.columns:
        return
    now = datetime.utcnow()
    for idx, row in df.iterrows():
        if str(row.get("submission_status","")).lower() == "claimed":
            try:
                t = datetime.fromisoformat(str(row.get("claimed_at","")))
                if now - t > timedelta(minutes=CLAIM_TTL_MIN):
                    write_row(ws, idx+2, header, {
                        "submission_status": "Pending",
                        "claimed_by": "",
                        "claimed_at": ""
                    })
            except Exception:
                pass

def pick_and_claim(ws, header, df, reviewer_name):
    release_stale_claims(ws, header, df)
    df, header = read_all(ws)
    mask = pending_mask(df, header)
    candidates = df[mask].index.tolist()
    if not candidates:
        return None, df, header
    choice = random.choice(candidates)
    # double-check still pending
    fresh_df, _ = read_all(ws)
    ss = str(fresh_df.iloc[choice].get("submission_status","")).strip().lower()
    pred = str(fresh_df.iloc[choice].get("expert_prediction","")).strip()
    if (ss not in ["","pending"]) or (pred != ""):
        return pick_and_claim(ws, header, df, reviewer_name)
    write_row(ws, choice+2, header, {
        "submission_status": "Claimed",
        "claimed_by": reviewer_name,
        "claimed_at": datetime.utcnow().isoformat()
    })
    return choice, df, header

def combined_csv_bytes(ws):
    df, _ = read_all(ws)
    return df.to_csv(index=False).encode("utf-8")

def iter_patient_fields(df_row):
    for c in DISPLAY_ORDER:
        if c in df_row.index:
            yield c
    admin_and_pred = set(ADMIN_COLS) | set(PRED_COLS.keys())
    for c in df_row.index:
        if (c not in DISPLAY_ORDER) and (c not in admin_and_pred):
            yield c


st.set_page_config(page_title="Expert Surgical Outcome Survey", page_icon="🩺", layout="centered")
ws = get_ws()
df, header = read_all(ws)
header = append_missing_admin(ws, header)
df, header = read_all(ws) 


pend = pending_mask(df, header).sum()
done = (df.get("submission_status","").astype(str).str.lower() == "submitted").sum() \
       + (df.get("expert_prediction","").astype(str).str.strip() != "").sum()

st.title("Expert Surgical Outcome Survey")
st.caption("Predict surgery outcome (0/1), confidence, and 6-month SNOT-22. Each patient can be submitted once.")
c1, c2 = st.columns(2)
c1.metric("Patients available", int(pend))
c2.metric("Submitted", int(done))
st.divider()

page = st.sidebar.radio("Navigate", ["I want to predict", "View dataset", "Download dataset"], index=0)

if page == "View dataset":
    st.subheader("Live dataset (same tab)")
    st.dataframe(df, use_container_width=True, height=520)

elif page == "Download dataset":
    st.subheader("Download CSV")
    st.download_button("⬇️ Download", combined_csv_bytes(ws),
                       file_name="expert_predictions_live.csv", mime="text/csv")

else:
    st.subheader("Start a prediction")
    with st.form("gate"):
        name  = st.text_input("Your name*", "")
        email = st.text_input("Your email*", "")
        ok = st.form_submit_button("Find me a patient")
    if ok:
        if not name or not email:
            st.error("Please enter name and email.")
            st.stop()

        choice_idx, df0, header = pick_and_claim(ws, header, df, name)
        if choice_idx is None:
            st.success("All patients have been completed. 🙌")
            st.stop()

        df_now, _ = read_all(ws)
        row = df_now.iloc[choice_idx]
        sheet_row = choice_idx + 2

        st.subheader(f"Patient (row {sheet_row})")
        for col in iter_patient_fields(row):
            label = FRIENDLY.get(col, col)
            st.write(f"**{label}**: {row[col]}")

        st.markdown("---")
        st.subheader("Your prediction")
        with st.form("pred"):
            outcome = st.selectbox("Surgical outcome",
                                   options=[0,1],
                                   format_func=lambda x: f"{x} — {'Successful' if x==1 else 'Unsuccessful'}")
            confidence = st.radio("How confident are you?", [
                "Very confident","Somewhat confident","Neutral","Somewhat unsure","Not at all confident"
            ], index=1)
            snot22 = st.slider("Estimated postoperative SNOT-22 at 6 months", 0, 110, 24, 1)
            submit = st.form_submit_button("Submit")

        if submit:
            fresh, _ = read_all(ws)
            ss = str(fresh.iloc[choice_idx].get("submission_status","")).strip().lower()
            claimed_by = str(fresh.iloc[choice_idx].get("claimed_by","")).strip()
            pred_now = str(fresh.iloc[choice_idx].get("expert_prediction","")).strip()
            if (ss not in ["claimed",""]) or (claimed_by != name) or (pred_now != ""):
                st.error("This patient is no longer editable. Please start again.")
                st.stop()

            updates = {
                "submission_status": "Submitted",
                "reviewer_name": name,
                "reviewer_email": email,
                "expert_prediction": outcome,
                "expert_confidence": confidence,
                "expert_SNOT22score_prediction": snot22,
                "submitted_at": datetime.utcnow().isoformat()
            }
            write_row(ws, sheet_row, header, updates)
            st.success("Thank you! Your submission has been saved.")
            st.balloons()
