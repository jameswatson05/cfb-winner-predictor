# cfb_winner_predictor.py — SIMPLE VERSION
# Auto-trains on last 3 completed seasons, predicts current season games.
# Add CFBD_API_KEY in Streamlit: ⋮ → Settings → Secrets

import os, math
from typing import Optional, List
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

CFBD_BASE = "https://api.collegefootballdata.com"
NOW = datetime.now()
CURRENT_YEAR = NOW.year
CFBD_API_KEY = os.getenv("CFBD_API_KEY", "")  # fallback if not in Secrets

st.set_page_config(page_title="CFB Winner Predictor", layout="wide")
st.title("🏈 College Football Winner Predictor (Moneyline)")
st.caption("Trains itself on the last 3 completed seasons (SP+, SRS, EPA/PPA, talent, spreads). Just pick this year's week & teams.")
# --- quick diagnostics ---
with st.expander("🧪 Diagnostics (temporary)", expanded=True):
    show = st.button("Run key test")
    if show:
        k = st.secrets.get("CFBD_API_KEY", os.getenv("CFBD_API_KEY",""))
        st.write("Key present:", bool(k), "(len:", len(k or ""), ")")
        try:
            r = requests.get(
                "https://api.collegefootballdata.com/teams/fbs",
                headers={"Authorization": f"Bearer {k}"} if k else {},
                timeout=20,
            )
            st.write("Status:", r.status_code)
            st.write((r.text or "")[:200])
        except Exception as e:
            st.write("Request error:", str(e))
# -------------------- Keys & HTTP --------------------
def _key() -> str:
    return st.secrets.get("CFBD_API_KEY", CFBD_API_KEY)

def cfbd_headers() -> dict:
    k = _key()
    if not k:
        st.error("Missing CFBD_API_KEY. Add it in ⋮ → Settings → Secrets.")
        return {}
    return {"Authorization": f"Bearer {k}"}

def get_json(url: str, params: Optional[dict] = None) -> list | dict:
    try:
        r = requests.get(url, params=params or {}, headers=cfbd_headers(), timeout=30)
        r.raise_for_status()
        if "application/json" in r.headers.get("Content-Type", ""):
            return r.json()
        return []
    except Exception as e:
        st.warning(f"GET failed {url}: {e}")
        return []

def df_empty(cols: List[str]) -> pd.DataFrame:
    d = {c: pd.Series(dtype="float64") for c in cols}
    if "team" in cols: d["team"] = pd.Series(dtype="object")
    return pd.DataFrame(d)

# -------------------- Safe CFBD wrappers --------------------
@st.cache_data(show_spinner=False)
def cfbd_schedule(year: int, season_type: str = "regular") -> pd.DataFrame:
    return pd.DataFrame(get_json(f"{CFBD_BASE}/games", {"year": year, "seasonType": season_type}))

@st.cache_data(show_spinner=False)
def get_sp(year: int) -> pd.DataFrame:
    need = ["team","rating","offenseRating","defenseRating"]
    df = pd.DataFrame(get_json(f"{CFBD_BASE}/ratings/sp", {"year": year}))
    if df.empty or not set(need).issubset(df.columns):
        return df_empty(need)
    return df[need]

@st.cache_data(show_spinner=False)
def get_srs(year: int) -> pd.DataFrame:
    df = pd.DataFrame(get_json(f"{CFBD_BASE}/ratings/srs", {"year": year}))
    if df.empty:
        return pd.DataFrame({"team": pd.Series(dtype="object"), "srs": pd.Series(dtype="float64")})
    if "srs" not in df.columns and "rating" in df.columns:
        df = df.rename(columns={"rating":"srs"})
    keep = ["team","srs"]
    if not set(keep).issubset(df.columns):
        return pd.DataFrame({"team": pd.Series(dtype="object"), "srs": pd.Series(dtype="float64")})
    return df[keep]

@st.cache_data(show_spinner=False)
def get_ppa(year: int) -> pd.DataFrame:
    df = pd.DataFrame(get_json(f"{CFBD_BASE}/metrics/ppa/teams", {"year": year}))
    if df.empty or not {"team","side","ppa"}.issubset(df.columns):
        return pd.DataFrame({"team": pd.Series(dtype="object"),
                             "offense_ppa": pd.Series(dtype="float64"),
                             "defense_ppa": pd.Series(dtype="float64")})
    off = df[df["side"]=="offense"][["team","ppa"]].rename(columns={"ppa":"offense_ppa"})
    de  = df[df["side"]=="defense"][["team","ppa"]].rename(columns={"ppa":"defense_ppa"})
    return off.merge(de, on="team", how="outer")

@st.cache_data(show_spinner=False)
def get_talent(year: int) -> pd.DataFrame:
    df = pd.DataFrame(get_json(f"{CFBD_BASE}/talent", {"year": year}))
    if df.empty:
        return pd.DataFrame({"team": pd.Series(dtype="object"), "talent": pd.Series(dtype="float64")})
    if "school" in df.columns:
        df = df.rename(columns={"school":"team"})
    return df[["team","talent"]] if set(["team","talent"]).issubset(df.columns) else pd.DataFrame({"team": pd.Series(dtype="object"), "talent": pd.Series(dtype="float64")})

@st.cache_data(show_spinner=False)
def get_lines(year: int, week: Optional[int] = None) -> pd.DataFrame:
    params = {"year": year}
    if week is not None: params["week"] = int(week)
    return pd.DataFrame(get_json(f"{CFBD_BASE}/lines", params))

# -------------------- Features --------------------
def merge_feat(base: pd.DataFrame, feat: pd.DataFrame, side: str) -> pd.DataFrame:
    if feat.empty: return base
    f = feat.copy().add_suffix(side)
    return base.merge(f, how="left",
                      left_on=("homeTeam" if side=="_home" else "awayTeam"),
                      right_on=f"team{side}")

def build_feature_row(r: pd.Series) -> dict:
    def diff(a,b):
        try: return float(a) - float(b)
        except: return np.nan
    feats = {
        "sp_rating_diff":  diff(r.get("rating_home"), r.get("rating_away")),
        "sp_offense_diff": diff(r.get("offenseRating_home"), r.get("offenseRating_away")),
        "sp_defense_diff": diff(r.get("defenseRating_home"), r.get("defenseRating_away")) * -1.0,  # lower def better
        "srs_diff":        diff(r.get("srs_home"), r.get("srs_away")),
        "talent_diff":     diff(r.get("talent_home"), r.get("talent_away")),
        "ppa_off_diff":    diff(r.get("offense_ppa_home"), r.get("offense_ppa_away")),
        "ppa_def_diff":    diff(r.get("defense_ppa_away"), r.get("defense_ppa_home")),
        "home_field":      0.0 if bool(r.get("neutralSite")) else 1.0,
    }
    try: feats["spread_home"] = float(r.get("spread_home"))
    except: feats["spread_home"] = np.nan
    return feats

FEATURES = ["sp_rating_diff","sp_offense_diff","sp_defense_diff","srs_diff",
            "talent_diff","ppa_off_diff","ppa_def_diff","home_field","spread_home"]

def default_training_years(now_year: int) -> List[int]:
    # Last 3 fully completed seasons = (now_year - 1), (now_year - 2), (now_year - 3)
    return [now_year-1, now_year-2, now_year-3]

# -------------------- Training data --------------------
@st.cache_data(show_spinner=True)
def assemble_training(seasons: List[int]) -> pd.DataFrame:
    frames=[]
    for year in seasons:
        sched = cfbd_schedule(year)
        if sched.empty: 
            continue
        sched = sched[sched["homeConference"].notna()].copy()

        sp, srs, ppa, tal = get_sp(year), get_srs(year), get_ppa(year), get_talent(year)
        df = sched[["id","season","week","homeTeam","awayTeam","homePoints","awayPoints","neutralSite"]].copy()

        for feat, side in [(sp,"_home"),(srs,"_home"),(ppa,"_home"),(tal,"_home")]:
            df = merge_feat(df, feat, side)
        for feat, side in [(sp,"_away"),(srs,"_away"),(ppa,"_away"),(tal,"_away")]:
            df = merge_feat(df, feat, side)

        lines = get_lines(year)
        if not lines.empty:
            rows=[]
            for _, lr in lines.iterrows():
                gid = lr.get("gameId")
                for l in lr.get("lines", []) if isinstance(lr.get("lines"), list) else []:
                    s, hf = l.get("spread"), l.get("homeFavorite")
                    if s is not None and hf is not None:
                        try: rows.append({"game_id": gid, "home_spread": float(s) if hf else -float(s)})
                        except: pass
            le = pd.DataFrame(rows)
            if not le.empty:
                df = df.merge(le.groupby("game_id").home_spread.median().reset_index(),
                              left_on="id", right_on="game_id", how="left").rename(columns={"home_spread":"spread_home"})
        if "spread_home" not in df.columns:
            df["spread_home"] = np.nan

        feat_rows, labels = [], []
        for _, r in df.iterrows():
            feat_rows.append(build_feature_row(r))
            try: labels.append(1 if int(r["homePoints"]) > int(r["awayPoints"]) else 0)
            except: labels.append(None)

        out = pd.concat([df[["id","season","week","homeTeam","awayTeam","neutralSite"]].reset_index(drop=True),
                         pd.DataFrame(feat_rows)], axis=1)
        out["home_win"] = labels
        out["year"] = year
        frames.append(out)

    full = pd.concat(frames, ignore_index=True).dropna(subset=["home_win"]) if frames else pd.DataFrame()
    if full.empty:
        return full
    need = [c for c in full.columns if c.endswith("_diff")] + ["home_field"]
    full = full.dropna(axis=0, how="any", subset=need)
    return full

@st.cache_resource(show_spinner=True)
def train_model(train_df: pd.DataFrame):
    X = train_df[FEATURES].fillna(0.0)
    y = train_df["home_win"].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    model = GradientBoostingClassifier(random_state=42).fit(Xtr, ytr)
    yhat = model.predict_proba(Xte)[:,1]
    metrics = {"AUC": float(roc_auc_score(yte, yhat)),
               "Accuracy": float(accuracy_score(yte, (yhat>=0.5).astype(int)))}
    return model, metrics

# -------------------- Auto-train (no picker) --------------------
TRAIN_YEARS = default_training_years(CURRENT_YEAR)
st.info(f"Training on seasons: {', '.join(map(str, TRAIN_YEARS))}")

with st.spinner("Assembling training data…"):
    train_df = assemble_training(TRAIN_YEARS)

if train_df.empty:
    st.error("No training data assembled (rate limit or missing key). Try again in a minute.")
    st.stop()

with st.spinner("Training model…"):
    model, metrics = train_model(train_df)

st.subheader("📈 Backtest (holdout)")
c1, c2 = st.columns(2)
c1.metric("AUC", f"{metrics['AUC']:.3f}")
c2.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
st.divider()

# -------------------- Predict current season --------------------
st.header("🔮 Predict a matchup (current season)")
year_pick = CURRENT_YEAR
week_pick = st.number_input("Week", min_value=1, max_value=20, value=5, step=1)

sched_now = cfbd_schedule(year_pick)
if sched_now.empty:
    st.warning("Schedule fetch empty (rate limit or season not posted). Try again shortly.")
    st.stop()

cand = sched_now[sched_now["week"] == int(week_pick)]
if cand.empty:
    st.info("No games for that week (or data not yet posted).")
    st.stop()

teams = sorted(set(cand["homeTeam"].dropna().tolist() + cand["awayTeam"].dropna().tolist()))
home = st.selectbox("Home team", options=["(pick)"] + teams, index=0)
away = st.selectbox("Away team", options=["(pick)"] + teams, index=0)

if home != "(pick)" and away != "(pick)":
    sp, srs, ppa, tal = get_sp(year_pick), get_srs(year_pick), get_ppa(year_pick), get_talent(year_pick)
    row = pd.DataFrame([{"homeTeam": home, "awayTeam": away, "neutralSite": False}])
    for feat, side in [(sp,"_home"),(srs,"_home"),(ppa,"_home"),(tal,"_home")]:
        row = merge_feat(row, feat, side)
    for feat, side in [(sp,"_away"),(srs,"_away"),(ppa,"_away"),(tal,"_away")]:
        row = merge_feat(row, feat, side)

    lines_now = get_lines(year_pick, int(week_pick))
    spread_val = np.nan
    if not lines_now.empty:
        vals=[]
        for _, r in lines_now.iterrows():
            if r.get("homeTeam")==home and r.get("awayTeam")==away:
                for l in r.get("lines", []) if isinstance(r.get("lines"), list) else []:
                    s, hf = l.get("spread"), l.get("homeFavorite")
                    if s is not None and hf is not None:
                        try: vals.append(float(s) if hf else -float(s))
                        except: pass
        if vals: spread_val = float(np.nanmedian(vals))
    row["spread_home"] = spread_val

    feats = pd.DataFrame([{
        **build_feature_row(row.iloc[0])
    }])
    X = feats[FEATURES].fillna(0.0)
    prob_home = float(model.predict_proba(X)[:,1])
    pick = home if prob_home >= 0.5 else away
    conf = prob_home if pick == home else (1.0 - prob_home)

    st.subheader("✅ Pick")
    st.markdown(f"**Winner:** {pick}  \n**Confidence:** {conf:.1%}")

    def safe(df, name):
        try:
            v = df.get(name)
            if v is None: return np.nan
            return float(v.iloc[0]) if hasattr(v,"iloc") else float(v)
        except: return np.nan

    reasons=[]
    try:
        d = safe(row,"rating_home") - safe(row,"rating_away")
        if not np.isnan(d): reasons.append(f"• SP+ edge: **{home if d>=0 else away} {abs(d):.1f} pts**.")
    except: pass
    try:
        d = safe(row,"srs_home") - safe(row,"srs_away")
        if not np.isnan(d): reasons.append(f"• SRS edge: **{home if d>=0 else away} {abs(d):.1f} pts**.")
    except: pass
    try:
        d = safe(row,"offense_ppa_home") - safe(row,"offense_ppa_away")
        if not np.isnan(d): reasons.append(f"• Offensive EPA/play: **{'home' if d>=0 else 'away'} {abs(d):.3f}**.")
    except: pass
    try:
        d = safe(row,"defense_ppa_away") - safe(row,"defense_ppa_home")
        if not np.isnan(d): reasons.append(f"• Defensive EPA/play: **{'home' if d>=0 else 'away'} {abs(d):.3f}**.")
    except: pass
    try:
        d = safe(row,"talent_home") - safe(row,"talent_away")
        if not np.isnan(d): reasons.append(f"• Roster talent: **{home if d>=0 else away} {abs(d):.1f} pts (247)**.")
    except: pass
    if not np.isnan(spread_val):
        reasons.append(f"• Median book spread: **{home if spread_val>0 else away} -{abs(spread_val):.1f}**." if spread_val!=0 else "• Median book spread: **Pick'em**.")
    st.text("\n".join(reasons) if reasons else "No specific edges found (data sparse/rate-limited).")
else:
    st.info("Pick both teams to get a prediction.")
