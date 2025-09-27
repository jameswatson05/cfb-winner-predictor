# cfb_winner_predictor.py
# Streamlit app: Predict outright winners for college football matchups
# Data via CollegeFootballData.com (you supply CFBD_API_KEY in app Secrets)
# Optional: moneylines via The Odds API (THE_ODDS_API_KEY) – not required

import os
import math
from typing import List, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from datetime import datetime

# ----------------------------
# Config
# ----------------------------
CFBD_API_KEY = os.getenv("CFBD_API_KEY", "")  # On Streamlit Cloud: Settings → Secrets
CFBD_BASE = "https://api.collegefootballdata.com"
CURRENT_YEAR = datetime.now().year

st.set_page_config(page_title="CFB Winner Predictor", layout="wide")
st.title("🏈 College Football Winner Predictor (Moneyline)")
st.caption("Trains on recent seasons using SP+, SRS, EPA/PPA, roster talent, and book spreads (via CFBD).")

# ----------------------------
# Helpers
# ----------------------------
def cfbd_headers() -> dict:
    key = st.secrets.get("CFBD_API_KEY", CFBD_API_KEY)
    if not key:
        st.error("Missing CFBD_API_KEY. Add it in the app’s Secrets.")
        return {}
    return {"Authorization": f"Bearer {key}"}

def get_json(url: str, params: Optional[dict] = None) -> Optional[dict]:
    try:
        r = requests.get(url, params=params or {}, headers=cfbd_headers(), timeout=30)
        r.raise_for_status()
        if "application/json" in r.headers.get("Content-Type", ""):
            return r.json()
        return None
    except Exception as e:
        st.warning(f"GET failed: {e}")
        return None

@st.cache_data(show_spinner=False)
def cfbd_schedule(year: int, season_type: str = "regular") -> pd.DataFrame:
    data = get_json(f"{CFBD_BASE}/games", {"year": year, "seasonType": season_type}) or []
    return pd.DataFrame(data)

@st.cache_data(show_spinner=False)
def cfbd_sp(year: int) -> pd.DataFrame:
    data = get_json(f"{CFBD_BASE}/ratings/sp", {"year": year}) or []
    return pd.DataFrame(data)

@st.cache_data(show_spinner=False)
def cfbd_srs(year: int) -> pd.DataFrame:
    data = get_json(f"{CFBD_BASE}/ratings/srs", {"year": year}) or []
    return pd.DataFrame(data)

@st.cache_data(show_spinner=False)
def cfbd_team_ppa(year: int) -> pd.DataFrame:
    data = get_json(f"{CFBD_BASE}/metrics/ppa/teams", {"year": year}) or []
    return pd.DataFrame(data)

@st.cache_data(show_spinner=False)
def cfbd_talent(year: int) -> pd.DataFrame:
    data = get_json(f"{CFBD_BASE}/talent", {"year": year}) or []
    return pd.DataFrame(data)

@st.cache_data(show_spinner=False)
def cfbd_lines(year: int, week: Optional[int] = None) -> pd.DataFrame:
    params = {"year": year}
    if week is not None:
        params["week"] = int(week)
    data = get_json(f"{CFBD_BASE}/lines", params) or []
    return pd.DataFrame(data)

def wide_merge(left: pd.DataFrame, feat_df: pd.DataFrame, left_key: str, suffix: str) -> pd.DataFrame:
    """Merge a team-level feature frame onto left using left[left_key] == feat_df['team']."""
    if feat_df.empty or left.empty:
        return left
    if "team" not in feat_df.columns:
        # Try common alternatives
        if "school" in feat_df.columns:
            feat_df = feat_df.rename(columns={"school": "team"})
        else:
            return left
    r = feat_df.copy()
    r = r.add_suffix(suffix)
    return left.merge(r, how="left", left_on=left_key, right_on=f"team{suffix}")

def build_feature_row(row: pd.Series) -> dict:
    def diff(a, b):
        try: return float(a) - float(b)
        except: return np.nan

    feats = {}
    feats["sp_rating_diff"]   = diff(row.get("rating_home"), row.get("rating_away"))
    feats["sp_offense_diff"]  = diff(row.get("offenseRating_home"), row.get("offenseRating_away"))
    # lower (better) defense → invert so positive favors home
    try:
        feats["sp_defense_diff"] = (diff(row.get("defenseRating_home"), row.get("defenseRating_away"))) * -1.0
    except:
        feats["sp_defense_diff"] = np.nan
    feats["srs_diff"]         = diff(row.get("srs_home"), row.get("srs_away"))
    feats["talent_diff"]      = diff(row.get("talent_home"), row.get("talent_away"))
    feats["ppa_off_diff"]     = diff(row.get("offense_ppa_home"), row.get("offense_ppa_away"))
    # defensive PPA lower is better → invert as away_home
    feats["ppa_def_diff"]     = diff(row.get("defense_ppa_away"), row.get("defense_ppa_home"))
    feats["home_field"]       = 0.0 if bool(row.get("neutralSite")) else 1.0
    try:
        feats["spread_home"]  = float(row.get("spread_home"))
    except:
        feats["spread_home"]  = np.nan
    return feats

# ----------------------------
# Training data
# ----------------------------
@st.cache_data(show_spinner=True)
def assemble_training(seasons: List[int]) -> pd.DataFrame:
    frames = []
    for year in seasons:
        sched = cfbd_schedule(year)
        if sched.empty:
            continue
        # Keep FBS vs FBS (approx. – rely on homeConference presence)
        sched = sched[sched["homeConference"].notna()].copy()

        # Fetch features
        sp   = cfbd_sp(year)[["team","rating","offenseRating","defenseRating"]] if not cfbd_sp(year).empty else pd.DataFrame(columns=["team","rating","offenseRating","defenseRating"])
        srs  = cfbd_srs(year)
        if not srs.empty:
            if "srs" not in srs.columns and "rating" in srs.columns:
                srs = srs.rename(columns={"rating": "srs"})
            srs = srs[["team","srs"]]
        ppa  = cfbd_team_ppa(year)
        if not ppa.empty:
            off = ppa[ppa["side"]=="offense"][["team","ppa"]].rename(columns={"ppa":"offense_ppa"})
            de  = ppa[ppa["side"]=="defense"][["team","ppa"]].rename(columns={"ppa":"defense_ppa"})
            pwide = off.merge(de, on="team", how="outer")
        else:
            pwide = pd.DataFrame(columns=["team","offense_ppa","defense_ppa"])
        tal  = cfbd_talent(year)
        if not tal.empty:
            tal = tal.rename(columns={"school":"team"})[["team","talent"]]
        else:
            tal = pd.DataFrame(columns=["team","talent"])

        df = sched[["id","season","week","homeTeam","awayTeam","homePoints","awayPoints","neutralSite"]].copy()
        # Merge features for home & away
        for feat in [(sp,"_home"), (srs,"_home"), (pwide,"_home"), (tal,"_home")]:
            df = wide_merge(df, feat[0], "homeTeam", feat[1])
        for feat in [(sp,"_away"), (srs,"_away"), (pwide,"_away"), (tal,"_away")]:
            df = wide_merge(df, feat[0], "awayTeam", feat[1])

        # Book spreads (median across books), home perspective
        lines = cfbd_lines(year)
        if not lines.empty:
            rows = []
            for _, r in lines.iterrows():
                gid = r.get("gameId")
                for l in r.get("lines", []) if isinstance(r.get("lines"), list) else []:
                    s  = l.get("spread")
                    hf = l.get("homeFavorite")
                    if s is not None and hf is not None:
                        try:
                            hs = float(s) if hf else -float(s)
                            rows.append({"game_id": gid, "home_spread": hs})
                        except:
                            pass
            le = pd.DataFrame(rows)
            if not le.empty:
                agg = le.groupby("game_id").home_spread.median().reset_index()
                df = df.merge(agg, left_on="id", right_on="game_id", how="left").rename(columns={"home_spread":"spread_home"})
        else:
            df["spread_home"] = np.nan

        # Build features + label
        feat_rows, labels = [], []
        for _, r in df.iterrows():
            feat_rows.append(build_feature_row(r))
            try:
                labels.append(1 if int(r["homePoints"]) > int(r["awayPoints"]) else 0)
            except:
                labels.append(None)  # games not finished

        feats_df = pd.DataFrame(feat_rows)
        out = pd.concat([df[["id","season","week","homeTeam","awayTeam","neutralSite"]].reset_index(drop=True), feats_df], axis=1)
        out["home_win"] = labels
        out["year"] = year
        frames.append(out)

    full = pd.concat(frames, ignore_index=True).dropna(subset=["home_win"])
    must_have = [c for c in full.columns if c.endswith("_diff")] + ["home_field"]
    full = full.dropna(axis=0, how="any", subset=must_have)
    return full

@st.cache_resource(show_spinner=True)
def train_model(train_df: pd.DataFrame):
    features = ["sp_rating_diff","sp_offense_diff","sp_defense_diff","srs_diff","talent_diff","ppa_off_diff","ppa_def_diff","home_field","spread_home"]
    X = train_df[features].fillna(0.0)
    y = train_df["home_win"].astype(int)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    model = GradientBoostingClassifier(random_state=42)
    model.fit(Xtr, ytr)

    yhat = model.predict_proba(Xte)[:, 1]
    metrics = {
        "AUC": float(roc_auc_score(yte, yhat)),
        "Accuracy": float(accuracy_score(yte, (yhat >= 0.5).astype(int))),
    }
    return model, features, metrics

# ----------------------------
# UI
# ----------------------------
with st.sidebar:
    st.header("⚙️ Training")
    default_years = list(range(max(2019, CURRENT_YEAR-6), CURRENT_YEAR))  # previous ~6 seasons
    seasons = st.multiselect("Seasons to train on", list(range(2013, CURRENT_YEAR+1)), default=default_years)
    st.caption("Tip: start with 2021–2023 if you hit rate limits.")

if not seasons:
    st.stop()

with st.spinner("Assembling training data…"):
    train_df = assemble_training(seasons)

if train_df.empty:
    st.error("No training data. Check CFBD key or reduce seasons.")
    st.stop()

with st.spinner("Training model…"):
    model, feat_cols, metrics = train_model(train_df)

st.subheader("📈 Backtest (holdout)")
c1, c2 = st.columns(2)
c1.metric("AUC", f"{metrics['AUC']:.3f}")
c2.metric("Accuracy @ 0.50", f"{metrics['Accuracy']:.3f}")
st.divider()

st.header("🔮 Predict an upcoming matchup")
year_pick = st.number_input("Season", min_value=2013, max_value=CURRENT_YEAR+1, value=CURRENT_YEAR, step=1)
week_pick = st.number_input("Week (regular season)", min_value=1, max_value=20, value=5, step=1)

sched_now = cfbd_schedule(int(year_pick))
if sched_now.empty:
    st.warning("Could not fetch schedule for this season.")
    st.stop()

cand = sched_now[sched_now["week"] == int(week_pick)]
if cand.empty:
    st.info("No games for that week (or data not yet posted). Try a different week.")
    st.stop()

teams = sorted(set(cand["homeTeam"].dropna().tolist() + cand["awayTeam"].dropna().tolist()))
home = st.selectbox("Home team", options=["(pick)"] + teams, index=0)
away = st.selectbox("Away team", options=["(pick)"] + teams, index=0)

if home != "(pick)" and away != "(pick)":
    # Build one-row feature frame for this matchup
    sp  = cfbd_sp(int(year_pick))[["team","rating","offenseRating","defenseRating"]]
    srs = cfbd_srs(int(year_pick))
    if not srs.empty:
        if "srs" not in srs.columns and "rating" in srs.columns:
            srs = srs.rename(columns={"rating":"srs"})
        srs = srs[["team","srs"]]
    ppa = cfbd_team_ppa(int(year_pick))
    if not ppa.empty:
        off = ppa[ppa["side"]=="offense"][["team","ppa"]].rename(columns={"ppa":"offense_ppa"})
        de  = ppa[ppa["side"]=="defense"][["team","ppa"]].rename(columns={"ppa":"defense_ppa"})
        pwide = off.merge(de, on="team", how="outer")
    else:
        pwide = pd.DataFrame(columns=["team","offense_ppa","defense_ppa"])
    tal = cfbd_talent(int(year_pick))[["school","talent"]].rename(columns={"school":"team"})

    row = pd.DataFrame([{"homeTeam": home, "awayTeam": away, "neutralSite": False}])
    for fdf, suf in [(sp,"_home"), (srs,"_home"), (pwide,"_home"), (tal,"_home")]:
        row = wide_merge(row, fdf, "homeTeam", suf)
    for fdf, suf in [(sp,"_away"), (srs,"_away"), (pwide,"_away"), (tal,"_away")]:
        row = wide_merge(row, fdf, "awayTeam", suf)

    # Median home spread from books for this matchup
    lines_now = cfbd_lines(int(year_pick), int(week_pick))
    spread_val = np.nan
    if not lines_now.empty:
        vals = []
        for _, r in lines_now.iterrows():
            if r.get("homeTeam") == home and r.get("awayTeam") == away:
                for l in r.get("lines", []) if isinstance(r.get("lines"), list) else []:
                    s  = l.get("spread")
                    hf = l.get("homeFavorite")
                    if s is not None and hf is not None:
                        try:
                            hs = float(s) if hf else -float(s)
                            vals.append(hs)
                        except:
                            pass
        if vals:
            spread_val = float(np.nanmedian(vals))
    row["spread_home"] = spread_val

    feats = pd.DataFrame([build_feature_row(row.iloc[0])])
    X = feats[feat_cols].fillna(0.0)
    prob_home = float(model.predict_proba(X)[:, 1])
    pick = home if prob_home >= 0.5 else away
    conf = prob_home if pick == home else (1.0 - prob_home)

    st.subheader("✅ Pick")
    st.markdown(f"**Winner:** {pick}  \n**Confidence:** {conf:.1%}")

    # Why box
    st.subheader("🧠 Why this pick?")
    reasons = []

    def add_reason(text: str):
        reasons.append("• " + text)

    def gcol(df: pd.DataFrame, name: str):
        v = df.get(name)
        if v is None:
            return np.nan
        return float(v.iloc[0]) if hasattr(v, "iloc") else float(v)

    try:
        diff = gcol(row, "rating_home") - gcol(row, "rating_away")
        add_reason(f"SP+ rating edge: **{home if diff>=0 else away} {abs(diff):.1f} pts**.")
    except: pass

    try:
        diff = gcol(row, "srs_home") - gcol(row, "srs_away")
        add_reason(f"SRS edge: **{home if diff>=0 else away} {abs(diff):.1f} pts**.")
    except: pass

    try:
        off = gcol(row, "offense_ppa_home") - gcol(row, "offense_ppa_away")
        add_reason(f"Offensive EPA/play advantage: **{'home' if off>=0 else 'away'} {abs(off):.3f}**.")
    except: pass

    try:
        de = gcol(row, "defense_ppa_away") - gcol(row, "defense_ppa_home")  # lower is better -> invert
        add_reason(f"Defensive EPA/play advantage: **{'home' if de>=0 else 'away'} {abs(de):.3f}**.")
    except: pass

    try:
        tal = gcol(row, "talent_home") - gcol(row, "talent_away")
        add_reason(f"Roster talent edge: **{home if tal>=0 else away} {abs(tal):.1f} points** (247 composite).")
    except: pass

    if not np.isnan(spread_val):
        if spread_val > 0:
            add_reason(f"Median book spread: **{home} -{abs(spread_val):.1f}**.")
        elif spread_val < 0:
            add_reason(f"Median book spread: **{away} -{abs(spread_val):.1f}**.")
        else:
            add_reason("Median book spread: **Pick'em**.")

    st.text("\n".join(reasons) if reasons else "No specific edges found (check data availability).")
else:
    st.info("Pick both teams to get a prediction.") 
