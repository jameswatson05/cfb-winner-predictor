# cfb_winner_predictor.py — QUICK PICK (no training)
# Uses CFBD endpoints: games, SP+, SRS, lines. No model fitting required.
# Add CFBD_API_KEY in Streamlit: ⋮ → Settings → Secrets (TOML):
# CFBD_API_KEY = "your_key_here"

import os, math
from typing import Optional
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st

CFBD_BASE = "https://api.collegefootballdata.com"
NOW = datetime.now()
CURRENT_YEAR = NOW.year
CFBD_API_KEY = os.getenv("CFBD_API_KEY", "")

st.set_page_config(page_title="CFB Winner Predictor (Quick)", layout="wide")
st.title("🏈 College Football Winner Predictor (Quick Mode)")
st.caption("No training. Blends SP+, SRS, and market spread for a fast pick with explanations.")

# ---------- Keys / HTTP ----------
def _key() -> str:
    return st.secrets.get("CFBD_API_KEY", CFBD_API_KEY)

def _hdr() -> dict:
    k = _key()
    if not k:
        st.error("Missing CFBD_API_KEY (add in ⋮ → Settings → Secrets).")
        return {}
    return {"Authorization": f"Bearer {k}"}

def jget(url: str, params: Optional[dict]=None) -> list | dict:
    try:
        r = requests.get(url, headers=_hdr(), params=params or {}, timeout=25)
        r.raise_for_status()
        if "application/json" in r.headers.get("Content-Type",""):
            return r.json()
        return []
    except Exception as e:
        st.warning(f"GET failed {url}: {e}")
        return []

# ---------- CFBD wrappers ----------
@st.cache_data(show_spinner=False)
def cfbd_schedule(year: int, season_type="regular") -> pd.DataFrame:
    return pd.DataFrame(jget(f"{CFBD_BASE}/games", {"year": year, "seasonType": season_type}))

@st.cache_data(show_spinner=False)
def cfbd_sp(year: int) -> pd.DataFrame:
    df = pd.DataFrame(jget(f"{CFBD_BASE}/ratings/sp", {"year": year}))
    need = ["team","rating","offenseRating","defenseRating"]
    if df.empty or not set(need).issubset(df.columns):
        return pd.DataFrame(columns=need)
    return df[need]

@st.cache_data(show_spinner=False)
def cfbd_srs(year: int) -> pd.DataFrame:
    df = pd.DataFrame(jget(f"{CFBD_BASE}/ratings/srs", {"year": year}))
    if df.empty:
        return pd.DataFrame(columns=["team","srs"])
    if "srs" not in df.columns and "rating" in df.columns:
        df = df.rename(columns={"rating":"srs"})
    if not {"team","srs"}.issubset(df.columns):
        return pd.DataFrame(columns=["team","srs"])
    return df[["team","srs"]]

@st.cache_data(show_spinner=False)
def cfbd_lines(year: int, week: Optional[int]=None) -> pd.DataFrame:
    params = {"year": year}
    if week is not None: params["week"] = int(week)
    return pd.DataFrame(jget(f"{CFBD_BASE}/lines", params))

# ---------- Helpers ----------
def median_home_spread(lines_df: pd.DataFrame, home: str, away: str) -> float:
    if lines_df.empty: return np.nan
    vals = []
    for _, r in lines_df.iterrows():
        if r.get("homeTeam")==home and r.get("awayTeam")==away:
            for l in r.get("lines", []) if isinstance(r.get("lines"), list) else []:
                s = l.get("spread"); hf = l.get("homeFavorite")
                if s is not None and hf is not None:
                    try:
                        vals.append(float(s) if hf else -float(s))
                    except:
                        pass
    return float(np.nanmedian(vals)) if vals else np.nan
# ==== Calibration on last season (uses only games/SP+/SRS/lines) ====
from sklearn.linear_model import LogisticRegression

@st.cache_resource(show_spinner=False)
def calibrate_weights(cal_year: int = CURRENT_YEAR - 1):
    g = cfbd_schedule(cal_year)
    if g.empty:
        return None
    g = g[g["homePoints"].notna() & g["awayPoints"].notna()].copy()

    sp  = cfbd_sp(cal_year)
    srs = cfbd_srs(cal_year)
    lines = cfbd_lines(cal_year)

    g = g.merge(sp.rename(columns={"team": "homeTeam", "rating": "sp_home"})[["homeTeam","sp_home"]],
                on="homeTeam", how="left")
    g = g.merge(sp.rename(columns={"team": "awayTeam", "rating": "sp_away"})[["awayTeam","sp_away"]],
                on="awayTeam", how="left")
    g = g.merge(srs.rename(columns={"team": "homeTeam", "srs": "srs_home"})[["homeTeam","srs_home"]],
                on="homeTeam", how="left")
    g = g.merge(srs.rename(columns={"team": "awayTeam", "srs": "srs_away"})[["awayTeam","srs_away"]],
                on="awayTeam", how="left")

    def med_spread(game_id: int) -> float:
        if lines.empty: return np.nan
        vals = []
        for _, r in lines[lines.get("gameId")==game_id].iterrows():
            for l in r.get("lines", []) if isinstance(r.get("lines"), list) else []:
                s, hf = l.get("spread"), l.get("homeFavorite")
                if s is not None and hf is not None:
                    try:
                        vals.append(float(s) if hf else -float(s))
                    except:
                        pass
        return float(np.nanmedian(vals)) if vals else np.nan

    g["sp_diff"] = g["sp_home"] - g["sp_away"]
    g["srs_diff"] = g["srs_home"] - g["srs_away"]
    g["spread_home"] = g["id"].apply(med_spread)
    g["home_win"] = (g["homePoints"].astype(float) > g["awayPoints"].astype(float)).astype(int)

    X = g[["sp_diff","srs_diff","spread_home"]].fillna(0.0)
    y = g["home_win"].astype(int)

    if len(X) < 200:
        return None

    lr = LogisticRegression(max_iter=200)
    lr.fit(X, y)
    return {
        "coefs": dict(zip(["sp_diff","srs_diff","spread_home"], lr.coef_[0])),
        "intercept": float(lr.intercept_[0]),
        "n": int(len(X)),
        "year": cal_year,
    }

CAL = calibrate_weights(CURRENT_YEAR - 1)
def quick_score(sp_diff: float, srs_diff: float, spread_home: float) -> float:
    """
    Simple weighted blend (no training):
      score = 0.55*SP+ + 0.35*SRS + 0.10*spread
    Positive -> favors home. Negative -> favors away.
    """
    a, b, c = 0.55, 0.35, 0.10
    x = (sp_diff if not np.isnan(sp_diff) else 0.0)
    y = (srs_diff if not np.isnan(srs_diff) else 0.0)
    z = (spread_home if not np.isnan(spread_home) else 0.0)
    return a*x + b*y + c*z

def score_to_conf(score: float) -> float:
    # Convert score to 0-1 via logistic with scale ~6 points
    k = 6.0
    p_home = 1.0/(1.0 + math.exp(-(score/k)))
    return p_home

# ---------- UI ----------
with st.sidebar:
    st.header("⚙️ Inputs")
    season = st.number_input("Season", min_value=2013, max_value=CURRENT_YEAR+1, value=CURRENT_YEAR, step=1)
    week = st.number_input("Week (regular)", min_value=1, max_value=20, value=1, step=1)
    st.caption("Pick 2025 and your week, then select teams.")

# Fetch schedule for the season/week
sched = cfbd_schedule(int(season))
if sched.empty:
    st.error("Schedule fetch returned empty. Check key or try again in a minute.")
    st.stop()

games = sched[sched["week"]==int(week)]
if games.empty:
    st.info("No games found for that week (or not posted yet). Try another week.")
    st.stop()

teams = sorted(set(games["homeTeam"].dropna().tolist() + games["awayTeam"].dropna().tolist()))
col1, col2 = st.columns(2)
home = col1.selectbox("Home team", ["(pick)"] + teams, index=0)
away = col2.selectbox("Away team", ["(pick)"] + teams, index=0)

if home != "(pick)" and away != "(pick)":
    # Ratings for season
    sp  = cfbd_sp(int(season))
    srs = cfbd_srs(int(season))
    lines = cfbd_lines(int(season), int(week))

    # Pull values
    def val(df, team, col):
        try:
            return float(df[df["team"]==team][col].iloc[0])
        except:
            return np.nan

    sp_home = val(sp, home, "rating")
    sp_away = val(sp, away, "rating")
    srs_home = val(srs, home, "srs")
    srs_away = val(srs, away, "srs")
    spread_home = median_home_spread(lines, home, away)

    sp_diff = (sp_home - sp_away) if (not np.isnan(sp_home) and not np.isnan(sp_away)) else np.nan
    srs_diff = (srs_home - srs_away) if (not np.isnan(srs_home) and not np.isnan(srs_away)) else np.nan

    score = quick_score(sp_diff, srs_diff, spread_home)
    p_home = score_to_conf(score)
    pick = home if p_home >= 0.5 else away
    conf = p_home if pick == home else (1.0 - p_home)

    st.subheader("✅ Pick")
    st.markdown(f"**Winner:** {pick}  \n**Confidence:** {conf:.1%}")

    st.subheader("🧠 Why this pick?")
    reasons = []
    def add(txt): reasons.append("• " + txt)

    if not np.isnan(sp_diff):
        add(f"SP+ edge: **{home if sp_diff>=0 else away} {abs(sp_diff):.1f} pts**.")
    else:
        add("SP+ data missing for one or both teams.")

    if not np.isnan(srs_diff):
        add(f"SRS edge: **{home if srs_diff>=0 else away} {abs(srs_diff):.1f} pts**.")
    else:
        add("SRS data missing for one or both teams.")

    if not np.isnan(spread_home):
        if spread_home > 0:
            add(f"Market spread: **{home} -{abs(spread_home):.1f}**.")
        elif spread_home < 0:
            add(f"Market spread: **{away} -{abs(spread_home):.1f}**.")
        else:
            add("Market spread: **Pick'em**.")
    else:
        add("No spread available for this matchup yet.")

    add(f"Blended score (0.55·SP+ + 0.35·SRS + 0.10·spread) → **{score:+.2f}**.")
    st.text("\n".join(reasons))

# Maintenance helpers
with st.expander("🧹 Maintenance"):
    if st.button("Clear caches"):
        st.cache_data.clear()
        st.success("Cleared. Click Rerun ↻ at top-right.")
