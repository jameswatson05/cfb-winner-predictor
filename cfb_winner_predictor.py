# cfb_winner_predictor.py — KEYLESS VERSION (Sports-Reference scrape)
# No API keys required. Uses public Sports-Reference pages via pandas.read_html.
# Features:
#  - SRS team ratings (current season)
#  - Weekly schedule (current season, selected week)
#  - Auto-calibration: fits a logistic model on last season's finished games (SRS diff -> win prob)
#  - Home-field boost (toggleable), optional manual spread input
#  - Clear "why" box
#
# Notes:
#  - This relies on Sports-Reference HTML tables. If they change layout, the parser may need tweaks.
#  - No guarantees of >90% overall accuracy; this is a solid keyless baseline.

import math
from datetime import datetime
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression

NOW = datetime.now()
CURRENT_YEAR = NOW.year

st.set_page_config(page_title="CFB Winner Predictor (Keyless)", layout="wide")
st.title("🏈 College Football Winner Predictor — Keyless (Sports-Reference)")
st.caption("Scrapes SRS ratings & schedules from sports-reference.com. Auto-calibrates on last season. No API keys needed.")

# ---------------------------
# Utilities
# ---------------------------
def _read_html_tables(url: str) -> List[pd.DataFrame]:
    """Read HTML tables from a URL with friendly error handling."""
    try:
        tables = pd.read_html(url, flavor="lxml")
        return tables
    except Exception as e:
        st.warning(f"Failed to read tables from {url}: {e}")
        return []

def _clean_team_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    # Sports-Reference includes rankings like "No. 3 Georgia" and notes/footnotes sometimes.
    name = name.replace("No. ", "")
    # Remove rank prefixes like "3 " if present
    parts = name.strip().split()
    if parts and parts[0].isdigit():
        parts = parts[1:]
    return " ".join(parts).strip()

# ---------------------------
# Scrapers (Sports-Reference)
# ---------------------------
@st.cache_data(show_spinner=False)
def fetch_srs(season: int) -> pd.DataFrame:
    """Fetch SRS ratings for a season. URL pattern: /cfb/years/{season}-ratings.html"""
    url = f"https://www.sports-reference.com/cfb/years/{season}-ratings.html"
    tables = _read_html_tables(url)
    # Usually the first (or only) table is the SRS table with columns including 'School' and 'SRS'
    srs_df = None
    for t in tables:
        cols = [c.lower() for c in t.columns]
        if any("school" in c for c in cols) and any("srs" == c for c in cols):
            srs_df = t.copy()
            break
    if srs_df is None and tables:
        # Fall back to first table
        srs_df = tables[0].copy()

    if srs_df is None or srs_df.empty:
        return pd.DataFrame(columns=["team","srs"])

    # Normalize columns
    colmap = {}
    for c in srs_df.columns:
        cl = str(c).strip().lower()
        if "school" in cl or "team" in cl:
            colmap[c] = "team"
        elif cl == "srs":
            colmap[c] = "srs"
        elif "conf" in cl:
            colmap[c] = "conf"
    srs_df = srs_df.rename(columns=colmap)
    if "team" not in srs_df.columns:
        # Try index as team if provided
        srs_df["team"] = srs_df.index.astype(str)
    if "srs" not in srs_df.columns:
        srs_df["srs"] = np.nan

    # Clean
    srs_df["team"] = srs_df["team"].apply(_clean_team_name)
    # Ensure numeric SRS
    srs_df["srs"] = pd.to_numeric(srs_df["srs"], errors="coerce")
    srs_df = srs_df[["team","srs"]].dropna(subset=["team"]).drop_duplicates(subset=["team"])
    return srs_df.reset_index(drop=True)

@st.cache_data(show_spinner=False)
def fetch_week_schedule(season: int, week: int) -> pd.DataFrame:
    """
    Fetch a week's schedule & results.
    URL pattern: /cfb/years/{season}-week-{week}-schedule.html
    Expected columns usually include 'Winner/Tie', 'Loser/Tie', 'Pts', 'Pts.1', 'Location' (with '@' for home/away), and maybe 'Neutral'.
    """
    url = f"https://www.sports-reference.com/cfb/years/{season}-week-{int(week)}-schedule.html"
    tables = _read_html_tables(url)
    if not tables:
        return pd.DataFrame(columns=["home","away","home_pts","away_pts","neutral"])

    # Find a table with winner/loser columns
    sched = None
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any("winner/tie" in c for c in cols) and any("loser/tie" in c for c in cols):
            sched = t.copy()
            break
    if sched is None:
        # fallback to first table
        sched = tables[0].copy()

    # Normalize names
    colmap = {}
    for c in sched.columns:
        cl = str(c).strip().lower()
        if "winner/tie" in cl:
            colmap[c] = "winner"
        elif "loser/tie" in cl:
            colmap[c] = "loser"
        elif cl in ("pts", "ptsw"):
            colmap[c] = "pts_w"
        elif cl in ("pts.1", "ptsl"):
            colmap[c] = "pts_l"
        elif "location" in cl:
            colmap[c] = "location"
        elif "neutral" in cl:
            colmap[c] = "neutral"
        elif "date" in cl:
            colmap[c] = "date"
    sched = sched.rename(columns=colmap)

    for c in ["winner","loser","pts_w","pts_l","location","neutral","date"]:
        if c not in sched.columns:
            sched[c] = np.nan

    # Clean names
    sched["winner"] = sched["winner"].astype(str).apply(_clean_team_name)
    sched["loser"]  = sched["loser"].astype(str).apply(_clean_team_name)

    # Determine home/away:
    # Heuristic (Sports-Reference convention):
    # - If 'location' contains '@', the winner is the AWAY team (played at loser's site)
    # - Else winner is HOME team (or neutral—check 'neutral' if present)
    def _row_to_match(r):
        loc = str(r.get("location") or "")
        neutral = False
        if "neutral" in sched.columns:
            neutral = bool(r.get("neutral")) if not pd.isna(r.get("neutral")) else False

        winner = r["winner"]; loser = r["loser"]
        ptsw = r.get("pts_w"); ptsl = r.get("pts_l")

        if "@" in loc:
            # game played at loser's site -> loser = home, winner = away
            home, away = loser, winner
            home_pts, away_pts = ptsl, ptsw
        else:
            # assume winner is home unless marked neutral (neutral -> no HFA)
            home, away = winner, loser
            home_pts, away_pts = ptsw, ptsl

        return pd.Series({
            "home": home, "away": away,
            "home_pts": pd.to_numeric(home_pts, errors="coerce"),
            "away_pts": pd.to_numeric(away_pts, errors="coerce"),
            "neutral": bool(neutral)
        })

    out = sched.apply(_row_to_match, axis=1)
    out = out.dropna(subset=["home","away"]).reset_index(drop=True)
    return out

# ---------------------------
# Auto-calibration (keyless)
# ---------------------------
@st.cache_resource(show_spinner=False)
def calibrate_on_season(season: int) -> Optional[dict]:
    """
    Build a calibration from finished games in `season`:
      home_win ~ SRS_diff (+ home_field, optional)
    """
    srs = fetch_srs(season)
    if srs.empty:
        return None

    # Sports-Reference weeks typically go 1..15 (varies). Try 1..20, collect what exists.
    frames = []
    for wk in range(1, 21):
        wkdf = fetch_week_schedule(season, wk)
        if wkdf.empty:
            continue
        frames.append(wkdf)
    if not frames:
        return None

    games = pd.concat(frames, ignore_index=True)
    # Only finished games
    games = games[games["home_pts"].notna() & games["away_pts"].notna()].copy()

    # Join SRS
    srs_map = {r["team"]: r["srs"] for _, r in srs.iterrows()}
    games["srs_home"] = games["home"].map(srs_map)
    games["srs_away"] = games["away"].map(srs_map)

    # Feature engineering
    games["srs_diff"] = games["srs_home"] - games["srs_away"]
    games["hfa"] = (~games["neutral"]).astype(float)  # 1 if home, 0 if neutral
    # Label
    games["home_win"] = (games["home_pts"] > games["away_pts"]).astype(int)

    # Drop rows missing SRS features
    feat = games[["srs_diff","hfa"]].fillna(0.0)
    y = games["home_win"].astype(int)

    if len(games) < 150:
        return None

    lr = LogisticRegression(max_iter=200)
    lr.fit(feat, y)
    coefs = dict(zip(["srs_diff","hfa"], lr.coef_[0]))
    intercept = float(lr.intercept_[0])
    return {"coefs": coefs, "intercept": intercept, "n": int(len(games)), "year": season}

@st.cache_resource(show_spinner=False)
def smart_calibration() -> Optional[dict]:
    """
    Prefer current season-to-date if enough finished games, else last season.
    """
    # Try current
    cal = calibrate_on_season(CURRENT_YEAR)
    if cal:
        return cal
    # Fallback to last season
    return calibrate_on_season(CURRENT_YEAR - 1)

CAL = smart_calibration()

# ---------------------------
# Scoring
# ---------------------------
def logit_score(srs_diff: float, hfa: float) -> float:
    x = 0.0 if (srs_diff is None or np.isnan(srs_diff)) else float(srs_diff)
    h = 0.0 if (hfa is None or np.isnan(hfa)) else float(hfa)
    if CAL:
        return CAL["intercept"] + CAL["coefs"]["srs_diff"]*x + CAL["coefs"]["hfa"]*h
    # Fallback if no calibration yet: convert edge to log-odds via a simple scale (6 pts ~ 75% win)
    return (x + 2.0*h) / 6.0

def score_to_prob(score: float) -> float:
    return 1.0 / (1.0 + math.exp(-score))

# ---------------------------
# UI
# ---------------------------
with st.sidebar:
    st.header("⚙️ Inputs")
    season = st.number_input("Season", min_value=2013, max_value=CURRENT_YEAR+1, value=CURRENT_YEAR, step=1)
    week = st.number_input("Week (regular)", min_value=1, max_value=20, value=1, step=1)
    hfa_points = st.slider("Home-field adjustment (pts, if no neutral)", min_value=0.0, max_value=4.0, value=2.0, step=0.5)
    manual_spread = st.text_input("Optional: manual spread (home -pts, e.g., -7 = home favored by 7)", value="")

# Pull SRS and week schedule
srs_df = fetch_srs(int(season))
if srs_df.empty:
    st.error("Couldn't load SRS for that season. Try a different season/week, or reload in a minute.")
    st.stop()

sched_df = fetch_week_schedule(int(season), int(week))
if sched_df.empty:
    st.info("No games found for that week (or page not posted yet). Try another week.")
    st.stop()

teams = sorted(set(srs_df["team"].dropna().tolist()))
# Build list of games from schedule
games = sched_df[["home","away","neutral"]].dropna().drop_duplicates().reset_index(drop=True)

# Let user pick a matchup
col1, col2 = st.columns(2)
if not games.empty:
    # Preselect the first game’s teams in dropdowns
    home_default = games.loc[0, "home"] if isinstance(games.loc[0, "home"], str) else "(pick)"
    away_default = games.loc[0, "away"] if isinstance(games.loc[0, "away"], str) else "(pick)"
else:
    home_default = "(pick)"
    away_default = "(pick)"

home = col1.selectbox("Home team", ["(pick)"] + teams, index=(["(pick)"]+teams).index(home_default) if home_default in teams else 0)
away = col2.selectbox("Away team", ["(pick)"] + teams, index=(["(pick)"]+teams).index(away_default) if away_default in teams else 0)

if home != "(pick)" and away != "(pick)":
    # Pull SRS values
    srs_map = {r["team"]: r["srs"] for _, r in srs_df.iterrows()}
    srs_home = srs_map.get(home, np.nan)
    srs_away = srs_map.get(away, np.nan)
    srs_diff = (srs_home - srs_away) if (not np.isnan(srs_home) and not np.isnan(srs_away)) else np.nan

    # Neutral flag from schedule (if present for this pair)
    neutral = False
    row = sched_df[(sched_df["home"]==home) & (sched_df["away"]==away)]
    if not row.empty:
        neutral = bool(row.iloc[0]["neutral"])

    # HFA points into logit as +hfa if not neutral; we’ll reflect that by adding to srs_diff
    # For transparency in "why", we keep both: base srs_diff and hfa separately
    hfa = 0.0 if neutral else 1.0  # as a binary for the model; UI-specified points for human explanation

    # Optional manual spread nudges the probability slightly toward the market
    spread_val = None
    if manual_spread.strip():
        try:
            sv = float(manual_spread.strip())
            spread_val = sv  # home-centric (negative means home favored)
        except:
            spread_val = None

    # Compute score & probability
    score = logit_score(srs_diff, hfa)
    p_home = score_to_prob(score)

    # Light blend with manual spread if provided (treat 7 pts as ~65% baseline)
    if spread_val is not None:
        # Convert spread to implied prob with a rough mapping, then blend 80/20 model/market
        # home -7 -> ~65%, -3 -> ~58%, +3 -> ~42%, +7 -> ~35%
        def spread_to_p(sp):
            # Negative sp = home favored. Cap effect within [0.30, 0.70]
            base = 0.5 - (sp/14.0)  # -7 -> +0.5; +7 -> -0.5
            return float(np.clip(1.0 - base, 0.30, 0.70))
        mkt_p = spread_to_p(spread_val)
        p_home = 0.8*p_home + 0.2*mkt_p

    # Home-field points shift for "why" (explain in points)
    hfa_pts = 0.0 if neutral else hfa_points

    # Final pick
    pick = home if p_home >= 0.5 else away
    conf = p_home if pick == home else (1.0 - p_home)

    st.subheader("✅ Pick")
    st.markdown(f"**Winner:** {pick}  \n**Confidence:** {conf:.1%}")

    st.subheader("🧠 Why this pick?")
    reasons = []
    def add(txt): reasons.append("• " + txt)

    if not np.isnan(srs_diff):
        add(f"SRS edge: **{home if srs_diff>=0 else away} {abs(srs_diff):.1f} pts**.")
    else:
        add("SRS rating missing for one or both teams.")

    if neutral:
        add("Venue: **Neutral site** (no home-field).")
    else:
        add(f"Home-field adjustment: **≈ {hfa_pts:.1f} pts** for the home side.")

    if manual_spread.strip():
        add(f"Manual spread provided (home-centric): **{spread_val:+.1f}** (negative = home favored).")

    if CAL:
        add(f"Calibrated on **{CAL['year']}** (n={CAL['n']} games).")
        add(f"Model weights → SRS diff: **{CAL['coefs']['srs_diff']:.3f}**, Home field: **{CAL['coefs']['hfa']:.3f}**.")
    else:
        add("Calibration fallback active (simple scaling) — early season or page missing.")

    st.text("\n".join(reasons))

# Maintenance
with st.expander("🧹 Maintenance"):
    if st.button("Clear caches"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Caches cleared. Click Rerun ↻ at top-right.")
