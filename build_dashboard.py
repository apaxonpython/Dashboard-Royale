"""
Dashboard Royale: rebuilds every number and the dashboard page from the raw data.

Run:   python build_dashboard.py --data path/to/dashboard-casino/data --out output
Needs: pip install -r requirements.txt   (Python 3.11 or newer; pandas, numpy, pyarrow)

What it does, in order:
  1. Q1  Luck-adjusted machine scorecard: manager's bottom 10, a peer-adjusted bottom 10,
         payout flags, zone and bank views. Verdicts are generated from the numbers.
  2. Q2  Monthly demand, event lift (two baselines), weekday forecast vs flat baseline
         (time-based split, predictions saved for the actual-vs-predicted chart)
  3. Q3  Downtime: lost machine-hours and estimated lost house win
  4. Q4  Player outcomes and loss concentration
  5. Q5  Comp cost and points-per-comp-dollar by month
  6. Writes output/dashboard.html (open it in any browser) plus two CSV tables.
The $100,000 plan on the dashboard is written by hand in dashboard_template.html (list P),
using the numbers this script produces.

Guide rules this script follows (GUIDE.md section 5):
  - ratio of sums, never a mean of daily ratios
  - only days the machine was on the floor (active)
  - hours per day come from calendar_day (March 8 has 23 hours)
  - luck is judged against theo before any machine is called weak
  - split by time for the forecast, no future information
  - flags are leads to inspect, never accusations
"""
import argparse, json, math, warnings
from decimal import Decimal
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent

# Thresholds live here so every flag on the dashboard can be traced to a rule.
MIN_DAYS = 90        # fewer floor days than this: too early to judge a machine
Z_FLAG = 3.0         # |luck score| at or above this: payout looks unusual, inspect
LOC_WEAK = 0.80      # zone wagers per hour below 80% of the floor average = weak location
AV_LOW = 0.92        # available for under 92% of scheduled hours = often down
PEER_MIN = 5         # smallest peer group (zone x volatility) trusted for the peer median
EXTRA_N = 3          # extra payout-flagged machines to show beside the bottom 10
TRAIN_END = "2025-05-01"   # forecast: train before this date (Jan-Apr), test on or after


def rnd(x, n=0):
    """Round for the dashboard; missing stays missing (becomes null in the page)."""
    if x is None or pd.isna(x):
        return None
    return int(round(float(x))) if n == 0 else round(float(x), n) + 0.0   # + 0.0 turns -0.0 into 0.0


def clean(o):
    """Make the data JSON-safe: NaN becomes null, numpy types become plain Python."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if math.isnan(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def load(root, name):
    """Find <name>.parquet (or .csv.gz / .csv) anywhere under the data folder and load it."""
    root = Path(root)
    hits = sorted(root.rglob(name + ".parquet"))
    if hits:
        df = pd.read_parquet(hits[0])
    else:
        hits = sorted(root.rglob(name + ".csv.gz")) or sorted(root.rglob(name + ".csv"))
        if not hits:
            raise FileNotFoundError(f"Could not find {name}.parquet or {name}.csv.gz under {root}")
        df = pd.read_csv(hits[0])
    if len(hits) > 1:
        warnings.warn(f"{name}: {len(hits)} matching files, using {hits[0]}")
    for c in df.columns:                      # money columns can arrive as Decimal; use floats
        s = df[c].dropna()
        if df[c].dtype == object and len(s) and isinstance(s.iloc[0], Decimal):
            df[c] = df[c].astype(float)
    if "gaming_date" in df:
        df["gaming_date"] = df["gaming_date"].astype(str)
    return df


def verdict(r):
    """One plain-language reading per machine, built only from that machine's own numbers."""
    if r.days < MIN_DAYS:
        return f"Only {int(r.days)} days on the floor. Too early to judge."
    if abs(r.z) >= Z_FLAG:
        way = "Pays out far above its design" if r.z < 0 else "Holds far more than its design"
        return (f"{way} (hold {r.hold:.0f}% vs {r.th:.0f}%, luck score {r.z:+.1f}). "
                "Inspect the payout setting before any decision; this is a lead, not proof of a fault.")
    why = []
    if r.zone_idx < LOC_WEAK:
        why.append(f"{r.zone} location (zone wagers per hour are {r.zone_idx:.0%} of the floor average)")
    if r.av < AV_LOW:
        why.append(f"often unavailable ({r.av:.0%} of scheduled hours)")
    luck = ("Luck is normal." if -1.5 < r.z < 1 else
            "Some bad luck." if r.z <= -1.5 else "Slightly lucky, still low.")
    if why:
        text = "; ".join(why)
        return text[0].upper() + text[1:] + ". " + luck
    if r.peer_idx < 0.75:
        return (f"Expected win per hour is {r.peer_idx:.0%} of its zone peers with normal luck and uptime. "
                "Candidate for replacement.")
    return f"No single cause stands out. {luck}"


def q1_machines(d, root, out):
    a = d[d["active"]].copy()          # drop the 1,018 rows from before install / after retire
    cal = load(root, "calendar_day")[["gaming_date", "scheduled_seconds"]]
    a = a.merge(cal, on="gaming_date", how="left")
    a["sched_h"] = a.scheduled_seconds / 3600           # March 8 has 23 hours, so don't assume 24
    a["lost_h"] = (a.sched_h - a.available_hours).clip(lower=0)

    # Luck score: (actual - expected win) / typical swing for that volatility class.
    a["r"] = a.luck_residual_usd / np.sqrt(a.coin_in_usd.clip(lower=1))
    sd = a.groupby("volatility_class").r.std()
    g = a.groupby("machine_id").agg(
        zone=("zone_name", lambda s: s.mode().iat[0]),       # most days, because machines move
        zones=("zone_name", "nunique"), vol=("volatility_class", "first"), man=("manufacturer", "first"),
        days=("gaming_date", "count"), ci=("coin_in_usd", "sum"), hw=("house_win_usd", "sum"),
        theo=("theo_usd", "sum"), ah=("available_hours", "sum"), sh=("sched_h", "sum"),
        faults=("faults_opened", "sum"))
    g["av"] = g.ah / g.sh                    # ratio of sums, not a mean of daily ratios
    g["z"] = (g.hw - g.theo) / (g.vol.map(sd) * np.sqrt(g.ci))
    g["hold"] = g.hw / g.ci * 100            # ratio of sums, never a mean of daily ratios
    g["th"] = g.theo / g.ci * 100
    g["cph"] = g.ci / g.ah.replace(0, np.nan)    # wagers per available hour = exposure-adjusted

    # Zone view first: the machine scorecard needs the zone's wagers per hour.
    zt = a.groupby("zone_name").agg(m=("machine_id", "nunique"), ci=("coin_in_usd", "sum"),
                                    hw=("house_win_usd", "sum"), theo=("theo_usd", "sum"), ah=("available_hours", "sum"))
    zt["cph"] = zt.ci / zt.ah
    zt.round(4).to_csv(out / "zone_summary.csv")
    floor_cph = a.coin_in_usd.sum() / a.available_hours.sum()
    g["zone_idx"] = g.zone.map(zt.cph) / floor_cph      # location effect: below 1 = weak zone

    # Peer index: expected win (theo, so luck-free) per available hour vs same zone and volatility.
    g["theo_ph"] = g.theo / g.ah.replace(0, np.nan)
    zone_med = g.groupby("zone").theo_ph.transform("median")
    grp_med = g.groupby(["zone", "vol"]).theo_ph.transform("median")
    grp_n = g.groupby(["zone", "vol"]).theo_ph.transform("size")
    g["peer_idx"] = g.theo_ph / np.where(grp_n >= PEER_MIN, grp_med, zone_med)

    g["verdict"] = g.apply(verdict, axis=1)
    bottom = list(g.sort_values("hw").head(10).index)    # the manager's naive bottom 10
    g["bottom10"] = g.index.isin(bottom)
    g.round(4).to_csv(out / "machine_scorecard.csv")

    # Machines whose payout looks unusual but who are not in the bottom 10: a lead to inspect.
    flagged = g[(g.days >= MIN_DAYS) & (g.z.abs() >= Z_FLAG) & ~g.index.isin(bottom)].z.abs()
    extra = list(flagged.sort_values(ascending=False).head(EXTRA_N).index)
    mach = [dict(id=m, zone=g.loc[m, "zone"], days=int(g.loc[m, "days"]), hw=rnd(g.loc[m, "hw"]),
                 theo=rnd(g.loc[m, "theo"]), z=rnd(g.loc[m, "z"], 1), hold=rnd(g.loc[m, "hold"], 1),
                 th=rnd(g.loc[m, "th"], 1), cph=rnd(g.loc[m, "cph"]), bot=m in bottom, v=g.loc[m, "verdict"])
            for m in bottom + extra]
    # Peer-adjusted bottom 10: lowest expected win per hour among machines with enough days.
    adj = [dict(id=m, zone=r.zone, days=int(r.days), peer=rnd(r.peer_idx, 2), z=rnd(r.z, 1),
                av=rnd(r.av * 100, 1), cph=rnd(r.cph), bot=m in bottom, v=r.verdict)
           for m, r in g[g.days >= MIN_DAYS].sort_values("peer_idx").head(10).iterrows()]

    zone = [dict(z=k, m=int(r.m), cph=rnd(r.ci / r.ah, 1), hold=rnd(r.hw / r.ci * 100, 1),
                 th=rnd(r.theo / r.ci * 100, 1)) for k, r in zt.iterrows()]
    bk = a.groupby(["zone_name", "bank_id"]).agg(m=("machine_id", "nunique"), ci=("coin_in_usd", "sum"),
                                                 hw=("house_win_usd", "sum"), ah=("available_hours", "sum")).reset_index()
    banks = [dict(z=r.zone_name, b=r.bank_id, m=int(r.m), cph=rnd(r.ci / r.ah),
                  hold=rnd(r.hw / r.ci * 100, 1)) for r in bk.itertuples()]

    # Q3: lost play. Lost hours x that machine's own wagers per hour x floor hold.
    a["lost_ci"] = a.lost_h * a.machine_id.map(g.cph)
    hold = a.house_win_usd.sum() / a.coin_in_usd.sum()
    exp_hold = a.theo_usd.sum() / a.coin_in_usd.sum()    # expected hold: the same estimate without luck
    q3 = a.groupby("month").agg(lost_h=("lost_h", "sum"), lost_ci=("lost_ci", "sum"), sched=("sched_h", "sum"))
    q3["lost_hw"] = q3.lost_ci * hold
    q3["lost_hw_exp"] = q3.lost_ci * exp_hold
    q3["pct"] = q3.lost_h / q3.sched * 100
    mf = a.groupby("manufacturer").agg(lost_h=("lost_h", "sum"), sched=("sched_h", "sum"),
                                       faults=("faults_opened", "sum"), m=("machine_id", "nunique"))
    mf["pct"] = mf.lost_h / mf.sched * 100
    mf["faults_per_machine"] = mf.faults / mf.m
    mf["lost_hw"] = a.groupby("manufacturer").lost_ci.sum() * hold
    ops = load(root, "operations")
    ops = ops[ops.event_type == "fault_repair"].copy()
    ops["dur"] = (pd.to_datetime(ops.available_at_utc) - pd.to_datetime(ops.opened_at_utc)).dt.total_seconds() / 3600
    ops.loc[ops.dur < 0, "dur"] = np.nan                 # a repair cannot finish before it opened
    ops["hr"] = pd.to_datetime(ops.opened_at_local).dt.hour
    h = ops.groupby("hr").agg(n=("dur", "size"), avg=("dur", "mean"))
    print("Manager's bottom 10 vs peer-adjusted bottom 10: %d machines in both" %
          len(set(bottom) & {x["id"] for x in adj}))
    return dict(mach=mach, adj=adj, zone=zone, banks=banks, hold=round(hold * 100, 2),
                exp_hold=round(exp_hold * 100, 2),
                man=mf.round(2).reset_index().to_dict("records"),
                hr=h.round(2).reset_index().to_dict("records")), q3, a


def q2_demand(p):
    p = p.copy()
    p["dt"] = pd.to_datetime(p.gaming_date)
    p["month"] = p.gaming_date.str[:7]
    # event lift = actual vs a normal (non-event) day with the same weekday.
    non = p[p.events_active.isna()]
    wdm = non.groupby("local_weekday").coin_in_usd.mean()
    ev = p[p.events_active.notna()].copy()
    ev["exp"] = ev.local_weekday.map(wdm)

    def local_baseline(r):
        """Same weekday, non-event days within 35 days: removes the Jan-Jun trend from the lift."""
        w = non[(non["local_weekday"] == r["local_weekday"]) & ((non["dt"] - r["dt"]).abs() <= pd.Timedelta(days=35))]
        return w["coin_in_usd"].mean() if len(w) else r["exp"]
    ev["exp_loc"] = ev.apply(local_baseline, axis=1)
    e = ev.groupby("events_active").agg(d=("gaming_date", "count"), act=("coin_in_usd", "sum"),
                                        exp=("exp", "sum"), exp_loc=("exp_loc", "sum"))
    events = [dict(n=k, d=int(r.d), l=rnd((r.act / r.exp - 1) * 100, 1),
                   l_local=rnd((r.act / r.exp_loc - 1) * 100, 1)) for k, r in e.iterrows()]

    # forecast: weekday average vs flat average. Train Jan-Apr, test May-Jun (split by TIME, no leakage).
    tr, te = p[p.dt < TRAIN_END], p[p.dt >= TRAIN_END].copy()
    wd = tr.groupby("local_weekday").coin_in_usd.mean()
    base = tr.coin_in_usd.mean()                 # constant training-period daily mean
    te["pred"] = te.local_weekday.map(wd)
    mw = (te.coin_in_usd - te.pred).abs().mean()
    mc = (te.coin_in_usd - base).abs().mean()
    fc = dict(mae_w=rnd(mw), mae_c=rnd(mc), base=rnd(base), wd={k: rnd(v) for k, v in wd.items()},
              split=TRAIN_END, n_train=len(tr), n_test=len(te), gain_pct=rnd((1 - mw / mc) * 100, 1),
              test=[[r.gaming_date, rnd(r.coin_in_usd / 1000, 1), rnd(r.pred / 1000, 1)] for r in te.itertuples()])

    evn = sorted(p.events_active.dropna().unique())
    daily = [[r.gaming_date, rnd(r.coin_in_usd / 1000, 1), rnd(r.total_zone_entries_observed / 1000, 2),
              evn.index(r.events_active) if isinstance(r.events_active, str) else -1] for r in p.itertuples()]
    pm = p.groupby("month").agg(ci=("coin_in_usd", "sum"), avail=("available_machine_hours", "sum"),
                                occ=("occupied_machine_hours", "sum"), ent=("total_zone_entries_observed", "mean"),
                                am=("active_machines", "mean"), days=("gaming_date", "count"))
    pm["ci_day"], pm["occ_rate"] = pm.ci / pm.days, pm.occ / pm.avail * 100
    pm["ci_per_mh"], pm["ent_day"] = pm.ci / pm.avail, pm.ent   # mean skips days the counter was offline
    return dict(events=events, fc=fc, evn=evn, daily=daily), pm


def q4_q5_players(pp):
    g = pp.groupby("player_id").agg(net=("tracked_net_usd", "sum"), ci=("tracked_coin_in_usd", "sum"))
    loss = (-g.net).clip(lower=0).sort_values(ascending=False)
    conc = {str(q): round(float(loss.iloc[:int(len(g) * q)].sum() / loss.sum() * 100), 1) for q in [.01, .05, .1, .2]}
    c = g.ci.sort_values(ascending=False)
    q4 = dict(players=len(g), losers=int((g.net < 0).sum()), winners=int((g.net > 0).sum()), ci=round(g.ci.sum()),
              net=round(g.net.sum()), back=round(g.ci.sum() + g.net.sum()),   # wagers - losses = returned to players
              loser_pct=round(float((g.net < 0).mean() * 100), 1), conc=conc,
              ci_top10=round(float(c.iloc[:int(len(g) * .1)].sum() / c.sum() * 100), 1))

    def seg(col):
        r = pp.groupby(col).agg(n=("player_id", "nunique"), ci=("tracked_coin_in_usd", "sum"), net=("tracked_net_usd", "sum"))
        r["loss_pct_ci"] = -r.net / r.ci * 100
        return r.round(1).reset_index().to_dict("records")
    m = pp.groupby("period_month").agg(pts_r=("points_redeemed", "sum"), pts_e=("points_earned", "sum"),
                                       cf=("comp_face_value_usd", "sum"), cc=("comp_cost_usd", "sum"), theo=("theo_usd", "sum"))
    m["ppc"], m["rr"], m["ccp"] = m.pts_r / m.cf, m.pts_r / m.pts_e * 100, m.cc / m.theo * 100   # comp cost counted once
    return dict(q4=q4, reg=seg("home_region"), tier=seg("opening_tier")), m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="the data folder from the dashboard-casino repo")
    ap.add_argument("--out", default="output")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(exist_ok=True)

    d = load(args.data, "prepared_daily_view")            # loaded once and reused
    if "month" not in d:
        d["month"] = d.gaming_date.str[:7]
    D, q3, a = q1_machines(d, args.data, out)
    d2, pm = q2_demand(load(args.data, "property_day"))
    d45, q5 = q4_q5_players(load(args.data, "player_period"))
    D.update(d2)
    D["ev"] = D.pop("events")
    D.update(d45)
    # zone-by-month wagers per hour (small table for the zone drill-down)
    zm = a.groupby(["zone_name", "month"]).agg(ci=("coin_in_usd", "sum"), ah=("available_hours", "sum"))
    zm = (zm.ci / zm.ah).unstack()
    D["zm"] = {k: [rnd(x) for x in v] for k, v in zm.iterrows()}
    D["mo"] = [dict(m=m, ci_day=pm.loc[m, "ci_day"], occ_rate=pm.loc[m, "occ_rate"], ci_per_mh=pm.loc[m, "ci_per_mh"],
                    ent_day=pm.loc[m, "ent_day"], am=pm.loc[m, "am"], lost_h=q3.loc[m, "lost_h"],
                    lost_hw=q3.loc[m, "lost_hw"], lost_hw_exp=q3.loc[m, "lost_hw_exp"], pct=q3.loc[m, "pct"],
                    cc=q5.loc[m, "cc"], cf=q5.loc[m, "cf"], ccp=q5.loc[m, "ccp"], ppc=q5.loc[m, "ppc"],
                    rr=q5.loc[m, "rr"]) for m in sorted(pm.index)]
    html = (HERE / "dashboard_template.html").read_text(encoding="utf-8")
    html = html.replace("__D__", json.dumps(clean(D), separators=(",", ":"), allow_nan=False))
    (out / "dashboard.html").write_text(html, encoding="utf-8")
    print("Done. Open", out / "dashboard.html", "in your browser.")
    print("Weekday forecast error $%s vs flat baseline $%s (train %d days, test %d days)" %
          (f"{D['fc']['mae_w']:,}", f"{D['fc']['mae_c']:,}", D["fc"]["n_train"], D["fc"]["n_test"]))


if __name__ == "__main__":
    main()
