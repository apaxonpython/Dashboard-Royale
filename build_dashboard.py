"""
Dashboard Royale: rebuilds every number and the dashboard page from the raw data.

Run:   python build_dashboard.py --data path/to/dashboard-casino/data --out output
Needs: pip install -r requirements.txt   (Python 3.11 or newer)

What it does, in order:
  1. Q1  Luck-adjusted machine scorecard, zone and bank views
  2. Q2  Monthly demand, event lift, weekday forecast vs flat baseline (time-based split)
  3. Q3  Downtime: lost machine-hours and estimated lost house win
  4. Q4  Player outcomes and loss concentration
  5. Q5  Comp cost and points-per-comp-dollar by month
  6. Writes output/dashboard.html (open it in any browser) plus two CSV tables.
The $100,000 plan on the dashboard is written by hand in dashboard_template.html (list P),
using the numbers this script produces.
"""
import argparse, json
from decimal import Decimal
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent

def load(root, name):
    """Find <name>.parquet anywhere under the data folder and load it."""
    hits = list(Path(root).rglob(name + ".parquet"))
    if not hits:
        raise FileNotFoundError(f"Could not find {name}.parquet under {root}")
    df = pd.read_parquet(hits[0])
    for c in df.columns:                      # money columns can arrive as Decimal; use floats
        s = df[c].dropna()
        if df[c].dtype == object and len(s) and isinstance(s.iloc[0], Decimal):
            df[c] = df[c].astype(float)
    if "gaming_date" in df:
        df["gaming_date"] = df["gaming_date"].astype(str)
    return df

# Hand-written notes for the machines we discuss (the numbers next to them are computed below).
VERDICT = {
 "M-5076": "Only 49 days on the floor, in the weakest zone. Too early to judge.",
 "M-1158": "Pays out far above its design (hold 3% vs 10%). Inspect the payout setting before any decision.",
 "M-1120": "Back Forty location. Luck is normal.",
 "M-1118": "Only 55 days on the floor. Plays well per hour (high-limit).",
 "M-1238": "Back Forty location, plus some bad luck.",
 "M-1181": "Back Forty location. Luck is normal.",
 "M-1035": "Back Forty location. Luck is normal.",
 "M-1179": "Back Forty location. Luck is normal.",
 "M-1017": "Back Forty location. Slightly lucky, still low.",
 "M-1265": "Back Forty location. Luck is normal.",
 "M-1210": "Unusually low hold (4% vs 5%). Inspect.",
 "M-1010": "Unusually low hold. Inspect.",
 "M-1214": "Unusually high hold (16% vs 10%). Inspect.",
}
EXTRA_FLAGS = ["M-1210", "M-1010", "M-1214"]

def q1_machines(root, out):
    d = load(root, "prepared_daily_view")
    a = d[d["active"]].copy()          # drop the 1,018 rows from before install / after retire
    cal = load(root, "calendar_day")[["gaming_date", "scheduled_seconds"]]
    a = a.merge(cal, on="gaming_date")
    a["sched_h"] = a.scheduled_seconds / 3600           # March 8 has 23 hours, so don't assume 24
    a["lost_h"] = (a.sched_h - a.available_hours).clip(lower=0)

    # Luck score: (actual - expected win) / typical swing for that volatility class.
    a["r"] = a.luck_residual_usd / np.sqrt(a.coin_in_usd.clip(lower=1))
    sd = a.groupby("volatility_class").r.std()
    g = a.groupby("machine_id").agg(
        zone=("zone_name", "last"), vol=("volatility_class", "first"), man=("manufacturer", "first"),
        days=("gaming_date", "count"), ci=("coin_in_usd", "sum"), hw=("house_win_usd", "sum"),
        theo=("theo_usd", "sum"), ah=("available_hours", "sum"), av=("availability_ratio", "mean"),
        faults=("faults_opened", "sum"))
    g["z"] = (g.hw - g.theo) / (g.vol.map(sd) * np.sqrt(g.ci))
    g["hold"] = g.hw / g.ci * 100            # ratio of sums, never a mean of daily ratios
    g["th"] = g.theo / g.ci * 100
    g["cph"] = g.ci / g.ah                   # wagers per available hour = exposure-adjusted
    g.round(4).to_csv(out / "machine_scorecard.csv")
    bottom = list(g.sort_values("hw").head(10).index)    # the manager's naive bottom 10
    mach = [dict(id=m, zone=g.loc[m, "zone"], days=int(g.loc[m, "days"]), hw=round(g.loc[m, "hw"]),
                 theo=round(g.loc[m, "theo"]), z=round(g.loc[m, "z"], 1), hold=round(g.loc[m, "hold"], 1),
                 th=round(g.loc[m, "th"], 1), cph=round(g.loc[m, "cph"]), bot=m in bottom, v=VERDICT.get(m, ""))
            for m in bottom + EXTRA_FLAGS]

    zt = a.groupby("zone_name").agg(m=("machine_id", "nunique"), ci=("coin_in_usd", "sum"),
                                    hw=("house_win_usd", "sum"), theo=("theo_usd", "sum"), ah=("available_hours", "sum"))
    zt["cph"] = zt.ci / zt.ah
    zt.round(4).to_csv(out / "zone_summary.csv")
    zone = [dict(z=k, m=int(r.m), cph=round(r.ci / r.ah, 1), hold=round(r.hw / r.ci * 100, 1),
                 th=round(r.theo / r.ci * 100, 1)) for k, r in zt.iterrows()]
    bk = a.groupby(["zone_name", "bank_id"]).agg(m=("machine_id", "nunique"), ci=("coin_in_usd", "sum"),
                                                 hw=("house_win_usd", "sum"), ah=("available_hours", "sum")).reset_index()
    banks = [dict(z=r.zone_name, b=r.bank_id, m=int(r.m), cph=round(r.ci / r.ah),
                  hold=round(r.hw / r.ci * 100, 1)) for r in bk.itertuples()]

    # Q3: lost play. Lost hours x that machine's own wagers per hour x floor hold.
    a["lost_ci"] = a.lost_h * a.machine_id.map(g.cph)
    hold = a.house_win_usd.sum() / a.coin_in_usd.sum()
    q3 = a.groupby("month").agg(lost_h=("lost_h", "sum"), lost_ci=("lost_ci", "sum"), sched=("sched_h", "sum"))
    q3["lost_hw"] = q3.lost_ci * hold
    q3["pct"] = q3.lost_h / q3.sched * 100
    mf = a.groupby("manufacturer").agg(lost_h=("lost_h", "sum"), sched=("sched_h", "sum"),
                                       faults=("faults_opened", "sum"), m=("machine_id", "nunique"))
    mf["pct"] = mf.lost_h / mf.sched * 100
    mf["faults_per_machine"] = mf.faults / mf.m
    mf["lost_hw"] = a.groupby("manufacturer").lost_ci.sum() * hold
    ops = load(root, "operations")
    ops["dur"] = (pd.to_datetime(ops.available_at_utc) - pd.to_datetime(ops.opened_at_utc)).dt.total_seconds() / 3600
    ops["hr"] = pd.to_datetime(ops.opened_at_local).dt.hour
    h = ops[ops.event_type == "fault_repair"].groupby("hr").agg(n=("dur", "size"), avg=("dur", "mean"))
    return dict(mach=mach, zone=zone, banks=banks, hold=round(hold * 100, 2),
                man=mf.round(2).reset_index().to_dict("records"),
                hr=h.round(2).reset_index().to_dict("records")), q3

def q2_demand(root):
    p = load(root, "property_day")
    p["dt"] = pd.to_datetime(p.gaming_date)
    p["month"] = p.gaming_date.str[:7]
    # event lift = actual vs the average non-event day with the same weekday
    wdm = p[p.events_active.isna()].groupby("local_weekday").coin_in_usd.mean()
    ev = p[p.events_active.notna()].copy()
    ev["exp"] = ev.local_weekday.map(wdm)
    e = ev.groupby("events_active").agg(d=("gaming_date", "count"), act=("coin_in_usd", "sum"), exp=("exp", "sum"))
    events = [dict(n=k, d=int(r.d), l=round((r.act / r.exp - 1) * 100, 1)) for k, r in e.iterrows()]
    # forecast: weekday average vs flat average, train Jan-Apr, test May-Jun (split by TIME, no leakage)
    tr, te = p[p.dt < "2025-05-01"], p[p.dt >= "2025-05-01"].copy()
    wd = tr.groupby("local_weekday").coin_in_usd.mean()
    base = tr.coin_in_usd.mean()
    te["pred"] = te.local_weekday.map(wd)
    fc = dict(mae_w=round((te.coin_in_usd - te.pred).abs().mean()), mae_c=round((te.coin_in_usd - base).abs().mean()),
              base=round(base), wd={k: round(float(v)) for k, v in wd.items()})
    evn = sorted(p.events_active.dropna().unique())
    daily = [[r.gaming_date, round(r.coin_in_usd / 1000, 1), round(r.total_zone_entries_observed / 1000, 2),
              evn.index(r.events_active) if isinstance(r.events_active, str) else -1] for r in p.itertuples()]
    pm = p.groupby("month").agg(ci=("coin_in_usd", "sum"), avail=("available_machine_hours", "sum"),
                                occ=("occupied_machine_hours", "sum"), ent=("total_zone_entries_observed", "sum"),
                                am=("active_machines", "mean"), days=("gaming_date", "count"))
    pm["ci_day"], pm["occ_rate"] = pm.ci / pm.days, pm.occ / pm.avail * 100
    pm["ci_per_mh"], pm["ent_day"] = pm.ci / pm.avail, pm.ent / pm.days
    return dict(events=events, fc=fc, evn=evn, daily=daily), pm

def q4_q5_players(root):
    pp = load(root, "player_period")
    g = pp.groupby("player_id").agg(net=("tracked_net_usd", "sum"), ci=("tracked_coin_in_usd", "sum"))
    loss = (-g.net).clip(lower=0).sort_values(ascending=False)
    conc = {str(q): round(float(loss.iloc[:int(len(g) * q)].sum() / loss.sum() * 100), 1) for q in [.01, .05, .1, .2]}
    c = g.ci.sort_values(ascending=False)
    q4 = dict(players=len(g), losers=int((g.net < 0).sum()), winners=int((g.net > 0).sum()), ci=round(g.ci.sum()),
              net=round(g.net.sum()), conc=conc, ci_top10=round(float(c.iloc[:int(len(g) * .1)].sum() / c.sum() * 100), 1))
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

    D, q3 = q1_machines(args.data, out)
    d2, pm = q2_demand(args.data)
    d45, q5 = q4_q5_players(args.data)
    D.update(d2)
    D["ev"] = D.pop("events")
    D.update(d45)
    # zone-by-month wagers per hour (small table for the zone drill-down)
    d = load(args.data, "prepared_daily_view"); a = d[d["active"]]
    zm = a.groupby(["zone_name", "month"]).apply(lambda x: x.coin_in_usd.sum() / x.available_hours.sum()).unstack().round(0)
    D["zm"] = {k: list(v) for k, v in zm.iterrows()}
    D["mo"] = [dict(m=m, ci_day=pm.loc[m, "ci_day"], occ_rate=pm.loc[m, "occ_rate"], ci_per_mh=pm.loc[m, "ci_per_mh"],
                    ent_day=pm.loc[m, "ent_day"], am=pm.loc[m, "am"], lost_h=q3.loc[m, "lost_h"],
                    lost_hw=q3.loc[m, "lost_hw"], pct=q3.loc[m, "pct"], cc=q5.loc[m, "cc"], cf=q5.loc[m, "cf"],
                    ccp=q5.loc[m, "ccp"], ppc=q5.loc[m, "ppc"], rr=q5.loc[m, "rr"]) for m in sorted(pm.index)]
    html = (HERE / "dashboard_template.html").read_text(encoding="utf-8")
    html = html.replace("__D__", json.dumps(D, separators=(",", ":"), default=float))
    (out / "dashboard.html").write_text(html, encoding="utf-8")
    print("Done. Open", out / "dashboard.html", "in your browser.")
    print("Weekday forecast error $%s vs flat baseline $%s" % (f"{D['fc']['mae_w']:,}", f"{D['fc']['mae_c']:,}"))

if __name__ == "__main__":
    main()
