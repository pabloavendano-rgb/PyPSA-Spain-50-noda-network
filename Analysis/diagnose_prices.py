"""
Quick diagnostic: check nodal price distribution across ES buses.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "Analysis"))

from config import MODEL_CONFIG
from refinery import apply_non_linear_refinements

cfg = MODEL_CONFIG
val = cfg["validation"]

import pypsa
n = pypsa.Network(str(ROOT / val["network_path"]))
n = apply_non_linear_refinements(n, cfg)

start = pd.Timestamp(val["start_date"])
n_days = int(val["n_days"])
end = start + pd.Timedelta(hours=n_days * 24 - 1)
snap = n.snapshots[(n.snapshots >= start) & (n.snapshots <= end)]
n.set_snapshots(snap)

n.optimize(solver_name=val["solver"])

# Nodal price analysis
es_buses = n.buses.index[n.buses.index.str.startswith("ES")]
price_cols = [b for b in es_buses if b in n.buses_t.marginal_price.columns]

print(f"\nES buses with prices: {len(price_cols)}")
print(f"\n{'Bus':<20} {'Mean Price':>12} {'Min Price':>12} {'Max Price':>12} {'p95':>12}")
print("-" * 68)
for b in price_cols:
    p = n.buses_t.marginal_price[b]
    print(f"{b:<20} {p.mean():>10.1f} €/MWh  {p.min():>10.1f}  {p.max():>10.1f}  {np.percentile(p, 95):>10.1f}")

# Check which buses have load shedding
print(f"\nLoad shedding check:")
for g in n.generators.index:
    if "VOLL" in g or n.generators.loc[g, "carrier"] == "load_shedding":
        if g in n.generators_t.p.columns:
            shed = n.generators_t.p[g].sum()
            if shed > 0.1:
                print(f"  {g}: {shed:.1f} MWh shed over period")

# Check which generators are running at/near capacity
print(f"\nTop 20 generators by avg dispatch (ES):")
es_gen = n.generators[n.generators["bus"].isin(es_buses)]
dispatch = {}
for g in es_gen.index:
    if g in n.generators_t.p.columns:
        dispatch[g] = n.generators_t.p[g].mean()
sorted_d = sorted(dispatch.items(), key=lambda x: -x[1])[:20]
for g, avg in sorted_d:
    p_nom = n.generators.loc[g, "p_nom"]
    mc = n.generators.loc[g, "marginal_cost"]
    carrier = n.generators.loc[g, "carrier"]
    util = avg / p_nom * 100 if p_nom > 0 else 0
    print(f"  {g:<35} carrier={carrier:<12} p_nom={p_nom:>8.0f} avg={avg:>8.0f} util={util:>5.1f}% MC={mc:.0f}")
