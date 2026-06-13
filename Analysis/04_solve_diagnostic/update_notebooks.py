#!/usr/bin/env python3
"""
Update all solving .ipynb notebooks with shoulder_constrained logic:

1. Scale ES-ES line s_nom by 0.35×
2. Set s_max_pu = 0.5 (50% capacity)
3. No capacity expansion
4. Hard-coded MCs and ramping
5. Hydro at 50% SOC with max_hours reductions
6. No interconnectors
7. Apr-Jun 2024 (3-month shoulder period)
8. Updated solver settings
"""

import json
import os
import re
import shutil

# ── Configuration ──────────────────────────────────────────────────────────

TRANS_BLOCK = """
# ── TRANSMISSION CONSTRAINT: Break the copper plate ────────────────────
TRANS_FACTOR = 0.35
es_lines = n.lines[
    n.lines.bus0.astype(str).str[:2].isin(['ES']) &
    n.lines.bus1.astype(str).str[:2].isin(['ES'])
].index
print(f'Scaling {len(es_lines)} ES-ES lines by {TRANS_FACTOR}×')
n.lines.loc[es_lines, 's_nom'] *= TRANS_FACTOR
n.lines.loc[es_lines, 's_nom_extendable'] = False
n.lines['s_max_pu'] = 0.5
print(f's_max_pu set to {n.lines.s_max_pu.iloc[0]} for all lines')
""".strip()

NO_EXPANSION_BLOCK = """
# --- Disable all capacity expansion (dispatch-only) ---
for attr in ['generators', 'links', 'stores', 'storage_units']:
    df = getattr(n, attr)
    if 'p_nom_extendable' in df.columns:
        df.loc[df.p_nom_extendable, 'p_nom_extendable'] = False
if 's_nom_extendable' in n.lines.columns:
    n.lines.loc[n.lines.s_nom_extendable, 's_nom_extendable'] = False
""".strip()

HYDRO_BLOCK = """
# --- Set hydro state_of_charge_initial to 50% ---
es_hyd = (n.storage_units.bus.map(lambda b: str(b).startswith('ES')) &
          (n.storage_units['carrier'] == 'hydro'))
n.storage_units.loc[es_hyd, 'state_of_charge_initial'] = (
    0.50 * n.storage_units.loc[es_hyd, 'p_nom'] * n.storage_units.loc[es_hyd, 'max_hours']
)
print(f'Hydro SOC init set to 50% for {es_hyd.sum()} units')
""".strip()

IC_BLOCK = """
# --- Border Lockdown: isolate Spain from FR/PT ---
cross_border_links = n.links[
    (n.links.bus0.str.contains('ES') & ~n.links.bus1.str.contains('ES')) |
    (~n.links.bus0.str.contains('ES') & n.links.bus1.str.contains('ES'))
].index
n.links.loc[cross_border_links, 'p_nom'] = 0
n.links.loc[cross_border_links, 'p_nom_extendable'] = False
print(f'Disabled {len(cross_border_links)} cross-border links')
""".strip()

# ── Notebook-specific modifications ────────────────────────────────────────

def update_notebook(notebook_path, dry_run=False):
    """Update a single notebook with shoulder_constrained logic."""
    
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    modified = False
    changes = []
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        
        source = ''.join(cell['source'])
        
        # ── 1. Update date parameters ──────────────────────────────────
        
        # Update START_DATE
        if re.search(r"START_DATE\s*=\s*['\"]", source):
            new_source = re.sub(
                r"(START_DATE\s*=\s*['\"])[^'\"]+(['\"])",
                r"\g<1>2024-04-01\g<2>",
                source
            )
            if new_source != source:
                changes.append(f"  Cell {i}: Updated START_DATE to 2024-04-01")
                source = new_source
                modified = True
        
        # Update N_DAYS
        if re.search(r"N_DAYS\s*=\s*\d+", source):
            new_source = re.sub(
                r"(N_DAYS\s*=\s*)\d+",
                r"\g<1>91",
                source
            )
            if new_source != source:
                changes.append(f"  Cell {i}: Updated N_DAYS to 91")
                source = new_source
                modified = True
        
        # Update start_time / end_time patterns
        if re.search(r"start_time\s*=\s*['\"]2024-0[0-9]", source):
            new_source = re.sub(
                r"(start_time\s*=\s*['\"])[^'\"]+(['\"])",
                r"\g<1>2024-04-01 00:00\g<2>",
                source
            )
            if new_source != source:
                changes.append(f"  Cell {i}: Updated start_time to 2024-04-01 00:00")
                source = new_source
                modified = True
        
        if re.search(r"end_time\s*=\s*['\"]2024-0[0-9]", source):
            new_source = re.sub(
                r"(end_time\s*=\s*['\"])[^'\"]+(['\"])",
                r"\g<1>2024-07-01 23:00\g<2>",
                source
            )
            if new_source != source:
                changes.append(f"  Cell {i}: Updated end_time to 2024-07-01 23:00")
                source = new_source
                modified = True
        
        # Update WINDOW_START
        if re.search(r"WINDOW_START\s*=\s*['\"]2024-0[0-9]", source):
            new_source = re.sub(
                r"(WINDOW_START\s*=\s*['\"])[^'\"]+(['\"])",
                r"\g<1>2024-04-01 00:00\g<2>",
                source
            )
            if new_source != source:
                changes.append(f"  Cell {i}: Updated WINDOW_START to 2024-04-01 00:00")
                source = new_source
                modified = True
        
        # Update WINDOW_HOURS (91 days = 2184 hours)
        if re.search(r"WINDOW_HOURS\s*=\s*\d+\s*#", source):
            new_source = re.sub(
                r"(WINDOW_HOURS\s*=\s*)\d+(\s*#.*)",
                r"\g<1>2184\g<2>",
                source
            )
            if new_source != source:
                changes.append(f"  Cell {i}: Updated WINDOW_HOURS to 2184 (91 days)")
                source = new_source
                modified = True
        elif re.search(r"WINDOW_HOURS\s*=\s*\d+\s*$", source):
            new_source = re.sub(
                r"(WINDOW_HOURS\s*=\s*)\d+",
                r"\g<1>2184",
                source
            )
            if new_source != source:
                changes.append(f"  Cell {i}: Updated WINDOW_HOURS to 2184 (91 days)")
                source = new_source
                modified = True
        
        # Update window = N * 24
        if re.search(r"window\s*=\s*\d+\s*\*\s*24", source):
            new_source = re.sub(
                r"(window\s*=\s*)\d+(\s*\*\s*24)",
                r"\g<1>91\g<2>",
                source
            )
            if new_source != source:
                changes.append(f"  Cell {i}: Updated window to 91*24")
                source = new_source
                modified = True
        
        # Update END_DATE
        if re.search(r"END_DATE\s*=\s*['\"]2024-0[0-9]", source):
            new_source = re.sub(
                r"(END_DATE\s*=\s*['\"])[^'\"]+(['\"])",
                r"\g<1>2024-07-01\g<2>",
                source
            )
            if new_source != source:
                changes.append(f"  Cell {i}: Updated END_DATE to 2024-07-01")
                source = new_source
                modified = True
        
        # ── 2. Update solver settings ──────────────────────────────────
        
        # Update TimeLimit (notebooks with 600s → 1800s, 7200s stays)
        if re.search(r"'TimeLimit':\s*600", source) and 'optimize(' in source:
            new_source = re.sub(
                r"'TimeLimit':\s*600",
                "'TimeLimit': 1800",
                source
            )
            if new_source != source:
                changes.append(f"  Cell {i}: Updated TimeLimit 600→1800")
                source = new_source
                modified = True
        
        # Add Threads: 5 if not present and optimize is called
        if 'optimize(' in source and "'Threads'" not in source and 'solver_options' in source:
            new_source = re.sub(
                r"('TimeLimit':\s*\d+)",
                r"\g<1>, 'Threads': 5",
                source
            )
            if new_source != source:
                changes.append(f"  Cell {i}: Added Threads: 5")
                source = new_source
                modified = True
        
        # ── 3. Inject transmission constraint block BEFORE solve ────────
        
        if 'n.optimize(' in source or 'n_sub.optimize(' in source:
            # Check if transmission constraint already exists
            if 'TRANS_FACTOR' not in source and 's_max_pu' not in source:
                # Find the right place to inject - before the optimize call
                # We need to add the block before the print/optimize section
                
                # Build the full injection block
                injection = ""
                
                # Add no-expansion if not present
                if 'p_nom_extendable' not in source and 'p_nom_extendable' not in ''.join(
                    [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
                ):
                    injection += NO_EXPANSION_BLOCK + "\n\n"
                
                # Add transmission constraint
                injection += TRANS_BLOCK + "\n\n"
                
                # Add hydro SOC init if not present
                if 'state_of_charge_initial' not in source:
                    injection += HYDRO_BLOCK + "\n\n"
                
                # Add IC lockdown if not present
                if 'cross_border_links' not in source:
                    injection += IC_BLOCK + "\n\n"
                
                if injection:
                    # Insert injection before the optimize call line
                    # Find the line with optimize(
                    lines = source.split('\n')
                    new_lines = []
                    injected = False
                    for line in lines:
                        if 'optimize(' in line and not injected:
                            new_lines.append(injection)
                            injected = True
                        new_lines.append(line)
                    source = '\n'.join(new_lines)
                    changes.append(f"  Cell {i}: Injected transmission constraint block")
                    modified = True
        
        # ── 4. Fix TypeError in 07_marginal_cost_analysis ──────────────
        # Fix: n.snapshots[start_time:end_time] → use get_loc
        if 'n.snapshots[start_time:end_time]' in source:
            new_source = source.replace(
                "n_sub.set_snapshots(n.snapshots[start_time:end_time])",
                "start_idx = n.snapshots.get_loc(start_time)\nend_idx = n.snapshots.get_loc(end_time)\nn_sub.set_snapshots(n.snapshots[start_idx:end_idx])"
            )
            if new_source != source:
                changes.append(f"  Cell {i}: Fixed TypeError (get_loc instead of string slice)")
                source = new_source
                modified = True
        
        # Update cell source
        if modified:
            nb['cells'][i]['source'] = source.split('\n')
    
    # ── Write back ─────────────────────────────────────────────────────
    if modified and not dry_run:
        # Backup original
        backup_path = notebook_path + '.bak'
        if not os.path.exists(backup_path):
            shutil.copy2(notebook_path, backup_path)
            changes.append(f"  Backup: {backup_path}")
        
        with open(notebook_path, 'w') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        
        print(f"\n✅ Updated: {notebook_path}")
        for c in changes:
            print(c)
        return True
    elif modified and dry_run:
        print(f"\n🔍 Dry-run: {notebook_path}")
        for c in changes:
            print(c)
        return True
    else:
        print(f"\n⏭️  No changes: {notebook_path}")
        return False


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Update notebooks with shoulder_constrained logic')
    parser.add_argument('--dry-run', action='store_true', help='Show what would change without modifying')
    parser.add_argument('--notebook', type=str, help='Update only a specific notebook path')
    args = parser.parse_args()
    
    # Notebooks to update (those that call optimize())
    notebooks = [
        'Analysis/03_prices/03_spain_price_analysis.ipynb',
        'Analysis/02_dispatch/02_dispatch_by_country.ipynb',
        'Analysis/04_solve_diagnostic/04_solve_diagnostic.ipynb',
        'Analysis/04_cross_border/04_cross_border_flows.ipynb',
        'Analysis/04_cross_border/04_cross_border_flows_executed.ipynb',
        'Analysis/05_load/05_load_analysis.ipynb',
        'Analysis/06_curtailment/06_curtailment_analysis.ipynb',
        'Analysis/07_marginal_costs/07_marginal_cost_analysis.ipynb',
        'Analysis/07_marginal_costs/07b_mc_calibration.ipynb',
        'Analysis/07_marginal_costs/08_carbon_cost_interactive.ipynb',
        'Analysis/07_marginal_costs/09_gurobi_solver_walkthrough.ipynb',
        'Analysis/10_transmission/10_transmission_flow_map.ipynb',
        'Analysis/11_parameter_calibration/11_parameter_calibration.ipynb',
    ]
    
    if args.notebook:
        notebooks = [args.notebook]
    
    updated = 0
    for nb_path in notebooks:
        if os.path.exists(nb_path):
            if update_notebook(nb_path, dry_run=args.dry_run):
                updated += 1
        else:
            print(f"\n❌ Not found: {nb_path}")
    
    print(f"\n{'='*60}")
    print(f"Done. {updated}/{len(notebooks)} notebooks updated.")
    if args.dry_run:
        print("(dry run — no files modified)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
