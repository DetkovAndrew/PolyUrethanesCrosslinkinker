import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance
import numpy as np
import sys
import random
from collections import defaultdict

# ==============================================================================
# --- ATOM TYPES ---
TYPE_OH_O, TYPE_OH_H = 58, 27
TYPE_NCO_C, TYPE_NCO_N, TYPE_NCO_O = 4, 39, 59

# --- NEW ATOM TYPES (GAFF) ---
NEW_O_ETHER, NEW_H_AMIDE, NEW_C_CARB, NEW_N_AMIDE, NEW_O_CARB = 59, 26, 1, 36, 57

# Идеальные заряды ядра (RESP) для сшитого состояния
CORE_CHARGES = {"o": -0.4409, "c": 0.7281, "n": -0.5189, "oc": -0.5920, "h": 0.3125}

MAX_BONDS_PER_CYCLE = 20
CYLINDER_RADIUS = 1.0 
# ==============================================================================

def load_dynamic_topology(ttree_path="ttree_assignments.txt"):
    mapping = {}
    try:
        with open(ttree_path, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[0].startswith("@/"): 
                    mapping[parts[0]] = int(parts[1])
    except FileNotFoundError:
        print(f"ERROR: Topology file {ttree_path} not found!")
        sys.exit(1)
    
    def find_item(item_type, *atoms):
        s1, s2 = f"@/{item_type}:GAFF/" + "-".join(atoms), f"@/{item_type}:GAFF/" + "-".join(reversed(atoms))
        if s1 in mapping: return mapping[s1]
        if s2 in mapping: return mapping[s2]
        raise KeyError(f"Cannot find '{s1}' or '{s2}' in ttree!")

    return {
        "Bonds": {"C_OS": find_item("bond", "c", "os"), "C_N": find_item("bond", "c", "n"), "C_O": find_item("bond", "c", "o"), "N_HN": find_item("bond", "hn", "n")},
        "Angles": {"C3_OS_C": find_item("angle", "c", "os", "c3"), "N_C_OS": find_item("angle", "n", "c", "os"), "O_C_OS": find_item("angle", "o", "c", "os"), "N_C_O": find_item("angle", "n", "c", "o"), "C_N_C3": find_item("angle", "c", "n", "c3"), "HN_N_C": find_item("angle", "hn", "n", "c"), "HN_N_C3": find_item("angle", "hn", "n", "c3")},
        "Dihedrals": {"C_OS": find_item("dihedral", "X", "c", "os", "X"), "C_N": find_item("dihedral", "X", "c", "n", "X"), "C3_OS": find_item("dihedral", "X", "c3", "os", "X"), "C3_N": find_item("dihedral", "X", "c3", "n", "X")},
        "Impropers": {"CENTER_C": find_item("improper", "X", "X", "c", "o"), "CENTER_N": find_item("improper", "X", "X", "n", "hn")}
    }

# Загружаем топологию из текущей директории
try: TOPOLOGY_CONFIG = load_dynamic_topology("ttree_assignments.txt")
except Exception as e: print(e); sys.exit(1)

def mic_vector(p1, p2, box):
    d = p1 - p2
    return d - box[:3] * np.round(d / box[:3])

def check_steric_clash(u, p_o, p_c, exclude_ids):
    box = u.dimensions
    vec_oc = mic_vector(p_c, p_o, box)
    length = np.linalg.norm(vec_oc)
    dir_oc = vec_oc / length
    midpoint = p_o + vec_oc * 0.5
    search_radius = (length / 2) + CYLINDER_RADIUS + 1.0
    neighbors = u.select_atoms(f"point {midpoint[0]} {midpoint[1]} {midpoint[2]} {search_radius}")
    
    for atom in neighbors:
        if atom.id in exclude_ids or int(atom.type) == TYPE_OH_H: continue
        v_atom = mic_vector(atom.position, p_o, box)
        t = np.dot(v_atom, dir_oc)
        if 0 < t < length and np.linalg.norm(v_atom - t * dir_oc) < CYLINDER_RADIUS:
            return True
    return False

def get_planar_h_position(pos_n, pos_c_carb, pos_c_pdi, box):
    v1 = mic_vector(pos_c_carb, pos_n, box)
    v2 = mic_vector(pos_c_pdi, pos_n, box)
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    v_h = -(v1 + v2)
    norm = np.linalg.norm(v_h)
    if norm < 1e-5:
        rand_vec = np.random.randn(3)
        v_h = rand_vec - np.dot(rand_vec, v1) * v1
        norm = np.linalg.norm(v_h)
    v_h /= norm
    return pos_n + v_h * 1.01

def crosslink_system(input_file, output_file, cutoff):
    print(f"\n=== CROSSLINKING (Physical Cutoff: {cutoff:.2f} A) ===")
    
    # Исправленный формат atom_style для MDAnalysis 2.x+
    u = mda.Universe(input_file, format='DATA', atom_style="id mol type q x y z")
    
    o_atoms = u.select_atoms(f"type {TYPE_OH_O}")
    c_atoms = u.select_atoms(f"type {TYPE_NCO_C}")

    pairs, dists = capped_distance(o_atoms.positions, c_atoms.positions, max_cutoff=cutoff, box=u.dimensions, return_distances=True)
    order = np.argsort(dists)

    reacted, reactions = set(), []

    for idx in order:
        if len(reactions) >= MAX_BONDS_PER_CYCLE: break
        
        o_idx, c_idx = pairs[idx]
        atom_o, atom_c = o_atoms[o_idx], c_atoms[c_idx]
        o_id, c_id = int(atom_o.id), int(atom_c.id)

        if o_id in reacted or c_id in reacted: continue

        h_neigh = [a for a in atom_o.bonded_atoms if int(a.type) == TYPE_OH_H]
        n_neigh = [a for a in atom_c.bonded_atoms if int(a.type) == TYPE_NCO_N]
        o_neigh = [a for a in atom_c.bonded_atoms if int(a.type) == TYPE_NCO_O]
        if not (h_neigh and n_neigh and o_neigh): continue

        atom_h, atom_n, atom_oc = h_neigh[0], n_neigh[0], o_neigh[0]
        c_pdi = [a for a in atom_n.bonded_atoms if int(a.id) != c_id]
        if not c_pdi: continue
        atom_c_pdi = c_pdi[0]

        if check_steric_clash(u, atom_o.position, atom_c.position, {o_id, c_id, int(atom_h.id), int(atom_n.id), int(atom_oc.id)}): continue

        new_h = get_planar_h_position(atom_n.position, atom_c.position, atom_c_pdi.position, u.dimensions)
        reactions.append({"o": o_id, "h": int(atom_h.id), "c": c_id, "n": int(atom_n.id), "oc": int(atom_oc.id), "new_h": new_h})
        reacted.add(o_id); reacted.add(c_id)

    print(f"Reactions queued this cycle: {len(reactions)}")
    if not reactions: 
        with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
            f_out.write(f_in.read())
        return

    with open(input_file) as f: lines = f.readlines()
    sections_order = ["Masses", "Atoms", "Velocities", "Bonds", "Angles", "Dihedrals", "Impropers"]
    sec_data = {s: [] for s in sections_order}
    header_lines, current_sec = [], None
    
    for line in lines:
        s_line = line.split('#')[0].strip()
        if s_line in sections_order: current_sec = s_line; sec_data[current_sec].append(line); continue
        if current_sec is None: header_lines.append(line)
        else: sec_data[current_sec].append(line)

    adj = defaultdict(set)
    bonds_raw = []
    for line in sec_data["Bonds"]:
        parts = line.split()
        if len(parts) >= 4 and parts[0].isdigit():
            b_id, b_type, a1, a2 = map(int, parts[:4])
            adj[a1].add(a2); adj[a2].add(a1); bonds_raw.append((b_id, b_type, a1, a2))

    old_charges = {int(p.split()[0]): float(p.split()[3]) for p in sec_data["Atoms"] if p.split() and p.split()[0].isdigit()}
    r_extended, charge_updates = [], {}
    
    for r in reactions:
        O_eth, C_carb, N_am, O_carb, H_am = r["o"], r["c"], r["n"], r["oc"], r["h"]
        C_poly = next((a for a in adj[O_eth] if a != H_am), None)
        C_pdi  = next((a for a in adj[N_am] if a != C_carb), None)
        
        if C_poly and C_pdi:
            q_old_cluster = sum(old_charges[x] for x in [O_eth, C_carb, N_am, O_carb, H_am, C_poly, C_pdi])
            q_new_core = sum(CORE_CHARGES.values())
            remainder = q_old_cluster - q_new_core
            charge_updates.update({
                O_eth: CORE_CHARGES["o"], C_carb: CORE_CHARGES["c"], 
                N_am: CORE_CHARGES["n"], O_carb: CORE_CHARGES["oc"], 
                H_am: CORE_CHARGES["h"],
                C_poly: remainder / 2.0, C_pdi: remainder / 2.0
            })
        r_extended.append({"O_eth": O_eth, "C_carb": C_carb, "N_am": N_am, "O_carb": O_carb, "H_am": H_am, "C_poly": C_poly, "C_pdi": C_pdi})
        if H_am in adj[O_eth]: adj[O_eth].remove(H_am)
        if O_eth in adj[H_am]: adj[H_am].remove(O_eth)
        adj[O_eth].add(C_carb); adj[C_carb].add(O_eth)
        adj[N_am].add(H_am);    adj[H_am].add(N_am)

    visited = set()
    mol_mapping = {}
    current_mol_id = 1
    for atom_id in sorted(adj.keys()):
        if atom_id not in visited:
            stack = [atom_id]
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node); mol_mapping[node] = current_mol_id
                    stack.extend(adj[node] - visited)
            current_mol_id += 1

    # Исправленное заполнение словарей (без SyntaxError)
    type_change = {}
    pos_change = {}
    for r in reactions:
        type_change[r["o"]] = NEW_O_ETHER
        type_change[r["h"]] = NEW_H_AMIDE
        type_change[r["c"]] = NEW_C_CARB
        type_change[r["n"]] = NEW_N_AMIDE
        type_change[r["oc"]] = NEW_O_CARB
        pos_change[r["h"]] = r["new_h"]

    new_atoms = []
    for line in sec_data["Atoms"]:
        parts = line.split()
        if parts and parts[0].isdigit():
            a_id = int(parts[0])
            m_id = mol_mapping.get(a_id, int(parts[1]))
            t_id = type_change.get(a_id, int(parts[2]))
            q = charge_updates.get(a_id, float(parts[3]))
            pos = pos_change.get(a_id, (float(parts[4]), float(parts[5]), float(parts[6])))
            new_line = f"{a_id} {m_id} {t_id} {q:.6f} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}"
            if len(parts) > 7: new_line += " " + " ".join(parts[7:])
            new_atoms.append(new_line + "\n")
        else:
            new_atoms.append(line)
    sec_data["Atoms"] = new_atoms

    bonds_to_delete = {tuple(sorted((r["O_eth"], r["H_am"]))) for r in r_extended}
    bonds_out = sec_data["Bonds"][:2] if len(sec_data["Bonds"])>1 else ["Bonds\n", "\n"]
    max_b_id = max((b[0] for b in bonds_raw), default=0)
    for b_id, b_type, a1, a2 in bonds_raw:
        if tuple(sorted((a1, a2))) in bonds_to_delete: continue
        for r in r_extended:
            if tuple(sorted((a1, a2))) == tuple(sorted((r["C_carb"], r["N_am"]))): b_type = TOPOLOGY_CONFIG["Bonds"]["C_N"]
            elif tuple(sorted((a1, a2))) == tuple(sorted((r["C_carb"], r["O_carb"]))): b_type = TOPOLOGY_CONFIG["Bonds"]["C_O"]
        bonds_out.append(f"{b_id} {b_type} {a1} {a2}\n")
    for r in r_extended:
        max_b_id += 1; bonds_out.append(f"{max_b_id} {TOPOLOGY_CONFIG['Bonds']['C_OS']} {r['C_carb']} {r['O_eth']}\n")
        max_b_id += 1; bonds_out.append(f"{max_b_id} {TOPOLOGY_CONFIG['Bonds']['N_HN']} {r['N_am']} {r['H_am']}\n")
    sec_data["Bonds"] = bonds_out + ["\n"]

    def init_block(sec_name):
        block = sec_data.get(sec_name, [])
        if not block: return [f"{sec_name}\n", "\n"], 0
        return block[:2] if len(block) > 1 else [block[0], "\n"], max([int(l.split()[0]) for l in block if l.split() and l.split()[0].isdigit()] + [0])

    angles_out, max_a_id = init_block("Angles")
    seen_angles = set()
    for line in sec_data["Angles"][2:]:
        parts = line.split()
        if not parts or not parts[0].isdigit(): (angles_out.append(line) if line.strip() else None); continue
        a_id, a_type, a1, a2, a3 = map(int, parts[:5])
        skip = False
        for r in r_extended:
            grp = {a1, a2, a3}
            if r["H_am"] in grp and r["O_eth"] in grp: skip = True
            if grp == {r["N_am"], r["C_carb"], r["O_carb"]}: a_type = TOPOLOGY_CONFIG["Angles"]["N_C_O"]
            elif grp == {r["C_pdi"], r["N_am"], r["C_carb"]}: a_type = TOPOLOGY_CONFIG["Angles"]["C_N_C3"]
        if not skip:
            tup = tuple(sorted((a1, a3))) + (a2,); seen_angles.add(tup); angles_out.append(f"{a_id} {a_type} {a1} {a2} {a3}\n")
    for r in r_extended:
        for a_type, a1, a2, a3 in [(TOPOLOGY_CONFIG["Angles"]["C3_OS_C"], r["C_poly"], r["O_eth"], r["C_carb"]), (TOPOLOGY_CONFIG["Angles"]["N_C_OS"], r["N_am"], r["C_carb"], r["O_eth"]), (TOPOLOGY_CONFIG["Angles"]["O_C_OS"], r["O_carb"], r["C_carb"], r["O_eth"]), (TOPOLOGY_CONFIG["Angles"]["HN_N_C"], r["H_am"], r["N_am"], r["C_carb"]), (TOPOLOGY_CONFIG["Angles"]["HN_N_C3"], r["H_am"], r["N_am"], r["C_pdi"])]:
            if a1 and a2 and a3:
                tup = tuple(sorted((a1, a3))) + (a2,)
                if tup not in seen_angles: seen_angles.add(tup); max_a_id += 1; angles_out.append(f"{max_a_id} {a_type} {a1} {a2} {a3}\n")
    sec_data["Angles"] = angles_out + ["\n"]

    dihedrals_out, max_d_id = init_block("Dihedrals")
    seen_dihedrals = set()
    for line in sec_data["Dihedrals"][2:]:
        parts = line.split()
        if not parts or not parts[0].isdigit(): (dihedrals_out.append(line) if line.strip() else None); continue
        d_id, d_type, a1, a2, a3, a4 = map(int, parts[:6])
        cb = tuple(sorted((a2, a3)))
        skip = False
        for r in r_extended:
            if r["H_am"] in {a1, a2, a3, a4}: skip = True
            if cb in [tuple(sorted((r["C_carb"], r["O_eth"]))), tuple(sorted((r["C_carb"], r["N_am"]))), tuple(sorted((r["O_eth"], r["C_poly"]))), tuple(sorted((r["N_am"], r["C_pdi"]))), tuple(sorted((r["C_carb"], r["O_carb"])))]: skip = True
        if not skip:
            tup = (a1, a2, a3, a4) if a1 < a4 else (a4, a3, a2, a1); seen_dihedrals.add(tup); dihedrals_out.append(f"{d_id} {d_type} {a1} {a2} {a3} {a4}\n")
    def add_dihedrals(a2, a3, d_type):
        nonlocal max_d_id
        for a1 in adj[a2]:
            if a1 == a3: continue
            for a4 in adj[a3]:
                if a4 == a2 or a4 == a1: continue
                tup = (a1, a2, a3, a4) if a1 < a4 else (a4, a3, a2, a1)
                if tup not in seen_dihedrals: seen_dihedrals.add(tup); max_d_id += 1; dihedrals_out.append(f"{max_d_id} {d_type} {a1} {a2} {a3} {a4}\n")
    for r in r_extended:
        if r["C_poly"]: add_dihedrals(r["C_poly"], r["O_eth"], TOPOLOGY_CONFIG["Dihedrals"]["C3_OS"])
        if r["C_pdi"]:  add_dihedrals(r["C_pdi"], r["N_am"], TOPOLOGY_CONFIG["Dihedrals"]["C3_N"])
        add_dihedrals(r["O_eth"], r["C_carb"], TOPOLOGY_CONFIG["Dihedrals"]["C_OS"])
        add_dihedrals(r["C_carb"], r["N_am"], TOPOLOGY_CONFIG["Dihedrals"]["C_N"])
    sec_data["Dihedrals"] = dihedrals_out + ["\n"]

    impropers_out, max_i_id = init_block("Impropers")
    for line in sec_data["Impropers"][2:]:
        parts = line.split()
        if not parts or not parts[0].isdigit(): (impropers_out.append(line) if line.strip() else None); continue
        i_id, i_type, a1, a2, a3, a4 = map(int, parts[:6])
        skip = False
        for r in r_extended:
            if r["C_carb"] in {a1, a2, a3, a4} or r["N_am"] in {a1, a2, a3, a4}: skip = True
        if not skip: impropers_out.append(f"{i_id} {i_type} {a1} {a2} {a3} {a4}\n")
    for r in r_extended:
        max_i_id += 1; impropers_out.append(f"{max_i_id} {TOPOLOGY_CONFIG['Impropers']['CENTER_C']} {r['O_eth']} {r['N_am']} {r['C_carb']} {r['O_carb']}\n")
        if r["C_pdi"]: max_i_id += 1; impropers_out.append(f"{max_i_id} {TOPOLOGY_CONFIG['Impropers']['CENTER_N']} {r['C_pdi']} {r['C_carb']} {r['N_am']} {r['H_am']}\n")
    sec_data["Impropers"] = impropers_out + ["\n"]

    counts = {"bonds": len([l for l in sec_data["Bonds"] if l.split() and l.split()[0].isdigit()]), "angles": len([l for l in sec_data["Angles"] if l.split() and l.split()[0].isdigit()]), "dihedrals": len([l for l in sec_data["Dihedrals"] if l.split() and l.split()[0].isdigit()]), "impropers": len([l for l in sec_data["Impropers"] if l.split() and l.split()[0].isdigit()])}
    new_lines = []
    for line in header_lines:
        parts = line.split()
        if len(parts) == 2 and parts[1] in counts: new_lines.append(f"{counts[parts[1]]} {parts[1]}\n")
        else: new_lines.append(line)
    for sec in sections_order: new_lines.extend(sec_data[sec])
    with open(output_file, "w") as f: f.writelines(new_lines)
    print("=== Topology & Molecules updated successfully ===")

if __name__ == "__main__":
    if len(sys.argv) < 4: sys.exit(1)
    crosslink_system(sys.argv[1], sys.argv[2], float(sys.argv[3]))