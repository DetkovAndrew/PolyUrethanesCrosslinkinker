# PolyUrethanesCrosslinkinker
Автоматизированный пайплайн для МД-моделирования процесса сшивки полимеров (LAMMPS + MDAnalysis). Включает итеративный алгоритм образования связей (OH + NCO) с проверкой стерических препятствий, динамическое перестроение топологии (GAFF/RESP) и температурный отжиг для обхода стеклования.

# LAMMPS Polymer Cross-linking Pipeline

An automated molecular dynamics pipeline written in Python and LAMMPS to simulate the chemical cross-linking of polymers (e.g., the formation of polyurethane networks from OH and NCO reactive groups). 

The script iteratively cross-links the system, dynamically updates the topology (atom types, charges, bonds, angles, dihedrals, and impropers), and runs intermediate relaxation steps (simulated annealing) while progressively increasing the capture radius of the reactive groups.

## 🌟 Key Features

* **Dynamic Capture Radius:** Gradually increases the cutoff distance from 3.5 Å to 6.0 Å if the reaction rate drops.
* **Reaction Filtering:** Performs steric clash checks considering molecular geometry to prevent unphysical overlaps before forming a new bond.
* **On-the-fly Topology Rebuilding:** Uses `MDAnalysis` and graph traversal to dynamically generate and update topological parameters based on `ttree_assignments.txt` (GAFF force field).
* **Charge Recalculation:** Adjusts partial charges (RESP) for the cross-linked regions while strictly maintaining the overall neutrality of the system.
* **Smart Relaxation (LAMMPS):** 
  * Soft-core relaxation (`nve/limit`) to prevent the system from "exploding" due to overlapping atoms after creating new bonds.
  * Standard simulated annealing (`relax.in`) to promote chain diffusion.
  * High-temperature rescue annealing (`extra.in`) — automatically triggered if the conversion rate drops to zero (system vitrification) at the maximum cutoff.

## 🛠 Dependencies

* **LAMMPS** (Must be compiled with the `GPU` package, as the script uses `-pk gpu 1 -sf gpu` flags).
* **Python 3.x**
* **Python Libraries:**
  * `MDAnalysis`
  * `numpy`

## 📂 File Structure

* `master_loop.py` — The main orchestrator. Manages the cross-linking cycles, controls the cutoff logic, and invokes LAMMPS.
* `crosslink.py` — The core algorithm. Parses the `.data` file, finds reactive pairs, checks steric hindrances, updates bond graphs, and rewrites the topology.
* `relax.in` — Base LAMMPS relaxation script (heating to 400 K, diffusion, quenching to 300 K).
* `extra.in` — "Rescue" LAMMPS relaxation for stuck conversion (heating to 500 K, extended diffusion time).
* `system.in.init` / `system.in.settings` — Base force field settings (must be provided by the user).
* `ttree_assignments.txt` — File containing reference topology parameters (GAFF).
* `equilibrated.data` — The initial uncross-linked system data file.

## 🚀 Usage

1. Ensure your starting file `equilibrated.data`, force field files (`system.in.*`), and `ttree_assignments.txt` are placed in the working directory.
2. Open `master_loop.py` and configure the parameters for your system:
   ```python
   INITIAL_DATA = "equilibrated.data"               
   LAMMPS_EXEC  = "/path/to/your/lmp"  # <--- IMPORTANT: Set the correct path to your LAMMPS executable
   TARGET_NEW_BONDS = 1540                          
   MAX_CYCLES = 500  
   ```
3. Run the pipeline:
   ```bash
   python3 master_loop.py
   ```

### Logging & Output
* Progress is printed to the console and appended to `bonds_history.log`.
* The LAMMPS output for each cycle is saved in `log_step_{i}.txt`.
* Intermediate data files (`*_xlinked.data`, `*_relaxed.data`) are generated at each step, allowing you to trace the entire process.

## ⚙️ Pipeline Workflow

1. **Initialization:** Reads the initial topology. Sets the starting search radius (`CUTOFF`).
2. **Search & Cross-link (`crosslink.py`):** Finds O and C atoms within the `CUTOFF`. Forms up to 20 bonds per cycle and rewrites the topology.
3. **Yield Evaluation:** 
   * If many bonds are formed (>=10), the radius resets to its minimum value.
   * If the yield is low (<10), the `CUTOFF` is increased by 0.5 Å.
4. **Relaxation (`relax.in`):** Runs LAMMPS. Performs a soft stress minimization followed by an MD annealing protocol to relax newly formed nodes and bring unreacted groups closer together.
5. **Overcoming Vitrification (`extra.in`):** If the maximum `CUTOFF` (6.0 Å) is reached and no bonds are formed, the system is assumed to be in a glassy state. A high-temperature, prolonged annealing step is triggered to restore mobility.
6. The loop repeats until `TARGET_NEW_BONDS` is reached.
