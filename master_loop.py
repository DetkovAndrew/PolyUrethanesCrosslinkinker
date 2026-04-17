# -*- coding: utf-8 -*-
import os
import subprocess
import shutil

# --- ФУНКЦИЯ ПОДСЧЕТА СВЯЗЕЙ ---
def get_num_bonds(filepath):
    """Считает количество строк в секции Bonds файла .data"""
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if 'bonds' in line:
                    parts = line.split()
                    if len(parts) >= 2 and parts[1] == 'bonds':
                        return int(parts[0])
    except Exception as e:
        print(f"Error reading bonds from {filepath}: {e}")
    return 0

# ==============================================================================
# --- КОНФИГУРАЦИЯ ---
# ==============================================================================
INITIAL_DATA = "equilibrated.data"               
LAMMPS_EXEC  = "/home/detkov_andrew/LAMMPS/build/lmp"  
TARGET_NEW_BONDS = 1540                          
MAX_CYCLES = 500                                 

# Настройки радиуса
START_CUTOFF = 3.5      # Начальный радиус
MAX_CUTOFF   = 6.0      # Потолок радиуса
CUTOFF_STEP  = 0.5      # Шаг увеличения радиуса
MIN_BONDS_PER_STEP = 10 # Порог "успешного" шага (если меньше - увеличиваем радиус)

current_data = "current.data"
shutil.copy(INITIAL_DATA, current_data)

initial_bonds = get_num_bonds(current_data)
total_new_bonds = 0
current_cutoff = START_CUTOFF

# Лог для отслеживания прогресса
with open("bonds_history.log", "w") as f:
    f.write("=== CROSSLINKING PROCESS START ===\n")
    f.write(f"Initial bonds: {initial_bonds}\n\n")

# ==============================================================================
# --- ОСНОВНОЙ ЦИКЛ СШИВКИ ---
# ==============================================================================
i = 1
while i <= MAX_CYCLES:
    print(f"\n" + "="*50)
    print(f" CYCLE {i} | Current Cutoff: {current_cutoff:.2f} A")
    print(f" Progress: {total_new_bonds} / {TARGET_NEW_BONDS} new bonds")
    print("="*50)
    
    temp_xlinked = f"step_{i}_xlinked.data"
    temp_relaxed = f"step_{i}_relaxed.data"
    
    # --- ШАГ 1: ПОПЫТКА СШИВКИ (crosslink.py) ---
    subprocess.run(["python3", "crosslink.py", current_data, temp_xlinked, str(current_cutoff)], check=True)
    
    current_bonds_total = get_num_bonds(temp_xlinked)
    cycle_new_bonds = (current_bonds_total - initial_bonds) - total_new_bonds

    # --- ШАГ 2: ЛОГИКА УПРАВЛЕНИЯ РАДИУСОМ ---
    if cycle_new_bonds >= MIN_BONDS_PER_STEP:
        print(f"SUCCESS: Formed {cycle_new_bonds} new bonds (>= {MIN_BONDS_PER_STEP}).")
        current_cutoff = START_CUTOFF  # Сбрасываем радиус на минимум после успеха
    
    else:
        print(f"LOW YIELD: Formed only {cycle_new_bonds} bonds at {current_cutoff:.2f} A.")
        
        if current_cutoff < MAX_CUTOFF:
            current_cutoff += CUTOFF_STEP
            if current_cutoff > MAX_CUTOFF:
                current_cutoff = MAX_CUTOFF
            print(f"Increasing cutoff to {current_cutoff:.2f} A for the next cycle...")
            
        else:
            # Если уперлись в потолок (6.0) И за шаг создано < 10 связей
            print("\n!!! REACHED MAX CUTOFF & LOW YIELD. TRIGGERING EXTRA RELAXATION (extra.in) !!!")
            temp_extra = f"step_{i}_extra.data"
            
            # Запуск спасительного отжига (extra.in)
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            cmd_extra = [
                "mpirun", "-np", "1", LAMMPS_EXEC,
                "-pk", "gpu", "1", "neigh", "no", "-sf", "gpu",
                "-in", "extra.in",
                "-var", "DATA_IN", temp_xlinked, "-var", "DATA_OUT", temp_extra
            ]
            subprocess.run(cmd_extra, check=True)
            
            # Пробуем до-шить ПОСЛЕ отжига на максимальном радиусе (6.0)
            subprocess.run(["python3", "crosslink.py", temp_extra, temp_xlinked, str(MAX_CUTOFF)], check=True)
            
            current_bonds_total = get_num_bonds(temp_xlinked)
            total_cycle_bonds = (current_bonds_total - initial_bonds) - total_new_bonds
            
            if total_cycle_bonds == 0:
                print("\n[!] VITRIFICATION DETECTED: No bonds formed even after extra high-temp relaxation.")
                print(f"Final conversion: {total_new_bonds} bonds created.")
                break
            
            print(f"EXTRA RELAXATION helped: Yielded total {total_cycle_bonds} bonds this cycle.")
            current_cutoff = START_CUTOFF  # Сбрасываем радиус после спасения

    # Обновляем глобальный счетчик
    total_new_bonds = current_bonds_total - initial_bonds
    
    # Записываем шаг в историю
    with open("bonds_history.log", "a") as f:
        f.write(f"Cycle {i}: Cutoff {current_cutoff:.2f}, New {cycle_new_bonds}, Total {total_new_bonds}\n")

    # --- ШАГ 3: РЕЛАКСАЦИЯ В LAMMPS (relax.in) ---
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cmd_relax = [
        "mpirun", "-np", "1", LAMMPS_EXEC, 
        "-pk", "gpu", "1", "neigh", "no", "-sf", "gpu",
        "-in", "relax.in",
        "-var", "DATA_IN", temp_xlinked, "-var", "DATA_OUT", temp_relaxed
    ]
    
    print(f"Running normal LAMMPS Relaxation for cycle {i}...")
    with open(f"log_step_{i}.txt", "w") as log_file:
        subprocess.run(cmd_relax, stdout=log_file, stderr=subprocess.STDOUT, check=True)
    
    shutil.copy(temp_relaxed, current_data)

    if total_new_bonds >= TARGET_NEW_BONDS:
        print(f"\n" + "*"*50)
        print(f"DONE! Target reached: {total_new_bonds} new bonds created.")
        print("*"*50)
        break
        
    i += 1

print("\n!!! CROSS-LINKING PIPELINE FINISHED !!!")