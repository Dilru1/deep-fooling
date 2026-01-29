import os
import glob
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ================= CONFIGURATION =================
ROOT_DIR = "."  
METRIC = 'rollout/ep_rew_mean' 
SMOOTHING = 0.8
OUTPUT_SUMMARY = "experiment_records.csv"
OUTPUT_STABLE = "stable_champions_data.csv"

CHECKPOINT_PREFIX = "rl_model_"  
CHECKPOINT_SUFFIX = "_steps.zip" 
# =================================================

def verify_checkpoint_path(log_path, best_step, interval=5000):
    """
    Snaps the best_step to the nearest physical checkpoint and verifies it.
    """
    log_dir = os.path.dirname(log_path)
    
    # Snap to the nearest 5000
    snapped_step = int(round(best_step / interval) * interval)
    
    filename = f"{CHECKPOINT_PREFIX}{snapped_step}{CHECKPOINT_SUFFIX}"
    potential_path = os.path.join(log_dir, filename)
    
    if os.path.exists(potential_path):
        return potential_path, snapped_step
    
    # If nearest isn't there, try the one before it (floor)
    floor_step = int((best_step // interval) * interval)
    filename = f"{CHECKPOINT_PREFIX}{floor_step}{CHECKPOINT_SUFFIX}"
    potential_path = os.path.join(log_dir, filename)
    
    if os.path.exists(potential_path):
        return potential_path, floor_step
        
    return "NOT_FOUND", None


def extract_all_data(root_dir):
    log_files = glob.glob(os.path.join(root_dir, "**", "*tfevents*"), recursive=True)
    all_records = []
    print(f"Reading {len(log_files)} logs...")

    for log_path in log_files:
        try:
            parts = os.path.normpath(log_path).split(os.sep)
            # jan24 / model / arch / seed / something / event
            arch_name = parts[-4] if len(parts) >= 4 else "Unknown_Arch"
            seed_name = parts[-3] if len(parts) >= 3 else "Unknown_Seed"
            group_name = parts[-5] if len(parts) >= 5 else "Experiment"

            ea = EventAccumulator(log_path, size_guidance={'scalars': 0})
            ea.Reload()
            if METRIC not in ea.Tags()['scalars']: continue
            events = ea.Scalars(METRIC)
            
            steps = [e.step for e in events]; values = [e.value for e in events]
            smoothed_values = []
            if SMOOTHING > 0:
                last = values[0]
                for v in values:
                    last = last * SMOOTHING + (1 - SMOOTHING) * v
                    smoothed_values.append(last)
            else: smoothed_values = values

            for step, val in zip(steps, smoothed_values):
                all_records.append({
                    "Timesteps": step, "Reward": val, "Group": group_name,
                    "Architecture": arch_name, "Seed": seed_name,
                    "ID": f"{group_name}_{arch_name}_{seed_name}",
                    "Full_Path": log_path 
                })
        except Exception as e: print(f"Error: {e}")
    return pd.DataFrame(all_records)

def filter_best_seeds(df):
    if df.empty: return df
    def calculate_fitness(group):
        tail = max(1, int(len(group) * 0.25))
        recent = group["Reward"].tail(tail)
        return recent.mean() - recent.std()

    run_stats = df.groupby("ID").apply(calculate_fitness, include_groups=False).reset_index(name="Fitness_Score")
    meta = df.groupby("ID")[["Group", "Architecture", "Seed", "Full_Path"]].first().reset_index()
    run_stats = pd.merge(run_stats, meta, on="ID")

    best_ids = []
    print("\n--- STABILITY-BASED SEED SELECTION ---")
    for _, config_group in run_stats.groupby(["Group", "Architecture"]):
        winner = config_group.loc[config_group["Fitness_Score"].idxmax()]
        best_ids.append(winner["ID"])
        print(f"{winner['Architecture']:<25} | {winner['Seed']:<10} | Score: {winner['Fitness_Score']:,.2f}")

    return df[df["ID"].isin(best_ids)]

def find_best_checkpoint_and_scan(df, window_size_steps=25000):
    best_checkpoints = []
    print(f"\n{'ID':<40} | {'BEST STEP':<12} | {'FILE STATUS'}")
    print("-" * 75)

    for run_id, run_data in df.groupby("ID"):
        step_diffs = run_data['Timesteps'].diff().dropna()
        interval = step_diffs.iloc[0] if not step_diffs.empty else 5000
        window = int(window_size_steps // interval)
        
        run_data = run_data.copy()
        run_data['fitness'] = run_data['Reward'].rolling(window=window).mean() - \
                             run_data['Reward'].rolling(window=window).std()
        
        valid_data = run_data.dropna(subset=['fitness'])
        if valid_data.empty: continue
        
        best_row = valid_data.loc[valid_data['fitness'].idxmax()]
        
        # SCAN FOR FILE
        file_path = verify_checkpoint_path(best_row['Full_Path'], best_row['Timesteps'])
        status = "✅ FOUND" if file_path != "NOT_FOUND" else "❌ MISSING"
        
        print(f"{run_id[:40]:<40} | {int(best_row['Timesteps']):<12,} | {status}")
        
        best_checkpoints.append({
            "ID": run_id, "Best_Step": int(best_row['Timesteps']),
            "Reward": best_row['Reward'], "Architecture": best_row['Architecture'],
            "Checkpoint_Path": file_path
        })

    return pd.DataFrame(best_checkpoints)

def plot_data(df, best_steps_df):
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")
    sns.lineplot(data=df, x="Timesteps", y="Reward", hue="Architecture", style="Group", linewidth=2.5)

    for _, row in best_steps_df.iterrows():
        plt.axvline(x=row['Best_Step'], color='red', linestyle=':', alpha=0.6)
        plt.annotate(f"BEST STEP: {row['Best_Step']:,}", 
                     xy=(row['Best_Step'], row['Reward']), xytext=(10, 20),
                     textcoords='offset points', arrowprops=dict(arrowstyle='->', color='red'))

    plt.title("Stable Seeds & Optimized Checkpoint Locations")
    plt.savefig("stable_seed_comparison.png", dpi=300)
    print("\nPlot saved to stable_seed_comparison.png")

if __name__ == "__main__":
    data = extract_all_data(ROOT_DIR)
    if not data.empty:
        # 1. Select the best overall seeds
        champions_full_data = filter_best_seeds(data)
        champions_full_data.to_csv(OUTPUT_STABLE, index=False)
        
        # 2. Find the specific best checkpoint within those seeds and scan folders
        best_checkpoints = find_best_checkpoint_and_scan(champions_full_data)
        best_checkpoints.to_csv("best_checkpoints_found.csv", index=False)
        
        # 3. Plot with annotations
        plot_data(champions_full_data, best_checkpoints)
        print("\nDeployment list saved to 'best_checkpoints_found.csv'")