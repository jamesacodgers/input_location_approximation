# %%
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# %%

def collect_all_results(*args):
    """Collect results from all Hydra multirun outputs."""

    
    all_results = []

    for multirun_dir in args:
        # Find all results.csv files in the multirun directory
        pattern = os.path.join(multirun_dir, "*", "results.csv")
        print(pattern)
        result_files = glob.glob(pattern)

        for file_path in result_files:
            df = pd.read_csv(file_path)
            all_results.append(df)
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv("combined_results.csv", index=False)
    print(f"Combined {len(result_files)} result files into combined_results.csv")
    
    return combined_df

# %%

df = collect_all_results("multirun/2025-09-19/20-14-56", "multirun/2025-09-19/18-31-17", "multirun/2025-09-19/18-31-16", "multirun/2025-09-19/18-29-31", "multirun/2025-09-19/17-43-27")

# %%
fig, ax = plt.subplots()
for temp,group in df.groupby("temperature"):
    ax.scatter(np.log(temp), group['final_val_ll'].mean(), color="blue")
    ax.errorbar(np.log(temp), group['final_val_ll'].mean(), yerr=group['final_val_ll'].std()*2, color="blue")
#%%
fig, ax = plt.subplots()
for temp,group in df.groupby("temperature"):
    ax.scatter(np.log(temp), group['final_ood_ll'].mean(), color="blue")
    ax.errorbar(np.log(temp), group['final_ood_ll'].mean(), yerr=group['final_ood_ll'].std()*2, color="blue")