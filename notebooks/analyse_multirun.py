# %%
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# %%

def collect_all_results(multirun_dir="multirun"):
    """Collect results from all Hydra multirun outputs."""

    
    all_results = []
    
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

df = collect_all_results("multirun/2025-09-19/11-03-16")

# %%

plt.scatter(np.log(df['temperature']), df['final_val_ll'])