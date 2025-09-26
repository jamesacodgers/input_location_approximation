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

# df = collect_all_results("multirun/2025-09-23/19-54-47", "multirun/2025-09-25/18-49-32")
# df = collect_all_results("multirun/2025-09-25/17-21-48", "multirun/2025-09-23/19-54-47")
df = collect_all_results("multirun/2025-09-23/19-54-47", "multirun/2025-09-25/18-49-32","multirun/2025-09-25/17-21-48", "multirun/2025-09-23/19-54-47")

# %%
for key in df.keys()[:-2]:
    fig, ax = plt.subplots()
    for temp,group in df.groupby("temperature"):
        ax.scatter(np.log(temp), group[key].mean())
        ax.errorbar(np.log(temp), group[key].mean(), yerr=group[key].std()/np.sqrt(len(group[key]))*2)
    ax.set_title(key)
    ax.set_xlabel("log(temperature)")
    ax.set_ylabel(key)
    # ax.set_xlim(right=0.1)
    # ax.set_ylim(bottom=-2)
    
        # print("best temp:", group[key].mean().idxmax(), "value:", group[key].mean().max())
    plt.show()

# # %%
# temps_vs_dist = {"0.01": [],
#                  "0.03": [],
#                  "0.1": [],
#                  "0.3": [],
#                  "1.0": [],
#                  "3.0": [],
#                  "10.0": [], 
#                  }
# dist_std = [0.01, 0.1, 1.0, 10.0, 100.0]

# for key in df.keys()[1:-2]:
#     for temp,group in df.groupby("temperature"):
#        temps_vs_dist[str(temp)].append(group[key].mean())
#     #    temps_vs_dist[std_key].append(group[key].std()/np.sqrt(len(group[key]))*2)

#         # print("best temp:", group[key].mean().idxmax(), "value:", group[key].mean().max())
# # %%
# for key in temps_vs_dist.keys():
#     plt.figure()
#     plt.plot(np.log(dist_std), temps_vs_dist[key], marker='o')
#     # plt.errorbar(np.log(temps_vs_dist["temperatures"]), temps_vs_dist[key], yerr=temps_vs_dist[key+"_2std"], fmt='o')
#     plt.title(key)
#     plt.xlabel("log(input variance)")
#     plt.ylabel("Log Likelihood")
#     plt.show()
# plt.plot(np.log(temps_vs_dist["dist_std"]), temps_vs_dist["0.01"], marker='o')
# # plt.errorbar(np.log(temps_vs_dist["temperatures"]), temps_vs_dist[key], yerr=temps_vs_dist[key+"_2std"], fmt='o')
# plt.title("0.01")
# plt.xlabel("log(temperature)")
# plt.ylabel("Log Likelihood")
# plt.show()


# %%
