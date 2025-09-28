
# %%
from src.collect_results import collect_all_results
import matplotlib.pyplot as plt
import numpy as np

# df = collect_all_results("multirun/2025-09-26/13-19-07")
# df = collect_all_results("multirun/2025-09-26/13-24-16")
df = collect_all_results("multirun/2025-09-26/13-19-07", "multirun/2025-09-26/13-24-16","multirun/2025-09-26/17-53-52","multirun/2025-09-26/17-54-25","multirun/2025-09-26/18-23-31")


# for key in df.keys()[:5]:
#     plt.figure()
#     plt.scatter(np.log(df["temperature"]), np.log(-(df[key] - df[key].max())), marker='o', c=df["frequency"], cmap='viridis')

#     plt.legend(title="Frequency")
#     plt.colorbar(label='Frequency')
#     plt.xlabel('Log Temperature')
#     plt.ylabel('Log OOD NLL Var (normalised)')
#     plt.title(f'Frequency vs Temperature and OOD NLL Var {key}')
#     plt.show()



# %%
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

for key in df.keys()[:5]:
    plt.figure()
    
    # Get unique frequencies and create a color map
    frequencies = sorted(df['frequency'].unique())
    colors = cm.viridis(np.linspace(0, 1, len(frequencies)))
    
    # Create a color dictionary for consistent coloring
    color_dict = dict(zip(frequencies, colors))
    
    for freq, group in df.groupby("frequency"):
        group_sorted = group.sort_values("temperature")
        x_vals = np.log(group_sorted["temperature"])
        y_vals = np.log(-(group_sorted[key] - df[key].max()))
        
        # Plot with color based on frequency
        plt.plot(x_vals, y_vals, marker='o', 
                color=color_dict[freq], label=f'{freq}', alpha=0.8)
    
    plt.legend(title="Frequency")
    plt.xlabel('Log Temperature')
    plt.ylabel('Log OOD NLL Var (normalised)')
    plt.title(f'Frequency vs Temperature and OOD NLL Var {key}')
    plt.grid(True, alpha=0.3)
    plt.show()

