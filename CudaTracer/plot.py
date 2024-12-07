import pandas as pd
import matplotlib.pyplot as plt

# Load benchmark results
df = pd.read_csv("benchmark_results.csv")

# Map allocation type integers to labels
alloc_type_map = {
    0: "Pinned",
    #1: "Unpinned",
    #2: "CustomAllocatorManager:Optim",
    3: "CustomAllocatorManager:Profile"

}
df["AllocType"] = df["AllocType"].map(alloc_type_map)

# Plot runtime vs. allocation size for each allocation type
plt.figure(figsize=(10, 6))
for alloc_type, group in df.groupby("AllocType"):
    plt.plot(group["AllocSizeKB"], group["RuntimeMS"], marker='o', label=alloc_type)

# Add a vertical line for 32 MB
#plt.axvline(x=32 * 1024, color='r', linestyle='--', label="32 MB")

# Configure plot
plt.title("Runtime vs. Allocated Size for Different Allocation Strategies")
plt.xlabel("Allocated Size (KB)")
plt.ylabel("Runtime (ms)")
plt.legend()
plt.grid(True)

# Save and show the plot
plt.savefig("benchmark_plot.png")
plt.show()