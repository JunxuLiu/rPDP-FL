# Plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import product

sns.set_theme(style="darkgrid")

results = pd.read_csv("results_fed_heart_disease.csv")
results = results.rename(columns={"perf": "Performance"})
print(results)

linestyle_str = [
    ("solid", "solid"),  # Same as (0, ()) or '-'
    ("dotted", "dotted"),  # Same as (0, (1, 1)) or ':'
    ("dashed", "dashed"),  # Same as '--'
    ("dashdot", "dashdot"),
]
linestyle_tuple = [
    ("loosely dotted", (0, (1, 10))),
    ("densely dotted", (0, (1, 1))),
    ("loosely dashed", (0, (5, 10))),
    ("densely dashed", (0, (5, 1))),
    ("loosely dashdotted", (0, (3, 10, 1, 10))),
    ("densely dashdotted", (0, (3, 1, 1, 1))),
    ("dashdotdotted", (0, (3, 5, 1, 5, 1, 5))),
    ("loosely dashdotdotted", (0, (3, 10, 1, 10, 1, 10))),
    ("densely dashdotdotted", (0, (3, 1, 1, 1, 1, 1))),
]
linestyles = linestyle_tuple + linestyle_str
nmdeltas = [(d, nm) for d, nm in zip(results["d"].unique(), results["nm"].unique()) if not (np.isnan(d))]
edeltas = [(e, d) for e, d in list(product(results["e"].unique(), results["d"].unique())) if not (np.isnan(d))]
print(results["e"].unique(), results["d"].unique(), edeltas)
fig, ax = plt.subplots()
for i, (e, d) in enumerate(edeltas):
    cdf = results.loc[results["e"] == e].loc[results["d"] == d]
    # print(cdf)
    sns.lineplot(
        data=cdf,
        x="nm",
        y="Performance",
        label=f"{e}, delta={d}",
        linestyle=linestyles[::-1][i][1],
        ax=ax,
    )

# ax.set_xscale("log")
xtick_values = [nm for nm in results["nm"].unique() if not (np.isnan(nm))]
# print(xtick_values)
xlabels = [str(v) for v in xtick_values]
ax.set_xticks(xtick_values, xlabels)
# ax.axhline(
#     np.array(results.loc[results["d"].isnull(), "Performance"].tolist()).mean(),
#     color="black",
#     label="Baseline wo DP",
# )
# ax.set_xlim(0.1, 50)
plt.legend()
plt.xlabel("noise multiplier")
plt.ylabel("Performance")
plt.savefig("perf_function_of_dp_heart_disease.pdf", dpi=100, bbox_inches="tight")
