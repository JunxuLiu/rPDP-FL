# Plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import product

sns.set_theme(style="darkgrid")

results = pd.read_csv("results.csv")
target_metric = "min_q"
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
        y=target_metric,
        label=f"epsilon dist={e}, delta={d}",
        linestyle=linestyles[::-1][i][1],
        ax=ax,
    )

xtick_values = [nm for nm in results["nm"].unique() if not (np.isnan(nm))]
xlabels = [str(v) for v in xtick_values]
ax.set_xticks(xtick_values, xlabels, rotation=45)
# ax.axhline(
#     np.array(results.loc[results["d"].isnull(), "Performance"].tolist()).mean(),
#     color="black",
#     label="Baseline wo DP",
# )
# ax.set_xlim(0.1, 50)
plt.legend()
plt.xlabel("nm")
plt.ylabel(f"{target_metric}")
plt.savefig(f"{target_metric}_estimation.pdf", dpi=100, bbox_inches="tight")
