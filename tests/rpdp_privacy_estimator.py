import sys
sys.path.append("..")
from itertools import product
import pandas as pd

from fedrpdp.utils.rpdp_utils import (
    get_sample_rate_curve,
    MultiLevels, 
    MixGauss, 
    Pareto,
)

STYLES = ["ThreeLevels", "BoundedPareto", "BoundedMixGauss"]
GENERATE_EPSILONS = {
    "ThreeLevels": lambda n, params: MultiLevels(3, *params, n),
    "BoundedPareto": lambda n, params: Pareto(*params, n), 
    "BoundedMixGauss": lambda n, params: MixGauss(*params, n),
}
SETTINGS = {
    "ThreeLevels": [[[0.7,0.2,0.1], [0.5, 1.0, 5.0]]],
    "BoundedPareto": [[3.0, 0.5]], 
    "BoundedMixGauss": [[[0.7,0.2,0.1], [(0.5, 0.1), (1.0, 0.1), (5.0, 1.0)]]],
}
BoundedFunc = lambda values: [min(max(x, 0.5), 5.0) for x in values]

deltas = [1e-3, 1e-5]
noise_multipliers = [5.0, 10.0, 20.0, 30.0]
nmdelta_list = list(product(noise_multipliers, deltas))

num_updates = 20
num_rounds = 50
client_sample_rate = 1.

epsilons = []
for name in STYLES:
    epsilons.extend([f"{name}-{_}" for _ in range(len(SETTINGS[name]))])

results_all_reps = []
for nm, d in nmdelta_list:
    curve_fn = get_sample_rate_curve(
        target_delta = d,
        noise_multiplier=nm,
        num_updates=num_updates,
        num_rounds=num_rounds,
        client_rate = client_sample_rate
    )

    for ename in epsilons:
        name, p_id = ename.split('-')
        e = BoundedFunc(GENERATE_EPSILONS[name](6000, SETTINGS[name][int(p_id)]))

        sample_rates = [curve_fn(v) for v in e]

        print(f"{name}: delta={d}, nm={nm}")
        print(f"min_eps={min(e):.2f} \t min_sample_rate={min(sample_rates):.4f}\nmax_eps={max(e):.2f} \t max_sample_rate={max(sample_rates):.4f}")

        results_all_reps.append({"e": ename, "d": d, "nm": nm, "min_e": round(min(e), 4), "min_q": round(min(sample_rates), 4), "max_e": round(max(e),4), "max_q": round(max(sample_rates),4)})

results = pd.DataFrame.from_dict(results_all_reps)
results.to_csv("results.csv", index=False)