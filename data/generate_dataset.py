"""
generate_dataset.py
--------------------
Builds a synthetic, but methodologically realistic, dataset of country-level
Human Development Index (HDI) indicators.

The four raw indicators follow the ranges actually used by the UNDP:
    - Life expectancy at birth      : 20  -> 85  years
    - Mean years of schooling       : 0   -> 15  years
    - Expected years of schooling   : 0   -> 18  years
    - GNI per capita (PPP $)        : 100 -> 75,000

For every simulated country we compute the *true* HDI using the official
UNDP geometric-mean formula, add a small amount of realistic noise to mimic
real-world measurement/reporting error, and then bucket the result into the
four official development tiers:

    Very High  : HDI >= 0.800
    High       : 0.700 <= HDI < 0.800
    Medium     : 0.550 <= HDI < 0.700
    Low        : HDI < 0.550

The dataset is intentionally generated (rather than scraped) so the project
is fully self-contained and reproducible without any external data license.
"""

import numpy as np
import pandas as pd

RNG_SEED = 42
N_SAMPLES = 6000

# UNDP min / max goalposts used for indicator normalisation
LE_MIN, LE_MAX = 20, 85
MYS_MAX = 15
EYS_MAX = 18
GNI_MIN, GNI_MAX = 100, 75000


def life_expectancy_index(le):
    return (le - LE_MIN) / (LE_MAX - LE_MIN)


def education_index(mys, eys):
    mysi = np.clip(mys / MYS_MAX, 0, 1)
    eysi = np.clip(eys / EYS_MAX, 0, 1)
    return (mysi + eysi) / 2


def income_index(gni):
    gni = np.clip(gni, GNI_MIN, GNI_MAX)
    return (np.log(gni) - np.log(GNI_MIN)) / (np.log(GNI_MAX) - np.log(GNI_MIN))


def hdi_score(le, mys, eys, gni):
    lei = np.clip(life_expectancy_index(le), 1e-6, 1)
    ei = np.clip(education_index(mys, eys), 1e-6, 1)
    ii = np.clip(income_index(gni), 1e-6, 1)
    return (lei * ei * ii) ** (1 / 3)


def hdi_tier(score):
    if score >= 0.800:
        return "Very High"
    if score >= 0.700:
        return "High"
    if score >= 0.550:
        return "Medium"
    return "Low"


def simulate(n=N_SAMPLES, seed=RNG_SEED):
    rng = np.random.default_rng(seed)

    # Sample a "development level" latent factor so the four indicators are
    # correlated the way real countries are (rich countries tend to also be
    # healthier and better educated), then add per-indicator noise.
    dev_level = rng.beta(2.0, 1.6, size=n)  # 0 (low) .. 1 (very high)

    life_expectancy = LE_MIN + dev_level * (LE_MAX - LE_MIN)
    life_expectancy += rng.normal(0, 4, size=n)
    life_expectancy = np.clip(life_expectancy, LE_MIN, LE_MAX)

    mean_years_schooling = dev_level * MYS_MAX + rng.normal(0, 1.5, size=n)
    mean_years_schooling = np.clip(mean_years_schooling, 0, MYS_MAX)

    expected_years_schooling = dev_level * EYS_MAX + rng.normal(0, 1.8, size=n)
    expected_years_schooling = np.clip(expected_years_schooling, 0, EYS_MAX)

    # Income is log-scale and has the widest spread between rich/poor nations
    log_gni = np.log(GNI_MIN) + dev_level * (np.log(GNI_MAX) - np.log(GNI_MIN))
    log_gni += rng.normal(0, 0.35, size=n)
    gni_per_capita = np.clip(np.exp(log_gni), GNI_MIN, GNI_MAX)

    scores = hdi_score(life_expectancy, mean_years_schooling,
                        expected_years_schooling, gni_per_capita)
    # small reporting noise on the final score, then re-clip
    scores = np.clip(scores + rng.normal(0, 0.01, size=n), 0, 1)
    tiers = [hdi_tier(s) for s in scores]

    df = pd.DataFrame({
        "life_expectancy": life_expectancy.round(1),
        "mean_years_schooling": mean_years_schooling.round(2),
        "expected_years_schooling": expected_years_schooling.round(2),
        "gni_per_capita": gni_per_capita.round(0),
        "hdi_score": scores.round(4),
        "hdi_tier": tiers,
    })
    return df


if __name__ == "__main__":
    df = simulate()
    out_path = "hdi_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(df["hdi_tier"].value_counts())
