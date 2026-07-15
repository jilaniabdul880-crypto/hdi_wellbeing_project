# A Comprehensive Measure of Well-Being

A machine-learning web app that estimates a country's **Human Development
Index (HDI) tier** — Low, Medium, High, or Very High — from four standard
UNDP indicators: life expectancy, mean years of schooling, expected years
of schooling, and GNI per capita.

Built for the *Artificial Intelligence and Machine Learning* group project.

---

## What's inside

```
hdi_project/
├── app.py                     Flask application (routes + API)
├── requirements.txt
├── data/
│   ├── generate_dataset.py    Builds the synthetic training dataset
│   └── hdi_dataset.csv        6,000 simulated country profiles
├── models/
│   ├── train_model.py         EDA + training + evaluation pipeline
│   ├── hdi_model.pkl          Trained classifier (best of 3 models)
│   ├── scaler.pkl             Fitted StandardScaler
│   ├── label_encoder.pkl      Tier label encoder
│   └── metadata.json          Model name, accuracy, classification report
├── plots/                     EDA & evaluation charts (PNG)
├── templates/
│   └── index.html             UI markup
└── static/
    ├── css/style.css          Design system + layout
    └── js/script.js           Live sliders, spectrum gauge, API calls
```

## How it works

1. **Dataset** — `data/generate_dataset.py` simulates 6,000 country-like
   profiles. Each profile's *true* HDI score is computed with the official
   UNDP geometric-mean formula, then bucketed into a tier, with realistic
   noise added so the relationship isn't perfectly linear.

2. **Model** — `models/train_model.py` runs exploratory data analysis
   (class balance, correlations, pairplots, boxplots — saved to `/plots`),
   then trains and 5-fold cross-validates three candidates (Logistic
   Regression, Random Forest, Gradient Boosting) on standardized features,
   picks the best performer, and evaluates it on a held-out test set
   (confusion matrix + classification report also saved to `/plots`).

3. **App** — `app.py` loads the trained model and exposes:
   - `GET /` — the interactive UI
   - `POST /api/predict` — takes the four indicators as JSON and returns
     the predicted tier, per-tier probabilities, the exact formula-based
     HDI score (for transparency), and a short interpretation.
   - `GET /api/model-info` — model metadata.

4. **UI** — a single page built around a "Well-Being Spectrum": a
   horizontal gradient bar, proportioned to the real UNDP thresholds
   (Low 0–0.550, Medium 0.550–0.700, High 0.700–0.800, Very High
   0.800–1.000), with a marker that moves live as you drag the sliders
   and lands on the model's prediction after you submit.

## Running it locally

**macOS / Linux:**
```bash
cd hdi_project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

**Windows (PowerShell):**
```powershell
cd hdi_project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
On Windows, use `python`, not `python3` — most Windows Python installs only
register the `python` command.

Then open **http://localhost:5000** in your browser.

To regenerate the dataset and retrain the model from scratch (optional —
trained artifacts are already included):
```bash
python data/generate_dataset.py
python models/train_model.py
```

### Troubleshooting

- **`ModuleNotFoundError` / pip tries to compile numpy from source and
  fails with a compiler error (`cl`, `gcc`, etc. not found):** this
  happens when `requirements.txt` pins an exact version that has no
  prebuilt wheel for your Python version. Fix: `pip install --upgrade pip`
  then `pip install -r requirements.txt` again — the versions in this
  file are minimums (`>=`), so pip will pick a compatible prebuilt wheel
  instead of trying to build one.
- **`cd` says the folder doesn't exist:** you're probably already inside
  `hdi_project` — run `dir` (Windows) or `ls` (macOS/Linux) to check
  before `cd`-ing again.

## Retraining

The dataset and model are already included, so the app runs out of the
box. To regenerate them from scratch — e.g. after changing the simulation
parameters in `data/generate_dataset.py` — just re-run the two scripts
above in order; `train_model.py` will overwrite `models/*.pkl` and
`plots/*.png`.

## Tech stack

Python · Flask · scikit-learn · Pandas · NumPy · Matplotlib · Seaborn ·
HTML/CSS/vanilla JavaScript

## Notes on the data

The training data is synthetically generated rather than scraped from a
live source, so the project is fully reproducible and license-free. It is
built directly from the official UNDP HDI formula (with added noise) so
the relationships a model learns from it mirror the real methodology.
For production use, this dataset could be swapped for the UNDP's
published *Human Development Report* indicator tables without changing
any code beyond `data/generate_dataset.py`.
