# Power system load type classification

Classifies each 15-minute reading of a steel plant's power system as `Light_Load`, `Medium_Load` or `Maximum_Load`, using 2018 consumption data. December (the last month) is the test set.

Everything is in [`load_type_classification.ipynb`](load_type_classification.ipynb), saved with outputs.

## Results on December (test set)

| Model | Accuracy | Macro F1 | Macro precision | Macro recall |
|---|---:|---:|---:|---:|
| XGBoost (tuned) | **0.958** | **0.955** | 0.965 | 0.945 |

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Light_Load | 0.949 | 0.982 | 0.965 |
| Medium_Load | 0.974 | 0.913 | 0.943 |
| Maximum_Load | 0.973 | 0.939 | 0.956 |

Cross-validation on January-November (5 time-ordered folds, macro F1):

| Model | Macro F1 | Accuracy |
|---|---:|---:|
| XGBoost | 0.893 | 0.910 |
| LightGBM | 0.892 | 0.909 |
| Random forest | 0.888 | 0.906 |
| Logistic regression | 0.780 | 0.820 |
| Most common label per hour of week | 0.746 | 0.791 |

The spread across folds is large because fold 1 trains on January-February only and has never seen the March-October schedule it's tested on. Folds 2-5 are all 0.95-0.97.

## Approach

**Cleaning.** Missing values in every measurement column. Power factors go up to 300 (valid range is 0-100), and `NSM` (seconds since midnight) goes up to ~249,000. Impossible power factors are set to missing. NSM is recalculated from the timestamp, because even in-range values often disagree with it. One duplicated timestamp is dropped. Nothing is interpolated, since that would use the next reading.

**What the data shows.** The label follows a time-of-use schedule:
- Midnight to 9am is always Light_Load.
- Sundays and holidays are mostly Light_Load.
- The weekday peak hours change with the season. Maximum_Load is in the late morning and afternoon from March to October, and in the late morning and evening from November to February. The switch happens on 1 March and 1 November.

**Features.**
- **Calendar:** hour, seconds since midnight, weekday, weekend flag, a winter-schedule flag, and cyclical time encodings.
- **Current readings:** the six electrical measurements.
- **Past readings:** usage and reactive power 15 minutes ago, 1 hour ago, and the mean of the previous hour.

The raw month number is left out on purpose. Trees can't extrapolate it to months they haven't seen, and replacing it with the season flag took December macro F1 from 0.944 to 0.955.

**Validation.** December is split off right after cleaning and used once, at the end. Model choice and tuning use `TimeSeriesSplit` on January-November only. Imputation and scaling for logistic regression sit inside the pipeline. The main metric is macro F1, because Light_Load is over half the data.

**Ablation on December.**

| Features | Macro F1 |
|---|---:|
| Calendar only | 0.882 |
| + current readings | 0.950 |
| + past hour of readings (final) | 0.955 |

Most remaining errors are weekend Medium/Maximum readings predicted as Light_Load.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate          # source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace load_type_classification.ipynb
```

It runs in about 3 minutes on a laptop CPU.

## Files

```
data/load_data.csv                  raw data
load_type_classification.ipynb      analysis, models, evaluation
models/load_type_model.joblib       final model, label encoder and feature list
outputs/december_predictions.csv    test-set predictions (Date_Time, actual, predicted)
```
