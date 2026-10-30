# Power system load type classification

Classifies each 15-minute reading of a steel plant's power system as `Light_Load`, `Medium_Load` or `Maximum_Load`, using 2018 consumption data. December (the last month) is the test set.

Everything is in [`load_type_classification.ipynb`](load_type_classification.ipynb), saved with outputs.

## Results on December (test set)

| Model | Accuracy | Macro F1 | Macro precision | Macro recall |
|---|---:|---:|---:|---:|
| XGBoost, class-weighted (final) | **0.998** | **0.998** | 0.998 | 0.999 |
| My first version | 0.958 | 0.955 | 0.965 | 0.945 |
| Calendar-only lookup (time, weekday, season) | 0.887 | 0.882 | | |

95% confidence intervals, from resampling whole December days 1,000 times: accuracy 0.995-1.000, macro F1 0.995-1.000.

| Class | Precision | Recall | F1 | Readings |
|---|---:|---:|---:|---:|
| Light_Load | 0.999 | 0.998 | 0.999 | 1,744 |
| Medium_Load | 0.996 | 0.999 | 0.997 | 704 |
| Maximum_Load | 0.998 | 1.000 | 0.999 | 528 |

Other measures on December: weighted F1 0.998, Cohen's kappa 0.997, MCC 0.997, macro one-vs-rest ROC-AUC 1.000, log loss 0.0066.

5 of the 2,976 December readings are wrong, all on weekends (4 between 9am and 1pm). My first version got 124 wrong.

## The data was deliberately corrupted

The file looks like the public UCI steel plant dataset with problems added. I found them, then had them checked by an independent audit.

- **Inflated values.** About 10% of the non-zero values in every measurement column have been multiplied by roughly 1.5-3 and stored with long decimals (`8.753692425450835`; the real readings have 2 decimals). Most stay inside a plausible range, so a range check misses them. Only the power factors above 100 and NSM above 86,400 give some of them away.
- **Blank cells.** 1-4.5% of each column.
- **Midnight timestamps.** Each 00:00 reading carries the date of the day that is ending: the file goes `01-01 23:45`, `01-01 00:00`, `02-01 00:15`. Sorting by date moved all 365 midnight readings back a day, which made the "previous reading" features wrong around midnight.
- **A duplicated reading.** 31 Dec 20:00 appears twice; the second copy is noisy.
- **Smaller things.** CO2 is 0 on 41 high-load readings on 2 January, and one reading is all zeros.

**How I fixed it.** The columns obey exact physical relationships: power factor = 100 × kWh / √(kWh² + kVarh²), and CO2 ≈ kWh × 0.000458, rounded. On untouched rows these hold every time. So I mark the inflated values as missing, then rebuild them and the blanks from the other readings in the same row. Hiding known values and rebuilding them gives a mean error of 0.007 kWh for usage. Nothing is interpolated over time, because that would use future readings. NSM is recalculated from the corrected timestamp.

I found no label noise. Every reading from midnight to 9am is Light_Load, and no label flips for a single reading.

## Approach

**What the label means.** The label is a time-of-use tariff period, not a consumption level. Some Maximum_Load readings use under 5 kWh, and some Light_Load readings use over 60. There are three day patterns:
- a March-October working day;
- a November-February working day (peak hours move to the evening);
- an all-Light day: Sundays, some Saturdays and holidays (62 in the year).

The calendar alone gets 88.7% on December. What it can't know is whether the plant is off on a given day, and that's where all the errors were.

**Features.** All of them use only the current reading and earlier ones.
- **Calendar:** hour, seconds since midnight, weekday, weekend flag, a winter-schedule flag, cyclical time encodings. I leave out the raw month on purpose: trees can't extrapolate it to November and December.
- **Current readings:** the six electrical measurements.
- **Recent history:** usage and reactive power 15 minutes and 1 hour ago, and the mean of the previous hour.
- **Plant activity so far today:** mean and total usage since midnight and since 6am, 2- and 4-hour means and maximum, and how many readings in the last hour were above 10 kWh. This answers "is the plant working today?", which the calendar can't.
- **Same slot yesterday and last week:** usage and reactive power, and their difference from now.

**Validation.** December is split off at the start and used once. Every choice (features, model, class weights, tuning) is made with 5 time-ordered folds on January-November, scored by macro F1, because Light_Load is over half the data.

| January-November cross-validation | Macro F1, all folds | Folds 2-5 |
|---|---:|---:|
| Most common label per hour of week | 0.746 | 0.785 |
| Logistic regression | 0.777 | 0.828 |
| Random forest | 0.888 | 0.958 |
| XGBoost, first version's features | 0.894 | 0.966 |
| + plant activity so far today | 0.921 | 0.993 |
| + same slot yesterday and last week | 0.922 | 0.993 |
| **Tuned, class-weighted XGBoost (final)** | **0.925** | **0.997** |
| Tuned LightGBM / XGBoost + LightGBM average | 0.924 | 0.997 |

Fold 1 trains on January-February only and has never seen the March-October schedule it's tested on, so it stays around 0.63 whatever the model. Folds 2-5 show the real differences.

**What each step added on December.**

| Change | Macro F1 |
|---|---:|
| Calendar only | 0.882 |
| + current and past-hour readings (first version) | 0.955 |
| Same, with the new cleaning | 0.956 |
| + plant activity, same slot yesterday / last week, tuning (final) | 0.998 |

The cleaning barely moves the score on its own: it makes the data right, and the gain comes from the activity features.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate          # source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace load_type_classification.ipynb
```

It runs in about 5 minutes on a laptop CPU.

## Files

```
data/load_data.csv                  raw data
load_type_classification.ipynb      cleaning, analysis, models, evaluation
models/load_type_model.joblib       final model, label encoder and feature list
outputs/december_predictions.csv    test-set predictions (Date_Time, actual, predicted)
```

## Limitations

- This classifies a reading once it arrives; it isn't a forecast.
- One year of data means each season is seen once.
- Cross-validation was used for several choices, so the CV scores may be slightly optimistic. December was used only once.
