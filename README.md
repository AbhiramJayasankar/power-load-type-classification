# Power system load type classification

Classifies each 15-minute reading of a steel plant's power system in 2018 as `Light_Load`, `Medium_Load` or `Maximum_Load`. As the brief asks, the last month (December) is the test set.

Everything is in [`load_type_classification.ipynb`](load_type_classification.ipynb), saved with outputs.

## Result on December (test set)

| Accuracy | Macro F1 | Macro precision | Macro recall |
|---:|---:|---:|---:|
| **0.998** | **0.998** | 0.998 | 0.999 |

95% confidence intervals, from resampling whole December days 1,000 times: accuracy 0.995-1.000, macro F1 0.995-1.000.

| Class | Precision | Recall | F1 | Readings |
|---|---:|---:|---:|---:|
| Light_Load | 0.999 | 0.998 | 0.999 | 1,744 |
| Medium_Load | 0.996 | 0.999 | 0.997 | 704 |
| Maximum_Load | 0.998 | 1.000 | 0.999 | 528 |

Also on December: weighted F1 0.998, Cohen's kappa 0.997, MCC 0.997, macro one-vs-rest ROC-AUC 1.000, log loss 0.0066. 5 of the 2,976 readings are wrong, all on two Saturdays. December was used once, at the end; every decision was made on January-November.

## How I got there

### 1. Looking at the data

The file has 35,041 readings: usage (kWh), lagging and leading reactive power (kVarh), the two power factors, CO2, seconds since midnight (NSM) and the label. It looks like the public UCI steel plant dataset, and before any modelling I checked every column against what the real data should look like. It has been deliberately corrupted:

- **Inflated values.** About 10% of the non-zero values in every measurement column are the true value multiplied by roughly 1.5-3, stored with long decimals (`8.753692425450835`; the real readings have 2 decimals). Most of them stay inside a plausible range, so a range check alone misses them. Only the power factors above 100 and NSM above 86,400 give some of them away.
- **Blank cells.** 1-4.5% of each column.
- **Midnight timestamps.** Each 00:00 reading carries the date of the day that is ending: the file goes `01-01 23:45`, `01-01 00:00`, `02-01 00:15`. Taken at face value, every midnight reading lands a day early.
- **A duplicated reading** at 31 Dec 20:00; the second copy is noisy.
- **Smaller things:** CO2 is 0 on 41 high-load readings on 2 January, and one reading is all zeros.

The labels are clean. Every reading from midnight to 9am is Light_Load, no label flips for a single reading, and there are no stray spellings.

### 2. Repairing it with physics

The columns obey exact relationships: power factor = 100 × kWh / √(kWh² + kVarh²), and CO2 ≈ kWh × 0.000458, rounded. On untouched rows these hold every time, so they tell me which values are corrupted and let me rebuild them. I mark every long-decimal value as missing, then rebuild those and the blanks from the other readings in the same row (usage from power factor and reactive power, and so on). Hiding known values and rebuilding them gives a mean error of 0.007 kWh for usage. Nothing is interpolated over time, because that would use future readings. I also move the midnight readings to the right day, recompute NSM from the timestamp, drop the noisy duplicate, fix the zero CO2 values and leave the power factors undefined on the all-zero row.

### 3. Understanding the label

The exploration showed the label isn't a consumption level. Some Maximum_Load readings use under 5 kWh, and some Light_Load readings use over 60. It is a time-of-use tariff period, with three day patterns:

- a March-October working day;
- a November-February working day, with the peak moved to the evening (the switch is on 1 March and 1 November);
- an all-Light day: Sundays, some Saturdays and holidays (62 in the year).

So the clock decides most labels, and a lookup on time of day, weekday and season already gets 88.7% on December. What the calendar can't know is whether the plant is working on a given day. That became the focus.

### 4. Features and models

I built features step by step, using only the current reading and earlier ones:

- **Calendar:** hour, seconds since midnight, weekday, weekend flag, a winter-schedule flag, cyclical encodings. I leave out the raw month: trees can't extrapolate it to November and December.
- **Current readings:** the six measurements.
- **Previous hour:** usage and reactive power 15 minutes and 1 hour ago, and the mean of the previous hour.
- **Plant activity so far today:** mean and total usage since midnight and since 6am, 2- and 4-hour means and maximum, and how many recent readings were above 10 kWh. This answers "is the plant working today?"
- **Same slot yesterday and last week:** usage and reactive power, and their difference from now.

Validation uses 5 time-ordered folds on January-November, scored by macro F1 because Light_Load is over half the data.

| January-November cross-validation | Macro F1, all folds | Folds 2-5 |
|---|---:|---:|
| Most common label per hour of week | 0.746 | 0.785 |
| Logistic regression | 0.777 | 0.828 |
| Random forest | 0.888 | 0.958 |
| XGBoost, calendar + readings + previous hour | 0.894 | 0.966 |
| + plant activity so far today | 0.921 | 0.993 |
| + same slot yesterday and last week | 0.922 | 0.993 |
| **Tuned, class-weighted XGBoost (final)** | **0.925** | **0.997** |
| Tuned LightGBM / XGBoost + LightGBM average | 0.924 | 0.997 |

Fold 1 trains on January-February only and has never seen the March-October schedule it's tested on, so it stays around 0.63 whatever the model. Folds 2-5 show the real differences.

I also tried two more activity groups with the final settings fixed: the same "today so far" summaries for reactive power and power factor, and how long the plant has been running. Both lowered CV macro F1 (0.9239 and 0.9228, against 0.9245; 0.9191 together), so I left them out.

### 5. What each step adds on December

Computed after the model was chosen, to explain the result:

| Features | December accuracy | Macro F1 |
|---|---:|---:|
| Calendar only | 0.887 | 0.882 |
| + current readings | 0.957 | 0.951 |
| + previous hour | 0.959 | 0.955 |
| + plant activity and same slot, tuned (final) | **0.998** | **0.998** |

Almost the whole gain comes from the plant-activity features. The data repair doesn't change these scores: with the same features, range checks alone give 0.956 macro F1 against 0.955 for the full repair. The repair is about having correct data, and the activity features are what fix the off-days.

The 5 remaining errors are two ambiguous Saturdays. One is the first reading of a start-up, before anything earlier that day shows the plant is working. The other four come from a brief usage rise during the usual peak hours on a day the plant stayed off.

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
- A plant shutdown or holiday calendar would be the natural next input for the remaining off-day errors.
