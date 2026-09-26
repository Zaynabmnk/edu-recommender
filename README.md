# Pathway-aware hybrid recommender for workplace e-learning

This is the code for my CM3070 final project, which follows template 1.1 (CM3005 Data Science,
Project Idea 1: Data-Driven Personalised Educational Content Recommendation).

The recommender suggests the next Microsoft 365 training video for a learner on the Mandarine Academy
MOOC, using the MARS dataset. It mixes three signals:

- **collaborative**: videos opened by the same learners (item-kNN)
- **content**: videos that cover similar material (TF-IDF over titles, tags and cleaned subtitles)
- **learning path**: videos learners usually open next (a first-order Markov model)

The weights of the mix depend on how many videos the learner has opened, and a learner with no history
gets the videos trending over the last 30 days. Every recommendation comes with a short reason, for
example *Often opened next after "Share documents"*.

## Results

Each learner's most recent video is held out, and the model has to rank it among every video the
learner has not opened yet. Settings are tuned with five-fold cross-validation over learners.
Results for 1,898 learners:

| Method | HR@10 | NDCG@10 |
|---|---|---|
| **Pathway hybrid, adaptive weights (final)** | **0.440** | **0.311** |
| Learning path only (Markov) | 0.424 | 0.296 |
| Item-kNN collaborative filtering | 0.394 | 0.269 |
| ALS matrix factorisation | 0.388 | 0.253 |
| My June prototype (CF + content) | 0.364 | 0.240 |
| Item-kNN, whole history weighted equally | 0.352 | 0.241 |
| Content only (TF-IDF) | 0.256 | 0.138 |
| Popularity | 0.171 | 0.082 |

All results, including significance tests, are in `results/`, and the charts are in `figures/`.

A stricter check trains and tunes the models only on activity before 1 July 2019. For learners who arrived
after that date, my prototype does better than the final model (HR@10 0.341 against 0.301). The final model is
still better on videos that already existed (0.410 against 0.388), but 27% of the videos those learners opened
next had no earlier activity, and only the content signal can recommend those.

## Data

The MARS dataset (Hafsa, 2022) is released under CC0 1.0, so the four English files are included in
`data/raw/`. See `data/README.md` for the source and citation.

## Running it

I used Python 3.13. To set up:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To reproduce everything in the report:

```bash
python -m edurec.explore       # dataset statistics and the dataset figure
python -m edurec.experiments   # all 16 methods, 5-fold tuning, significance tests (under three minutes)
python -m edurec.coldstart     # the brand-new learner experiment
python -m edurec.train         # fit the final model on all data and save it to models/
python -m edurec.timesplit     # stricter check: train only on activity before July 2019
python -m edurec.examples      # worked examples and timings
python -m edurec.figures       # the result charts
```

To try the final model:

```bash
python -m edurec.cli learner 111316      # a learner from the dataset
python -m edurec.cli videos 510 511      # someone who has watched these two videos
python -m edurec.cli new                 # a brand-new learner
python -m edurec.webserver               # web API and demo page on http://127.0.0.1:8000
```

On the demo page, a history can be built by searching and adding videos, marking recommended videos as
watched, or loading a learner from the dataset (the "Heaviest learner" and "Random learner" buttons pick one
for you). MARS has no video files, so opening a video shows its
details and subtitles instead, and each recommendation shows how much each signal added to it. The page can
also open with a history already filled in, for example
http://127.0.0.1:8000/?videos=510,511 or http://127.0.0.1:8000/?learner=111316.

Example API calls:

```bash
curl http://127.0.0.1:8000/learners/111316/recommendations?k=5
curl -X POST http://127.0.0.1:8000/recommendations -H "Content-Type: application/json" -d '{"history": [510, 511], "k": 5}'
```


## Tests

```bash
python -m unittest discover -s tests -t .
```

The tests use a tiny made-up dataset where the right answers are known, plus the saved model for the
command line tool and the web API (which the tests start on a free port and call over HTTP).

## Project structure

```
edurec/
  data.py, text.py         loading MARS, merging views and watch events, repairing the subtitle text
  split.py                 time-ordered test/validation split and the five folds
  models/                  random, popularity, trending, item-kNN, content, learning path, ALS,
                           my June prototype and the final hybrid
  tuning.py                weight search over a 0.1 grid
  metrics.py, stats.py     ranking metrics, bootstrap intervals, Wilcoxon, McNemar and Holm
  experiments.py           the main experiment
  coldstart.py             the brand-new learner experiment
  timesplit.py             the single cut-off date check and the new-video slot test
  train.py, service.py     the final model and the service that serves it
  cli.py                   command line tool
  webserver.py, static/    web API and demo page (standard library only)
  explore.py, figures.py   dataset statistics and charts
  examples.py              worked examples and timings
tests/                     unit tests
data/raw/                  the MARS files
results/, figures/         everything the scripts produce
models/                    the saved final model
```

The random seed is fixed (42), so running the scripts again gives the same numbers.

## Reference

Hafsa, M. (2022) *E-learning Recommender System Dataset*, version 2.0. Harvard Dataverse.
https://doi.org/10.7910/DVN/BMY3UD

Hafsa, M., Wattebled, P., Jacques, J. and Jourdan, L. (2023) 'E-learning recommender system dataset',
*Data in Brief*, 47, 108942. https://doi.org/10.1016/j.dib.2023.108942
