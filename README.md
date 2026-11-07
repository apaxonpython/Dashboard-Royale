# Dashboard Royale: The Gilded Ace

Rebuilds every number and the dashboard from the raw casino data.

## Files

- `build_dashboard.py`: reads the raw data, computes every number, writes the dashboard
- `dashboard_template.html`: the dashboard page (charts, filters, the $100,000 plan)
- `dashboard.html`: the finished dashboard; open it in any browser, no install needed
- `requirements.txt`: Python packages (pandas, numpy, pyarrow)

## Requirements

- Python 3.11 or newer
- The data from https://github.com/tamu-datathon-org/dashboard-casino (Code, then Download ZIP, then unzip)

## Running

From this folder:

```
pip install -r requirements.txt
python build_dashboard.py --data path/to/dashboard-casino-main/data --out output
```

On Mac, use `pip3` and `python3`.

`--data` is the unzipped data folder. The script searches it for each table by name, so the folder layout does not matter. `--out` is where results are written (default: `output`).

## Output

Written to the `--out` folder:

- `dashboard.html`: the finished dashboard
- `machine_scorecard.csv`: one row per machine, with luck score, availability, peer index and verdict
- `zone_summary.csv`: wagers, hold and available hours by zone

The script also prints the forecast error against the flat baseline when it finishes.

## Troubleshooting

- `Could not find <name>.parquet`: `--data` points to the wrong folder. Use the unzipped `data` folder.
- `ModuleNotFoundError`: run the `pip install` line again, in the same terminal.
- The script reads `dashboard_template.html` from its own folder, so keep the two files together.

## AI-use disclosure

Claude (Anthropic) was used to explore the data, write and revise the analysis code and dashboard, and draft the explanations, write-up and this README. Our team reviewed the results.