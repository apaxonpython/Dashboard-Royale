# Dashboard Royale: The Gilded Ace

Everything needed to reproduce our analysis and dashboard.

## Files
- `build_dashboard.py`: reads the raw data, computes every number, writes the dashboard
- `dashboard_template.html`: the dashboard page (charts, filters, the $100,000 plan)
- `dashboard.html`: the finished dashboard. Double-click it to open in a browser, no install needed
- `requirements.txt`: Python packages

## Run it yourself
1. Install Python 3.11 or newer from python.org (tick "Add Python to PATH" on Windows).
2. Download the data: https://github.com/tamu-datathon-org/dashboard-casino (Code, then Download ZIP, then unzip).
3. Open a terminal in this folder and run:

       pip install -r requirements.txt
       python build_dashboard.py --data path/to/dashboard-casino-main/data --out output

   (On Mac, use `pip3` and `python3`.) Open `output/dashboard.html`. The script also writes
   `machine_scorecard.csv` and `zone_summary.csv`.

## Put the dashboard online
`dashboard.html` is one self-contained file. Drag it onto https://app.netlify.com/drop, or upload it to
a GitHub repository and turn on GitHub Pages. Test the link in a private browser window.

## Method notes
- Gaming days and exposure: rows before install or after retirement are dropped; hours come from `calendar_day`.
- Luck: actual house win vs theo, divided by the typical swing for the machine's volatility class (approximate).
- Ratios are sums over sums, not averages of daily ratios.
- Forecast: weekday average from Jan to Apr, tested on May to Jun against a flat Jan to Apr average. Split by time.
- Event lift: event-day wagers vs the average non-event day with the same weekday.
- Downtime loss: lost hours x the machine's wagers per available hour x floor hold.
- The $100,000 plan (list `P` in `dashboard_template.html`) is written by hand from these results; its assumptions are listed on the dashboard.

## AI-use disclosure
Claude (Anthropic) was used to explore the data, write the analysis code and dashboard, and draft explanations.
Our team reviewed the results.
