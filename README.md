
# smart-bbt-analysis-fixed

Equal-days leveling up to **Phase step** (not to coverage target), with strict donor floor at **Coverage target**.
No `apply(axis=1)` in core calculations — fully vectorized to avoid pandas assignment errors.

## How it works
- **AvgDaily** = Sales / Sales period (days).
- **Donor floor** = ceil(min(coverage_days * AvgDaily, MonthTarget)): donor never drops below coverage target.
- **Receiver target** this run = ceil(min(phase_step * AvgDaily, MonthTarget)): we only raise up to Phase step.
- Greedy allocation from largest-free donors to lowest-coverage receivers.
- Final output grouped by `Item Code, Item Name, From_Branch, To_Branch`.

## Run
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
