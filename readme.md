# Freight Rate Prediction Challenge

See `Freight_Rate_ML_Assessment.pdf` for the assessment instructions.

## What to do

1. Train and validate your model using `train-test.csv`.
2. Predict every load in `validation.csv`. Each load has a unique `load_id`.
3. Fill the matching `predicted_rate` values in `validation-predictions-template.csv` and save it as `validation_predictions.csv`.
4. Predict every row in `december-chart-inputs.csv` by filling its `predicted_rate` column.
5. Install the scorer requirements and run:

```bash
python -m pip install -r requirements.txt
python train_predict.py
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

The scorer validates both files and creates `scorer_results/candidate_december.png`.

## Submit

- GitHub repository containing your code, dependencies, and run instructions
- `validation_predictions.csv`
- PDF or DOCX report containing your validation, data split approach and `candidate_december.png`
- 2-3 minute Loom link covering:
  - Key findings from exploring the data
  - Any data-quality issues identified and how you addressed them
  - Reasoning behind the chosen model
  - Training and validation approach, including how the data was split
  - A brief walkthrough of the most important parts of your code
