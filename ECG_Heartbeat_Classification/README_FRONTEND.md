# ECG Assignment Front End

A Streamlit dashboard for the supplied ECG heartbeat classification project.

## Run it

Open a terminal in the `AML_Assignment3_ECG` project folder and run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`) for the dashboard.

## Pages

- **Overview:** project summary, key metrics and example ECG beats.
- **Model performance:** test metrics, per-class accuracy, confusion matrix and training plots.
- **Try a heartbeat:** upload a CSV with 187 numeric signal values to classify the first row using `results/ecgnet_best.pt`.
- **About the project:** dataset, architecture and workflow.

## Important files

The app expects the original project structure, including `src/model.py`, `results/test_results.json`, `results/data_info.json`, `results/best_model_info.json`, the result images, and `results/ecgnet_best.pt`.

The original dataset CSV files are not required to open the dashboard or classify an uploaded heartbeat. The saved trained model is required for predictions.

**Educational use only:** this project is not a medical diagnostic tool.
