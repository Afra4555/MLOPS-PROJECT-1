# Hotel Booking Cancellation Predictor (End-to-End MLOps)

An AI tool that predicts, at the moment a booking is made, whether a hotel reservation will be **cancelled** or **kept**, so a hotel can act early: reconfirm with the guest, ask for a deposit, or plan overbooking with confidence.

Built as a complete MLOps project: data in the cloud, a tuned machine learning model, experiment tracking, a web app, Docker packaging and an automated Jenkins pipeline that deploys to Google Cloud Run.

## Why this project?

About **1 in 3 bookings is cancelled** in the data used here (36,000+ reservations). A room that sits empty at the last minute is revenue that cannot be recovered, and late cancellations also distort pricing, staffing and forecasts. An early warning for risky bookings helps revenue managers, front-office teams and management.

The same approach applies to any business where a no-show or drop-out costs money, such as airlines, events, subscriptions and bank customer churn.

## Results

Evaluated on **6,047 held-out bookings the model never saw**, with the real mix of cancelled and kept bookings:

| Metric | Value |
|---|---|
| Overall accuracy | about 87% |
| Cancelled bookings caught (recall) | about 81% (1,493 of 1,842) |
| When it flags a cancellation, it is right (precision) | about 78% |

**What the data showed:** bookings made far ahead cancel the most. The median lead time is **122 days for cancelled bookings versus 39 days for kept ones**. Lead time and average room price are the strongest signals the model uses.

## How it works

```
Booking data (Google Cloud Storage)
    -> Ingestion and train/test split
    -> Cleaning, label encoding, skew handling, SMOTE (training data only), top-10 feature selection
    -> LightGBM model tuned with RandomizedSearchCV, runs logged in MLflow
    -> Flask web app (enter a booking, get a prediction)
    -> Docker image -> Jenkins CI/CD pipeline -> Google Cloud Run
```

## Tech stack

Python, pandas, NumPy, scikit-learn, LightGBM, imbalanced-learn (SMOTE), MLflow, Flask, Docker, Jenkins, GitHub, Google Cloud Storage and Google Cloud Run.

## Project structure

```
application.py            Flask web app (loads the trained model)
config/                   config.yaml, model parameters, path constants
src/                      data_ingestion, data_preprocessing, model_training, logging, exceptions
pipeline/                 training_pipeline.py (runs all stages in order)
utils/                    helper functions
templates/, static/       web form (HTML and CSS)
artifacts/                raw data, processed data and the trained model
notebook/                 exploratory data analysis
Dockerfile                container for the app
Jenkinsfile               CI/CD pipeline (clone, build, push, deploy)
custom_jenkins/           Jenkins image with Docker support
```

## Run the app locally

The trained model is included, so you can try the app without retraining.

```bash
git clone https://github.com/Afra4555/MLOPS-PROJECT-1.git
cd MLOPS-PROJECT-1
python -m venv venv
venv\Scripts\activate        # Windows (macOS/Linux: source venv/bin/activate)
pip install -e .
python application.py
```

Then open **http://localhost:8080**, enter a booking and click **Predict**.

Example to try: lead time 250, 0 special requests, Online segment gives *"going to cancel"*; lead time 5, 2 special requests, Offline segment gives *"not going to cancel"*.

## Retrain the model

Retraining downloads the dataset from a Google Cloud Storage bucket, so you need your own bucket and credentials:

1. Upload the Hotel Reservations dataset (CSV) to your bucket.
2. Set the bucket and file name in `config/config.yaml`.
3. Point `GOOGLE_APPLICATION_CREDENTIALS` to your service-account key file (never commit this key).
4. Run `python pipeline/training_pipeline.py`.
5. View experiments with `mlflow ui` and open http://127.0.0.1:5000.

## Docker and deployment

```bash
docker build -t hotel-cancellation .
docker run -p 8080:8080 hotel-cancellation
```

The `Jenkinsfile` defines four stages: clone from GitHub, set up the environment, build and push the Docker image, and deploy to Google Cloud Run.

## What I learned

- Building the whole pipeline around a model (data, tracking, packaging, automated delivery) matters as much as the model itself.
- Keep the test set untouched and report class-level metrics such as recall on cancelled bookings, not only overall accuracy.
- Automating delivery with CI/CD makes every change repeatable.

## Next steps

- Explain predictions with SHAP
- Monitor data drift and retrain on a schedule
- Wider hyperparameter search and added tests
- A dashboard of cancellation rate by segment, lead time and month

## Author

**Fathima Afra**. Feedback and ideas are welcome.
