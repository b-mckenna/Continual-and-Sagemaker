   # Run batch predictions and log
    model_version.batch_predictions.create(
        id="my_batch_prediction", model_version=model_version.name
    )