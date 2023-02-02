# Evaluate most recent with best
# if the most recent model version's eval metric is greater than X:

    # if there's a promoted model version, compare with most recent. If most recent is better than previous, promote it. 
    model_version.
    # if there's no promoted model version, promote most recent, deploy new model_version to endpoint
         
        # Promotions Create/Get with Id
            model.promotions.create(
                id="test-promotion",
                model_version=model_version.name,
                state="FAILED",
                reason="UPLIFT",
                base_improvement_metric_value=0.8,
                improvement_metric_value=0.9,
                improvement_metric="accuracy",
                improvement_metric_diff=0.1,
            )
        # Promote best performing
            #https://github.com/continual-ai/continual/blob/main/continual/python/sdk/promotions.py
            #model.promotions.create(model_version=model_version.name, reason="UPLIFT")

        # model.promotions.create(model_version=model_version.name, reason="UPLIFT")


        # Define a SKLearn Transformer from the trained SKLearn Estimator
        transformer = estimator.transformer(
            instance_count=1,
            instance_type="ml.m5.xlarge",
            assemble_with="Line",
            accept="text/csv"
        )

        test_data_path = sagemaker_session.upload_data('chimpanzee.txt')

        # Feed the test data
        transformer.transform(test_data_path, content_type="text/csv")
        print("Waiting for transform job: " + transformer.latest_transform_job.job_name)
        transformer.wait()
        output = transformer.output_path
        print(output)