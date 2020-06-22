from models.prediction import KerasModel

dropouts = [0.1, 0.2]
if __name__ == '__main__':
    for dropout in dropouts:
        keras_classifier = KerasModel(dropout)
        exp_id, run_id = keras_classifier.mlflow_run()
        print(f"MLflow Run completed with run_id {exp_id} and experiment_id {run_id}")
        print("<->" * 40)
