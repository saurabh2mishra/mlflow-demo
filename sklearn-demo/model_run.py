import mlflow

from models.prediction import TreeModel

max_depths = [5, 8, 10]
if __name__ == '__main__':
    for n in max_depths:
        params = {"max_depth": n, "random_state": 42}
        dtc = TreeModel.create_instance(**params)
        exp_id, run_id = dtc.mlflow_run()
        print(f"MLflow Run completed with run_id {exp_id} and experiment_id {run_id}")
        print("<->" * 40)

