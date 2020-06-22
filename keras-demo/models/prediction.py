import os

import mlflow.keras
import mlflow.sklearn
from keras import Sequential
from keras.layers import Dropout, Dense
from sklearn import metrics
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import utils


class KerasModel:
    """
    DecisionTree classifier to predict binary labels(malignant and benign) of  cancer dataset.
    """

    def __init__(self, dropout):
        """
        Constructor
        :param model_params: parameters (key-value) for the tree model such as no of estimators, depth of the tree, random_state etc
        """
        self.dropout = dropout
        self.keras_classifier = self._get_keras_classifier()
        self.data = load_breast_cancer()

    @property
    def model(self):
        """
        Getter for the property the model
        :return: return the trained decision tree model
        """

        return self._keras_classifier

    def mlflow_run(self, run_name="Breast Cancer Keras Classification Run"):
        """
        This method trains, computes metrics, and logs all metrics, parameters,
        and artifacts for the current run
        :param run_name: Name of the experiment as logged by MLflow
        :return: MLflow Tuple (experiment_id, run_id)
        """

        with mlflow.start_run(run_name=run_name) as run:

            # get current run and experiment id
            run_id = run.info.run_uuid
            experiment_id = run.info.experiment_id

            # split the data into train and test
            X_train, X_test, y_train, y_test = train_test_split(self.data.data,
                                                                self.data.target,
                                                                test_size=0.25,
                                                                random_state=23)

            # Feature Scaling
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            # Auto-logging
            mlflow.keras.autolog()
            # train and predict
            self.keras_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            self.keras_classifier.fit(X_train, y_train, batch_size=50, nb_epoch=100)
            y_probs = self.keras_classifier.predict(X_test)
            y_pred = (y_probs > 0.5)

            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)

            roc = metrics.roc_auc_score(y_test, y_pred)

            # confusion matrix values
            tp = conf_matrix[0][0]
            tn = conf_matrix[1][1]
            fp = conf_matrix[0][1]
            fn = conf_matrix[1][0]

            # get classification metrics
            class_report = classification_report(y_test, y_pred, output_dict=True)
            recall_0 = class_report['0']['recall']
            f1_score_0 = class_report['0']['f1-score']
            recall_1 = class_report['1']['recall']
            f1_score_1 = class_report['1']['f1-score']

            # create confusion matrix plot
            plt_cm, fig_cm, ax_cm = utils.plot_confusion_matrix(y_test, y_pred, y_test,
                                                                title="Classification Confusion Matrix")

            temp_name = "confusion-matrix.png"
            fig_cm.savefig(temp_name)
            mlflow.log_artifact(temp_name, "confusion-matrix-plots")
            try:
                os.remove(temp_name)
            except FileNotFoundError as e:
                print(f"{temp_name} file is not found")

            # create roc plot

            plot_file = "roc-auc-plot.png"
            print("**************************************")
            print(y_test, y_probs.ravel(), type(y_test), type(y_probs))
            probs = y_probs.ravel()  # [:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, probs)
            plt_roc, fig_roc, ax_roc = utils.create_roc_plot(fpr, tpr)
            fig_roc.savefig(plot_file)
            mlflow.log_artifact(plot_file, "roc-auc-plots")
            try:
                os.remove(plot_file)
            except FileNotFoundError as e:
                print(f"{temp_name} file is not found")

            print("<->" * 40)
            print("Inside MLflow Run with run_id {run_id} and experiment_id {experiment_id}")
            print("dropout of the Keras Model:", self.dropout)
            print(conf_matrix)
            print(classification_report(y_test, y_pred))
            print("Accuracy Score =>", acc)
            print("Precision      =>", precision)
            print("ROC            =>", roc)

            return experiment_id, run_id

    def _get_keras_classifier(self):
        classifier = Sequential()
        classifier.add(Dense(output_dim=16, init='uniform', activation='relu', input_dim=30))
        classifier.add(Dropout(p=self.dropout))

        # Adding 2nd layer
        classifier.add(Dense(output_dim=16, init='uniform', activation='relu'))

        # Adding dropout to prevent overfitting
        classifier.add(Dropout(p=self.dropout))

        # Adding the output layer
        classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

        return classifier
