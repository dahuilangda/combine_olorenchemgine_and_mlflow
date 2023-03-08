import argparse
import os
import shutil
import olorenchemengine as oce
import pandas as pd
import uuid
import numpy as np
import json

from rdkit import Chem 
from rdkit.Chem import AllChem, DataStructs

import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature

def get_cls_dataset_and_model(args, df):
    cls_dataset = oce.BaseDataset(data=df.to_csv(), structure_col=args.mol_col, property_col=args.prop_col)
    cls_splitter = oce.RandomSplit(split_proportions=[0.8,0.1,0.1])
    cls_dataset = cls_splitter.transform(cls_dataset)

    cls_model = oce.BaseBoosting([
        oce.RandomForestModel(oce.DescriptastorusDescriptor("morgan3counts"), n_estimators=1000),
        oce.RandomForestModel(oce.DescriptastorusDescriptor("rdkit2dnormalized"), n_estimators=1000),
        oce.RandomForestModel(oce.OlorenCheckpoint("default"), n_estimators=1000),
    ])

    return cls_dataset, cls_model

def get_reg_dataset_and_model(args, df):
    reg_dataset = oce.BaseDataset(data=df.to_csv(), structure_col=args.mol_col, property_col=args.prop_col)
    reg_splitter = oce.RandomSplit(split_proportions=[0.8,0.1,0.1])
    reg_dataset = reg_splitter.transform(reg_dataset)

    reg_model = oce.BaseBoosting([
        oce.RandomForestModel(oce.DescriptastorusDescriptor("morgan3counts"), n_estimators=1000),
        oce.RandomForestModel(oce.DescriptastorusDescriptor("rdkit2dnormalized"), n_estimators=1000),
        oce.RandomForestModel(oce.OlorenCheckpoint("default"), n_estimators=1000),
    ])

    return reg_dataset, reg_model

# Create an `artifacts` dictionary that assigns a unique name to the saved XGBoost model file.
# This dictionary will be passed to `mlflow.pyfunc.save_model`, which will copy the model file
# into the new MLflow Model's directory.

class OCEWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.oce_model = oce.load(context.artifacts["oce_model"])

    def predict(self, context, model_input):
        return self.oce_model.predict(model_input)


def main(args):
    df = pd.read_csv(args.input)
    oce_model_path = f"/tmp/{uuid.uuid4().hex}.oce"

    with mlflow.start_run():
        if args.task_type == 'cls':
            clf_data, clf_model = get_cls_dataset_and_model(args, df)
            clf_model.fit(*clf_data.train_dataset)
            # Predict on the test set
            results = clf_model.test(*clf_data.test_dataset)
            # Save the model
            oce.save(clf_model, oce_model_path)

        elif args.task_type == 'reg':
            reg_data, reg_model = get_reg_dataset_and_model(args, df)
            error_model = oce.Predicted()
            reg_model.fit(*reg_data.train_dataset)
            reg_model.create_error_model(error_model,
                    reg_data.train_dataset[0], reg_data.train_dataset[1],
                    reg_data.valid_dataset[0], reg_data.valid_dataset[1],
                    ci=0.80, method = "roll")
            results = reg_model.test(*reg_data.test_dataset)
            # Save the model
            oce.save(reg_model, oce_model_path)
        else:
            raise ValueError("Task type must be either cls(classification) or reg(regression)")

        for k, v in results.items():
            print(f"{k}: {v}")
            mlflow.log_metric(k, v)

        # # Save results (dict)
        # with open('cls_results.json', 'w') as f:
        #     json.dump(results, f)

        # Save the MLflow Model
        mlflow_pyfunc_model_path = args.output_path
        if os.path.exists(mlflow_pyfunc_model_path):
            shutil.rmtree(mlflow_pyfunc_model_path)

        artifacts = {
            "oce_model": f'{oce_model_path}',
        }

        mlflow.pyfunc.save_model(
                path=mlflow_pyfunc_model_path, python_model=OCEWrapper(), artifacts=artifacts)

        # infer the signature
        if args.task_type == 'cls':
            signature = infer_signature(clf_data.test_dataset[0], clf_model.predict(clf_data.test_dataset[0]))
        elif args.task_type == 'reg':
            signature = infer_signature(reg_data.test_dataset[0], reg_model.predict(reg_data.test_dataset[0]))
        else:
            raise ValueError("Task type must be either cls(classification) or reg(regression)")

        # log the model
        mlflow.pyfunc.log_model(
            artifact_path="pyfunc_model",
            python_model=OCEWrapper(),
            conda_env=f"{mlflow_pyfunc_model_path}/conda.yaml",
            artifacts=artifacts,
            signature=signature,
        )

        # delete the oce model
        if os.path.exists(oce_model_path):
            os.remove(oce_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='Input data file (Dataframe)')
    parser.add_argument('-m', '--mol_col', type=str, help='Molecule column name', default='SMILES')
    parser.add_argument('-p', '--prop_col', type=str, help='Property column name', default='Value')
    parser.add_argument('-t', '--task_type', type=str, help='Task type (cls or reg)', default='cls')
    parser.add_argument('-o', '--output_path', type=str, help='Output path', default='oce_pyfunc_model')
    args = parser.parse_args()

    main(args)