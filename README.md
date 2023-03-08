Install:
--------

### Prerequisites

*   Install Pytorch: [https://pytorch.org/TensorRT/tutorials/installation.html](https://pytorch.org/TensorRT/tutorials/installation.html)
*   Install Pytorch Geometric: [https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

### Create a conda environment and install dependencies

```bash
conda create -n ml python=3.9
conda activate ml
pip install olorenchemengine[full]
pip install pandas numpy 
pip install rdkit-pypi
pip install mlflow
pip install pytdc
```

Usage:
------

*   Download data from the 'hERG\_Karim' dataset in TDC (Therapeutics Data Commons) and saves it as a CSV file
*   'hERG\_Karim' dataset in TDC is a binary classification dataset.

```python
from tdc.single_pred import Tox
data = Tox(name='hERG_Karim')
df = data.get_data()
df.to_csv('data/hERG_Karim.csv', index=False)
```

*   Download data from the 'hERG\_central' dataset in TDC (Therapeutics Data Commons) and saves it as a CSV file
*   'hERG\_central' dataset in TDC is a regression dataset, which contains 3 labels. "hERG\_at\_10uM" is the label we want to use.

```python
from tdc.utils import retrieve_label_name_list
from tdc.single_pred import Tox
label_list = retrieve_label_name_list('herg_central')
print(label_list)
data = Tox(name = 'herg_central', label_name = label_list[1])
df = data.get_data()
df.to_csv('data/hERG_central_hERG_at_10uM.csv', index=False)
```

### Train a model using the 'hERG\_Karim' dataset

```bash
python training_oce_mlflow.py -i data/hERG_Karim.csv -m Drug -p Y -t cls -o models/hERG_Karim
```

*   After training the hERG classification model, set the trained model to production mode in MLflow ([https://mlflow.org/docs/latest/models.html#production-mode](https://mlflow.org/docs/latest/models.html#production-mode))
*   Assuming we named this model "hERG\_cls", we can deploy the service using the command below.
*   The service will be deployed at [http://127.0.0.1:1234/invocations](http://127.0.0.1:1234/invocations)

```bash
mlflow models serve -m "models:/hERG_cls/Production" -h 0.0.0.0 -p 1234 --env-manager local
```

*   Now we can use the standard API to predict the hERG classification of a small molecule.

```bash
curl -X POST -H "Content-Type:application/json" --data '{"dataframe_split": {"columns":["Drug"], "data": [["c1ccccc1"]]}}' http://127.0.0.1:1234/invocations
```

*   Use "predictor\_oce\_mlflow.py" script to predict both classification and sensitivity using "oce.VisualizePredictionSensitivity".

```bash
python predictor_oce_mlflow.py --smiles "CC(=O)Nc1ccc2c(c1)C(=O)N(Cc1ccc(Cl)cc1)C2=O" \
                               --vis vis.svg \
                               --ip "127.0.0.1" \
                               --port 1234
    args:
        --smiles: SMILES string of the molecule
        --vis: output file name of the visualization
        --ip: IP address of the MLflow service
        --port: port number of the MLflow service
```

### we can also train a regression model using the 'hERG_central' dataset in TDC by running the following command.
* The model is trained very slowly, please be patient.
* In addition to training the regression model, we also trained the confidence interval (80%).
```bash
python training_oce_mlflow.py -i data/hERG_central_hERG_at_10uM.csv \
                              -m Drug 
                              -p Y 
                              -t reg 
                              -o models/hERG_central
    args:
        -i: input file name
        -m: name of the molecule column
        -p: name of the label column
        -t: type of the task (cls or reg)
        -o: output directory
```
