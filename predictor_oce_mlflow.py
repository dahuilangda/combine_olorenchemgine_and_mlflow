import olorenchemengine as oce
from rdkit import Chem
from rdkit.Chem import Draw

import argparse
import requests
import json


class ModelWrapper:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port

    def predict(self, smiles_list):

        url = f'http://{self.ip}:{self.port}/invocations'
        headers = {'Content-Type': 'application/json'}
        data ={
            "dataframe_split": {
                "columns": ["Drug"],
                "data": smiles_list}
        }
        data = json.dumps(data)
        response = requests.post(url, headers=headers, data=data)
        preds = response.json()
        return preds['predictions'] if 'predictions' in preds else None
                                 

class ActiveLearning(object):
    """Scores based on active learning models"""

    def __init__(self, ip, port):
        self.model = ModelWrapper(ip, port)

    def _clf_score(self, smiles):
        return self.model.predict([smiles])[0]
    
    def _reg_score(self, smiles):
        pred = self.model.predict([smiles])
        value = pred.predicted.to_list()[0]
        ci = pred.ci.to_list()[0] # ci (down, up)
        value = f'{value:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]'
        return value
    
    def score(self, smiles, type='cls'):
        if type == 'cls':
            return self._clf_score(smiles)
        elif type == 'reg':
            return self._reg_score(smiles)
        else:
            raise ValueError(f'Unknown type: {type}, must be "cls" or "reg"')
    
    def _to_rgb(self, color):
        # '#fde725' -> (0.99, 0.90, 0.14)
        color = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        color = (color[0]/255, color[1]/255, color[2]/255)
        return color

    def visualizer(self, smiles, output):
        vis = oce.VisualizePredictionSensitivity(self.model, smiles)
        smi = vis.get_data()['SMILES']
        highlight_info_list = vis.get_data()['highlights']
        mol = Chem.MolFromSmiles(smi)

        highlights = []
        highlights_color = {}
        for atom in mol.GetAtoms():
            for highlight_info in highlight_info_list:
                if atom.GetAtomMapNum() == highlight_info[0]:
                    highlights.append(atom.GetIdx())
                    highlights_color[atom.GetIdx()] = self._to_rgb(highlight_info[1])
                    atom.SetAtomMapNum(0)

        view = Draw.rdMolDraw2D.MolDraw2DSVG(300, 300)
        tm = Draw.rdMolDraw2D.PrepareMolForDrawing(mol)
        highlight_size = 0.5
        view.DrawMolecule(tm, highlightAtoms=highlights, 
                        highlightAtomColors=highlights_color, 
                        highlightAtomRadii={atom.GetIdx(): highlight_size for atom in mol.GetAtoms()})
        view.FinishDrawing()
        svg = view.GetDrawingText()
        with open(output, 'w') as f:
            f.write(svg)
        return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smiles', type=str, required=True, help='SMILES string, e.g. "CC(=O)Nc1ccc2c(c1)C(=O)N(Cc1ccc(Cl)cc1)C2=O"')
    parser.add_argument('--ip', type=str, required=False, help='IP address of the server, e.g. "127.0.0.1" (default)', default='127.0.0.1')
    parser.add_argument('--port', type=str, required=False, help='Port of the server, e.g. "1234" (default)', default='1234')
    parser.add_argument('--type', type=str, required=False, help='Type of the model, e.g. "cls" (default) or reg', default='cls')
    parser.add_argument('--vis', type=str, required=False, help='Visualize the prediction sensitivity, e.g. visualize.svg')

    args = parser.parse_args()

    _active = ActiveLearning(args.ip, args.port)
    score = _active.score(args.smiles)
    if args.type == 'cls':
        print(f'Classification score: {score}')
    elif args.type == 'reg':
        print(f'Regression score: {score}')

    if args.vis:
        print(f'Visualize the prediction sensitivity: {args.vis}')
        _active.visualizer(args.smiles, args.vis)