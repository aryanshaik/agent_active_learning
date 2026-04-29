import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
import torch


class Minimol:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, smiles_list):
        embs = []
        for s in smiles_list:
            fp = self._get_fingerprint(s)
            embs.append(torch.tensor(fp, dtype=torch.float32))
        return torch.stack(embs).to(self.device)

    def _get_fingerprint(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(512, dtype=np.float32)
        morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
        arr = np.zeros((1,), dtype=np.float32)
        Chem.DataStructs.ConvertToNumpyArray(morgan, arr)
        return arr.astype(np.float32)
