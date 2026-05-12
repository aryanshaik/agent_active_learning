import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch


def _encode_chunk(args):
    smiles_list, n_bits, radius = args
    from rdkit import Chem
    from rdkit.Chem import AllChem
    embs = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    for i, s in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            arr = np.zeros((1,), dtype=np.float32)
            Chem.DataStructs.ConvertToNumpyArray(fp, arr)
            embs[i] = arr
    return embs


class Minimol:
    def __init__(self, n_bits=512, radius=2, n_jobs=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_bits = n_bits
        self.radius = radius
        self.n_jobs = n_jobs or int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

    def __call__(self, smiles_list):
        n = len(smiles_list)
        if self.n_jobs <= 1 or n < 5000:
            return torch.tensor(
                _encode_chunk((smiles_list, self.n_bits, self.radius)),
                dtype=torch.float32,
            ).to(self.device)

        chunk_size = max(1, n // self.n_jobs)
        chunks = [smiles_list[i:i + chunk_size] for i in range(0, n, chunk_size)]
        args = [(c, self.n_bits, self.radius) for c in chunks]

        with ThreadPoolExecutor(max_workers=self.n_jobs) as ex:
            results = list(ex.map(_encode_chunk, args))

        return torch.tensor(np.concatenate(results, axis=0), dtype=torch.float32).to(self.device)
