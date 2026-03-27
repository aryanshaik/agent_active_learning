import os
import subprocess
import time

def submit_slurm_job(job_name, command, partition="gpu_quad", gpus=1, memory="40G", time_limit="0-12:00", cpus=8):
    """
    Submits a job to SLURM using a temporary sbatch script.
    """
    log_dir = "run_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    sbatch_content = f"""#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -c {cpus}
#SBATCH -t {time_limit}
#SBATCH -p {partition}
#SBATCH --gres=gpu:{gpus}
#SBATCH --mem={memory}
#SBATCH -o {log_dir}/{job_name}_%j.out
#SBATCH -e {log_dir}/{job_name}_%j.err

module load conda/miniforge3/24.11.3-0
module load gcc/14.2.0
module load cuda/12.8
eval "$(conda shell.bash hook)"
conda activate /home/ars3983/miniforge3/envs/al-agent

{command}
"""
    
    script_path = f"{job_name}.sbatch"
    with open(script_path, "w") as f:
        f.write(sbatch_content)
    
    result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
    
    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        print(f"Submitted job {job_name} with ID {job_id}")
        return job_id
    else:
        print(f"Error submitting job: {result.stderr}")
        return None

def wait_for_job(job_id, check_interval=60):
    """
    Wait for a SLURM job to complete.
    """
    while True:
        result = subprocess.run(["squeue", "-j", job_id], capture_output=True, text=True)
        if job_id not in result.stdout:
            break
        time.sleep(check_interval)
    print(f"Job {job_id} finished.")
def get_tanimoto_similarity(smiles_list_a, smiles_list_b=None):
    """
    Calculates Tanimoto similarity between molecules.
    If smiles_list_b is None, calculates internal pairwise similarity (Diversity).
    If smiles_list_b is provided, calculates similarity to reference set (Novelty).
    """
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
    except ImportError:
        print("RDKit not found. Skipping similarity calculation.")
        return None

    fps_a = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2) for s in smiles_list_a if Chem.MolFromSmiles(s)]
    
    if smiles_list_b:
        fps_b = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2) for s in smiles_list_b if Chem.MolFromSmiles(s)]
        # Calculate max similarity of each A to any B
        similarities = [max(DataStructs.BulkTanimotoSimilarity(fp, fps_b)) for fp in fps_a]
    else:
        # Internal diversity: average pairwise similarity within A
        if len(fps_a) < 2: return 0
        sims = []
        for i in range(len(fps_a)):
            sims.extend(DataStructs.BulkTanimotoSimilarity(fps_a[i], fps_a[i+1:]))
        similarities = sims

    return np.mean(similarities)
