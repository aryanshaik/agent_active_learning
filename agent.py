#!/usr/bin/env python
"""Agent-driven AL loop using DeepSeek API directly (no opencode dependency)."""
import os, re, sys, subprocess, json
import requests

API_KEY = os.environ["DEEPSEEK_API_KEY"]
API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"

WORK_DIR = "/n/data1/hms/dbmi/farhat/aryan/AL/agent_active_learning"

SYSTEM_PROMPT = """You are an autonomous active learning researcher optimizing molecular property prediction.

Your goal: maximize test_auroc on a held-out test set through iterative active learning.

You control:
- Acquisition weights: W_INHIBITION, W_UNCERTAINTY, W_NOVELTY, W_DIVERSITY in al_optimizer_chemprop.py
- Model hyperparameters: NUM_FOLDS, EPOCHS, BATCH_SIZE, HIDDEN_SIZE, DEPTH
- Selection strategy: BATCH_DIVERSE (True/False), SELECTION_SIZE
- The acquisition_score() function itself

Architecture: Chemprop DMPNN (3-layer, hidden=300) with Dirichlet evidential loss.
Training: 5-fold scaffold-balanced CV, 200 epochs. ~60 min/iteration on GPU.
Prediction: ~10 min for 86K pool molecules.
Pool: ~96,000 unlabeled. Train starts at ~10,000. Select 1,000 per iteration.

Rules:
- No label leakage. Pool labels revealed only after selection.
- Keep every commit. Never git reset.
- First run = baseline (submitted weight-0).

Respond with a JSON object:
{"analysis": "...", "changes": {"W_INHIBITION": 0.5, "W_UNCERTAINTY": 1.0, "W_NOVELTY": 0.3, "W_DIVERSITY": 0.2, "SELECTION_SIZE": 1000, "BATCH_DIVERSE": false, "EPOCHS": 200}, "commit_message": "iter N: <description>", "continue": true}
"""


def call_agent(messages):
    """Call DeepSeek API and return response."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 2000,
    }
    resp = requests.post(API_URL, headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def apply_changes(changes):
    """Update al_optimizer_chemprop.py with new weights."""
    path = os.path.join(WORK_DIR, "al_optimizer_chemprop.py")
    with open(path) as f:
        code = f.read()
    for key, val in changes.items():
        if key in ["W_INHIBITION", "W_UNCERTAINTY", "W_NOVELTY", "W_DIVERSITY"]:
            code = re.sub(rf"^{key} = [\d.]+", f"{key} = {val}", code, flags=re.MULTILINE)
        elif key == "SELECTION_SIZE":
            code = re.sub(r"^SELECTION_SIZE = \d+", f"SELECTION_SIZE = {val}", code, flags=re.MULTILINE)
        elif key == "BATCH_DIVERSE":
            code = re.sub(r"^BATCH_DIVERSE = (True|False)", f"BATCH_DIVERSE = {val}", code, flags=re.MULTILINE)
        elif key == "EPOCHS":
            code = re.sub(r"^EPOCHS = \d+", f"EPOCHS = {val}", code, flags=re.MULTILINE)
    with open(path, "w") as f:
        f.write(code)


def run_iteration(iteration):
    """Run one AL iteration and return metrics."""
    cmd = [
        sys.executable, "-u", "al_optimizer_chemprop.py",
        "--train", "data/train_df_chemprop.csv",
        "--pool", "data/pool_df_chemprop.csv",
        "--test", "data/test_df_chemprop.csv",
        "--iters", "1",
        "--work_dir", "al_chemprop_runs",
    ]
    log_file = os.path.join(WORK_DIR, f"al_run_iter_{iteration}.log")
    if iteration == 0:
        log_file = os.path.join(WORK_DIR, "al_run.log")
    with open(log_file, "w") as f:
        result = subprocess.run(cmd, cwd=WORK_DIR, stdout=f, stderr=subprocess.STDOUT)
    with open(log_file) as f:
        stdout = f.read()

    # Parse FINAL_METRICS
    match = re.search(r"FINAL_METRICS iter=\d+: test_auroc=([\d.]+) test_auprc=([\d.]+) hit_rate=([\d.]+)", stdout)
    if match:
        return {"auroc": float(match.group(1)), "auprc": float(match.group(2)), "hit_rate": float(match.group(3))}
    return None


def main():
    os.chdir(WORK_DIR)
    # Ensure we are on the dmpnn branch (force — discard any local changes)
    subprocess.run(["git", "checkout", "-f", "al/dmpnn_may16"], cwd=WORK_DIR)
    subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=WORK_DIR)
    # Verify the required file exists
    if not os.path.exists(os.path.join(WORK_DIR, "al_optimizer_chemprop.py")):
        print("ERROR: al_optimizer_chemprop.py not found! Check branch.")
        sys.exit(1)
    print("Branch OK:", subprocess.run(["git", "branch", "--show-current"], cwd=WORK_DIR, capture_output=True, text=True).stdout.strip())
    history = []

    # Read results.tsv for trajectory
    tsv_path = os.path.join(WORK_DIR, "results.tsv")
    if os.path.exists(tsv_path):
        with open(tsv_path) as f:
            history = f.read()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    for iteration in range(10):
        print(f"\n{'='*60}")
        print(f"  Agent Iteration {iteration}")
        print(f"{'='*60}")

        # Build context for agent
        context = f"Iteration {iteration} of 10. Results so far:\n{history}\n\n"
        if iteration == 0:
            context += "This is the BASELINE. Use default weights: W_INHIBITION=0.5, W_UNCERTAINTY=1.0, W_NOVELTY=0.3, W_DIVERSITY=0.2. Run unmodified to establish baseline."
        else:
            context += "Analyze the trajectory. What weights would improve test_auroc? Modify acquisition function to beat the baseline. Be creative."

        messages.append({"role": "user", "content": context})

        # Get agent decision
        resp = call_agent(messages)
        content = resp["choices"][0]["message"]["content"]
        messages.append({"role": "assistant", "content": content})

        # Parse JSON from response
        try:
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            decision = json.loads(json_match.group(0))
        except (json.JSONDecodeError, AttributeError):
            print(f"  Agent response not parseable: {content[:500]}")
            continue

        print(f"  Analysis: {decision.get('analysis', 'N/A')[:200]}")
        changes = decision.get("changes", {})
        print(f"  Changes: {changes}")

        # Apply changes and commit
        apply_changes(changes)
        commit_msg = decision.get("commit_message", f"iter {iteration}: agent update")

        cmd = ["git", "add", "al_optimizer_chemprop.py"]
        subprocess.run(cmd, cwd=WORK_DIR)
        cmd = ["git", "commit", "-m", commit_msg]
        subprocess.run(cmd, cwd=WORK_DIR)

        # Run iteration
        print(f"  Running iteration {iteration}...")
        metrics = run_iteration(iteration)
        if metrics is None:
            print("  ERROR: iteration failed!")
            with open(tsv_path, "a") as f:
                f.write(f"unknown\t{iteration}\t0.0\t0.0\t0.0\tcrash\tAgent iteration failed\n")
            continue

        # Log results
        auroc, auprc, hit_rate = metrics["auroc"], metrics["auprc"], metrics["hit_rate"]

        # Get commit hash
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=WORK_DIR, capture_output=True, text=True)
        commit = result.stdout.strip()

        with open(tsv_path, "a") as f:
            f.write(f"{commit}\t{iteration}\t{auroc:.6f}\t{auprc:.6f}\t{hit_rate:.6f}\t{'baseline' if iteration == 0 else 'run'}\t{commit_msg}\n")

        # Push every 5
        if iteration % 5 == 0 and iteration > 0:
            subprocess.run(["git", "push", "origin", "HEAD"], cwd=WORK_DIR)

        print(f"  FINAL_METRICS iter={iteration}: test_auroc={auroc:.6f} test_auprc={auprc:.6f} hit_rate={hit_rate:.6f}")

        # Update history for next iteration
        with open(tsv_path) as f:
            history = f.read()

        if not decision.get("continue", True):
            break

    print(f"\n=== ALL DONE ===")
    subprocess.run(["git", "push", "origin", "HEAD"], cwd=WORK_DIR)


if __name__ == "__main__":
    main()
