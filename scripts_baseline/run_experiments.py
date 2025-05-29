import subprocess

import yaml

with open("experiments.yaml") as f:
    config = yaml.safe_load(f)

for exp in config["experiments"]:
    args = [
        "python",
        "experiment.py",
        f"--model={exp['model']}",
        f"--feeding={exp['feeding']}",
        f"--augmentation={exp['augmentation']}",
        f"--learning_rate={exp['learning_rate']}",
        f"--batch_size={exp['batch_size']}",
        f"--run_name={exp['name']}" f"--run_name={exp['img_size']}",
    ]
    subprocess.run(args)
