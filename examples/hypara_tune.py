# %%
import hydra
import mlflow
import optuna
import random

from mlflow import log_metric
from omegaconf import DictConfig, OmegaConf, open_dict


import sys
sys.path.append("..")
import hytunaflow

from train import evaluate

hytunaflow.enable_hydra_with_ipython(__builtins__)


# %%
def objective_wrapper(hyflow):
    def objective(trial: optuna.trial.Trial):
        train_config = hyflow.create_optuna_train_config(trial)
        return evaluate(train_config)
    return objective


# %%
@hydra.main(config_path="./conf", config_name="hypara_tune", version_base=None)
def tuning(config: DictConfig):
    hyflow = hytunaflow.Hytunaflow(config)
    print("tuning:", OmegaConf.to_yaml(config))

    study = hyflow.create_optuna_study()
    study.optimize(objective_wrapper(hyflow), n_trials=config.params.n_trials, callbacks=[hyflow.save_optuna_hist_callback])
    mlflow.end_run()


@hydra.main(config_path="./conf", config_name="hypara_tune", version_base=None)
def tuning_wrapper(config: DictConfig):
    optuna_config = config.optuna
    hytunaflow.set_keyval2DictConfig(optuna_config, "train", config.train)
    tuning(optuna_config)


if __name__ == "__main__":
    tuning_wrapper()

# %%
import glob
import yaml
import os

from typing import Tuple
# %%
mlruns_dir_path = "mlruns"
src_exp_name = "test_tune1"
src_run_name = "lyrical-gnu-501"
def exp_run_name2id(mlruns_dir_path: str, src_exp_name: str, src_run_name: str) -> Tuple[str, str]:
    """mlrun's {exp/run} name to id.

    Args:
        mlruns_dir_path (str): 
        src_exp_name (str): 
        src_run_name (str): run_name should be in src_exp_name.

    Returns:
        Tuple[str, str]: (exp_id, run_id)
    """
    exp_id = None
    exp_paths = glob.glob(f"{mlruns_dir_path}/*")
    for exp_path in exp_paths:
        exp_meta_path = f"{exp_path}/meta.yaml"
        if not os.path.exists(exp_meta_path):
            continue
        with open(exp_meta_path) as file:
            obj = yaml.safe_load(file)
        exp_name = obj["name"]
        if exp_name == src_exp_name:
            exp_id = os.path.basename(exp_path)
            break
    # print(f"exp_id:{exp_id}")

    run_paths = glob.glob(f"{mlruns_dir_path}/{exp_id}/*")
    for run_path in run_paths:
        run_meta_path = f"{run_path}/meta.yaml"
        if not os.path.exists(run_meta_path):
            continue
        with open(run_meta_path) as file:
            obj = yaml.safe_load(file)
        run_name = obj["run_name"]
        if run_name == src_run_name:
            exp_id = os.path.basename(exp_path)
            run_id = os.path.basename(run_path)
            break
    # print(f"run_id:{run_id}")

    return exp_id, run_id

# %%
mlruns_dir_path = "mlruns"
exp_paths = glob.glob(f"{mlruns_dir_path}/*")
exp_name2id = {}
exp_run_name2id = {}
for exp_path in exp_paths:
    exp_meta_path = f"{exp_path}/meta.yaml"
    if not os.path.exists(exp_meta_path):
        continue
    with open(exp_meta_path) as file:
        obj = yaml.safe_load(file)
    exp_name = obj["name"]
    exp_id = os.path.basename(exp_path)
    exp_name2id[exp_name] = exp_id

    exp_run_name2id[exp_name] = {}
    run_paths = glob.glob(f"{exp_path}/*")
    for run_path in run_paths:
        run_meta_path = f"{run_path}/meta.yaml"
        if not os.path.exists(run_meta_path):
            continue
        with open(run_meta_path) as file:
            obj = yaml.safe_load(file)
        run_name = obj["run_name"]
        run_id = os.path.basename(run_path)
        exp_run_name2id[exp_name][run_name] = run_id

# %%
exp_name2id
exp_run_name2id
# %%
