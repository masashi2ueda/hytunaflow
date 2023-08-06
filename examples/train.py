# %%
import hydra
import mlflow
import random

from mlflow import log_metric
from omegaconf import DictConfig, OmegaConf


import sys
sys.path.append("..")
import hytunaflow

hytunaflow.enable_hydra_with_ipython(__builtins__)


# %%
def evaluate(config: DictConfig):
    hyflow = hytunaflow.Hytunaflow(config)
    print("evaluate:", OmegaConf.to_yaml(config))

    temp_val1 = random.randint(0, 10)
    hyflow.save_add_result_yaml("temp_val1", temp_val1)
    log_metric("temp_val1", temp_val1)
    temp_val2 = random.randint(0, 10)
    hyflow.save_add_result_yaml("temp_val2", temp_val2)
    log_metric("temp_val2", temp_val2)

    val = random.randint(config.params.p1, config.params.p2)
    log_metric("val", val)
    mlflow.end_run()

    return val


@hydra.main(config_path="./conf", config_name="train", version_base=None)
def evaluate_wrapper(config: DictConfig):
    evaluate(config.train)


# %%
if __name__ == "__main__":
    evaluate_wrapper()

# %%
