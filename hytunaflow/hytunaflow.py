# %%
import hydra
import mlflow
import optuna
import os
import random
import shutil
import subprocess
import yaml

from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from typing import Any

from . import utils


class Hytunaflow:
    def __init__(
            self,
            cfg: DictConfig,
            is_set_random_seed: bool = True
    ):
        """
        Args:
            cfg (DictConfig): config dict from hydra.
            is_set_random_seed (bool, optional): Initialize a random number to avoid having same run_names.
                Defaults to True.
        """
        self.cfg = cfg

        if is_set_random_seed:
            random.seed(datetime.now().timestamp())

        if self.is_dummy:
            return

        mlflow.set_experiment(cfg.mlflow.experiment_name)

        self.flow_run = mlflow.start_run(nested=self.is_nested)

        self._save_git_hash()

        utils.log_params_from_omegaconf_dict(self.cfg)

        OmegaConf.save(cfg, f"{self.artifact_dir_path}/config.yaml")

    @property
    def artifact_dir_path(self) -> str:
        """get ml flow's artifact dir path.
        Returns:
            str: ml flow's artifact dir path.
        """
        return self.flow_run.info.artifact_uri.replace("file://", "")

    @property
    def is_nested(self) -> bool:
        """Whether the mlflow run is contained in a parent run.

        Returns:
            bool: nested or not.
        """
        if "is_nested" in self.cfg["mlflow"].keys():
            return self.cfg["mlflow"]["is_nested"]
        return False

    @property
    def is_dummy(self) -> bool:
        """If is_dummy is true. All materials are not saved.

        Returns:
            bool: is dummy or not.
        """
        if "is_dummy" in self.cfg["mlflow"].keys():
            return self.cfg["mlflow"]["is_dummy"]
        return False

    def _save_git_hash(self) -> None:
        """save git hash to mlflow log_param.
        """
        try:
            cmd = "git rev-parse --short HEAD"
            hash = subprocess.check_output(cmd.split()).strip().decode('utf-8')
            mlflow.log_param("git_commit_hash", hash)
        except:
            print("failed to get git commit hash id")

    def create_optuna_study(self) -> optuna.Study:
        """create optuna study config.
        If optuna.src_storage_path or {optuna.src_storage_experimentid, optuna.src_storage_runid} is specified, use it.
        direction = optuna.direction.
        studya_name = cfg.optuna.study_name.

        Returns
            optuna.Study: optuna studay.
        """
        # save to mlflow artifact path
        dst_storage_path = f"sqlite:///{self.artifact_dir_path}/optuna_study.db"
        dst_storage_file_path = dst_storage_path.replace("sqlite:///", "")

        if self.cfg["optuna"]["src_storage_path"] is not None:
            src_storage_path = self.cfg["optuna"]["src_storage_path"]
            shutil.copy(src_storage_path, dst_storage_file_path)
        elif self.cfg["optuna"]["src_storage_runid"] is not None:
            splts = dst_storage_file_path.split("/")
            splts[-2] = self.cfg["optuna"]["src_storage_experimentid"]
            splts[-3] = self.cfg["optuna"]["src_storage_runid"]
            src_storage_path = "/".join(splts)
            shutil.copy(src_storage_path, dst_storage_file_path)

        self.study = optuna.create_study(
            study_name=self.cfg.optuna.study_name,
            storage=dst_storage_path,
            direction=self.cfg["optuna"]["direction"],
            load_if_exists=True)
        return self.study

    def create_optuna_train_config(self):
        train_config_path = self.cfg.train.config_path
        train_cfg = DictConfig(yaml.load(open(train_config_path).read(), Loader=yaml.SafeLoader))
        train_cfg.mlflow.experiment_name = self.cfg.mlflow.experiment_name
        train_cfg["mlflow"]["nest"] = True
        return train_cfg

    def log_artifacts_hydra_output(self) -> None:
        """hydra output directory is registerd at mlflow artifact.
        """
        mlflow.log_artifacts(hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir'])

    def save_add_result_yaml(self, key: str, val: Any, file_name: str = "result_param") -> None:
        """save additional result or params to mlflow artifact's yaml file

        Args:
            key (str): name.
            val (Any): value.
            file_name (str, optional): save to {mlflow ardifact's dir}/{file_name}.yaml. Defaults to "result_param".
        """
        yaml_path = f"{self.artifact_dir_path}/{file_name}.yaml"
        if os.path.exists(yaml_path):
            d = DictConfig(yaml.load(open(yaml_path).read(), Loader=yaml.SafeLoader))
        else:
            d = {}
        d[key] = val
        with open("test.yaml", "w") as fp:
            OmegaConf.save(config=d, f=fp.name)
