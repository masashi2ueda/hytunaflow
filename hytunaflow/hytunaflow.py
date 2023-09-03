# %%
import copy
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

        mlflow.set_experiment(cfg.mlflow.exp_name)

        self.flow_run = mlflow.start_run(nested=self.is_nested)
        self.cfg = utils.set_keyval2DictConfig(self.cfg, "mlflow.run_id", self.flow_run.info.run_id)

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
            return self.cfg.mlflow.is_nested
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
        study_name = self.cfg.params.study_name
        if utils.get_dict_val(self.cfg, "params.restart_storage_path") is not None:
            src_storage_path = self.cfg.params.restart_storage_path
            shutil.copy(src_storage_path, dst_storage_file_path)
        elif utils.get_dict_val(self.cfg, "params.restart_expeid") is not None:
            splts = dst_storage_file_path.split("/")
            splts[-2] = self.cfg.params.restart_expeid
            splts[-3] = self.cfg.params.restart_runid
            src_storage_path = "/".join(splts)
            shutil.copy(src_storage_path, dst_storage_file_path)
        elif utils.get_dict_val(self.cfg, "params.restart_expname") is not None:
            splts = dst_storage_file_path.split("/")
            exp_name = self.cfg.params.restart_expname
            run_name = self.cfg.params.restart_runname
            mlrootpath = "/".join(splts[:-4])
            exp_id, run_id = utils.exp_run_name2id(mlrootpath, exp_name, run_name)
            splts[-3] = run_id
            splts[-4] = exp_id
            src_storage_path = "/".join(splts)
            shutil.copy(src_storage_path, dst_storage_file_path)

            f = open(f"{mlrootpath}/{exp_id}/{run_id}/params/params.study_name", 'r')
            study_name = f.read()
            f.close()
            

        self.study = optuna.create_study(
            study_name=study_name,
            storage=dst_storage_path,
            direction=self.cfg.params.direction,
            load_if_exists=True)

        if utils.get_dict_val(self.cfg, "params.enqueue_trials") is not None:
            for kv in self.cfg.params.enqueue_trials:
                self.study.enqueue_trial(eval(kv))

        return self.study

    def create_optuna_train_config(self, trial: optuna.trial.Trial):
        train_cfg = copy.deepcopy(self.cfg.train)
        train_cfg = utils.set_keyval2DictConfig(train_cfg, "mlflow.exp_name", self.cfg.mlflow.exp_name)
        train_cfg = utils.set_keyval2DictConfig(train_cfg, "mlflow.is_nested", True)

        for sg in self.cfg.params.suggets:
            param_name = sg[0]
            val = eval(f"trial.{sg[1]}(param_name, {sg[2]}, {sg[3]})")
            train_cfg = utils.set_keyval2DictConfig(train_cfg, param_name, val)

        return train_cfg

    def save_optuna_hist_callback(self, trial: optuna.trial.Trial, val: float) -> None:
        hist_df = self.study.trials_dataframe(multi_index=True)
        hist_df.to_csv(f"{self.artifact_dir_path}/hist_df.csv", index=False)
        for k, v in self.study.best_trials[0].params.items():
            mlflow.log_metric(f"best_{k}", v)
        mlflow.log_metric("best_value", self.study.best_trials[0].values[0])

    def copy2atifact(self, src_path: str) -> None:
        """copy file 2 mlflow artifact dir path.
        """
        dst_path = f"{self.artifact_dir_path}/{os.path.basename(src_path)}"
        shutil.copy(src_path, dst_path)

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
