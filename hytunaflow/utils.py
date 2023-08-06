# %%
import copy
import glob
import mlflow
import os
import sys
import yaml

from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from typing import Any, Dict, Optional, Tuple


# %%
def enable_hydra_with_ipython(btin) -> None:
    """By calling this function, you can use hydra with ipython.

    Examples:
        >>> import hytunaflow
        >>> hytunaflow.ipython_hydra(__builtins__)

    """
    if hasattr(btin, '__IPYTHON__'):
        sys.argv = ['self.py']


def _log_param_recursive(parent_name: str, element: Any) -> None:
    """log param key, value to mlflow.

    Args:
        parent_name (str): parent key name.
        element (Any): key's value.
    """
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _log_param_recursive(f'{parent_name}.{k}', v)
            else:
                mlflow.log_param(f'{parent_name}.{k}', v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            mlflow.log_param(f'{parent_name}.{i}', v)
    else:
        mlflow.log_param(f'{parent_name}', element)


def log_params_from_omegaconf_dict(params: Dict) -> None:
    """log param key, value to mlflow.

    Args:
        params (Dict): parameter directory
    """
    for param_name, element in params.items():
        _log_param_recursive(param_name, element)


def set_keyval2DictConfig(
        config: DictConfig,
        key: str,
        val: Any,
        inplace: bool = True) -> DictConfig:
    """set key and value to DictConfig.

    Args:
        config (DictConfig): target_config
        key (str): dict key
        val (Any): dict value.
        inplace (bool, optional): change input config or not. Defaults to True.

    Returns:
        DictConfig: dict new key and val are added.

    Examples:
        >>> conf = OmegaConf.create({"a": {"d":2, "e":3}, "b": [1,2,3]})
        >>> print(conf)
        {'a': {'d': 2, 'e': 3}, 'b': [1, 2, 3]}
        >>> conf = hytunaflow.set_keyval2DictConfig(conf, 3, "a")
        >>> print(conf)
        {'a': 3, 'b': [1, 2, 3]}
        >>> conf = hytunaflow.set_keyval2DictConfig(conf, 3, "aa")
        >>> print(conf)
        {'a': 3, 'b': [1, 2, 3], 'aa': 3}
        >>> conf = hytunaflow.set_keyval2DictConfig(conf, 3, "ff.dd.cc")
        >>> print(conf)
        {'a': 3, 'b': [1, 2, 3], 'aa': 3, 'ff': {'dd': {'cc': 3}}}
    """
    if not inplace:
        config = copy.deepcopy(config)
    OmegaConf.set_struct(config, True)
    with open_dict(config):
        keys = key.split(".")
        temp_conf = config
        for ki, key in enumerate(keys):
            if ki == (len(keys) - 1):
                temp_conf[key] = val
            else:
                if key not in temp_conf.keys():
                    temp_conf[key] = {}
                temp_conf = temp_conf[key]
    OmegaConf.set_struct(config, False)
    return config


def get_dict_val(conf: DictConfig, key_name: str) -> Optional[Any]:
    """get val from dict.
    if key is not in dict return None.

    Args:
        conf (DictConfig): target dictionary.
        key_name (str): key name. if key is nested, dot(.) is useed to separate.

    Returns:
        Optional[Any]: key's value. if key it not in dict, None.

    Examples:
        >>> from omegaconf import OmegaConf
        >>> import hytunaflow
        >>> conf = OmegaConf.create({"a": {"d":2, "e":3}, "b": [1,2,3]})
        >>> print(hytunaflow.get_dict_val(conf, "a"))
        {'d': 2, 'e': 3}
        >>> print(hytunaflow.get_dict_val(conf, "a.d"))
        2
        >>> print(hytunaflow.get_dict_val(conf, "b"))
        [1, 2, 3]
        >>> print(hytunaflow.get_dict_val(conf, "f"))
        None
        >>> print(hytunaflow.get_dict_val(conf, "b.a"))
        None
    """
    key_names = key_name.split(".")

    def _get_dict_val(conf, key_names):
        if len(key_names) == 0:
            return conf

        if type(conf) is not DictConfig:
            return None

        if key_names[0] in conf.keys():
            return _get_dict_val(conf[key_names[0]], key_names[1:])
        else:
            return None
    return _get_dict_val(conf, key_names)


def exp_run_name2id(
        mlruns_dir_path: str,
        src_exp_name: str,
        src_run_name: str) -> Tuple[str, str]:
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
        if "name" not in obj.keys():
            continue
        exp_name = obj["name"]
        if exp_name == src_exp_name:
            exp_id = os.path.basename(exp_path)
            break

    run_paths = glob.glob(f"{mlruns_dir_path}/{exp_id}/*")
    for run_path in run_paths:
        run_meta_path = f"{run_path}/meta.yaml"
        if not os.path.exists(run_meta_path):
            continue
        with open(run_meta_path) as file:
            obj = yaml.safe_load(file)
        run_name = obj["run_name"]
        if run_name == src_run_name:
            run_id = os.path.basename(run_path)
            break

    return exp_id, run_id
