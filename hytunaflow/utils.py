# %%
import sys
import mlflow

from omegaconf import DictConfig, ListConfig
from typing import Any, Dict


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
