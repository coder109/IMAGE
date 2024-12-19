import argparse
import datetime
import json
import os
import re
import signal
import time
from functools import wraps
import pytz
import yaml
from pyinstrument import Profiler
from transformers import PreTrainedTokenizerBase
import collections.abc as abc
import dataclasses
from typing import Any
from omegaconf import DictConfig, ListConfig, OmegaConf
import pydoc
from typing import Any
import torch
from modeling.logger import logger
import itertools

def _convert_target_to_string(t: Any) -> str:
    """
    Inverse of ``locate()``.

    Args:
        t: any object with ``__module__`` and ``__qualname__``
    """
    module, qualname = t.__module__, t.__qualname__

    # Compress the path to this object, e.g. ``module.submodule._impl.class``
    # may become ``module.submodule.class``, if the later also resolves to the same
    # object. This simplifies the string, and also is less affected by moving the
    # class implementation.
    module_parts = module.split(".")
    for k in range(1, len(module_parts)):
        prefix = ".".join(module_parts[:k])
        candidate = f"{prefix}.{qualname}"
        try:
            if locate(candidate) is t:
                return candidate
        except ImportError:
            pass
    return f"{module}.{qualname}"


def locate(name: str) -> Any:
    """
    Locate and return an object ``x`` using an input string ``{x.__module__}.{x.__qualname__}``,
    such as "module.submodule.class_name".

    Raise Exception if it cannot be found.
    """
    obj = pydoc.locate(name)

    # Some cases (e.g. torch.optim.sgd.SGD) not handled correctly
    # by pydoc.locate. Try a private function from hydra.
    if obj is None:
        try:
            from hydra.utils import _locate
        except ImportError as e:
            raise ImportError(f"Cannot dynamically locate object {name}!") from e
        else:
            obj = _locate(name)  # it raises if fails

    return obj


def find_matching_parenthesis(expression, opening_index):
    if expression[opening_index] != "(":
        raise ValueError("The character at the provided index is not '('.")

    stack = 0

    for index in range(opening_index + 1, len(expression)):
        char = expression[index]
        if char == "(":
            stack += 1
        elif char == ")":
            if stack == 0:
                return index
            stack -= 1

    raise ValueError("No matching ')' found for '(' at index {}.".format(opening_index))

def pretty_format(obj, indent: int = 4) -> str:
    if isinstance(obj, dict):
        return yaml.dump(obj, sort_keys=True, indent=indent)
    elif isinstance(obj, PreTrainedTokenizerBase):
        repr_str = obj.__repr__()
        class_name, rest = repr_str.split("(", 1)
        idx = find_matching_parenthesis(f"({rest}", 0)
        other = rest[idx:]
        other = other.strip(",").strip(" ")
        rest = rest[:idx]
        rest = rest.rstrip(")")

        pairs = re.findall(r"(\w+)=({[^}]*}|[^,]*),?", rest)

        formatted_pairs = []
        for k, v in pairs:
            if v.startswith("{") and v.endswith("}"):
                try:
                    v_dict = json.loads(v.replace("'", '"'))
                    v_formatted = json.dumps(v_dict, indent=indent).replace("\n", "\n" + " " * indent)
                except json.JSONDecodeError:
                    v_formatted = v
            else:
                v_formatted = v

            formatted_pairs.append(f"{' ' * indent}{k}={v_formatted},")

        return f"{class_name}(\n" + "\n".join(formatted_pairs) + "\n),\n" + other.replace("\t", " " * indent)
    elif isinstance(obj, argparse.Namespace):
        args_dict = vars(obj)
        return yaml.dump(args_dict, sort_keys=True, indent=indent)
    else:
        return obj

def deep_instantiate(cfg: list | ListConfig | abc.Mapping):
    """
    Recursively instantiate objects defined in dictionaries by "_target_" and arguments.

    Args:
        cfg: a dict-like object with "_target_" that defines the caller, and
            other keys that define the arguments

    Returns:
        object instantiated by cfg
    """
    if isinstance(cfg, ListConfig):
        lst = [deep_instantiate(x) for x in cfg]
        return ListConfig(lst, flags={"allow_objects": True})
    if isinstance(cfg, list):
        # Specialize for list, because many classes take
        # list[objects] as arguments, such as ResNet, DatasetMapper
        return [deep_instantiate(x) for x in cfg]

    # If input is a DictConfig backed by dataclasses (i.e. omegaconf's structured config),
    # instantiate it to the actual dataclass.
    if isinstance(cfg, DictConfig) and dataclasses.is_dataclass(cfg._metadata.object_type):
        return OmegaConf.to_object(cfg)

    if isinstance(cfg, abc.Mapping):
        if "_target_" in cfg:
            # conceptually equivalent to hydra.utils.instantiate(cfg) with _convert_=all,
            # but faster: https://github.com/facebookresearch/hydra/issues/1200
            cfg = {k: deep_instantiate(v) for k, v in cfg.items()}
            cls = cfg.pop("_target_")
            cls = deep_instantiate(cls)

            if isinstance(cls, str):
                cls_name = cls
                cls = locate(cls_name)
                assert cls is not None, cls_name
            else:
                try:
                    cls_name = cls.__module__ + "." + cls.__qualname__
                except Exception:
                    # target could be anything, so the above could fail
                    cls_name = str(cls)
            assert callable(cls), f"_target_ {cls} does not define a callable object"
            try:
                return cls(**cfg)
            except TypeError:
                logger.error(f"Error when instantiating {cls_name}!")
                raise
        else:
            return DictConfig({k: deep_instantiate(v) for k, v in cfg.items()}, flags={"allow_objects": True})
    return cfg  # return as-is if don't know what to do

def randn_tensor(
    shape: tuple | list,
    generator: list[torch.Generator] | torch.Generator | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                logger.info(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slighly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents

def check_path_and_file(path, file):
    if os.path.isdir(path):
        file_path = os.path.join(path, file)
        return os.path.isfile(file_path)
    # elif os.path.isfile(path):
    #     if path.endswith(file):
    #         return True
    else:
        return False
    
def get_model_device(model: torch.nn.Module):
    try:
        parameters_and_buffers = itertools.chain(model.parameters(), model.buffers())
        return next(parameters_and_buffers).device
    except StopIteration:
        # For torch.nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: torch.nn.Module) -> list[tuple[str, torch.Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = model._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].device




def get_model_dtype(model: torch.nn.Module):
    try:
        params = tuple(model.parameters())
        if len(params) > 0:
            return params[0].dtype

        buffers = tuple(model.buffers())
        if len(buffers) > 0:
            return buffers[0].dtype

    except StopIteration:
        # For torch.nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: torch.nn.Module) -> list[tuple[str, torch.Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = model._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype