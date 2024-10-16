import json
import pathlib
import subprocess
import sys
import time
from typing import Dict
import os
import numpy as np
import io
import sys
import tempfile

path = str(pathlib.Path(__file__).parent.resolve())

class SmoothMetric:
    def __init__(self, init_value=0., lr=0.993, n_init=100):
        assert 0 <= lr < 1
        self.lr = lr
        self._v = init_value
        self.n_init = n_init

        self.init_value = None
        self.init_count = 0

    def update(self, value, weight=1.):

        if not np.any(np.isnan(value)):
            if np.any(np.isnan(self._v)):
                next_count = self.init_count + weight
                if self.init_value is None:
                    self.init_value = value
                else:

                    self.init_value = value * weight / next_count + self.init_value * self.init_count / next_count
                self.init_count = next_count
                if self.init_count >= self.n_init:
                    self.set(self.init_value)
            else:
                lr = np.maximum(1. - (1. - self.lr) * weight, 0.)
                self._v = lr * self._v + (1. - lr) * value

        return self._v

    def get(self):
        return self._v

    def set(self, v):
        self._v = v


def print_cmd(message, duration=2):
    subprocess.run("clear")
    sys.stdout.write(message + "\n")
    sys.stdout.flush()
    time.sleep(duration)
    sys.stdout.write('\r' + ' ' * len(message) + '\r')  # Clear the line
    sys.stdout.flush()


def run_fire_script(
        script_name,
        input_dict: Dict,
        print_output=False,
        output_dtype=None
):

    cmd = [sys.executable, '-m', script_name] + [
        f'--{arg_name}={str(value).replace(" ", "")}'
        for arg_name, value in input_dict.items()
    ]
    print_cmd("Running command:\n" + " ".join(cmd))
    try:
        output = subprocess.run(
            cmd,
            capture_output=not print_output, text=True, check=True, stdout=sys.stdout if print_output else None
        )
    except subprocess.CalledProcessError as e:
        print(e)
        output = None

    if output is not None and output.stdout is not None:
        output = output.stdout.strip().split("\n")[-1]

    if output_dtype == str:
        return output
    elif output_dtype in (list, dict):
        return json.loads(output)
    else:
        return output


def inject_callback(
        instance,
        callback,
        base_method_name,
        after=False,
):
    """

    :param instance:
    :param callback:
    :param base_method_name:
    :param after: if true, the callback is executed after the method, it also is called with the base method returned value
    :return:
    """
    new_base_method_name = f"base_{base_method_name}"
    setattr(instance, new_base_method_name, getattr(instance, base_method_name))
    if after:
        def injection(algo):
            ret = getattr(instance, new_base_method_name)()
            ret = callback(algo, ret)
            return ret

    else:
        def injection(algo):
            ret = callback(algo)
            getattr(instance, new_base_method_name)()
            return ret

    setattr(instance, base_method_name, injection.__get__(instance, type(instance)))


def get_subdir_names(dir_path):
    _, dirnames, _ = next(os.walk(dir_path))
    return dirnames

class SuppressStd(object):
    """Context to capture stderr and stdout at C-level.
    """

    def __init__(self):
        self.orig_stdout_fileno = sys.__stdout__.fileno()
        self.orig_stderr_fileno = sys.__stderr__.fileno()
        self.output = None

    def __enter__(self):
        # Redirect the stdout/stderr fd to temp file
        self.orig_stdout_dup = os.dup(self.orig_stdout_fileno)
        self.orig_stderr_dup = os.dup(self.orig_stderr_fileno)
        self.tfile = tempfile.TemporaryFile(mode='w+b')
        os.dup2(self.tfile.fileno(), self.orig_stdout_fileno)
        os.dup2(self.tfile.fileno(), self.orig_stderr_fileno)

        # Store the stdout object and replace it by the temp file.
        self.stdout_obj = sys.stdout
        self.stderr_obj = sys.stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        return self

    def __exit__(self, exc_class, value, traceback):

        # Make sure to flush stdout
        print(flush=True)

        # Restore the stdout/stderr object.
        sys.stdout = self.stdout_obj
        sys.stderr = self.stderr_obj

        # Close capture file handle
        os.close(self.orig_stdout_fileno)
        os.close(self.orig_stderr_fileno)

        # Restore original stderr and stdout
        os.dup2(self.orig_stdout_dup, self.orig_stdout_fileno)
        os.dup2(self.orig_stderr_dup, self.orig_stderr_fileno)

        # Close duplicate file handle.
        os.close(self.orig_stdout_dup)
        os.close(self.orig_stderr_dup)

        # Copy contents of temporary file to the given stream
        self.tfile.flush()
        self.tfile.seek(0, io.SEEK_SET)
        self.output = self.tfile.read().decode()
        self.tfile.close()





