# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import functools
import inspect
import itertools
import os
import io
import re
import shlex
import shutil
import subprocess
import sys
import typing as tp
import uuid
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..core import core, job_environment, logger, utils
from clustertools.schedulers import lsf

class CommandFunction:
    """Wraps a command as a function in order to make sure it goes through the
    pipeline and notify when it is finished.
    The output is a string containing everything that has been sent to stdout.
    WARNING: use CommandFunction only if you know the output won't be too big !
    Otherwise use subprocess.run() that also streams the outputto stdout/stderr.

    Parameters
    ----------
    command: list
        command to run, as a list
    verbose: bool
        prints the command and stdout at runtime
    cwd: Path/str
        path to the location where the command must run from

    Returns
    -------
    str
       Everything that has been sent to stdout
    """

    def __init__(
        self,
        command: tp.List[str],
        submission_path: str,
        verbose: bool = True,
        cwd: tp.Optional[tp.Union[str, Path]] = None,
        env: tp.Optional[tp.Dict[str, str]] = None,
    ) -> None:
        if not isinstance(command, list):
            raise TypeError("The command must be provided as a list")
        self.command = command
        self.verbose = verbose
        self.cwd = None if cwd is None else str(cwd)
        self.env = env
        self.submission_path = submission_path

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> str:
        """Call the cammand line with addidional arguments
        The keyword arguments will be sent as --{key}={val}
        The logs bufferized. They will be printed if the job fails, or sent as output of the function
        Errors are provided with the internal stderr.
        """
        full_command = (
            self.command + [str(x) for x in args] + [f"--{x}={y}" for x, y in kwargs.items()]
        )  # TODO bad parsing
        if self.verbose:
            print(f"The following command is sent: \"{' '.join(full_command)}\"")
        with subprocess.Popen(
            full_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=open(self.submission_path, 'r'),
            shell=False,
            cwd=self.cwd,
            env=self.env,
        ) as process:
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()

            try:
                utils.copy_process_streams(process, stdout_buffer, stderr_buffer, self.verbose)
            except Exception as e:
                process.kill()
                process.wait()
                raise utils.FailedJobError("Job got killed for an unknown reason.") from e
            stdout = stdout_buffer.getvalue().strip()
            stderr = stderr_buffer.getvalue().strip()
            retcode = process.wait()
            if stderr and (retcode and not self.verbose):
                # We don't print is self.verbose, as it already happened before.
                print(stderr, file=sys.stderr)
            if retcode:
                subprocess_error = subprocess.CalledProcessError(
                    retcode, process.args, output=stdout, stderr=stderr
                )
                raise utils.FailedJobError(stderr) from subprocess_error
        return stdout

class JobPaths:
    """Creates paths related to the lsf job and its submission"""

    def __init__(
        self, folder: tp.Union[Path, str], job_id: tp.Optional[str] = None, task_id: tp.Optional[int] = None
    ) -> None:
        self._folder = Path(folder).expanduser().absolute()
        self.job_id = job_id
        self.task_id = task_id or 0

    @property
    def folder(self) -> Path:
        return self._format_id(self._folder)

    @property
    def submission_file(self) -> Path:
        if self.job_id and "[" in self.job_id:
            # We only have one submission file per job array
            return self._format_id(self.folder / "%J_submission.sh")
        return self._format_id(self.folder / "%J_submission.sh")

    @property
    def submitted_pickle(self) -> Path:
        return self._format_id(self.folder / "%J_submitted.pkl")

    @property
    def result_pickle(self) -> Path:
        return self._format_id(self.folder / "%J_%t_result.pkl")

    def _format_id(self, path: tp.Union[Path, str]) -> Path:
        """Replace id tag by actual id if available"""
        if self.job_id is None:
            return Path(path)
        replaced_path = str(path).replace("%J", str(self.job_id)).replace("%t", str(self.task_id))
        array_id, *array_index = str(self.job_id).split("_", 1)
        if "%I" in replaced_path:
            if len(array_index) != 1:
                raise ValueError("%I is in the folder path but this is not a job array")
            replaced_path = replaced_path.replace("%I", array_index[0])
        return Path(replaced_path.replace("%J", array_id))

    def move_temporary_file(self, tmp_path: tp.Union[Path, str], name: str) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        Path(tmp_path).rename(getattr(self, name))
    
    @property
    def stderr(self) -> Path:
        return self._format_id(self.folder / "%J_%t_log.err")

    @property
    def stdout(self) -> Path:
        return self._format_id(self.folder / "%J_%t_log.out")

    @staticmethod
    def get_first_id_independent_folder(folder: tp.Union[Path, str]) -> Path:
        """Returns the closest folder which is id independent"""
        parts = Path(folder).expanduser().absolute().parts
        tags = ["%J", "%t", "%I"]
        indep_parts = itertools.takewhile(lambda x: not any(tag in x for tag in tags), parts)
        return Path(*indep_parts)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.folder})"



def read_job_id(job_id: str) -> tp.List[Tuple[str, ...]]:
    """Reads formated job id and returns a tuple with format:
    (main_id, [array_index, [final_array_index])
    """
    pattern = r"(?P<main_id>\d+)_\[(?P<arrays>(\d+(-\d+)?(,)?)+)(\%\d+)?\]"
    match = re.search(pattern, job_id)
    if match is not None:
        main = match.group("main_id")
        array_ranges = match.group("arrays").split(",")
        return [tuple([main] + array_range.split("-")) for array_range in array_ranges]
    else:
        main_id, *array_id = job_id.split("_", 1)
        if not array_id:
            return [(main_id,)]
        # there is an array
        array_num = str(int(array_id[0]))  # trying to cast to int to make sure we understand
        return [(main_id, array_num)]


class LsfInfoWatcher(core.InfoWatcher):
    def get_info(self, job_id: str, mode: str = "standard") -> tp.Dict[str, str]:
        """Returns a dict containing info about the job.
        State of finished jobs are cached (use watcher.clear() to remove all cache)

        Parameters
        ----------
        job_id: str
            id of the job on the cluster
        mode: str
            one of "force" (forces a call), "standard" (calls regularly) or "cache" (does not call)
        """
        if job_id is None:
            raise RuntimeError("Cannot call bjobs without a job id")
        if job_id not in self._registered:
            self.register_job(job_id)
        # check with a call to bjobs
        self.update_if_long_enough(mode)
        return self._info_dict.get(job_id, {})
    
    def _make_command(self) -> Optional[List[str]]:
        # asking for array id will return all status
        # on the other end, asking for each and every one of them individually takes a huge amount of time
        to_check = {x.split("_")[0] for x in self._registered - self._finished}
        if not to_check:
            return None
        command = ["bjobs", "-o", 'jobid stat exec_host delimiter="|"']
        for jid in to_check:
            command.extend(["--job_id", str(jid)])
        return command

    def get_state(self, job_id: str, mode: str = "standard") -> str:
        """Returns the state of the job
        State of finished jobs are cached (use watcher.clear() to remove all cache)

        Parameters
        ----------
        job_id: int
            id of the job on the cluster
        mode: str
            one of "force" (forces a call), "standard" (calls regularly) or "cache" (does not call)
        """
        info = self.get_info(job_id, mode=mode)
        return info.get("STAT") or "UNKWN"
    
    def is_done(self, job_id: str, mode: str = "standard") -> bool:
        """Returns whether the job is finished.

        Parameters
        ----------
        job_id: str
            id of the job on the cluster
        mode: str
            one of "force" (forces a call), "standard" (calls regularly) or "cache" (does not call)
        """
        state = self.get_state(job_id, mode=mode)
        return state.upper() not in ["PROV", "PEND", "RUN", "UNKWN"]
    
    def read_info(self, string: Union[bytes, str]) -> Dict[str, Dict[str, str]]:
        """Reads the output of sacct and returns a dictionary containing main information"""
        if not isinstance(string, str):
            string = string.decode()
        lines = string.splitlines()
        if len(lines) < 2:
            return {}  # one job id does not exist (yet)
        names = lines[0].split("|")
        # read all lines
        all_stats: Dict[str, Dict[str, str]] = {}
        for line in lines[1:]:
            stats = {x: y.strip() for x, y in zip(names, line.split("|"))}
            job_id = stats["JOBID"]
            if not job_id or "." in job_id:
                continue
            try:
                multi_split_job_id = read_job_id(job_id)
            except Exception as e:
                # Array id are sometimes displayed with weird chars
                warnings.warn(
                    f"Could not interpret {job_id} correctly (please open an issue):\n{e}", DeprecationWarning
                )
                continue
            for split_job_id in multi_split_job_id:
                all_stats[
                    "_".join(split_job_id[:2])
                ] = stats  # this works for simple jobs, or job array unique instance
                # then, deal with ranges:
                if len(split_job_id) >= 3:
                    for index in range(int(split_job_id[1]), int(split_job_id[2]) + 1):
                        all_stats[f"{split_job_id[0]}_{index}"] = stats
        return all_stats


class LsfJob(core.Job[core.R]):

    _cancel_command = "bkill"
    watcher = LsfInfoWatcher(delay_s=600)

    def __init__(self, folder: tp.Union[Path, str], job_id: str, tasks: tp.Sequence[int] = (0,)) -> None:
        super().__init__(folder, job_id, tasks)
        self._paths = JobPaths(folder, job_id=job_id, task_id=self.task_id)


    def _interrupt(self, timeout: bool = False) -> None:
        """Sends preemption or timeout signal to the job (for testing purpose)

        Parameter
        ---------
        timeout: bool
            Whether to trigger a job time-out (if False, it triggers preemption)
        """
        cmd = ["bkill", self.job_id, "-s"]
        # in case of preemption, SIGTERM is sent first
        if not timeout:
            subprocess.check_call(cmd + ["SIGTERM"])
        subprocess.check_call(cmd + [LsfJobEnvironment.USR_SIG])


class LsfParseException(Exception):
    pass


def _parse_node_list(node_list: str):
    try:
        node = 0
        parsed_nodes: List[str] = []
        node_list = node_list.split(' ')[:-1] # Split with space and ignore the last character
        while node < len(node_list):
            parsed_nodes.append(node_list[node])
            node+=2
        return parsed_nodes
    except ValueError as e:
        raise LsfParseException(f"Unrecognized format for LSB_MCPU_HOSTS: '{node_list}'", e) from e


class LsfJobEnvironment(job_environment.JobEnvironment):

    _env = {
        "job_id": "LSB_JOBID",
        "num_tasks": "LSB_DJOB_NUMPROC",
        "array_job_id": "LSB_JOBID",
        "array_task_id": "LSB_JOBINDEX",
        "nodes": "LSB_MCPU_HOSTS",
        "host": "HOSTNAME",
        "global_rank": "",
        "local_rank": "",
    }
        
    def _requeue(self, countdown: int) -> None:
        jid = self.job_id
        subprocess.check_call(["brequeue", jid], timeout=60)
        logger.get_logger().info(f"Requeued job {jid} ({countdown} remaining timeouts)")

    @property
    def hostname(self) -> str:
        return os.environ.get(self._env["host"], "")
    
    @property
    def hostnames(self) -> List[str]:
        """Parse the content of the "SLURM_JOB_NODELIST" environment variable,
        which gives access to the list of hostnames that are part of the current job.

        In SLURM, the node list is formatted NODE_GROUP_1,NODE_GROUP_2,...,NODE_GROUP_N
        where each node group is formatted as: PREFIX[1-3,5,8] to define the hosts:
        [PREFIX1, PREFIX2, PREFIX3, PREFIX5, PREFIX8].

        Link: https://hpcc.umd.edu/hpcc/help/slurmenv.html
        """

        node_list = os.environ.get(self._env["nodes"], "")
        if not node_list:
            return [self.hostname]
        return _parse_node_list(node_list)

    @property
    def num_tasks(self) -> int:
        """Total number of tasks for the job"""
        return int(os.environ.get(self._env["num_tasks"], 1))

    @property
    def num_nodes(self) -> int:
        """Total number of nodes for the job"""
        node_list = self.hostnames
        return int(len(node_list))

    @property
    def node(self) -> int:
        """Id of the current node"""
        hostname = self.hostname
        node_list = self.hostnames
        return int(node_list.index(hostname))

    def activated(self) -> bool:
        """Tests if we are running inside this environment.

        @plugin-dev: assumes that the SUBMITIT_EXECUTOR variable has been
        set to the executor name
        """
        return True if "LSF_SERVERDIR" in os.environ else False

    @property
    def paths(self) -> JobPaths:
        """Provides the paths used by submitit, including
        stdout, stderr, submitted_pickle and folder.
        """
        folder = os.environ["SUBMITIT_FOLDER"]
        return JobPaths(folder, job_id=self.job_id, task_id=self.global_rank)
    
    @property
    def job_id(self) -> str:
        if self.array_task_id != "0":
            return f"{self.array_job_id}_{self.array_task_id}"
        else:
            return self.raw_job_id

class LsfExecutor(core.PicklingExecutor):
    """LSF job executor
    This class is used to hold the parameters to run a job on lsf.
    In practice, it will create a batch file in the specified directory for each job,
    and pickle the task function and parameters. At completion, the job will also pickle
    the output. Logs are also dumped in the same directory.

    Parameters
    ----------
    folder: Path/str
        folder for storing job submission/output and logs.
    max_num_timeout: int
        Maximum number of time the job can be requeued after timeout (if
        the instance is derived from helpers.Checkpointable)

    Note
    ----
    - be aware that the log/output folder will be full of logs and pickled objects very fast,
      it may need cleaning.
    - the folder needs to point to a directory shared through the cluster. This is typically
      not the case for your tmp! If you try to use it, lsf will fail silently (since it
      will not even be able to log stderr.
    - use update_parameters to specify custom parameters (n_gpus etc...). If you
      input erroneous parameters, an error will print all parameters available for you.
    """

    job_class = LsfJob

    def __init__(self, folder: Union[Path, str], max_num_timeout: int = 3) -> None:
        super().__init__(folder, max_num_timeout)
        if not self.affinity() > 0:
            raise RuntimeError('Could not detect "bsub", are you indeed on a lsf cluster?')

    @classmethod
    def _equivalence_dict(cls) -> core.EquivalenceDict:
        return {
            "name": "job_name",
            "timeout_min": "duration",
            "mem_gb": "memory",
            "nodes": "nodes",
            "cpus_per_task": "num_cpus",
            "gpus_per_node": "num_gpus",
            "tasks_per_node": "ntasks_per_node",
        }
    @classmethod
    def _valid_parameters(cls) -> Set[str]:
        """Parameters that can be set through update_parameters"""
        return set(_get_default_parameters())

    def _convert_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params = super()._convert_parameters(params)
        return params

    def _internal_update_parameters(self, **kwargs: Any) -> None:
        """Updates sbatch submission file parameters

        Parameters
        ----------
        See slurm documentation for most parameters.
        Most useful parameters are: time, mem, gpus_per_node, cpus_per_task, partition
        Below are the parameters that differ from slurm documentation:

        signal_delay_s: int
            delay between the kill signal and the actual kill of the slurm job.
        setup: list
            a list of command to run in sbatch befure running srun
        array_parallelism: int
            number of map tasks that will be executed in parallel

        Raises
        ------
        ValueError
            In case an erroneous keyword argument is added, a list of all eligible parameters
            is printed, with their default values

        Note
        ----
        Best practice (as far as Quip is concerned): cpus_per_task=2x (number of data workers + gpus_per_task)
        You can use cpus_per_gpu=2 (requires using gpus_per_task and not gpus_per_node)
        """
        defaults = _get_default_parameters()
        in_valid_parameters = sorted(set(kwargs) - set(defaults))
        if in_valid_parameters:
            string = "\n  - ".join(f"{x} (default: {repr(y)})" for x, y in sorted(defaults.items()))
            raise ValueError(
                f"Unavailable parameter(s): {in_valid_parameters}\nValid parameters are:\n  - {string}"
            )
        # check that new parameters are correct
        _make_bsub_string(command="nothing to do", folder=self.folder, **kwargs)
        super()._internal_update_parameters(**kwargs)

    def _internal_process_submissions(
        self, delayed_submissions: tp.List[utils.DelayedSubmission]
    ) -> tp.List[core.Job[tp.Any]]:
        if len(delayed_submissions) == 1:
            return super()._internal_process_submissions(delayed_submissions)
        # array
        folder = JobPaths.get_first_id_independent_folder(self.folder)
        folder.mkdir(parents=True, exist_ok=True)
        timeout_min = self.parameters.get("time", 5)
        pickle_paths = []
        for d in delayed_submissions:
            pickle_path = folder / f"{uuid.uuid4().hex}.pkl"
            d.set_timeout(timeout_min, self.max_num_timeout)
            d.dump(pickle_path)
            pickle_paths.append(pickle_path)
        n = len(delayed_submissions)
        # Make a copy of the executor, since we don't want other jobs to be
        # scheduled as arrays.
        array_ex = LsfExecutor(self.folder, self.max_num_timeout)
        array_ex.update_parameters(**self.parameters)
        array_ex.parameters["map_count"] = n
        self._throttle()

        first_job: core.Job[tp.Any] = array_ex._submit_command(self._submitit_command_str)
        tasks_ids = list(range(first_job.num_tasks))
        jobs: List[core.Job[tp.Any]] = [
            LsfJob(folder=self.folder, job_id=f"{first_job.job_id}_{a}", tasks=tasks_ids) for a in range(1, n+1)
        ]
        for job, pickle_path in zip(jobs, pickle_paths):
            job.paths.move_temporary_file(pickle_path, "submitted_pickle")
        return jobs

    def _submit_command(self, command: str) -> core.Job[tp.Any]:
        """Submits a command to the cluster
        It is recommended not to use this function since the Job instance assumes pickle
        files will be created at the end of the job, and hence it will not work correctly.
        You may use a CommandFunction as argument to the submit function instead. The only
        problem with this latter solution is that stdout is buffered, and you will therefore
        not be able to monitor the logs in real time.

        Parameters
        ----------
        command: str
            a command string

        Returns
        -------
        Job
            A Job instance, providing access to the crun job information.
            Since it has no output, some methods will not be efficient
        """
        tmp_uuid = uuid.uuid4().hex
        submission_file_path = (
            JobPaths.get_first_id_independent_folder(self.folder) / f"submission_file_{tmp_uuid}.sh"
        )
        with submission_file_path.open("w") as f:
            command = self._make_submission_file_text(command, tmp_uuid)
            f.write(command)
        command_list = self._make_submission_command()
        # run
        output = CommandFunction(command_list,  submission_file_path, verbose=False)()  # explicit errors
        job_id = self._get_job_id_from_submission_command(output)
        tasks_ids = list(range(self._num_tasks()))
        job: core.Job[tp.Any] = self.job_class(folder=self.folder, job_id=job_id, tasks=tasks_ids)
        job.paths.move_temporary_file(submission_file_path, "submission_file")
        self._write_job_id(job.job_id, tmp_uuid)
        self._set_job_permissions(job.paths.folder)
        return job

    @property
    def _submitit_command_str(self) -> str:
        return " ".join(
            [shlex.quote(sys.executable), "-u -m submitit.core._submit", shlex.quote(str(self.folder))]
        )

    def _make_submission_file_text(self, command: str, uid: str) -> str:
        return _make_bsub_string(command=command, folder=self.folder, **self.parameters)

    def _num_tasks(self) -> int:
        nodes: int = self.parameters.get("nodes", 1)
        tasks_per_node: int = max(1, self.parameters.get("ntasks_per_node", 1))
        return nodes * tasks_per_node

    def _make_submission_command(self) -> List[str]:
        return ['bsub', '-L', '/bin/bash']

    @staticmethod
    def _get_job_id_from_submission_command(string: Union[bytes, str]) -> str:
        """Returns the job ID from the output of sbatch string"""
        if not isinstance(string, str):
            string = string.decode()
        output = re.search(r"Job <(?P<id>[0-9]+)>", string)
        if output is None:
            raise utils.FailedSubmissionError(
                f'Could not make sense of sbatch output "{string}"\n'
                "Job instance will not be able to fetch status\n"
                "(you may however set the job job_id manually if needed)"
            )
        return output.group("id")

    @classmethod
    def affinity(cls) -> int:
        return 2 if "LSF_SERVERDIR" in os.environ else -1


@functools.lru_cache()
def _get_default_parameters() -> Dict[str, Any]:
    """Parameters that can be set through update_parameters"""
    specs = inspect.getfullargspec(_make_bsub_string)
    zipped = zip(specs.args[-len(specs.defaults) :], specs.defaults)  # type: ignore
    return {key: val for key, val in zipped if key not in {"command", "folder", "map_count"}}

# pylint: disable=too-many-arguments,unused-argument, too-many-locals
def _make_bsub_string(
    command: str,
    folder: tp.Union[str, Path],
    job_name: str = "submitit",
    exec_backend: str = "plain",
    queue: tp.Optional[str] = None,
    duration: int = 5,
    nodes: int = 1,
    ntasks_per_node: tp.Optional[int] = None,
    num_cpus: tp.Optional[int] = 1,
    num_gpus: tp.Optional[int] = None,
    gpu_mode: tp.Optional[str] = None,
    gpu_option: tp.Optional[tp.Dict[str, tp.Any]] = None,
    memory: tp.Optional[int] = None,
    resources: tp.Optional[str] = None,
    env_variables: tp.Optional[tp.Dict[str, tp.Any]] = None,
    signal_delay_s: int = 60,
    array_parallelism: int = 256,
    stderr_to_stdout: bool = False,
    map_count: tp.Optional[int] = None,  # used internally
    additional_parameters: tp.Optional[tp.Dict[str, tp.Any]] = None,
) -> str:
    """Creates the content of a bsub file with provided parameters using clustertools

    Parameters
    ----------
    See clustertools documentation for most parameters:
    https://pages.github.boschdevcloud.com/bios-bcai/clustertools/autoapi/clustertools/schedulers/lsf/index.html

    Below are the parameters that differ from slurm documentation:

    folder: str/Path
        folder where print logs and error logs will be written
    signal_delay_s: int
        delay between the kill signal and the actual kill of the slurm job.
    map_size: int
        number of simultaneous map/array jobs allowed

    Raises
    ------
    ValueError
        In case an erroneous keyword argument is added, a list of all eligible parameters
        is printed, with their default values
    """
    nonlsf = [
        "nonlsf",
        "folder",
        "map_count",
        "array_parallelism",
        "additional_parameters",
        "signal_delay_s",
        "stderr_to_stdout",
    ]
    parameters = {k: v for k, v in locals().items() if v is not None and k not in nonlsf}
    # rename and reformat parameters
    lsf_options = [_as_bsub_flag("wt", f"{signal_delay_s // 60}")]
    lsf_options.append(_as_bsub_flag("wa", f"{LsfJobEnvironment.USR_SIG}"))

    # add necessary parameters
    # paths = utils.JobPaths(folder=folder)
    # Job arrays will write files in the form  <ARRAY_ID>_<ARRAY_TASK_ID>_<TASK_ID>
    if map_count is not None:
        assert isinstance(map_count, int) and map_count
        parameters["job_name"] = parameters["job_name"] + f"[1-{map_count}]%{min(map_count, array_parallelism)}"
        parameters["log_file_format"] = "%J_%I_log"
    parameters["log_dir"] = folder
    parameters["log_file_format"] = "%J_0_log"
    
    if additional_parameters is not None:
        for key, value in additional_parameters.items():
            lsf_options.append(_as_bsub_flag(key, value))
    # Convert to lsf style job distribution
    num_nodes = parameters.pop("nodes")
    parameters["resources"] = "span[ptile={}]".format(parameters["num_cpus"])
    parameters["num_cpus"] = num_nodes * parameters["num_cpus"]

    parameters["exec_backend"] = getattr(lsf.backend, parameters["exec_backend"])(["poetry"])
    # Remove ntasks_per_node since lsf doesn't support that as far as I know
    parameters.pop("ntasks_per_node")
    return lsf.submit(**parameters, lsf_options=lsf_options, dry_run=True)

def _as_bsub_flag(key: str, value: tp.Any) -> str:
    value = shlex.quote(str(value))
    return f"-{key} {value}"