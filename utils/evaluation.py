import os
import _jsonnet
import tempfile
import subprocess
import shutil
from pathlib import Path
import json
import gzip

from config import EvaluationConfig

from . import RUNS_DIR, CHECKPOINT_DIR


########################################################
#
# Paloma Evaluation
#
########################################################

PPL_METRICS_FILE = "ppl_metrics.jsonl.gz"
EVAL_DATA_PATH = "lib/paloma"
EVAL_LIB_DIR = "lib/olmo-eval"

jsonnet_template = """
    local utils = import 'lib/olmo-eval/configs/utils.libsonnet';
    local ppl_suite = import 'lib/olmo-eval/configs/task_sets/paloma_hf_release_val.libsonnet';

    local gsheet = null;
    local output_dir = std.extVar('output_dir');

    local model_path = std.extVar('model_path');
    local max_length = std.parseInt(std.extVar('max_length'));
    local limit = std.parseInt(std.extVar('limit'));

    local model = {
        model_path: model_path,
        revision: null,
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: max_length,
            limit: limit,
        }
    };

    local task_sets = [
        ppl_suite.task_set
    ];

    {
        steps: utils.create_fine_grained_pipeline([model], task_sets, gsheet, output_dir)
    }
"""


def setup_paloma_config(model_path: str, evaluation_config: EvaluationConfig) -> str:
    """Create Paloma config from evaluation configuration."""

    # Convert evaluation config to external vars
    output_dir = f"{os.getcwd()}/{RUNS_DIR}/{evaluation_config.run_name}/eval_results"

    # create output dir
    os.makedirs(output_dir, exist_ok=True)

    ext_vars = {
        "output_dir": output_dir,
        "model_path": model_path,
        "max_length": str(evaluation_config.paloma.max_length),
        "limit": "null"
        if evaluation_config.paloma.limit_eval_examples is None
        else str(evaluation_config.paloma.limit_eval_examples),
    }

    # Evaluate template with overrides
    json_str = _jsonnet.evaluate_snippet("config", jsonnet_template, ext_vars=ext_vars)

    # Write to temporary file
    temp_config = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonnet", delete=False)
    temp_config.write(json_str)
    temp_config.close()

    return temp_config.name


def run_paloma_evaluation(paloma_config_path: str) -> None:
    """Run tango evaluation from olmo-eval directory."""
    olmo_eval_dir = Path(EVAL_LIB_DIR)
    venv_activate = "env/bin/activate"
    tmp_workspace_name = "pico-tmp-eval-ws"

    # Construct the command with source activation
    cmd = f"source {venv_activate} && tango --settings tango.yml run {paloma_config_path} --workspace {tmp_workspace_name}"

    try:
        subprocess.run(
            cmd,
            cwd=olmo_eval_dir,
            shell=True,  # Required for source command
            executable="/bin/bash",  # Ensure bash is used
            check=True,
            text=True,
            capture_output=False,
            env=os.environ.copy(),
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Model evaluation failed: {e}")

    # Delete workspace cache
    shutil.rmtree(f"{os.getcwd()}/{EVAL_LIB_DIR}/{tmp_workspace_name}")


def process_tango_output(evaluation_config: EvaluationConfig) -> None:
    """Process the output of a tango evaluation."""
    output_dir = f"{os.getcwd()}/{RUNS_DIR}/{evaluation_config.run_name}/eval_results"
    # load in ppl metrics
    ppl_metrics_path = os.path.join(output_dir, PPL_METRICS_FILE)

    with gzip.open(ppl_metrics_path, "rt") as f:  # 'rt' mode for text reading
        # For a JSONL file, read line by line
        ppl_metrics = [json.loads(line) for line in f]

    # average together the ppl_primary metrics
    ppl_primary_metrics = [metric["ppl_primary"] for metric in ppl_metrics]
    avg_ppl_primary = sum(ppl_primary_metrics) / len(ppl_primary_metrics)

    # delete ppl metrics file -- it would be too messy to keep around
    shutil.rmtree(output_dir)

    return avg_ppl_primary


########################################################
#
# Evaluation
#
########################################################


def run_evaluation(evaluation_config: EvaluationConfig) -> None:
    """
    Run evaluation on a given model using a provided evaluation configuration.
    """

    if evaluation_config.checkpoint_path is not None:
        model_path = evaluation_config.checkpoint_path
    else:
        run_name = evaluation_config.run_name
        model_path = f"{os.getcwd()}/{RUNS_DIR}/{run_name}/{CHECKPOINT_DIR}/latest"
    os.makedirs(model_path, exist_ok=True)

    evaluation_results = {}

    for metric in evaluation_config.evaluation_metrics:
        if metric == "paloma":
            paloma_config_path = setup_paloma_config(model_path, evaluation_config)

            os.environ["EVAL_DATA_PATH"] = os.path.join(
                os.getcwd(), "lib"
            )  # No need for "paloma"
            run_paloma_evaluation(paloma_config_path)
            metric_result = process_tango_output(evaluation_config)
        else:
            raise ValueError(f"Metric {metric} not supported")

        evaluation_results[metric] = metric_result

    return evaluation_results
