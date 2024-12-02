"""
Evaluation Module for Pico Language Model

This module implements the evaluation pipeline for the Pico language model. It provides
functionality to evaluate model performance using various metrics and handles the complete
evaluation workflow.

NOTE: out of the box we only support Paloma, but the structure is designed to be flexible and
you are meant to add whatever metrics you want.

Main Workflow:
1. Setup evaluation configuration
2. Run evaluation using specified metrics
3. Process and aggregate results
4. Clean up temporary files and workspaces
"""

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

"""
Paloma is a comprehensive evaluation benchmark for large language models (LLMs) that focuses 
on measuring perplexity across diverse text domains. 

To evaluate on Paloma, we use the olmo-eval library, which provides a unified interface for
evaluating models on a variety of benchmarks.

For more details, see: https://huggingface.co/datasets/allenai/paloma
"""

TEMP_EVAL_RESULTS_DIR = "_temp_paloma_results"
PPL_METRICS_FILE = "ppl_metrics.jsonl.gz"
EVAL_DATA_PATH = "lib/paloma"
EVAL_LIB_DIR = "lib/olmo-eval"

# NOTE: the jsonnet template is what is used by the olmo-eval library to run the evaluation.
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
    """Create Paloma config from evaluation configuration.

    This function generates a Jsonnet configuration file for Paloma evaluation by:
    1. Setting up the output directory structure
    2. Configuring model-specific parameters (max length, example limits)
    3. Applying the configuration template for the Paloma evaluation suite

    Args:
        model_path (str): Path to the model checkpoint to be evaluated
        evaluation_config (EvaluationConfig): Configuration object containing:
            - run_name (str): Name of the evaluation run
            - paloma.max_length (int): Maximum sequence length for evaluation
            - paloma.limit_eval_examples (Optional[int]): Number of examples to evaluate
                (None for full evaluation)

    Returns:
        str: Path to the generated temporary Jsonnet configuration file

    Example:
        config_path = setup_paloma_config(
            model_path="/checkpoints/model-1000",
            evaluation_config=EvaluationConfig(
                run_name="experiment_1",
                paloma=PalomaConfig(
                    max_length=2048,
                    limit_eval_examples=100
                )
            )
        )

    Note:
        The generated config uses the Paloma evaluation suite's standard template
        with custom parameters for the specific model evaluation run.
    """

    # Convert evaluation config to external vars
    output_dir = (
        f"{os.getcwd()}/{RUNS_DIR}/{evaluation_config.run_name}/{TEMP_EVAL_RESULTS_DIR}"
    )

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
    """Run Paloma evaluation using the Tango framework.

    This function executes the evaluation process for the Paloma benchmark by:
    1. Activating the virtual environment for the olmo-eval library
    2. Running the Tango command to perform evaluation based on the provided config
    3. Managing temporary workspaces and cleaning up after execution

    Args:
        paloma_config_path (str): Path to the Jsonnet configuration file for Paloma evaluation

    Note:
        Ensure that the environment is correctly set up with all dependencies
        before running this function. The function uses bash to execute commands.
    """
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
    """Process and aggregate the results from Paloma evaluation.

    This function handles the post-processing of Tango evaluation outputs by:
    1. Loading the compressed metrics file from the evaluation directory
    2. Processing the JSONL format containing per-example perplexity scores
    3. Computing the average perplexity across all evaluated examples
    4. Cleaning up temporary evaluation files

    Args:
        evaluation_config (EvaluationConfig): Configuration object containing:
            - run_name (str): Name of the evaluation run used for directory paths

    Returns:
        float: Average perplexity score across all evaluated examples

    File Structure:
        The function expects results in:
        {RUNS_DIR}/{run_name}/{TEMP_EVAL_RESULTS_DIR}/ppl_metrics.jsonl.gz

    Note:
        This function automatically cleans up temporary evaluation files
        after processing to maintain disk space efficiency.
    """
    output_dir = (
        f"{os.getcwd()}/{RUNS_DIR}/{evaluation_config.run_name}/{TEMP_EVAL_RESULTS_DIR}"
    )
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
    """Run model evaluation using specified metrics.

    This function orchestrates the complete evaluation pipeline by:
    1. Resolving the model checkpoint path (either specified or latest)
    2. Creating necessary directories and environment setup
    3. Executing each requested evaluation metric
    4. Aggregating results across all metrics

    Args:
        evaluation_config (EvaluationConfig): Configuration object containing:
            - checkpoint_path (Optional[str]): Specific checkpoint to evaluate
                If None, uses the latest checkpoint
            - run_name (str): Name of the evaluation run
            - evaluation_metrics (List[str]): Metrics to evaluate
                Currently supported: ["paloma"]
            - paloma (PalomaConfig): Configuration for Paloma evaluation
                - max_length (int): Maximum sequence length
                - limit_eval_examples (Optional[int]): Number of examples to evaluate

    Returns:
        Dict[str, float]: Dictionary mapping metric names to their values
            Example: {"paloma": 3.45}

    Raises:
        ValueError: If an unsupported evaluation metric is requested

    Example:
        results = run_evaluation(
            EvaluationConfig(
                run_name="experiment_1",
                evaluation_metrics=["paloma"],
                paloma=PalomaConfig(max_length=2048)
            )
        )

    Note:
        The function automatically handles checkpoint resolution, directory
        creation, and cleanup of temporary files. For each metric, it ensures
        proper setup and teardown of evaluation environment.
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
