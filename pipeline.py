import datetime
import os

from tfx.components import CsvExampleGen
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Transform
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.airflow.airflow_dag_runner import AirflowDagRunner
from tfx.orchestration.airflow.airflow_dag_runner import AirflowPipelineConfig
from tfx.utils.dsl_utils import external_input

_root_dir = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.path.join(_root_dir, "data", "interim")

# Path to the module containing preprocessing_fn and _build_estimator required by TFX.
_MODULE_FILE = os.path.join(_root_dir, "tfx_functions.py")

_PIPELINE_NAME = "amazon_reviews_sentiment"

_TFX_ROOT = os.path.join(_root_dir, ".tfx")
_PIPELINE_ROOT = os.path.join(_TFX_ROOT, "pipelines", _PIPELINE_NAME)
_METADATA_PATH = os.path.join(_TFX_ROOT, "metadata", _PIPELINE_NAME, "metadata.db")
_SERVING_MODEL_DIR = os.path.join(_TFX_ROOT, "serving")

# 0 workers => number of workers is set based on number of available CPUs.
_BEAM_WORKERS_COUNT = 0

_AIRFLOW_CONFIG = {
    "schedule_interval": None,
    "start_date": datetime.datetime(2020, 1, 1),
}


def _create_pipeline(
    pipeline_name: str,
    pipeline_root: str,
    data_root: str,
    module_file: str,
    serving_model_dir: str,
    metadata_path: str,
    direct_workers_count: int,
) -> pipeline.Pipeline:

    examples = external_input(data_root)
    example_gen = CsvExampleGen(input=examples)

    statistics_gen = StatisticsGen(examples=example_gen.outputs["examples"])

    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs["statistics"], infer_feature_shape=False
    )

    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=module_file,
    )

    metadata_connection_config = metadata.sqlite_metadata_connection_config(
        metadata_path
    )

    beam_pipeline_args = [
        "--direct_num_workers={workers}".format(workers=direct_workers_count)
    ]

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=[example_gen, statistics_gen, schema_gen, transform],
        enable_cache=True,
        metadata_connection_config=metadata_connection_config,
        beam_pipeline_args=beam_pipeline_args,
    )


_pipeline = _create_pipeline(
    pipeline_name=_PIPELINE_NAME,
    pipeline_root=_PIPELINE_ROOT,
    data_root=_DATA_ROOT,
    module_file=_MODULE_FILE,
    serving_model_dir=_SERVING_MODEL_DIR,
    metadata_path=_METADATA_PATH,
    direct_workers_count=_BEAM_WORKERS_COUNT,
)

DAG = AirflowDagRunner(AirflowPipelineConfig(_AIRFLOW_CONFIG)).run(_pipeline)
