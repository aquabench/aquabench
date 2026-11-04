"""AQUABENCH: underwater semantic segmentation benchmark for prepared encoders.

    import aquabench
    result = aquabench.run(encoder, run_type="linear", seed=0, data_root="/data/aquabench")
"""

__version__ = "1.1.0"

from aquabench.api import Result, RunResults, evaluate, run  # noqa: E402
from aquabench.heads import RUN_TYPES, EncoderContractError  # noqa: E402
from aquabench.loggers import Logger, PrintLogger  # noqa: E402

__all__ = ["run", "evaluate", "Result", "RunResults", "RUN_TYPES", "EncoderContractError",
           "Logger", "PrintLogger", "__version__"]
