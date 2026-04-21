from .base import BaseTask, UnivariateCRPSTask
from .tasks.cxr_daily_volume_tasks import (
    CXRDailyForecastTask,
    CXRDailyForecastUnder1YearTask,
    CXRDailyForecastUnder6MonthsTask,
    __TASKS__,
    __CLUSTERS__,
)

__version__ = "0.0.2"

ALL_TASKS = __TASKS__

__all__ = [
    "__version__",
    "BaseTask",
    "UnivariateCRPSTask",
    "ALL_TASKS",
    "__TASKS__",
    "__CLUSTERS__",
    "CXRDailyForecastTask",
    "CXRDailyForecastUnder6MonthsTask",
    "CXRDailyForecastUnder1YearTask",
]
