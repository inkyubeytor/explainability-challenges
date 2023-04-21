import pprint
from dataclasses import dataclass
from functools import wraps

from typing import Any, Callable, Dict, ParamSpec, Tuple, TypeVar

P = ParamSpec("P")
R = TypeVar("R")
TraceData = Dict[str, Any]


def trace(f: Callable[P, Tuple[R, TraceData]]) -> Callable[P, R]:
    """
    Wraps all transformations to store a record of the undertaken action.

    :param f: The method to trace.
    :return: A function that extracts the trace data and returns the
        expected output.
    """

    @wraps(f)
    def _wrapper(self, *args, **kwargs):
        out, trace_data = f(self, *args, **kwargs)
        self._traces.append(Trace(operation=f.__name__, data=trace_data))
        return out

    return _wrapper


@dataclass
class Trace:
    operation: str
    data: TraceData

    def __str__(self):
        return "\n".join(["", self.operation, pprint.pformat(self.data), ""])
