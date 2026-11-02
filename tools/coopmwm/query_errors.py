"""Narrow per-plan code-fault isolation; safety/physics/accounting stay fatal."""
import traceback
from .protocol import digest


def execute(record,call):
    try:
        return call()
    except (KeyError,TypeError,AttributeError,NameError,NotImplementedError) as error:
        # ValueError/AssertionError/RuntimeError include validation, parity,
        # accounting, physics and CUDA failures: deliberately propagate those.
        return dict(accepted=False,reason='per-query code error; policy continues',
            record_sha256=digest(record),compute_counts_unknown=True,
            query_error=dict(kind='code',error=repr(error),traceback=traceback.format_exc(),
                inference_retry=False))


def validate_response(record,result):
    error=result.get('query_error')
    if error is None:return
    if (error.get('kind')!='code' or not error.get('traceback') or
            error.get('inference_retry') is not False or result.get('accepted') is not False or
            result.get('record_sha256')!=digest(record)):
        raise ValueError('invalid per-query error response')
