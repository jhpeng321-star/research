"""Narrow per-plan code-fault isolation; safety/physics/accounting stay fatal."""
import traceback
from .protocol import digest


# ValueErrors naming an invariant stay fatal (worker death -> slot quarantine).
FATAL_WORDS=('parity','immutable','accounting','physics','safety','budget','deadline','cuda','identity')


def per_query(error):
    """Owner 17:26: any per-query error is a decline, never a worker death.

    AssertionError/RuntimeError (CUDA, executor) still propagate; a ValueError is
    per-query unless its message names a numerical/accounting invariant.
    """
    if isinstance(error,(KeyError,TypeError,AttributeError,NameError,NotImplementedError,
            IndexError,ZeroDivisionError,StopIteration)):return True
    return type(error) is ValueError and not any(w in str(error).lower() for w in FATAL_WORDS)


def execute(record,call):
    try:
        return call()
    except Exception as error:
        if not per_query(error):raise
        # kind stays 'code' so already-deployed brokers accept the decline.
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
