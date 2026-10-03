class PipelineError(Exception):
    pass


class StageError(PipelineError):
    pass


class GateError(PipelineError):
    pass


class TriggerError(PipelineError):
    pass
