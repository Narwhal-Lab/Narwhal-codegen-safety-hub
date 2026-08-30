


class QProbeError(Exception):
    pass


class ConfigurationError(QProbeError):
    pass


class DataValidationError(QProbeError):
    pass


class ModelCallError(QProbeError):
    pass


class ScientificDefinitionError(QProbeError):
    pass


class ArtifactValidationError(QProbeError):
    pass
