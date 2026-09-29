"""Version shims so the pipeline runs on scikit-learn 1.4 through 1.8+."""

from sklearn.calibration import CalibratedClassifierCV


def prefit_calibrator(estimator, method: str) -> CalibratedClassifierCV:
    """Calibrate an already-fitted estimator on held-out data.

    scikit-learn < 1.6 spells this cv="prefit"; 1.6 deprecated it and 1.8 removed it in
    favour of wrapping the estimator in FrozenEstimator. Same maths either way.
    """
    try:
        from sklearn.frozen import FrozenEstimator  # scikit-learn >= 1.6
        return CalibratedClassifierCV(estimator=FrozenEstimator(estimator), method=method)
    except ImportError:  # pragma: no cover - older scikit-learn
        return CalibratedClassifierCV(estimator=estimator, method=method, cv="prefit")

