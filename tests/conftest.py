"""Shared fixtures: offscreen Qt app, trained MLP models, viewer widget.

Qt runs on the offscreen platform so the suite needs no display.
"""

import os
import traceback
import warnings

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from AnyQt.QtCore import QPoint
from AnyQt.QtWidgets import QApplication
from Orange.data import ContinuousVariable, Domain, Table
from Orange.regression.neural_network import NNRegressionLearner
from sklearn.exceptions import ConvergenceWarning


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class _DummyHoverEvent:
    """Stand-in for QGraphicsSceneHoverEvent, which PyQt6 refuses to
    instantiate. The widget's handlers run their own code first and then
    forward the event via super(); that forward raises TypeError on the
    dummy, which `fire_hover` is allowed to swallow -- and *only* that."""

    def screenPos(self):
        return QPoint(100, 100)


@pytest.fixture()
def fire_hover():
    def fire(handler):
        try:
            handler(_DummyHoverEvent())
        except TypeError as exc:
            # PyQt6 raises TypeError when the handler forwards the dummy
            # event to super().hover*Event(). That forward is the only
            # TypeError this fixture may swallow: verify the innermost
            # traceback frame is the handler's `super()` line, so a
            # genuine bug in the handler body is not masked.
            tb = traceback.extract_tb(exc.__traceback__)
            innermost = tb[-1]
            assert innermost.name.startswith("hover"), tb
            assert innermost.line and "super()" in innermost.line, (
                f"TypeError raised by handler itself, not the super() "
                f"forward: {exc}")
    return fire


@pytest.fixture(scope="session")
def make_model():
    """Train (and cache) an MLP regression model of the given shape."""
    models = {}

    def make(n_features=21, hidden=(3,), seed=0):
        key = (n_features, hidden, seed)
        if key not in models:
            rs = np.random.RandomState(seed)
            dom = Domain(
                [ContinuousVariable(f"F{i:02d}")
                 for i in range(n_features)],
                ContinuousVariable("Target"))
            data = Table.from_numpy(dom, rs.rand(150, n_features),
                                    rs.rand(150, 1))
            # The fixture only needs a model with the right layer shapes,
            # not a converged one; keep the test output clean.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                with np.errstate(all="ignore"):
                    models[key] = NNRegressionLearner(
                        hidden_layer_sizes=hidden,
                        max_iter=50, random_state=seed)(data)
        return models[key]

    return make


@pytest.fixture()
def widget(qapp):
    from orangecontrib.nnviewer.widgets.ow_nn_viewer import OWNeuralNetworkViewer
    w = OWNeuralNetworkViewer()
    w.show()
    yield w
    w.close()