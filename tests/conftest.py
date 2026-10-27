"""Shared fixtures: offscreen Qt app, trained MLP models, viewer widget.

Qt runs on the offscreen platform so the suite needs no display.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from AnyQt.QtCore import QPoint
from AnyQt.QtWidgets import QApplication

from Orange.data import ContinuousVariable, Domain, Table
from Orange.regression.neural_network import NNRegressionLearner


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class _DummyHoverEvent:
    """Stand-in for QGraphicsSceneHoverEvent, which PyQt6 refuses to
    instantiate. Handlers run their own code first; the super() forward
    then raises TypeError, which `fire_hover` swallows."""

    def screenPos(self):
        return QPoint(100, 100)


@pytest.fixture()
def fire_hover():
    def fire(handler):
        try:
            handler(_DummyHoverEvent())
        except TypeError:
            pass
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
            models[key] = NNRegressionLearner(
                hidden_layer_sizes=hidden, random_state=seed)(data)
        return models[key]

    return make


@pytest.fixture()
def widget(qapp):
    from orangecontrib.nnviewer.widgets.ow_nn_viewer import (
        OWNeuralNetworkViewer)
    w = OWNeuralNetworkViewer()
    w.show()
    yield w
    w.close()