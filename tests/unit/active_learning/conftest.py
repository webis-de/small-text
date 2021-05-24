import random as rand

import numpy as np
import pytest


@pytest.fixture
def random():
    rand.seed(0)
    np.random.seed(0)
