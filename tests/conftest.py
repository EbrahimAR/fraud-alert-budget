import pytest
from fraud_alert_budget.synth import CATEGORIES, generate


@pytest.fixture(scope="session")
def small_df():
    return generate(n_cards=300, n_days=40, seed=7)


@pytest.fixture(scope="session")
def categories():
    return CATEGORIES
