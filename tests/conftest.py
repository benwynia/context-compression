import pytest

from ctxc.synth import synth_session
from ctxc.tokens import TokenCounter


@pytest.fixture(scope="session")
def counter():
    return TokenCounter()


@pytest.fixture(scope="session")
def session_messages():
    """A long synthetic coding-agent session (~hundreds of k tokens)."""
    return synth_session(rounds=40, seed=7)


@pytest.fixture(scope="session")
def small_session():
    return synth_session(rounds=8, seed=3, result_lines=(10, 40))
