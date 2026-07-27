"""The `net` fixture must restore ALLOW_MODEL_REQUESTS even when it fails.

``ALLOW_MODEL_REQUESTS`` is a module-level global. The `net` fixture turns it
off so a test that forgets to inject a fake gets a loud refusal instead of a
real provider call -- but anything raising between the flip and the fixture's
try/finally leaves it off for the rest of the session. Every later test that
legitimately exercises `client.chat()` then fails, citing a guard it never
touched, and the real cause is nowhere in the failure output.

This is not hypothetical. Constructing `DMOrchestrator` used to sit in that
gap, and on a machine with no ANTHROPIC_API_KEY it raised there. One missing
environment variable produced 52 setup errors and 43 unrelated-looking
failures across two other test files; the first CI run this project ever did
is where that surfaced.

The fixture is driven manually here rather than requested, because a test that
*requests* a deliberately-failing fixture reports as an ERROR -- which would
leave a permanent red mark in the suite it is meant to protect.
"""

import inspect

import pytest

from dnd_bot.llm import client as llm_client
from tests.integration import test_process_action as net_module


class _ExplodingOrchestrator:
    def __init__(self):
        raise ValueError("simulated ANTHROPIC_API_KEY failure")


class TestNetFixtureRestoresTheRequestGuard:
    @pytest.mark.asyncio
    async def test_guard_is_restored_when_the_fixture_raises_mid_setup(
        self, tmp_path, monkeypatch
    ):
        # `@pytest.fixture` wraps the function so direct calls error out, and
        # keeps the original on __wrapped__. Assert rather than getattr-default:
        # if pytest ever stops doing this, this test must fail loudly, not
        # quietly stop testing anything.
        assert hasattr(net_module.net, "__wrapped__"), (
            "cannot reach the underlying fixture function -- pytest's wrapping "
            "changed, and this test is no longer exercising anything"
        )
        underlying = net_module.net.__wrapped__
        assert inspect.isasyncgenfunction(underlying)

        monkeypatch.setattr(net_module, "DMOrchestrator", _ExplodingOrchestrator)
        assert llm_client.ALLOW_MODEL_REQUESTS is True, "precondition"

        generator = underlying(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="simulated ANTHROPIC_API_KEY failure"):
            await generator.__anext__()

        assert llm_client.ALLOW_MODEL_REQUESTS is True, (
            "the failing fixture left real provider calls blocked for the rest "
            "of the session -- set_model_requests_allowed(False) must be "
            "immediately followed by the try whose finally restores it"
        )
