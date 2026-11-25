# Embedder tests.
#
# Nearly everything here runs against an injected loader, so the suite never
# needs the real weights. The one test that does is marked `model` and skipped
# unless explicitly enabled; see the gate in conftest.py.

from __future__ import annotations

import asyncio
import threading
from collections.abc import Sequence

import pytest

from app.cache.embedder import MINILM_DIMENSION, SUPPORTED_MODEL, EncodeFn, OnnxEmbedder


class RecordingModel:
    """A stand-in for the ONNX model that records which thread ran it."""

    def __init__(self, width: int = MINILM_DIMENSION) -> None:
        self.width = width
        self.threads: list[int] = []
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str]) -> Sequence[Sequence[float]]:
        self.threads.append(threading.get_ident())
        self.calls.append(texts)
        return [[0.5] * self.width for _ in texts]


def embedder_with(model: RecordingModel) -> OnnxEmbedder:
    """An embedder whose loader returns the given stand-in."""

    def loader() -> EncodeFn:
        return model

    return OnnxEmbedder(SUPPORTED_MODEL, loader=loader)


def test_an_unsupported_model_is_refused_at_construction() -> None:
    """Embedding with the wrong model would make every stored score meaningless."""
    with pytest.raises(ValueError, match="unsupported embedding model"):
        OnnxEmbedder("some-other-model")


def test_construction_loads_nothing() -> None:
    """Weights are loaded in the lifespan, not when the object is built."""
    loaded: list[bool] = []

    def loader() -> EncodeFn:
        loaded.append(True)
        return RecordingModel()

    OnnxEmbedder(SUPPORTED_MODEL, loader=loader)
    assert loaded == []


async def test_encode_before_load_is_an_error() -> None:
    """Using an unloaded embedder is a wiring bug and should say so."""
    embedder = embedder_with(RecordingModel())
    with pytest.raises(RuntimeError, match="before load"):
        await embedder.encode("hello")


def test_load_runs_a_warm_up_embedding() -> None:
    """The first request should not be the one that pays the load cost."""
    model = RecordingModel()
    embedder_with(model).load()
    assert model.calls == [["warm up"]]


def test_load_rejects_a_model_of_the_wrong_width() -> None:
    """Caught at startup rather than as a dimension error on the first write."""
    with pytest.raises(RuntimeError, match="expected 384"):
        embedder_with(RecordingModel(width=128)).load()


async def test_encode_returns_one_vector_of_plain_floats() -> None:
    """Callers get a list of float, not a numpy array."""
    embedder = embedder_with(RecordingModel())
    embedder.load()
    vector = await embedder.encode("What is the capital of France?")
    assert len(vector) == MINILM_DIMENSION
    assert all(type(value) is float for value in vector)


async def test_encode_does_not_run_on_the_event_loop_thread() -> None:
    """The point of the module: inference is CPU bound and must not stall the loop."""
    model = RecordingModel()
    embedder = embedder_with(model)
    embedder.load()
    loop_thread = threading.get_ident()

    await embedder.encode("hello")

    assert model.threads[-1] != loop_thread


def test_dimension_is_known_before_loading() -> None:
    """The collection can be set up without waiting for the weights."""
    assert embedder_with(RecordingModel()).dimension == MINILM_DIMENSION


@pytest.mark.model
async def test_real_model_produces_unit_vectors_of_the_expected_width() -> None:
    """Against the real weights. Opt in with SEMCACHE_RUN_MODEL_TESTS=1."""
    embedder = OnnxEmbedder(SUPPORTED_MODEL)
    embedder.load()
    vector = await embedder.encode("What is the capital of France?")
    assert len(vector) == MINILM_DIMENSION
    assert abs(sum(value * value for value in vector) - 1.0) < 1e-3


# --- recent embeddings ------------------------------------------------------


def cached_embedder(model: RecordingModel, cache_size: int = 2) -> OnnxEmbedder:
    """A loaded embedder over the stand-in, keeping cache_size recent vectors."""

    def loader() -> EncodeFn:
        return model

    embedder = OnnxEmbedder(SUPPORTED_MODEL, loader=loader, cache_size=cache_size)
    embedder.load()
    model.calls.clear()
    return embedder


async def test_a_repeated_prompt_is_embedded_once() -> None:
    model = RecordingModel()
    embedder = cached_embedder(model)
    first = await embedder.encode("hello")
    second = await embedder.encode("hello")
    assert first == second
    assert model.calls == [["hello"]]


async def test_the_least_recently_used_vector_is_dropped_first() -> None:
    model = RecordingModel()
    embedder = cached_embedder(model, cache_size=2)
    await embedder.encode("a")
    await embedder.encode("b")
    await embedder.encode("a")  # a is now the most recent
    await embedder.encode("c")  # so b goes
    await embedder.encode("a")
    await embedder.encode("b")
    assert model.calls == [["a"], ["b"], ["c"], ["b"]]


async def test_a_cache_size_of_zero_keeps_nothing() -> None:
    model = RecordingModel()
    embedder = cached_embedder(model, cache_size=0)
    await embedder.encode("hello")
    await embedder.encode("hello")
    assert model.calls == [["hello"], ["hello"]]


def test_a_negative_cache_size_is_refused() -> None:
    with pytest.raises(ValueError, match="negative"):
        OnnxEmbedder(SUPPORTED_MODEL, cache_size=-1)


async def test_changing_a_returned_vector_does_not_change_the_cache() -> None:
    embedder = cached_embedder(RecordingModel())
    vector = await embedder.encode("hello")
    vector[0] = 99.0
    assert (await embedder.encode("hello"))[0] == 0.5


async def test_concurrent_requests_for_one_prompt_share_one_embedding() -> None:
    model = RecordingModel()
    embedder = cached_embedder(model, cache_size=0)
    vectors = await asyncio.gather(*(embedder.encode("same") for _ in range(10)))
    assert model.calls == [["same"]]
    assert all(vector == vectors[0] for vector in vectors)


async def test_a_caller_that_gives_up_does_not_cancel_the_others() -> None:
    release = threading.Event()

    class SlowModel(RecordingModel):
        def __call__(self, texts: list[str]) -> Sequence[Sequence[float]]:
            if texts == ["slow"]:
                release.wait(5)
            return super().__call__(texts)

    model = SlowModel()
    embedder = cached_embedder(model)
    impatient = asyncio.create_task(embedder.encode("slow"))
    patient = asyncio.create_task(embedder.encode("slow"))
    await asyncio.sleep(0.05)
    impatient.cancel()
    release.set()
    vector = await patient
    assert len(vector) == MINILM_DIMENSION
    assert impatient.cancelled()
    assert model.calls == [["slow"]]


async def test_a_failed_embedding_is_not_remembered() -> None:
    attempts = []

    class FlakyModel(RecordingModel):
        def __call__(self, texts: list[str]) -> Sequence[Sequence[float]]:
            attempts.append(texts)
            if len(attempts) == 2:  # the first call is load()'s warm up
                raise RuntimeError("model hiccup")
            return super().__call__(texts)

    embedder = cached_embedder(FlakyModel())
    with pytest.raises(RuntimeError, match="hiccup"):
        await embedder.encode("hello")
    assert len(await embedder.encode("hello")) == MINILM_DIMENSION
