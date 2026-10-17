# Local sentence-transformers encoder.
#
# Will hold an Embedder class with:
#   load       pull the model weights from the image into memory at startup
#   encode     embed one prompt, running the model in a thread pool executor
#   dimension  the embedding width, needed when creating the Chroma collection
#
# Why encode never runs on the event loop: the call is CPU bound, and awaiting
# it directly would stall every other in flight request for the duration of a
# Tier 2 lookup. That would surface as a latency regression under concurrency
# and would be easy to misdiagnose as slow vector search.
#
# Why the model is local rather than a hosted API: a paid embedding call on
# every Tier 2 lookup, including the ones that miss, would have to be netted out
# of the headline cost figure or it would silently inflate it. It would also add
# a second external dependency to the request path of a project whose other
# claim is about surviving external failures.
#
# The PRD names sentence-transformers. This runs the same model, all-MiniLM-L6-v2,
# through the ONNX runtime that chromadb already ships, because sentence-
# transformers pulls in torch and fifteen nvidia CUDA packages to do the same job
# on a CPU. The output is the same 384 wide, unit length vector.

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Protocol

SUPPORTED_MODEL = "all-MiniLM-L6-v2"
MINILM_DIMENSION = 384

EncodeFn = Callable[[list[str]], Sequence[Sequence[float]]]


class Embedder(Protocol):
    """Anything that turns a prompt into a vector.

    A protocol rather than a base class, so the semantic cache depends on the
    shape and not on this module. Tests supply their own embedder with vectors
    chosen to produce an exact similarity, which a real model cannot do.
    """

    @property
    def dimension(self) -> int:
        """Width of the vectors this embedder produces."""
        ...

    async def encode(self, text: str) -> list[float]:
        """Embed one prompt."""
        ...


def _load_onnx_minilm() -> EncodeFn:
    """Construct chromadb's ONNX MiniLM function.

    Imported inside the function so that importing this module does not pull in
    onnxruntime, which keeps import time and the smoke test cheap for everyone
    running with Tier 2 switched off.
    """
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

    model = ONNXMiniLM_L6_V2()

    def encode(texts: list[str]) -> Sequence[Sequence[float]]:
        # The function returns float32 numpy arrays. Converting to plain floats
        # here keeps numpy out of every caller's types.
        return [vector.tolist() for vector in model(texts)]

    return encode


class OnnxEmbedder:
    """all-MiniLM-L6-v2 through the ONNX runtime, run off the event loop."""

    def __init__(self, model_name: str, loader: Callable[[], EncodeFn] = _load_onnx_minilm) -> None:
        """Record the model and how to load it; nothing is loaded here.

        Only one model is supported. A different name fails at construction,
        because silently embedding with the wrong model would make every stored
        similarity score meaningless against the configured one.

        The loader is injectable so tests can exercise this class without the
        weights, which are a 79 MB download.
        """
        if model_name != SUPPORTED_MODEL:
            raise ValueError(
                f"unsupported embedding model {model_name!r}; only {SUPPORTED_MODEL} is available"
            )
        self._loader = loader
        self._encode: EncodeFn | None = None

    def load(self) -> None:
        """Load the weights and run one warm up embedding.

        Called once during lifespan startup so that no request pays the load
        cost. In the container the weights are baked in at build time; outside
        it the first load downloads them, which is why Tier 2 is off by default.

        The warm up also checks the width, so a model that does not produce what
        the collection expects fails here rather than on the first write.
        """
        encode = self._loader()
        width = len(encode(["warm up"])[0])
        if width != MINILM_DIMENSION:
            raise RuntimeError(f"embedding width is {width}, expected {MINILM_DIMENSION}")
        self._encode = encode

    @property
    def dimension(self) -> int:
        """384, the width all-MiniLM-L6-v2 produces."""
        return MINILM_DIMENSION

    async def encode(self, text: str) -> list[float]:
        """Embed one prompt in a worker thread.

        Inference is CPU bound. Running it on the event loop would stall every
        other request for the length of a Tier 2 lookup, which under load looks
        exactly like slow vector search and would be diagnosed as that.
        """
        encode = self._encode
        if encode is None:
            raise RuntimeError("embedder used before load()")
        vectors = await asyncio.to_thread(encode, [text])
        return [float(value) for value in vectors[0]]
