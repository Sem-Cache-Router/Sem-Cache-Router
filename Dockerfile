# Gateway image.
#
# In order:
#   start from a slim Python base matching CI (3.14)
#   install the locked dependency set as its own layer, so a code change does
#   not reinstall the world
#   download the embedding model and the tiktoken vocabulary at build time
#   copy the application source
#   run as a non root user
#   start the ASGI server
#
# Why the model weights are baked in at build time rather than fetched on first
# use: a running container then needs no outbound network access to serve a
# request, which matters both for a locked down environment and for a project
# whose headline claim is about surviving external failures. The same holds for
# the tiktoken vocabulary the rate limiter loads at startup, which otherwise
# takes about 18 seconds to download on the first start.
#
# No secrets are baked into the image. Provider keys arrive through the
# environment at run time, and .dockerignore keeps .env out of the build context.

FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TIKTOKEN_CACHE_DIR=/home/app/.cache/tiktoken

RUN useradd --create-home --uid 10001 app

WORKDIR /srv

# The lockfile pins every package, transitive ones included, so the image gets
# exactly the set CI tests against.
COPY requirements.lock.txt .
RUN pip install -r requirements.lock.txt

USER app

# Bake the models into the app user's cache. Embedding one sentence forces the
# ONNX MiniLM download, and loading cl100k_base fetches the vocabulary.
RUN python -c "from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2; ONNXMiniLM_L6_V2()(['warm up'])" \
 && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

COPY --chown=app:app app ./app

EXPOSE 8000

# /health answers 200 whenever the process serves requests and reports Redis
# and Chroma state in its body, so this checks liveness, not dependencies.
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/health', timeout=2)" || exit 1

# The port comes from PORT when the host sets one, as Render does, and is 8000
# otherwise. exec hands the process to uvicorn, so it receives stop signals.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port \"${PORT:-8000}\""]
