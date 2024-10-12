FROM python:3.10-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch --index-url ${PYTORCH_INDEX_URL}

COPY deps/ ./deps/
RUN pip install ./deps/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir pytest

ENV PYTHONPATH=/app

CMD ["python"]

## Test that the package builds and can be installed
FROM base AS test_install

WORKDIR /build

RUN pip install --no-cache-dir build

COPY dcl/ ./dcl/
COPY setup.cfg .
COPY pyproject.toml .

RUN python -m build
RUN pip install dist/dcl-*.whl
RUN python -c "import dcl; assert dcl.__version__ == '0.0.1'"
