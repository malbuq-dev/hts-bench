# Pinned to the exact Python patch version this project was validated against
# (see requirements.txt/requirements-dev.txt for the exact dependency set
# pinned alongside it) - reproducibility is the whole point of this image, so
# nothing here is allowed to float to "whatever's newest."
FROM python:3.12.3-slim-bookworm

# LightGBM's wheel dynamically links against libgomp (OpenMP) at import time -
# not included in the slim base image by default, so `import lightgbm` fails
# without it. Nothing else in this project's dependency set needs a system
# package beyond what pip's wheels already bundle.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies in their own layer, before any source code is copied in, so an
# unchanged requirements file reuses the cached install on rebuild instead of
# reinstalling every dependency whenever a .py file changes.
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY pytest.ini ./
COPY hts_bench/ hts_bench/
COPY scripts/ scripts/
COPY tests/ tests/
COPY dataset/ dataset/

# Build-time correctness gate: the image only finishes building if the exact
# pinned dependency set still passes every test inside this exact base image -
# the same verification done by hand when validating the Python 3.12 upgrade,
# now enforced automatically on every rebuild. M5-specific tests skip cleanly
# rather than failing (dataset/m5/ isn't baked into the image - see
# .dockerignore - since it's ~70MB of generated CSVs excluded from git for the
# same reason; reproduce it inside a running container via
# `docker run --entrypoint python <image> scripts/convert_m5.py` if needed).
RUN pytest tests/ -q

ENTRYPOINT ["python", "scripts/run_benchmark.py"]
