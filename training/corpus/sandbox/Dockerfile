# Sandbox image for executing the pinned third-party test suites (v0.2 corpus).
# Build is the ONLY network-enabled step: it installs hash-pinned wheels.
# Runtime containers run with --network none (see sandbox_python.sh).
FROM python@sha256:4840d514c583a4d3d8cad3940e68dee6f0f6f3635c12fcd0a3d74c21a5b93a54
COPY requirements-sandbox.txt /tmp/requirements-sandbox.txt
RUN python -m venv /opt/venv \
 && /opt/venv/bin/python -m pip install --no-cache-dir --only-binary=:all: \
      --require-hashes --no-deps -r /tmp/requirements-sandbox.txt \
 && rm /tmp/requirements-sandbox.txt \
 && chmod -R a+rX /opt/venv
