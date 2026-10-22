FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git libglib2.0-0 libgl1 fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY data_fetching/requirements.txt data_fetching/requirements-quality.txt /workspace/data_fetching/
RUN python -m pip install --no-cache-dir -r data_fetching/requirements-quality.txt
COPY data_fetching /workspace/data_fetching
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "-m", "data_fetching"]
CMD ["--help"]
