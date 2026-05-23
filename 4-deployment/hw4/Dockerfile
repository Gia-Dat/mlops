FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv pip install --system pandas pyarrow scikit-learn==1.5.2

COPY starter.py .

ENTRYPOINT [ "python", "starter.py" ]