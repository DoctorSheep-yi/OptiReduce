FROM python:3.10-slim
WORKDIR /app
COPY node.py .

ENTRYPOINT ["python", "node.py"]