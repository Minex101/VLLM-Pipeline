FROM vllm/vllm-openai:latest
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
CMD ["python3", "src/run_pipeline.py"]