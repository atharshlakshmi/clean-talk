FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy frontend code and shared utilities
COPY src/app.py ./src/app.py
COPY src/utils ./src/utils
COPY src/core ./src/core

ENV PYTHONPATH=/app

EXPOSE 8501

CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]