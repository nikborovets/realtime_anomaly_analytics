FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Установка system dependencies
RUN apt-get update && apt-get install -y gcc

CMD ["bash"]
# CMD ["python", "-m", "ipykernel_launcher", "-f", "/dev/null"]