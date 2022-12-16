FROM google/cloud-sdk:411.0.0-slim

WORKDIR /app

COPY . /app

RUN apt update -y && apt-get update && \
    pip3 install --upgrade pip && \
    apt-get install python3 -y && pip3 install -r requirements.txt && \
    pip3 install torch==1.9.0 torchvision==0.10.0 && \
    python3 -m spacy download en_core_web_sm

CMD ["python3", "app.py"]
