FROM google/cloud-sdk:411.0.0-slim

WORKDIR /app

COPY . /app

# RUN apt update -y && apt-get update && pip install --upgrade pip && apt-get install python3 -y && pip install -r requirements.txt && \
#     conda install pytorch==1.9.0 torchvision==0.10.0 cpuonly -c pytorch -y && \
#     python -m spacy download en_core_web_sm

RUN apt update -y && apt-get update && pip3 install --upgrade pip && apt-get install python3 -y && pip3 install -r requirements.txt && pip3 install torch==1.9.0 torchvision==0.10.0 && python3 -m spacy download en_core_web_sm

# pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

CMD ["python3", "app.py"]
