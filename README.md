# code-generation-using-transformers

## Problem statement
Recently, Machine learning methods have been used to create powerful language models for a broad range of natural language processing tasks. An important subset of this field is that of generating code of programming languages for automatic software development. Given a textual data of python questions with answer find a correct python code for the given coding question.

## Solution proposed
This issue has been addressed as a Sequence to Sequence(Seq2Seq) learning.  The input or SRC sequence in this case will be an English text, and the output or TRG sequence will be a Python code. Transformers have become the standard architecture for handling Seq2Seq issues during the past few years. The majority of modern SOTA models, such as the BERT or GPT-3, use transformers internally. I have applied multi-headed attention when using transformers. Transformer architecture has been combined with encoder and decoder architecture.

## Dataset used
A random dataset which containes a python question and a code has been used for this project. 

## Tech stack used
1. Python 3.8
2. FastAPI
3. Deep learning
4. Natural language processing
5. Transformers
6. Encode and Decoder
7. Docker

## Infrastructure required
1. Google compute engine
2. Google cloud storage
3. Google artifact registry
4. Circle CI

## How to run?
Step 1. Cloning the repository.
```
git clone https://github.com/Deep-Learning-01/code-generation-using-transformers.git
```
Step 2. Create a conda environment.
```
conda create -p env python=3.8 -y
```
```
conda activate env/
```
Step 3. Install the Pytorch library with conda
```
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cpuonly -c pytorch -y
```
Step 4. Install the requirements
```
pip install -r requirements.txt
```
Step 5. Download the Spacy english model
```
python -m spacy download en_core_web_sm
```
Step 6. Install Google Cloud Sdk and configure
For windows -
```
https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe
```
For Ubuntu
```
sudo apt-get install apt-transport-https ca-certificates gnupg
```
```
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
```
```
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
```
```
sudo apt-get update && sudo apt-get install google-cloud-cli
```
```
gcloud init
```
Before running server application make sure your `Google Cloud Storage` bucket is available

Step 7. Run the application server
```
python app.py
```
Step 8. Train application
```
http://localhost:8080/train
```
Step 9. Prediction application
```
http://localhost:8080/predict
```
## Run locally
1. Check if the Dockerfile is available in the project directory/
2. Build the docker image
```
docker build -t hate-speech . 
```
3. Run the docker image
```
docker run -d -p 8080:8080 <IMAGEID>
```

## `src` is the main package folder which contains -

**Components** : Contains all components of this Project
- DataIngestion
- DataTransformation
- ModelTrainerAndEval
- ModelPusher

**Custom Logger and Exceptions** are used in the Project for better debugging purposes.

## Conclusion
It has been shown that code generation is an effective tool for beginners who wish to get their hands dirty with coding. We are all aware of how challenging it may be to comprehend syntax and optimization for someone who has only recently begun coding. Those who have recently begun to code will undoubtedly benefit from this application.