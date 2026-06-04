# EURO-5K: Web-Application

This is a companion repository for the paper: “EURO-5K:  When  Does  Domain  Pretraining  Matter? Benchmarking Transformers for EU Reporting Obligation Extraction”.

This repository provides users with an interactive web interface that acts as a complete pipeline for the legal NLP task of **reporting obligation extraction**.

You can find the main repository here: https://github.com/ntua-el21432/EURO-5K-paper.git .

---

## Project Overview

In this repository you can find the [Dockerfile](Dockerfile) that can be used to build an out-of-the-box working environment containing the [REST API](src-api), and the [web interface](src-ui).
Access to required API tokens can be requested via the following form: https://forms.gle/oVFJFcUJkRP9rBP39 .


## API Endpoints

| Endpoint        | Method | Description |
|----------------|--------|------------|
| `/models`      | GET    | Returns available models |
| `/predict`     | POST   | Extract reporting obligations |
| `/explain`     | POST   | Generate explainability output |
| `/export`      | POST   | Export results to RDF |

In-depth descriptions can be found at: http://147.102.74.52:5939/docs


## Authentication

All API endpoints (except `/models`) require an authentication token.

Include it in requests as:

Authorization: Bearer <AUTH_TOKEN>

The token is set via the environment variable:

-e AUTH_TOKEN="your_secret_token"

## Installation Notes

Depending on your deployment environment, you may need to adjust certain dependencies for optimal performance and compatibility.

If you are running the application **without GPU support**, keep the following lines **uncommented** in your requirements.txt:
```
--extra-index-url https://download.pytorch.org/whl/cpu \
#bitsandbytes \
#unsloth \
```


## Run the docker image

First, copy the .env.example file into your .env file with your actual tokens to set the required environmental variables.

To execute the software using the provided [Dockerfile](Dockerfile), clone/download the repository and build the image using the command:

```
docker build \
  --build-arg HF_TOKEN="your_token" \
  --build-arg LANGS="XXX" \
  -t <choose name> .
```
where `XXX` is a space-separated list of models (for instance, `bert_base_fft bert_eurlex_lora`).

When the build is complete, run the command:

```
docker run -d \
    --name <choose name> \
    -p 5939:5939 \
    -e AUTH_TOKEN="your_secret_token" \
    <image name>
```

where `5939` can be replaced with any other port number, depending on port availability.

You can also build the image with the following command:

```
docker-compose up --build -d
```
(Port 5939 is exposed automatically.)
