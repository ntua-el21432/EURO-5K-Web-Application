FROM node:18-alpine AS VueImage

WORKDIR /app/frontend

COPY ./src-ui/package.json ./src-ui/package-lock.jso[n] /app/frontend/
RUN npm install --omit=optional

COPY ./src-ui ./

RUN VUE_APP_API_URL= VUE_PUBLIC_PATH=/ui/ npm run build


FROM python:3.10-slim

WORKDIR /code

RUN apt-get update && apt-get install -y \
    jq tar curl build-essential git \
    && rm -rf /var/lib/apt/lists/* \
    
COPY ./src-api/requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade pip
RUN pip install -v --no-cache-dir -r /code/requirements.txt
RUN python -m spacy download en_core_web_sm

COPY ./src-api/mappings.json /code/mappings.json

ARG LANGS

COPY ./src-api/install.sh /code/install.sh
RUN chmod u+x install.sh 

RUN ./install.sh $LANGS

COPY ./src-api/ /code/

COPY --from=VueImage ./app/frontend/dist/. /code/dist/.

EXPOSE 5939

CMD ["uvicorn", "server:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "5939"]
