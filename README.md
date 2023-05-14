# LLMServer
LLMServer is an experimental REST server for testing Large Language Models (LLMs). In partticular the server uses quantized GPTQ-for-LLaMa models. Currently it has only been tested with 13B 4-bit models (group size 128), because that is the largest models I can use.

> The server should not be used in a production environment. There are absolutely no access restrictions and CORS is disabled. You would have to change this yourself, as I have no plans of doing so.

## Installation
* Install Python. (I am using Python 3.10)
* Clone this repository into a folder
* In the GPTQ-for-LLaMa folder install the requirements and install the library
* In the project folder install the requirements
* Copy the file .env.sample to .env and change the environment variables
    * LLM_MODEL_PATH Path to folder of the LLM model you want to use.
    * LLM_CHECKPOINT Name of the main model file (usually ends on `.pt` or `.safetensors`)
    * PORT Sets the port the server will use

> The server is hard-coded to run as `localhost` to not tempt you to run it on a production server.

```bash
$ git clone <repo>
$ cd GPTQ-for-LLaMa
$ python -m pip install -r requirements.txt
$ python setup_cuda.py install
$ cd ..
$ python -m pip install -r requirements.txt
```

If you don't have some models lying around already, you will have to download them manually.

## Starting
Excecute `launch.cmd`.
On Linux execute `uvicorn llmserver:app --reload` in a terminal.
The `--reload` parameter is only needed for development
If everything has been installed correctly, the server should start after a few minutes. Loading the model takes 5-10 minutes on my machine. So be patient.

Once the server has started fully, it should be reachable on `lcoalhost:<port>`.

## Usage
Documentation for the REST API can be found at `localhost:<port>/docs`.


## Disclaimer
I am taking no responsibility whatsoever for any problems caused by using this server. So don't blame me.

> If you have read this far, and haven't understand a word, don't bother and use [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui) instead.

## Dedicatiom
To @oobabooga, who wrote a fantastic Web UI for LLMs. His borderline unusable API inspired me to write my own server.