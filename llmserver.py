import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)

import sys
from os import environ, getenv
from pathlib import Path
from typing import Any, List

from dotenv import load_dotenv

sys.path.insert(0, str(Path("GPTQ-for-LLaMa")))

import random
from contextlib import asynccontextmanager

import torch
from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer
from yaspin import yaspin

from model import load_quantized, load_tokenizer

load_dotenv()

def set_manual_seed(seed):
    seed = int(seed)
    if seed == -1:
        seed = random.randint(1, 2**31)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return seed

embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large", model_kwargs={'device': 'cpu'})
tokenizer : AutoTokenizer = None
pipe : HuggingFacePipeline = None
llm = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm
    global pipe
    global tokenizer
    print("Initializing...")
    assert getenv("LLM_MODEL_PATH") and getenv("LLM_CHECKPOINT"), "LLM_MODEL_PATH and LLM_CHECKPOINT need to be defined."
    tokenizer = load_tokenizer(environ.get("LLM_MODEL_PATH"))
    model = load_quantized(environ.get("LLM_MODEL_PATH"), environ.get("LLM_CHECKPOINT"))
    pipe = pipeline("text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0,
        temperature=0.7,
        max_new_tokens=500,
        do_sample=True,
        top_p=0.5,
        typical_p=1,
        repetition_penalty=1.2,
        top_k=40,
        min_length=0,
        no_repeat_ngram_size=0,
        num_beams=1,
        penalty_alpha=0,
        length_penalty=1,
        early_stopping=False
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    print("Initialization finished.")

    yield

@yaspin("Generating text...")
def _generate(input):
    set_manual_seed(-1)
    return llm(input)

class Options(BaseModel):
    temperature: float | None = 0.7
    max_new_tokens: int | None = 500


class LLMProperties(BaseModel):
    prompt: str
    options: Options | None = None

class StrResult(BaseModel):
    result: str

class FloatListResult(BaseModel):
    result: List[float]

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/generate", summary="Generate text from a prompt.")
async def generate(input: LLMProperties) -> StrResult:
    pipe.temperature = input.options.temperature
    pipe.max_new_tokens = input.options.max_new_tokens
    text = _generate(input.prompt)
    return {"result": text}

@app.post("/api/embed/documents", summary="Create embeddings for a list of strings.")
async def embedDocuments(texts: list[str]) -> FloatListResult:
    ret = embeddings.embed_documents(texts)
    return {"result": ret}

@app.post("/api/embed/query", summary="Create embeddings for a single string.")
async def embedQuery(query: str = Body()) -> FloatListResult:
    ret = embeddings.embed_query(query)
    return {"result": ret}

@app.post("/api/tokens/count", response_class=PlainTextResponse, summary="Get the number of tokens of a string.")
async def tokenCount(text: str = Body(media_type="text/plain")) -> str:
    tokens = tokenizer(text)
    print(text, tokens)
    return str(len(tokens['input_ids']))
