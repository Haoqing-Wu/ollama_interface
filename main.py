import requests
import json
from pydantic import BaseModel
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

generate_url = "http://localhost:11434/api/generate"


app = FastAPI(title="llama3",version='0.0.1')


origins = ["*"]
app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins,  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"])  
        
class InputData(BaseModel):
    debug: bool = True
    prompt: str
    mirostat: int = 0
    mirostat_eta: float = 0.1
    mirostat_tau: float = 5.0
    num_ctx: int = 2048
    repeat_last_n: int = 64
    repeat_penalty: float = 1.1
    temperature: float = 0.8
    seed: int = 0
    stop: list = ["\n"]
    tfs_z: int = 1
    num_predict: int = 128
    top_k: int = 40
    top_p: float = 0.9



@app.post("/aigic")
def aigic(inputdata: InputData):
    print('+++++++++++++++++++++++++++++++++++++++++++++++++')
    print('1. Input prompt:', inputdata.prompt)

    print('2. Inferring ... ')

    # Prepare the data to send to localhost:11434/api/generate
    data = {
        "model": "llama3",
        "prompt": inputdata.prompt,
        "stream": False,
        "options": {
            "mirostat": inputdata.mirostat,
            "mirostat_eta": inputdata.mirostat_eta,
            "mirostat_tau": inputdata.mirostat_tau,
            "num_ctx": inputdata.num_ctx,
            "repeat_last_n": inputdata.repeat_last_n,
            "repeat_penalty": inputdata.repeat_penalty,
            "temperature": inputdata.temperature,
            "seed": inputdata.seed,
            "stop": inputdata.stop,
            "tfs_z": inputdata.tfs_z,
            "num_predict": inputdata.num_predict,
            "top_k": inputdata.top_k,
            "top_p": inputdata.top_p,
        }
    }
    response = requests.post(generate_url, json=data).json()
    response.pop('context')
    print('5. Output Info:', response)

    if inputdata.debug == False:
        response.pop('created_at')
        response.pop('done')
        response.pop('total_duration')
        response.pop('load_duration')
        response.pop('prompt_eval_duration')
        response.pop('eval_count')
        response.pop('eval_duration')

    ret = {
    "output": response
    }

    return {
    "code": 200,
    "msg": "success",
    "content": ret 
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app=app, host="0.0.0.0", port=8090)