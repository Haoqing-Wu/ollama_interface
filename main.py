import requests
import json
from pydantic import BaseModel, Field
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

class Parameters(BaseModel):
    debug: bool = Field(title="Debug", type="boolean", description="provide debugging output in logs", default=True)
    prompt: str = Field(title="Prompt", type="string", description="Prompt to send to the model.")
    mirostat: int = Field(title="Mirostat", type="integer", description="Enable Mirostat sampling for controlling perplexity. (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)", default=0)
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

class InputData(BaseModel):
    input: Parameters = Field(title="Input")



@app.get("/health-check")
def health_check():
    return {"status": "True"}

@app.post("/predictions")
def aigic(inputdata: InputData):
    print('+++++++++++++++++++++++++++++++++++++++++++++++++')
    print('1. Input prompt:', inputdata.input['prompt'])

    print('2. Inferring ... ')

    # Prepare the data to send to localhost:11434/api/generate
    data = {
        "model": "llama3",
        "prompt": inputdata.input['prompt'],
        "stream": False,
        "options": {
            "mirostat": inputdata.input['mirostat'],
            "mirostat_eta": inputdata.input['mirostat_eta'],
            "mirostat_tau": inputdata.input['mirostat_tau'],
            "num_ctx": inputdata.input['num_ctx'],
            "repeat_last_n": inputdata.input['repeat_last_n'],
            "repeat_penalty": inputdata.input['repeat_penalty'],
            "temperature": inputdata.input['temperature'],
            "seed": inputdata.input['seed'],
            "stop": inputdata.input['stop'],
            "tfs_z": inputdata.input['tfs_z'],
            "num_predict": inputdata.input['num_predict'],
            "top_k": inputdata.input['top_k'],
            "top_p": inputdata.input['top_p'],
        }
    }
    response = requests.post(generate_url, json=data).json()
    response.pop('context')
    print('5. Output Info:', response)

    if inputdata.input['debug'] == False:
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
    uvicorn.run(app=app, host="0.0.0.0", port=5001)