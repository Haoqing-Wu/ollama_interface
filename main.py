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
    input: dict = {
        "debug": False,
        "prompt": "string",
        "mirostat": 0,
        "mirostat_eta": 0.1,
        "mirostat_tau": 5.0,
        "num_ctx": 2048,
        "repeat_last_n": 64,
        "repeat_penalty": 1.1,
        "temperature": 0.8,
        "seed": 0,
        "stop": ["\n"],
        "tfs_z": 1,
        "num_predict": 128,
        "top_k": 40,
        "top_p": 0.9
    }


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