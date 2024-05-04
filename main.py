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
    mirostat_eta: float = Field(title="Mirostat Eta", type="number", description="Influences how quickly the algorithm responds to feedback from the generated text. A lower learning rate will result in slower adjustments, while a higher learning rate will make the algorithm more responsive.", default=0.1)
    mirostat_tau: float = Field(title="Mirostat Tau", type="number", description="Controls the balance between coherence and diversity of the output. A lower value will result in more focused and coherent text.", default=5.0)
    num_ctx: int = Field(title="Number of context", type="integer", description="Sets the size of the context window used to generate the next token.", default=2048)
    repeat_last_n: int = Field(title="Repeat last n", type="integer", description="Sets how far back for the model to look back to prevent repetition. (Default: 64, 0 = disabled, -1 = num_ctx)", default=64)
    repeat_penalty: float = Field(title="Repeat penalty", type="number", description="Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient. (Default: 1.1)", default=1.1)
    temperature: float = Field(title="Temperature", type="number", description="The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.8)", default=0.8)
    seed: int = Field(title="Seed", type="integer", description="Seed for the random number generator. (Default: 0)", default=0)
    stop: list = Field(title="Stop", type="array", description="List of tokens to stop generation at.", default=["\n"])
    tfs_z: float = Field(title="TFS Z", type="number", description="Tail free sampling is used to reduce the impact of less probable tokens from the output. A higher value (e.g., 2.0) will reduce the impact more, while a value of 1.0 disables this setting. (default: 1)", default=1.0)
    num_predict: int = Field(title="Number of predictions", type="integer", description="Number of tokens to predict at each step. (default: 128)", default=128)
    top_k: int = Field(title="Top k", type="integer", description="Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)", default=40)
    top_p: float = Field(title="Top p", type="number", description="Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)", default=0.9)

class InputData(BaseModel):
    input: Parameters = Field(title="Input")



@app.get("/health-check")
def health_check():
    return {"status": "True"}

@app.post("/predictions")
def aigic(inputdata: InputData):
    print('+++++++++++++++++++++++++++++++++++++++++++++++++')
    print('1. Input prompt:', inputdata.input.prompt)

    print('2. Inferring ... ')

    # Prepare the data to send to localhost:11434/api/generate
    data = {
        "model": "llama3",
        "prompt": inputdata.input.prompt,
        "stream": False,
        "options": {
            "mirostat": inputdata.input.mirostat,
            "mirostat_eta": inputdata.input.mirostat_eta,
            "mirostat_tau": inputdata.input.mirostat_tau,
            "num_ctx": inputdata.input.num_ctx,
            "repeat_last_n": inputdata.input.repeat_last_n,
            "repeat_penalty": inputdata.input.repeat_penalty,
            "temperature": inputdata.input.temperature,
            "seed": inputdata.input.seed,
            "stop": inputdata.input.stop,
            "tfs_z": inputdata.input.tfs_z,
            "num_predict": inputdata.input.num_predict,
            "top_k": inputdata.input.top_k,
            "top_p": inputdata.input.top_p,
        }
    }
    response = requests.post(generate_url, json=data).json()
    #response.pop('context')
    print('5. Output Info:', response)

    if inputdata.input.debug == False:
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