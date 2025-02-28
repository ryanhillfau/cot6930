##
## Prompt Engineering Lab
## Platform for Education and Experimentation with Prompt NEngineering in Generative Intelligent Systems
## _pipeline.py :: Simulated GenAI Pipeline 
## 
#  
# Copyright (c) 2025 Dr. Fernando Koch, The Generative Intelligence Lab @ FAU
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights 
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# Documentation and Getting Started:
#    https://github.com/GenILab-FAU/prompt-eng
#
# Disclaimer: 
# Generative AI has been used extensively while developing this package.
# 


import requests
import json
import os
import time

def load_config():
    """
    Load config file looking into multiple locations
    """
    config_locations = [
        "./_config",
        "prompt-eng/_config",
        "../_config"
    ]
    
    # Find CONFIG
    config_path = None
    for location in config_locations:
        if os.path.exists(location):
            config_path = location
            break
    
    if not config_path:
        raise FileNotFoundError("Configuration file not found in any of the expected locations.")
    
    # Load CONFIG
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()


def create_payload(model, prompt, target="ollama", **kwargs):
    """
    Create the Request Payload in the format required byt the Model Server
    @NOTE: 
    Need to adjust here to support multiple target formats
    target can be only ('ollama' or 'open-webui')

    @TODO it should be able to self_discover the target Model Server
    [Issue 1](https://github.com/genilab-fau/prompt-eng/issues/1)
    """

    payload = None
    if target == "ollama":
        payload = {
            "model": model,
            "prompt": prompt, 
            "stream": False,
        }
        if kwargs:
            payload["options"] = {key: value for key, value in kwargs.items()}

    elif target == "open-webui":
        '''
        @TODO need to verify the format for 'parameters' for 'open-webui' is correct.
        [Issue 2](https://github.com/genilab-fau/prompt-eng/issues/2)
        '''
        payload = {
            "model": model,
            "messages": [ {"role" : "user", "content": prompt } ]
        }

        # @NOTE: Taking not of the syntaxes we tested before; none seems to work so far 
        #payload.update({key: value for key, value in kwargs.items()})
        #if kwargs:
        #   payload["options"] = {key: value for key, value in kwargs.items()}
        
    else:
        print(f'!!ERROR!! Unknown target: {target}')
    return payload


def model_req(payload=None):
    """
    Issue request to the Model Server
    """
        
    # CUT-SHORT Condition
    try:
        load_config()
    except:
        return -1, f"!!ERROR!! Problem loading prompt-eng/_config"

    url = os.getenv('URL_GENERATE', None)
    api_key = os.getenv('API_KEY', None)
    delta = response = None

    headers = dict()
    headers["Content-Type"] = "application/json"
    if api_key: headers["Authorization"] = f"Bearer {api_key}"

    #print(url, headers)
    print(payload)

    # Send out request to Model Provider
    try:
        start_time = time.time()
        response = requests.post(url, data=json.dumps(payload) if payload else None, headers=headers)
        delta = time.time() - start_time
    except:
        return -1, f"!!ERROR!! Request failed! You need to adjust prompt-eng/config with URL({url})"

    # Checking the response and extracting the 'response' field
    if response is None:
        return -1, f"!!ERROR!! There was no response (?)"
    elif response.status_code == 200:

        ## @NOTE: Need to adjust here to support multiple response formats
        result = ""
        delta = round(delta, 3)

        response_json = response.json()
        if 'response' in response_json: ## ollama
            result = response_json['response']
        elif 'choices' in response_json: ## open-webui
            result = response_json['choices'][0]['message']['content']
        else:
            result = response_json 
        
        return delta, result
    elif response.status_code == 401:
        return -1, f"!!ERROR!! Authentication issue. You need to adjust prompt-eng/config with API_KEY ({url})"
    else:
        return -1, f"!!ERROR!! HTTP Response={response.status_code}, {response.text}"
    return


###
## FEW SHOTS PROMPTING
##

from _pipeline import create_payload, model_req

#### (1) Adjust the inbounding  Prompt, simulating inbounding requests from users or other systems
MESSAGE = "California"

#### (2) Adjust the Prompt Engineering Technique to be applied, simulating Workflow Templates
FEW_SHOT1 = "You are a real estate investor. You want to present a list of the average price per square foot for the bottom 15 counties in the following state for properties that sold for less than $300000, provide only the list, no other information:"
PROMPT1 = FEW_SHOT1 + '\n' + MESSAGE 

#### (3) Configure the Model request, simulating Workflow Orchestration
# Documentation: https://github.com/ollama/ollama/blob/main/docs/api.md
payload = create_payload(target="ollama",
                         model="llama3.2:latest", 
                         prompt=PROMPT1, 
                         temperature=1.0, 
                         num_ctx=100, 
                         num_predict=250)

### YOU DONT NEED TO CONFIGURE ANYTHING ELSE FROM THIS POINT
# Send out to the model
time, response = model_req(payload=payload)
print("Generated List 1")
print(response)
if time: print(f'Time taken: {time}s')

#### (1) Adjust the inbounding  Prompt, simulating inbounding requests from users or other systems
GENERATED_OUTPUT1 = response

#### (2) Adjust the Prompt Engineering Technique to be applied, simulating Workflow Templates
FEW_SHOT2 = "You are a real estate investor. You want to present a list of the average rent per square foot for the top 15 counties in the following state; provide only the list and no other information:"
PROMPT2 = FEW_SHOT2 + '\n' + MESSAGE 

#### (3) Configure the Model request, simulating Workflow Orchestration
# Documentation: https://github.com/ollama/ollama/blob/main/docs/api.md
payload2 = create_payload(target="ollama",
                         model="llama3.2:latest", 
                         prompt=PROMPT2, 
                         temperature=1.0, 
                         num_ctx=100, 
                         num_predict=250)

### YOU DONT NEED TO CONFIGURE ANYTHING ELSE FROM THIS POINT
# Send out to the model
time, response = model_req(payload=payload2)
print("Second Generated List")
print(response)
if time: print(f'Time taken: {time}s')
GENERATED_OUTPUT2 = response

# Attempt to get the LLM to combine the results of the two prompts
FEW_SHOT3 = "You are a data analyst crafting an executive summary. You have been given two lists of counties. Without making any inferences, you must list counties in" + MESSAGE + "that appear on both of the following lists."
PROMPT3 = FEW_SHOT3 + '\n' + "The first list is " + GENERATED_OUTPUT1 + '\n' + "The second list is " + GENERATED_OUTPUT2

#### (3) Configure the Model request, simulating Workflow Orchestration
# Documentation: https://github.com/ollama/ollama/blob/main/docs/api.md
payload3 = create_payload(target="ollama",
                         model="llama3.2:latest", 
                         prompt=PROMPT3, 
                         temperature=1.0, 
                         num_ctx=100, 
                         num_predict=1000)

### YOU DONT NEED TO CONFIGURE ANYTHING ELSE FROM THIS POINT
# Send out to the model
time, response = model_req(payload=payload3)
print("Final Response ")
print(response)
if time: print(f'Time taken: {time}s')
