import numpy as np
import time
import torch
import json
import os

try:
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
    print("ready!")
except:
    pass


def openai_result_to_json(result):
    try:
        return json.loads(json.dumps(result))
    except Exception as e:
        print(e)
        return None

def openai_result_to_fairseq_result(result):
    result_tokens = result["choices"][0]["logprobs"]["tokens"]
    result_logprobs = result["choices"][0]["logprobs"]["token_logprobs"]
    if result_logprobs[0] is None and len(result_logprobs)>1:
        result_logprobs[0] = np.mean(result_logprobs[1:])

    fairseq_result = {'tokens': result_tokens,
     'score': None,
     'attention': None,
     'alignment': None,
     'positional_scores': torch.tensor(result_logprobs, dtype=torch.float32)}
    fairseq_result["gpt3_response"] = openai_result_to_json(result)

    return fairseq_result

def call_openai_completion(prompt, engine, max_tokens_to_generate=0, temperature=0, logprobs_per_token=1, echo=True, stop_token=None, n=None, max_tries=60):
    response = None
    success = False

    for tries_cnt in range(max_tries):
        try:
            response = openai.Completion.create(engine=engine,
                                                prompt=prompt,
                                                max_tokens=max_tokens_to_generate,
                                                temperature=temperature,
                                                logprobs=logprobs_per_token,
                                                echo=echo,
                                                stop=stop_token,
                                                n=n)


            success = True
            break
        except openai.error.InvalidRequestError as error:
            print(f"InvalidRequestError:{error}\nPrompt sent:\n{prompt}\n")
            raise error
        except Exception as error:
            print(f"API error: {error}, retrying")
            time.sleep(1)

    if not success and tries_cnt >= max_tries:
        raise RuntimeError(f"Max {max_tries} tries to call the OpenAI api reached!")

    return response
