import os
import asyncio
from carag import CaRAG, QueryParam
from carag.llm import gpt_4o_mini_complete
import json
import re
from concurrent.futures import ProcessPoolExecutor

evaluation_level_match = """
You are tasked with evaluating how well a given response matches the intended audience level. Consider the following audience types and their criteria:

Basic level: This level targets early childhood to elementary school learners. It focuses on providing simplified explanations and structured guidance to support foundational understanding and cognitive growth.
Intermediate level: Geared towards middle and high school students, this level introduces more complex concepts and encourages moderate reasoning. It aims to bridge the gap between basic comprehension and advanced analytical skills, fostering critical thinking and problem-solving abilities.
Advanced level: Designed for undergraduate students and beyond, this level explores complex, abstract concepts that require strong analytical skills. It challenges learners to engage with sophisticated ideas, promoting deep understanding and intellectual development.

Audience Type 0 (Basic level):
Is the response fun and engaging?
Does it use simple knowledge points and avoid complex vocabulary?
Are analogies and metaphors used appropriately?
Is any difficult content explained in simple terms?

Audience Type 1 (Intermediate level):
Does the response provide normal knowledge points?
Is it based on common sense and easily understandable?

Audience Type 2 (Advanced level):
Is the response professional and detailed?
Does it use technical language appropriate for experts in the field?

Question: {question}

Answer for Audience Type 0: {answer_type_0}

Answer for Audience Type 1: {answer_type_1}

Answer for Audience Type 2: {answer_type_2}

Evaluate each response based on how well it aligns with the specified audience type and provide a score out of 100 for each. Output only the scores as numbers.
"""

os.environ["OPENAI_API_KEY"] = ""
llm = gpt_4o_mini_complete  # 确保这个函数是异步的

def read_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def ter2int_by_re(string):
    res = [int(x) for x in re.findall(r"\d+", string)]
    if len(res) == 1:
        return res[0]
    else:
        return res

async def evaluate_single_response_async(data, index=None):
    question = data["question"]
    if index != 4:
        resp = [data[f"resp{i}"][index] for i in range(3)]
    else:
        resp = [data[f"resp{i}"] for i in range(3)]

    re_level_match = ter2int_by_re(await llm(prompt=evaluation_level_match.format(
        question=question,
        answer_type_0=resp[0],
        answer_type_1=resp[1],
        answer_type_2=resp[2]
    )))

    response = {
        "question": question,
        "resp0": resp[0],
        "resp1": resp[1],
        "resp2": resp[2],
        "re_level_match": re_level_match
    }
    return response

def evaluate_single_response(data, index=0):
    return asyncio.run(evaluate_single_response_async(data, index))

async def evaluate_responses(test_data, index=0, num_processes=4):
    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        tasks = [
            loop.run_in_executor(executor, evaluate_single_response, data, index)
            for data in test_data
        ]
        responses = await asyncio.gather(*tasks)
    return responses

res_path = "path2jsonl"
test_data = read_jsonl(res_path)
print(len(test_data))
if res_path.find("data_dpo_test_responses") != -1:
    index = 0
else:
    index = 4

responses = asyncio.run(evaluate_responses(test_data, index=index, num_processes=10))

re_level_match_sum = [0, 0, 0]
for response in responses:
    re_level_match_sum[0] += response["re_level_match"][0]
    re_level_match_sum[1] += response["re_level_match"][1]
    re_level_match_sum[2] += response["re_level_match"][2]

print(f"re_level_match_avg:{[x/len(responses) for x in re_level_match_sum]}")

type_ = res_path.split("/")[-1].split(".")[0]
save_path = f"path2save"
if os.path.exists(save_path):
    i = 1
    while True:
        save_path = f"path2save/{type_}_evaluation_{i}.jsonl"
        if not os.path.exists(save_path):
            break
        i += 1

with open(save_path, "w") as f:
    for response in responses:
        f.write(json.dumps(response) + "\n")
    f.write(f"re_level_match_avg:{[x/len(responses) for x in re_level_match_sum]}\n")

print(f"save to {save_path}")
