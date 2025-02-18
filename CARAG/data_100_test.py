import os
from carag import CaRAG, QueryParam
from carag.llm import gpt_4o_mini_complete,llama_8b_31_complete,gpt_4o_complete
from tqdm import tqdm

WORKING_DIR = "path2working_dir"


reslut_path="path2reslut"


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


rag = CaRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_complete,  # Use gpt_4o_mini_complete LLM model
    # log_level="DEBUG",
    log_level="INFO",
    chunk_overlap_token_size=100,
    chunk_token_size=600
    # llm_model_func=gpt_4o_complete  # Optionally, use a stronger model
)
import json
def read_jsonl(file_path):
    data = []
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

already=read_jsonl(reslut_path)
already_set=set()
for al in already:
    already_set.add(al["question"])

use_llama = True
use_cache = False
difficulty_level_with_low = True
# Perform naive search

only_need_context = False
only_need_prompt = False
use_gtp4=True
use_rag=False
shot=True
responses=[]
data_path="path2data"
datas=read_jsonl(data_path)
quesion_set=set()
for data in tqdm(datas):
    if data_path.find("peom")!=-1:
        quesion=data["prompt"].split("的回答：\n 问题：")[1].strip()
    else:
        quesion=data["prompt"].split(" question: ")[1].strip()
    quesion_set.add(quesion)


for quesion in tqdm(quesion_set):
    if quesion in already_set:
        print(f"already have {quesion}")
        continue

    # 捕获异常
    try:
        resp0=(rag.query(query=quesion, param=QueryParam(mode="local",difficulty_level=0,use_llama=use_llama,use_gtp4=use_gtp4,use_rag=use_rag,
                                        use_cache=use_cache,only_need_context=only_need_context,only_need_prompt=only_need_prompt,shot=shot,
                                        difficulty_level_with_low=difficulty_level_with_low)))
        resp2=(rag.query(query=quesion, param=QueryParam(mode="local",difficulty_level=2,use_llama=use_llama,use_gtp4=use_gtp4,use_rag=use_rag,
                                        use_cache=use_cache,only_need_context=only_need_context,only_need_prompt=only_need_prompt,shot=shot,
                                        difficulty_level_with_low=difficulty_level_with_low)))
        resp1=(rag.query(query=quesion, param=QueryParam(mode="local",difficulty_level=1,use_llama=use_llama,use_gtp4=use_gtp4,use_rag=use_rag,
                                        use_cache=use_cache,only_need_context=only_need_context,only_need_prompt=only_need_prompt,shot=shot,
                                        difficulty_level_with_low=difficulty_level_with_low)))
    except Exception as e:
        print(f"error:{e}")
        continue
    response={"question":quesion,"resp0":resp0,"resp1":resp1,"resp2":resp2}
    with open(reslut_path, "a") as f:
        f.write(json.dumps(response,ensure_ascii=False)+"\n")
