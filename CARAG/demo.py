import os
from carag import CaRAG, QueryParam
from carag.llm import gpt_4o_mini_complete, gpt_4o_complete,llama_8b_31_complete

#########
# Uncomment the below two lines if running in a jupyter notebook to handle the async nature of rag.insert()
# import nest_asyncio
# nest_asyncio.apply()
#########

WORKING_DIR = "path2working_dir"

os.environ["OPENAI_API_KEY"] = ""

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = CaRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_complete,  
    # log_level="DEBUG",
    log_level="INFO",
    chunk_overlap_token_size=100,
    chunk_token_size=600
)


rag.insert(open("./books.txt").read())

use_llama = True
use_cache = False
difficulty_level_with_low = True
# Perform naive search

only_need_context = False
only_need_prompt = False
print(rag.query("What is the derivative of a function?", param=QueryParam(mode="local",difficulty_level=0,use_llama=use_llama,
                                    use_cache=use_cache,only_need_context=only_need_context,only_need_prompt=only_need_prompt,
                                    difficulty_level_with_low=difficulty_level_with_low)))
