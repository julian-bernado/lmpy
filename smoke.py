from lmpy.server.manager import LlamaServer
from lmpy.client import Client
from lmpy.paths import find

gguf = find("openai/gpt-oss-20b-q4")
with LlamaServer(model=gguf, port=8080, ctx_size=8192, n_gpu_layers=-1, alias="olmo13-q4") as srv:
    llm = Client(base_url=srv.base_url)
    question = "How do tadpoles become frogs?"
    llm.set_system_prompt("You are a helpful assistant that speaks like a Pirate")
    answer = llm.answer(question)
    print(answer)