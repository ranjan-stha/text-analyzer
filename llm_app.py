from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
#from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = Ollama(model="qwen2:0.5b")
