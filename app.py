from fastapi import FastAPI
from pydantic import BaseModel

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser

from langchain.chains import LLMChain
from llm_app import llm

app = FastAPI()

class RequestSchema(BaseModel):
    """ Request Schema """
    message: str

@app.post("/llm_response")
async def get_response(item: RequestSchema):
    user_message: str = item.message
    llm_request = LLMRequest(message=user_message)
    response = llm_request()
    
    return response["text"]


class LLMRequest:
    def __init__(self, message: str, temperature: int=0.1):
        self.message = message
        self.temperature = temperature

    def _define_prompt(self):
        template: str = """
        You act as a text analyzer. Go through the following message and return response in Yes, No or Unclear.
        If the message states if someone is going to be on leave or cannot join to office, respond Yes,
        If the message states if someone shall join office but will be late, respond No.
        If you are undecided or the message is not clear, respond Unclear.

        Message: {message}

        Reply:
        """
        return template

    def __call__(self):
        template = self._define_prompt()
        prompt = PromptTemplate.from_template(template)
        #chain = prompt | llm | StrOutputParser
        chain = LLMChain(
            llm=llm,
            prompt=prompt
        )

        response = chain.invoke({"message": self.message})

        print(response)

        return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)


# curl -X POST -d '{"message": " I have to go to the hospital, so i will be late by an hour or so."}' -H "Content-Type: application/json"  localhost:8000/llm_response
