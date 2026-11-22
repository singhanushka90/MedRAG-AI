from dotenv import load_dotenv

from langchain_groq import ChatGroq
from deepeval.models import DeepEvalBaseLLM


load_dotenv()
class GroqJudge(DeepEvalBaseLLM):

    def __init__(self):

        self.model = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0
        )

    def load_model(self):

        return self.model

    def get_model_name(self):

        return "Groq GPT OSS 20B"

    def generate(self, prompt: str, schema=None):

        model = self.load_model()

        if schema:

            structured_model = model.with_structured_output(
                schema,
                method="json_mode"
            )

            return structured_model.invoke(prompt)

        response = model.invoke(prompt)

        return response.content

    async def a_generate(self, prompt: str, schema=None):

        model = self.load_model()

        if schema:

            structured_model = model.with_structured_output(
                schema,
                method="json_mode"
            )

            return await structured_model.ainvoke(prompt)

        response = await model.ainvoke(prompt)

        return response.content