from util import get_chatbot, get_dnd_data, get_llm

dnd = get_dnd_data()

# System prompt describes information given to all conversations
prompt = """
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant for answering questions. Use only the context to answer questions.
<</SYS>>
"""


def context_aware_prompt(user_prompt, context):
    return (
        prompt
        + f"""
    [INST]
    Question:
    {user_prompt}
    Context:
    {context}

    Based on the information in the context above, please answer the question.
    [/INST]
    """
    )


class DndBot:
    def __init__(self):
        self.dnd = get_dnd_data()
        self.true_page_map = {x["page"] + 1: x["text"] for x in self.dnd}
        self._model = None
        self._llm = None

    @property
    def llm(self):
        if not self._llm:
            self._model = get_llm()
            self._llm = get_chatbot(self._model)
        return self._llm

    def page_text(self, page: int) -> str:
        """Return text from a page in the pdf"""
        return self.true_page_map[page]

    def ask(self, q: str, pages: list[int]) -> str:
        """Ask a question to the bot"""
        context = " ".join(self.page_text(x) for x in pages)
        return self.llm(context_aware_prompt(q, context))[0]["generated_text"]
