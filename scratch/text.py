# System prompt describes information given to all conversations
system_prompt = """
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant for labeling topics.
<</SYS>>
"""

# Example prompt demonstrating the output we are looking for
# things work better with an example
example_prompt = """
I have a topic that contains the following documents:
- Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.
- Meat, but especially beef, is the word food in terms of emissions.
- Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.

The topic is described by the following keywords: 'meat, beef, eat, eating, emissions, steak, food, health, processed, chicken'.

Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.
[/INST] Environmental impacts of eating meat
"""

# Our main prompt with documents ([DOCUMENTS]) and keywords ([KEYWORDS]) tags
main_prompt = """
[INST]
I have a topic that contains the following documents:
[DOCUMENTS]

The topic is described by the following keywords: '[KEYWORDS]'.

Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.
[/INST]
"""

# combine all the prompts together
prompt = system_prompt + example_prompt + main_prompt

# now add in whatever the users types, and the chat should respond as previously instructed
user_prompt = "[INST]I liked “Breaking Bad” and “Band of Brothers”. My TV show review gives it 4 out of 5 stars. I just love how Taylor Swift cameos in every episode[/INST]"

print(chat(prompt + user_prompt)["generated_text"])


# System prompt describes information given to all conversations
system_prompt = """
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant for answering questions. Use only the context to answer questions.
<</SYS>>
"""

# Example prompt demonstrating the output we are looking for
# things work better with an example, but you could skip this
half_orc_context = dnd[31]["text"]
example_prompt = f"""
Question:
What is a half-orc?
Context:
{half_orc_context}

Based on the information in the context above, please answer the question.
[/INST]
A half-orc are tough and violent warriors who are a cross between humans and orcs.
They still remember their savage ways, but can tolerate being in modern civilization.
"""

# Our main prompt with documents ([DOCUMENTS]) and keywords ([KEYWORDS]) tags
main_prompt = """
[INST]
Question:
[QUESTION]
Context:
[Context]

Based on the information in the context above, please answer the question.
[/INST]
"""

# combine all the prompts together
prompt = system_prompt  # + example_prompt # + main_prompt


dnd = get_dnd_data()


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


def page_text(page: int):
    print(next(x["text"] for x in dnd if x["page"] == page - 1))


def dnd_question_prompt(q: str, pages: list[int]):
    pages = {x - 1 for x in pages}
    text = " ".join(x["text"] for x in dnd if x["page"] in pages)
    return context_aware_prompt(q, text)


# chat = get_llm()
def dnd_question(q, pages: list[int]):
    print(chat(dnd_question_prompt(q, pages))[0]["generated_text"])
