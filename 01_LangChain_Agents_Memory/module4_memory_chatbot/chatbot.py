"""
Module 4 — Conversational Chatbot with Multi-Turn Memory
Use Case: Insurance policy advisor with session management
"""
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv; load_dotenv()

SYSTEM_PROMPT = """You are Priya, a friendly insurance advisor at SecureLife Insurance.
You help customers understand their policies, file claims, and make smart coverage decisions.
Be empathetic, clear, and always ask clarifying questions when needed."""

def build_buffer_chatbot():
    """Simple buffer memory — keeps all messages verbatim."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    memory = ConversationBufferMemory(return_messages=True)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    return ConversationChain(llm=llm, memory=memory, prompt=prompt, verbose=True)

def build_summary_chatbot():
    """Summary buffer memory — summarises old messages to stay token-efficient."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=500, return_messages=True)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    return ConversationChain(llm=llm, memory=memory, prompt=prompt, verbose=False)

def run_demo():
    print("=== Insurance Chatbot Demo (Buffer Memory) ===\n")
    chatbot = build_buffer_chatbot()
    conversation = [
        "Hi, I want to know about health insurance for my family of 4.",
        "What is the typical premium for a 10 lakh family floater plan?",
        "What about pre-existing diseases? My father has diabetes.",
        "Can you summarize what we discussed so far?",
    ]
    for message in conversation:
        print(f"Customer: {message}")
        response = chatbot.predict(input=message)
        print(f"Priya: {response}\n")

if __name__ == "__main__":
    run_demo()
