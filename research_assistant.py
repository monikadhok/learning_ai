import os

# For fetching data using http calls (i.e. webpage)
import requests

import asyncio
import tiktoken
import aiohttp
import multiprocessing

#Pulling data out of HTML files
from bs4 import BeautifulSoup
from concurrent.futures import ProcessPoolExecutor
from langchain.text_splitter import CharacterTextSplitter
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
#from langchain.llms import OpenAI
from langchain.schema import HumanMessage
from langchain.memory import ConversationBufferMemory
#from langchain.community.llms import OpenAI as OpenAICommunity
#from langchain.community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent, AgentType
#from langchain.community.document_loaders import WebBaseLoader
from langchain.indexes import VectorstoreIndexCreator
#from langchain.community.vectorstores import FAISS as CommunityFAISS

# Set API key
os.environ["OPENAI_API_KEY"] = ""

# Splits text in to the array of token, space being the default delimeter.
# This array is then truncated 'upto' 3000 words
# Later they are concatenated with '' space and returned
def truncate_text(text, max_tokens=3000):
    """ Truncate text to approximately max_tokens. """
    return ''.join(text.split()[:max_tokens])


def gather_information(topic):
    #f"..." indicates an f-string (formatted string literal), which allows embedding expressions inside string literals.
    url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
    response = requests.get(url)
    #This line uses the BeautifulSoup object (soup) to find all HTML paragraph elements.
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ''.join([p.get_text() for p in paragraphs])
    return truncate_text(text)

def create_knowledge_base(text):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(texts, embeddings)
    return knowledge_base

def query_knowledge_base(query, kb):
  docs = kb.similarity_search(query, k=1)
    return truncate_text(docs[0].page_content, max_tokens=1000)

def research_assistant(topic):
    info = gather_information(topic)
    kb = create_knowledge_base(info)
    return kb

def evaluate_agent(topic, kb):
    llm = OpenAI(temperature=0.7, max_tokens=256)
    test_queries = [
        f"What is {topic}?",
        f"What are the main applications of {topic}?",
        f"What are the challenges in {topic}?"
    ]
    scores = []
    for query in test_queries:
        result = query_knowledge_base(query, kb)
        evaluation_prompt = f"""
Evaluate the relevance and accuracy of the following response to the query: "{query}"
Response: {result}
Score the response from 1 to 10 (10 being perfect).
Provide only the numeric score without any additional text or explanation.
"""
        try:
            response = llm.invoke(evaluation_prompt)
            score = int(response.strip())
            if 1 <= score <= 10:
                scores.append(score)
            else:
                print(f"Invalid score received: {response}. Skipping this evaluation.")
        except ValueError as e:
            print(f"Error processing score: {e}. Skipping this evaluation.")
    if scores:
        return sum(scores) / len(scores)
    else:
        return "Unable to calculate score due to errors in evaluation."

def research_assistant_with_memory(topic):
    info = gather_information(topic)
    kb = create_knowledge_base(info)
    memory = ConversationBufferMemory(return_messages=True)
    return kb, memory

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def truncate_to_token_limit(text: str, max_tokens: int) -> str:
    """Truncates text to fit within a specified token limit."""
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])


def query_with_memory(query, kb, memory):
    llm = OpenAI(temperature=0.7, max_tokens=256)
    result = query_knowledge_base(query, kb)
    memory.chat_memory.add_user_message(query)
    memory.chat_memory.add_ai_message(result)

    # Calculate available tokens for context
    max_prompt_tokens = 4097 - 256 - 100  # Reserve 256 for completion and 100 for query and result
    query_result_tokens = num_tokens_from_string(query) + num_tokens_from_string(result)
    # The variable below seems to have a typo in the original PDF, assuming it meant to subtract query_result_tokens
    available_context_tokens = max_prompt_tokens - query_result_tokens

    # Build context starting from the most recent messages
    context = ""
    for message in reversed(memory.chat_memory.messages):
        message_text = f"{message.type}: {message.content}\n"
        message_tokens = num_tokens_from_string(message_text)
        if num_tokens_from_string(context) + message_tokens > available_context_tokens:
            break
        context = message_text + context

    response_prompt = f"""
Given the conversation history and the latest query, provide a coherent response:

Conversation history:
{context}
Latest query: {query}
Latest result: {truncate_to_token_limit(result, 500)}
Coherent response:
"""
    try:
        response = llm.invoke(truncate_to_token_limit(response_prompt, max_prompt_tokens))
        return response.strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I apologize, but I encountered an error while processing your query. Could you please try again?"

async def gather_information_multi_source(topic):
    urls = [
        f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}",
        f"https://www.britannica.com/technology/{topic.replace(' ', '-')}",
        f"https://www.sciencedaily.com/search/?keyword={topic.replace(' ', '+')}"
    ]
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        responses = await asyncio.gather(*tasks)
    all_text = ""
    for html in responses:
        soup = BeautifulSoup(html, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ''.join([p.get_text() for p in paragraphs])
        all_text += text + "\n\n"
    return truncate_text(all_text, max_tokens=4000)

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()

        def generate_summary(text):
    llm = OpenAI(temperature=0.7, max_tokens=256)
    prompt = f"""
Summarize the following text in no more than 3 paragraphs:
{truncate_text(text, max_tokens=2000)}
Summary:
"""
    return llm.invoke(prompt).strip()

def research_assistant_optimized(topic):
    # Corrected to use the async version if intended, or ensure 'gather_information' is the non-async one.
    # Assuming the non-async 'gather_information' was intended here as per previous function definitions.
    # If 'gather_information_multi_source' was intended, it needs `asyncio.run()`
    info = asyncio.run(gather_information_multi_source(topic)) # Corrected to use the async version with asyncio.run()
    kb = create_knowledge_base(info)
    summary = generate_summary(info)
    memory = ConversationBufferMemory(return_messages=True)
    return kb, memory, summary

def fine_tuned_query(query, kb, memory, summary):
    llm = OpenAI(temperature=0.7, max_tokens=256)
    result = query_knowledge_base(query, kb)
    memory.chat_memory.add_user_message(query)
    memory.chat_memory.add_ai_message(result)
    context = "\n".join([f"{m.type}: {m.content}" for m in memory.chat_memory.messages[-3:]])
    response_prompt = f"""
Given the topic summary, conversation history, and the latest query, provide a coherent and informed response.
Topic summary:
{truncate_text(summary, max_tokens=500)}
Conversation history:
{context}
Latest query: {query}
Latest result: {truncate_text(result, max_tokens=300)}
Coherent and informed response:
"""
    return llm.invoke(response_prompt).strip()

if __name__ == "__main__":
    topic = "Artificial Intelligence"
    kb, memory, summary = research_assistant_optimized(topic)
    print(f"Topic Summary:\n{summary}\n")
    queries = [
        "What is artificial intelligence?",
        "What are its main applications?",
        "What are some challenges in AI?"
    ]
    for query in queries:
        response = fine_tuned_query(query, kb, memory, summary)
        print(f"Query: {query}")
        print(f"Response: {response}\n")


    
