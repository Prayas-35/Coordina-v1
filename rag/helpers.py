from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import TiDBVectorStore
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from pydantic import BaseModel
from typing import Any, List, Optional
from groq import Groq
import dotenv
import os

dotenv.load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL = "llama-3.1-70b-versatile"
TEMP = 0.3


class CustomConfig(BaseModel):
    api_url: str
    api_key: str


class CustomAPILLM(LLM):
    api_key: str = None
    api_url: str = None

    def __init__(self, config: CustomConfig, callbacks: Optional[List] = None):
        super().__init__()
        self.api_url = config.api_url
        self.api_key = config.api_key
        self.callbacks = callbacks or []

    @property
    def _llm_type(self) -> str:
        return "custom_api"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        client = Groq(api_key=GROQ_API_KEY)
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=MODEL,
            temperature=TEMP,
        )
        return chat_completion.choices[0].message.content


loader = TextLoader(file_path="data.txt", encoding="utf-8")
docs = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
documents = text_splitter.split_documents(docs)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY,
)

capath = "/etc/ssl/certs/ca-certificates.crt" if os.path.exists("/etc/ssl/certs/ca-certificates.crt") else "isrgrootx1.pem"
vector_store = TiDBVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
    table_name="CodingGuidance",
    connection_string=f"mysql+mysqldb://2DXyH3NQNPFiCYW.root:UHsVdjbJrSqD2xpo@gateway01.ap-southeast-1.prod.aws.tidbcloud.com:4000/test?ssl_ca={capath}",
    distance_strategy="cosine",
    drop_existing_table=True,
)

retriever = vector_store.as_retriever(score_threshold=0.9)  # Increased threshold for better filtering

config = CustomConfig(api_url="", api_key=GROQ_API_KEY)
custom_llm = CustomAPILLM(config=config)

async def generate(quest, conversation_history):
    rephrasing_prompt = f"""
        Your name is Cody, an expert in natural language processing and domain-specific analysis for web development, blockchain, cybersecurity, and machine learning.

        Your task:
        1. Rephrase the question concisely for querying a vector database, ensuring relevance to web development, blockchain, cybersecurity, or machine learning.
        2. If the input asks for personal details (e.g., "What is your name?"), respond with: 
        "My name is Cody, your coding mentor specializing in web development, blockchain, cybersecurity, and machine learning."
        3. If the input is not a question (e.g., "Got it, thanks" or "Okay, Got it"), respond with an appropriate acknowledgment (e.g., "You're welcome!" or "Happy to help!").
        4. If the question is ambiguous or lacks sufficient detail:
        - Include a clarification note requesting more information.
        - If a potential connection to the domains can be assumed based on the conversation history, ask for confirmation using this format:
            "Are you asking about the intersection of [assumed topic] and [domain]? If yes, I can rephrase accordingly."
        5. If the input is entirely unrelated to the specified domains and does not fall into the above categories, respond with:
            "The question is outside the specified domains of expertise."
        6. if it seems that the conversation has concluded, provide a positive and concise closing statement (e.g., "Bye, See you soon!").

        ### Input:
        QUESTION: {quest}
        CONVERSATION HISTORY: {conversation_history}
    """

    client = Groq(api_key=GROQ_API_KEY)
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": rephrasing_prompt}],
        model=MODEL,
        temperature=TEMP,
    )
    refined_question = chat_completion.choices[0].message.content

    prompt_template = f"""
    You are Cody, an expert coding mentor specializing in web development, machine learning, blockchain, and cybersecurity. 
        Provide accurate, detailed, and helpful responses based strictly on the given context and conversation history. 

        ### Guidelines:
        1. **Strict Relevance**: Only answer questions directly related to the provided context or conversation history. If a question is irrelevant to the context, clearly state: 
        - "The question is outside the scope of the provided context, so I cannot answer it."
        2. **Insufficient Information**: If the context or history does not provide enough details, state explicitly: 
        - "I don't have enough information to provide a meaningful answer."
        3. **Code Examples**: Where applicable, include concise and relevant code snippets that directly address the question.
        4. **Clarity and Simplicity**: Break down complex concepts into simple, clear explanations for better understanding.
        5. **Best Practices**: Highlight recommended approaches and warn against common mistakes or potential pitfalls.
        6. **Closing Statement**: If the conversation appears to have concluded, provide a positive and concise closing statement.
        7. **No Speculation**: Avoid speculative or generic answers. Stick strictly to the context provided.

        ### Input Structure:
        - **Previous Conversation**: 
        {conversation_history}
        - **Context**: 
        {{context}}
        - **Question**: 
        {{question}}
    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": PROMPT}

    chain = RetrievalQA.from_chain_type(
        llm=custom_llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )

    response = chain({"query": refined_question, "question": quest})
    print(response)
    return response.get("result", "I don't know.")
