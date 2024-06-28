
import json
import random
from typing import Dict, Generator, List, Tuple
import uuid
from llama_index.core import VectorStoreIndex
from pathlib import Path
from typing import List
from llama_index_client import ChatMessage, Document
from llama_index.core import SimpleDirectoryReader

from advanced_chatbot.config import  (DATA_PATH, DEFAULT_RAG_CHUNK_OVERLAP, DEFAULT_RAG_CHUNK_SIZE,
                                     DEFAULT_RAG_SIMILARITY_TOP_K, DEFAULT_RAG_TOKEN_LIMIT, DEFAULT_RAG_WINDOW_SIZE, 
                                     OPENAI_API_KEY, USE_MOCK_MODELS)

from llama_index.core.llms import MockLLM
from llama_index.core import MockEmbedding

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import (
    SentenceSplitter,
    SentenceWindowNodeParser,
)
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory.chat_memory_buffer import (
    ChatMemoryBuffer,
)
from llama_index.core.schema import NodeWithScore
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.chat_engine.types import (
    StreamingAgentChatResponse,
)
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate
import shutil


DEFAULT_SYSTEM_PROMPT= """
You are a helpful assistant, here to provide answers to user queries.
You should always be polite and provide accurate information.
Don't provide false information or try to mislead the user.
"""


TRANSLATION_SYSTEM_PROMPT = """
You are translator tool, given a source text, you should translate into french.
Only respond with the translation of the text and nothing else.
No greetings or additional information.
"""

TRANSATION_USER_PROMPT = """
#Source text:
{source_text}
"""


SUMMARIZATION_SYSTEM_PROMPT = """
You will be given some text or some counter in french, you should summarize it in
very few words in french. 
DIrecty provide the summary and nothing else.(Don't writing résumé:)

"""

SUMMARIZATION_USER_PROMPT = """
#Source text:
 {source_text}
"""


LANGUAGE_DETECTION_SYSTEM_PROMPT = """
You are a language detection tool. Given a text, you should detect the language of the text.
If french, respon with 'fr', if english respond with 'en'. etc. 
Even though , a text comprises multiple languages, you should respond with the dominant language.
In the short version of the language.
Don't always with nothing but that short version of the language.
"""

RAG_STORAGE_PATH = DATA_PATH / "rag_storage"


class _RagService:
    """
    Service implement retrieval augmented generatoin primitives.
    """
    
    def __init__(self):
        self.__init_llm_and_embedding()
        RAG_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
        
        
        
    def __init_llm_and_embedding(self)-> None:
        """
        Initialize the language model and the embedding.
        """
        
        if USE_MOCK_MODELS:
            self._llm = MockLLM(max_tokens=256)
            self._embedding = MockEmbedding(embed_dim=1536)
        else:
            self._llm = OpenAI(api_key=OPENAI_API_KEY,model="gpt-3.5-turbo")
            self._embedding = OpenAIEmbedding(api_key=OPENAI_API_KEY, model="text-embedding-3-small")
        

        
    def parse_document(self, document_path: Path)-> List[Document]:
        """
        Read a document and return a list of pages.
        :param document_path: Path to the document( Absolute path of a pdf or docx file.)
        ex: document_path = "/home/user/toto.pdf"
        :return: List[Document] : A list of Document objects.
        
        """
        file_extension = document_path.suffix
        
        if file_extension not in [".pdf", ".docx"]:
            raise ValueError("The document must be a pdf or a docx file.")
        
        reader = SimpleDirectoryReader(input_files = [document_path])
        return reader.load_data()
    
    
    
    def __get_index_persist_dir(self, index_id: str)-> Path:
        """
        Get the path to the directory where the index will be persisted.
        :param index_id: The id of the index.
        :return: The path to the directory where the index will be persisted.
        """
        return RAG_STORAGE_PATH / index_id
    
    
    
    def create_vector_store_index(self,  document_path: str, persist=True)->Tuple[str, VectorStoreIndex]:
        
        """
        Create a vector store index using a given document.
        :param document_path: Path to the document( Absolute path of a pdf or docx file.)
        ex: document_path = "/home/user/toto.pdf"
        :param persist: Whether to persist the index or not.
        :return : Tuple[Dict, VectorStoreIndex] : A tuple containing the index config and the index object.
        
        """
        #1.Read the document
        sentence_splitter = SentenceSplitter.from_defaults(
            chunk_size=DEFAULT_RAG_CHUNK_SIZE,
            chunk_overlap=DEFAULT_RAG_CHUNK_OVERLAP
        )
        text_plitter_fn = lambda x: sentence_splitter.split_text(x)
        parser = SentenceWindowNodeParser.from_defaults(
            sentence_splitter=text_plitter_fn,
            window_size=DEFAULT_RAG_WINDOW_SIZE
        )
        
        #2. Parse nodes.
        nodes = parser.get_nodes_from_documents(self.parse_document(document_path))
        
        #3. Generate index UUID
        index_id = str(uuid.uuid4()).split("-")[0]  
        
        if persist: 
            persist_dir = self.__get_index_persist_dir(index_id)
            persist_dir.mkdir(parents=True, exist_ok=True)
            storage_context = StorageContext.from_defaults()
        else:
            storage_context = None
        
        
        index_config = {
            "index_id": index_id,
            "document_path": str(document_path),
        }
        
        #4. Create the index
        index = VectorStoreIndex(
            nodes=
            nodes,
            storage_context=storage_context,
            embed_model=self._embedding,
            show_progress=True,
        )
        
        if persist:
            storage_context.persist(persist_dir=persist_dir)
            #Save the index config in the persist directory
            with open(self.__get_index_persist_dir(index_id) / "index_config.json", "w") as f:
                json.dump(index_config, f)
        
        return index_id, index
        
  

    def delete_vector_store_index(self, index_id: str):
        """
        Delete a vector store index from a given path.
        :param document_name: The name of the document to delete.
        """

        if not self.__get_index_persist_dir(index_id).exists():
            raise ValueError(f"Index with id {index_id} does not exist.")
    
        #Delete the index directory
        shutil.rmtree(self.__get_index_persist_dir(index_id), ignore_errors=True)
        
    
    def update_index_config(self, index_id:str, new_config:Dict)-> None:
        """
        Update the index config with new values.
        :param index_id: The id of the index to update.
        :param new_config: The new config to update.
        """
        
        index_dir = self.__get_index_persist_dir(index_id)
        
        if not index_dir.exists():
            raise ValueError(f"Index with id {index_id} does not exist.")
        
        with open(index_dir / "index_config.json", "w") as f:
            json.dump(new_config, f)
        
    
    def load_index_config(self, index_id:str)-> Dict:
        """
        Load the index config from the index id.
        :param index_id: The id of the index to load.
        """
        
        index_dir = self.__get_index_persist_dir(index_id)
        
        if not index_dir.exists():
            raise ValueError(f"Index with id {index_id} does not exist.")
        
        with open(index_dir / "index_config.json", "r") as f:
            config = json.load(f)
            config["document_path"] = Path(config["document_path"])
            return config

    
    def load_vector_store_index(self, index_id: str)-> VectorStoreIndex:
        """
        Load a vector store index from a given path.
        :param index_id: The id of the index to load.
        :return: Tuple[Dict, VectorStoreIndex] : A tuple containing the index config and the index object.
        """
        storage_context = StorageContext.from_defaults(persist_dir=self.__get_index_persist_dir(index_id))
        return load_index_from_storage(storage_context=storage_context,
                                       embed_model=self._embedding)
                                       
    
    
    
    def complete_chat(self, 
                      query: str,
                      conversation_history: List[ChatMessage],
                      index_ids : List[str],
                      system_prompt:str = DEFAULT_SYSTEM_PROMPT,    
                      )-> Tuple[Generator[str,None,None], List[NodeWithScore]]:
        """
        Generate a response to a given question.
        :param query: The user query to generate a response to.
        
        :param conversation_history: The conversation history.
        Conversation history is a list of ChatMessage objects.
        ChatMessage(Role="user|assistant|system", content="The content of the message")
        
        :param document_list: A list of document paths to search for the answer.
        The index of document index to search for the answer.
        
        """
        
        #1. Load the indexes
        index_list  = [self.load_vector_store_index(index_id) for index_id in index_ids]
        
        
        #2.Retriever 
        retriever = QueryFusionRetriever(
            retrievers=[index.as_retriever(similarity_top_k=DEFAULT_RAG_SIMILARITY_TOP_K) for index in index_list],
            num_queries=1,
            similarity_top_k=DEFAULT_RAG_SIMILARITY_TOP_K,
        )
        
        
        
        memory = ChatMemoryBuffer(chat_history=conversation_history,
                                  token_limit=DEFAULT_RAG_TOKEN_LIMIT)
        
        
        
        chat_engine = ContextChatEngine.from_defaults(
            retriever=retriever,
            memory=memory,
            llm=self._llm,
            node_postprocessors = [
                MetadataReplacementPostProcessor(target_metadata_key="window")
            ],
            system_prompt=system_prompt)
        
        response :StreamingAgentChatResponse = chat_engine.stream_chat(query)
        
        response_generator = response.response_gen
        source_nodes = response.source_nodes
        
        return response_generator, source_nodes

        
        
        
        
        

    def list_vector_store_index(self)-> List[dict]:
        """
        List all the vector store indexes
        :return A list of vector _index configs
        """
        
        index_configs = []
        for index_dir in RAG_STORAGE_PATH.iterdir():
            with open(index_dir / "index_config.json", "r") as f:
                index_config = json.load(f)
                index_configs.append(index_config)
        return index_configs
        
    
 
    
    
    
    def translate_and_summarize_first_page_fr(self, index_id:str)-> str:
        """
        Translate a document to french.
        :param document_content: The content of the document to translate.
        :return: The translated document.
        """
        #1. Load the file path of the document
        index_dir = self.__get_index_persist_dir(index_id)
        with open(index_dir / "index_config.json", "r") as f:
            index_config = json.load(f)
            document_path = index_config["document_path"]
        
        
        #2. Load the document
        document_list = self.parse_document(document_path)
        #First page
        first_page_document = [k for k in document_list if k.page_label == '1']
        first_page_content = first_page_document[0].text
        
        #3. Translate the document to french
        chat_messages=  ChatPromptTemplate(
            message_templates=[ChatMessage(role=MessageRole.SYSTEM,content=TRANSLATION_SYSTEM_PROMPT),
                ChatMessage(role=MessageRole.USER,content=first_page_content)]
        )
        
        first_page_fr = self._llm.predict(
            prompt=chat_messages,
            source_text= first_page_content
        )
        return self.summarize_content(first_page_fr)
        
    


    
    def summarize_content(self, input_content:str)-> str:
        """
        Summarize a given :input_content into few words.
        :param prompt: The prompt to translate.
        :return: The translated prompt.
        """
        
        
        chat_messages = ChatPromptTemplate(
            message_templates=
            [ChatMessage(role=MessageRole.SYSTEM, content=SUMMARIZATION_SYSTEM_PROMPT),
             ChatMessage(role=MessageRole.USER, content=SUMMARIZATION_USER_PROMPT)]
        )
        
        return self._llm.predict(prompt=chat_messages, source_text=input_content)
        
        
    
        
    def summarize_document_index(self, index_id)->str:
        """
        Summarize the content of a vector store index. 
        :param index_id: The id of the index to summarize.
        """
        
        #1. First load the index
        index = self.load_vector_store_index(index_id)
        
        #2. Retriever (Dummy to get sources nodes)
        query = "__"
        nodes = index.as_retriever(similarity_top_k=50).retrieve(query)
        
        #3. Randomly select 10 sources
        source_nodes = random.sample(nodes, 20)
        
        #4. Content to use for summarization
        content = "\n".join([node.get_text() for node in source_nodes])
        
        return self.summarize_content(content)
    
    
    

    def detect_document_language(self, index_id:str)-> str:
        """
        :param index_id: The id of the index to detect the language of.
        :return: The language of the document.
        """
        
        #Load the index
        index = self.load_vector_store_index(index_id)
        
        #Retriever (Dummy to get sources nodes)
        query = "__"
        nodes = index.as_retriever(similarity_top_k=50).retrieve(query)
        
        
        #Randomly select 10 sources
        source_nodes = random.sample(nodes, 5)
        
        #Content to use for language detection
        content = "\n".join([node.get_text() for node in source_nodes])
        
        chat_messages = ChatPromptTemplate(
            [ChatMessage(role=MessageRole.SYSTEM, content=LANGUAGE_DETECTION_SYSTEM_PROMPT),
             ChatMessage(role=MessageRole.USER, content=content)]
        )
        
        language = self._llm.predict(prompt=chat_messages, source_text=content)
        return language
        
        
    
RagService = _RagService() #Singleton instance of the RagService class
    