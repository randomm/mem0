import hashlib
import importlib
import json

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.configs.llms.base import BaseLlmConfig


def load_class(class_type):
    module_path, class_name = class_type.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class LlmFactory:
    provider_to_class = {
        "ollama": "mem0.llms.ollama.OllamaLLM",
        "openai": "mem0.llms.openai.OpenAILLM",
        "groq": "mem0.llms.groq.GroqLLM",
        "together": "mem0.llms.together.TogetherLLM",
        "aws_bedrock": "mem0.llms.aws_bedrock.AWSBedrockLLM",
        "litellm": "mem0.llms.litellm.LiteLLM",
        "azure_openai": "mem0.llms.azure_openai.AzureOpenAILLM",
        "openai_structured": "mem0.llms.openai_structured.OpenAIStructuredLLM",
        "anthropic": "mem0.llms.anthropic.AnthropicLLM",
        "azure_openai_structured": "mem0.llms.azure_openai_structured.AzureOpenAIStructuredLLM",
        "gemini": "mem0.llms.gemini.GeminiLLM",
        "deepseek": "mem0.llms.deepseek.DeepSeekLLM",
        "xai": "mem0.llms.xai.XAILLM",
        "lmstudio": "mem0.llms.lmstudio.LMStudioLLM",
        "jina": "mem0.llms.jina.JinaLLM",
    }

    @classmethod
    def create(cls, provider_name, config):
        class_type = cls.provider_to_class.get(provider_name)
        if class_type:
            llm_instance = load_class(class_type)
            base_config = BaseLlmConfig(**config)
            return llm_instance(base_config)
        else:
            raise ValueError(f"Unsupported Llm provider: {provider_name}")


class EmbedderFactory:
    provider_to_class = {
        "openai": "mem0.embeddings.openai.OpenAIEmbedding",
        "ollama": "mem0.embeddings.ollama.OllamaEmbedding",
        "huggingface": "mem0.embeddings.huggingface.HuggingFaceEmbedding",
        "azure_openai": "mem0.embeddings.azure_openai.AzureOpenAIEmbedding",
        "gemini": "mem0.embeddings.gemini.GoogleGenAIEmbedding",
        "vertexai": "mem0.embeddings.vertexai.VertexAIEmbedding",
        "together": "mem0.embeddings.together.TogetherEmbedding",
        "lmstudio": "mem0.embeddings.lmstudio.LMStudioEmbedding",
        "model2vec": "mem0.embeddings.model2vec.Model2VecEmbedder",
    }
    _instances = {}

    @classmethod
    def create(cls, provider_name, config):
        config_str = json.dumps(config, sort_keys=True)
        cache_key = hashlib.md5(f"{provider_name}:{config_str}".encode()).hexdigest()

        if cache_key in cls._instances:
            return cls._instances[cache_key]

        class_type = cls.provider_to_class.get(provider_name)
        if class_type:
            embedder_class = load_class(class_type)
            base_config = BaseEmbedderConfig(**config)
            instance = embedder_class(base_config)
            cls._instances[cache_key] = instance
            return instance
        else:
            raise ValueError(f"Unsupported Embedder provider: {provider_name}")


class VectorStoreFactory:
    provider_to_class = {
        "qdrant": "mem0.vector_stores.qdrant.Qdrant",
        "chroma": "mem0.vector_stores.chroma.ChromaDB",
        "pgvector": "mem0.vector_stores.pgvector.PGVector",
        "milvus": "mem0.vector_stores.milvus.MilvusDB",
        "azure_ai_search": "mem0.vector_stores.azure_ai_search.AzureAISearch",
        "pinecone": "mem0.vector_stores.pinecone.PineconeDB",
        "redis": "mem0.vector_stores.redis.RedisDB",
        "elasticsearch": "mem0.vector_stores.elasticsearch.ElasticsearchDB",
        "vertex_ai_vector_search": "mem0.vector_stores.vertex_ai_vector_search.GoogleMatchingEngine",
        "opensearch": "mem0.vector_stores.opensearch.OpenSearchDB",
        "supabase": "mem0.vector_stores.supabase.Supabase",
        "weaviate": "mem0.vector_stores.weaviate.Weaviate",
        "faiss": "mem0.vector_stores.faiss.FAISS",
    }

    @classmethod
    def create(cls, provider_name, config):
        class_type = cls.provider_to_class.get(provider_name)
        if class_type:
            if not isinstance(config, dict):
                config = config.model_dump()
            vector_store_instance = load_class(class_type)
            return vector_store_instance(**config)
        else:
            raise ValueError(f"Unsupported VectorStore provider: {provider_name}")
