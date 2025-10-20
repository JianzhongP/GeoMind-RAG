# 阿秋
# 2025/10/05 17:11
import os

import httpx
from langchain.embeddings.base import Embeddings
from dotenv import load_dotenv
from openai import OpenAI

# 应继承 langchain.embedding.base 的 Embedding类，然后重写类的方法（embed_query）
class QWENEmbedding(Embeddings):
    def __init__(self,
                 model: str = None,
                 api_key: str = None,
                 base_url: str = None,
                 dimensions=1024,
                 encoding_format: str = "float"):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

        # 初始化参数
        self.dimensions = dimensions
        self.encoding_format = encoding_format
        # 初始化 OpenAI 客户端
        # 创建自定义 HTTP 客户端，避免 proxies 参数问题
        http_client = httpx.Client(
            timeout=60.0,
            follow_redirects=True
        )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=http_client,
        )

    def embed_documents(self, texts: list[str])  -> list[list[float]]:
        """文档向量化"""
        embeddings = []

        for text in texts:
            try:
                emb = self.client.embeddings.create(
                    model=self.model,
                    input=text,
                    dimensions=self.dimensions,
                    encoding_format=self.encoding_format,
                )
                embeddings.append(emb.data[0].embedding)

            except Exception as e:
                print(f"向量化失败！{e} 返回0向量")
                embeddings.append([0.0] * self.dimensions)

        return embeddings

    def embed_query(self, query: str):
        try:
            embeddings = self.client.embeddings.create(
                model=self.model,
                input=query,
                dimensions=self.dimensions,
                encoding_format=self.encoding_format,
            )
            return embeddings.data[0].embedding
        except Exception as e:
            print(f"向量化失败！{e}")
            return [0.0] * self.dimensions

if __name__ == "__main__":
    load_dotenv(f"D:\Pycharm_project\my_multimodal_RAG\\backend\.env", override=True)
    model = os.getenv("QWEN_EMBEDDING_MODEL_NAME")
    API_KEY = os.getenv("QWEN_API_KEY")
    # print(API_KEY)
    BASE_URL = os.getenv("QWEN_BASE_URL")

    QWEN = QWENEmbedding(model=model, api_key=API_KEY, base_url=BASE_URL)
    # 测试单个文本
    text = "用于用户指定输出向量维度，只适用于text-embedding-v3与text-embedding-v4模型。"
    vector = QWEN.embed_query(text)
    print(f"\n文本: {text}")
    print(f"向量维度: {len(vector)}")
    print(f"向量前10维: {vector[:10]}")

    # 测试批量文本
    texts = [
        "这是第一段文本",
        "这是第二段文本",
        "这是第三段文本"
    ]
    vectors = QWEN.embed_documents(texts)
    print(f"\n批量向量化:")
    print(f"  文本数量: {len(texts)}")
    print(f"  向量数量: {len(vectors)}")
    print(f"  每个向量维度: {len(vectors[0])}")


