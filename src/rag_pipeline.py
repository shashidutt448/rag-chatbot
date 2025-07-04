from transformers import AutoTokenizer, AutoModelForCausalLM
from src.embedder import Embedder
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class RAGPipeline:
    def __init__(self, chunks, model_id):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            local_files_only=True,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            local_files_only=True,
            trust_remote_code=True
        )
        self.chunks = chunks

class RAGPipeline:
    def __init__(self, chunks, model_id="./models/zephyr-7b"):
        self.chunks = chunks
        self.embedder = Embedder()
        self.embeddings = self.embedder.embed(chunks)
        self.index = self.embedder.build_index(self.embeddings)

        # Tell HF this is a local model
        AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True, torch_dtype=torch.float32)
        self.model.eval()



    def retrieve(self, query, top_k=3):
        query_emb = self.embedder.embed([query])
        D, I = self.index.search(query_emb, top_k)
        return [self.chunks[i] for i in I[0]]

    def query(self, user_input):
        retrieved_chunks = self.retrieve(user_input)
        context = "\n".join(retrieved_chunks)
        prompt = f"""You are a legal assistant. Use the context below to answer the question.

Context:
{context}

Question:
{user_input}

Answer:"""
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        output = self.model.generate(input_ids, max_new_tokens=300)
        answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
        final_answer = answer.split("Answer:")[-1].strip()
        return final_answer, retrieved_chunks
