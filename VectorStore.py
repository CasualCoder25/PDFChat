import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class VectorStore:
    def __init__(self, embedding_model = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model_tokenizer = self.get_embedding_model_tokenizer(embedding_model)
        self.embedding_model = self.get_embedding_model(embedding_model)
        self.index_to_text = {}
        self.index_to_vector = {}
        self.text_to_index = {}
        self.index = 0

    def add_vector(self, text):
        if text not in self.text_to_index.keys():
            current_index = self.index
            self.index += 1
            self.index_to_text[current_index] = text
            self.text_to_index[text] = current_index
            self.index_to_vector[current_index] = self.get_embeddings(text)

    def add_vectors(self, texts = [], chunk_size = 128):
        temp = ""
        for text in texts:
            if len(temp.split(" ")) < chunk_size:
                temp = temp + text
            else:
                self.add_vector(temp)
                temp = ""
        if not temp=="":
            self.add_vector(temp)

    def get_text(self, index):
        return self.index_to_text[index]

    def get_vector(self, index):
        return self.index_to_vector[index]
    
    def get_embedding_model_tokenizer(self, embedding_model):
        return AutoTokenizer.from_pretrained(embedding_model, cache_dir="./model/")
    
    def get_embedding_model(self, embedding_model):
        return AutoModel.from_pretrained(embedding_model, cache_dir="./model/")
    
    def get_embeddings(self, text):
        tokenized_text = self.embedding_model_tokenizer(text.lower(), padding = True, truncation = True, return_tensors = 'pt')
        embeddings = self.embedding_model(**tokenized_text)
        sentence_embeddings = self.mean_pooling(embeddings, tokenized_text["attention_mask"])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings[0]
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def find_similar_vectors(self, query_text, num_results=5):
        query_vector = self.get_embeddings(query_text)
        result = []
        for index, vector in self.index_to_vector.items():
            similarity = np.dot(query_vector.detach().numpy(), vector.detach().numpy()) / (np.linalg.norm(query_vector.detach().numpy()) * np.linalg.norm(vector.detach().numpy()))
            result.append((index, similarity))
        result.sort(key=lambda x: x[1], reverse=True)
        return [self.index_to_text[i] for (i,_) in result[:num_results]]