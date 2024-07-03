import pymupdf
import os
from typing import List
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration



def split_text_to_chunck(text: List[str], tokenizer, max_length) -> List[str]:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    chunked_texts = [tokenizer.decode(chunk) for chunk in chunks]
    return chunked_texts


def get_chunks(texts: List[str], tokenizer, max_length) -> List[str]:
    res = []
    for text in texts:
        chunks = split_text_to_chunck(text, tokenizer, max_length)
        for chunk in chunks:
            res.append(chunk)
    return res


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def generate_embedding(chunked_texts: List[str], model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    encoded_input = tokenizer(chunked_texts, padding=True, truncation=True, return_tensors='pt')
    encoded_input = {key: value.to(device) for key, value in encoded_input.items()}

    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    return sentence_embeddings.cpu().numpy()


def load_data_to_db(texts: List[str], model_name, device, client, max_length=256):
    print("Loading data into vector database...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    chunked_texts = get_chunks(texts, tokenizer, max_length)
    embeddings = generate_embedding(chunked_texts, model_name, device)
    dim = embeddings.shape[1]

    if client.has_collection(collection_name="demo_collection"):
        client.drop_collection(collection_name="demo_collection")

    client.create_collection(
        collection_name="demo_collection",
        dimension=dim,
    )
    data = [
        {"id": i, "vector": embeddings[i], "text": chunked_texts[i], "subject": "demo"}
        for i in range(len(chunked_texts))
    ]
    client.insert(collection_name="demo_collection", data=data)


def extract_info(query: str, model_name: str, client, limit=2):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(query)
    search_res = client.search(
        collection_name="demo_collection",
        data=[embeddings.tolist()],
        limit=limit,
        output_fields=["text"],
    )
    search_res = search_res[0]
    text_res = ""
    for item in search_res:
        text_res += item['entity']['text']

    return text_res


def get_llm_response(query, retrieved_info, device):
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base", legacy=False)

    prompt = f"{query}\n" \
             f"Answer the question using the following information: {retrieved_info}\n"
    if device == 'cuda':
        model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base", device_map="auto")
    else:
        model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
    attention_mask = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).attention_mask
    outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=100,
                             pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

if __name__ == '__main__':
    path = 'data'
    texts = []
    for filename in os.listdir(path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(path, filename)
            doc = pymupdf.open(pdf_path)
            s = ""
            for page in doc:
                s += page.get_text()
            texts.append(s)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    client = MilvusClient("milvus_demo.db")
    load_data_to_db(texts, model_name, device, client, max_length=256)

    query = "how to do rag research better"
    res = extract_info(query, model_name, client, limit=2)
    ans = get_llm_response(query, res, device)
    print(ans)
