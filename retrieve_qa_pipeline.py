import pymupdf
import os
from typing import List
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import numpy as np


def split_text_to_chunck(text: List[str], tokenizer, max_length: int) -> List[str]:
    '''
    Splits text into chunks based on the maximum token length.

    :param text: The text to be split.
    :param tokenizer: The tokenizer to be used for encoding and decoding.
    :param max_length: The maximum length of tokens in each chunk.

    :return: List[str], A list of text chunks.
    '''

    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    chunked_texts = [tokenizer.decode(chunk) for chunk in chunks]
    return chunked_texts


def get_chunks(texts: List[str], tokenizer, max_length: int) -> List[str]:
    '''
    Given a list of texts, split each text into chunks based on the maximum token length.

    :param texts: a list of text
    :param tokenizer: The tokenizer to be used for encoding and decoding.
    :param max_length: The maximum length of tokens in each chunk.

    :return: A list of text chunks.
    '''

    res = []
    for text in texts:
        chunks = split_text_to_chunck(text, tokenizer, max_length)
        for chunk in chunks:
            res.append(chunk)
    return res


def mean_pooling(model_output, attention_mask):
    '''
    helper function for embedding. Applies mean pooling to the model output using the attention mask.
    '''

    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def generate_embedding(chunked_texts: List[str], model_name: str, device: torch.device) -> np.ndarray:
    '''
    Generates embeddings for the given chunked texts.

    :param chunked_texts: The chunked texts to be embedded.
    :param model_name: The name of the model to be used.
    :param device: The device to be used for computation.

    :return: np.ndarray: The generated embeddings.
    '''

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    encoded_input = tokenizer(chunked_texts, padding=True, truncation=True, return_tensors='pt')
    encoded_input = {key: value.to(device) for key, value in encoded_input.items()}

    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    return sentence_embeddings.cpu().numpy()


def load_data_to_db(texts: List[str], model_name: str, device: torch.device, client: MilvusClient, max_length=256) -> None:
    '''
    1. split texts into chunks
    2. generate embeddings for each chunk
    3. Load text and embeddings into the vector database.

    :param texts: A list of texts read from pdf
    :param model_name: The name of the model to be used for embedding generation.
    :param device: The device to be used for computation. (cpu/gpu)
    :param client: MilvusClient, the Milvus client for database operations.
    :param max_length: The maximum length of tokens in each chunk. Default is 256.
    '''

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


def extract_info(query: str, model_name: str, client: MilvusClient, limit=2) -> str:
    '''
    Extracts information relevant to the query from the vector database.

    :param query: user input
    :param model_name: The name of the model to be used for query encoding.
    :param client: The Milvus client for database operations.
    :param limit: The maximum number of results to return. Default is 2.

    :return: The extracted information as a concatenated string.
    '''

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


def get_llm_response(query: str, retrieved_info: str, device: torch.device) -> str:
    '''
    Generates a response from the LLM(google/flan-t5-base) based on the query and retrieved information.

    :param query: The user input.
    :param retrieved_info: The relevant information.
    :param device: The device to be used for computation (cpu/gpu).

    :return: The response of LLM
    '''

    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base", legacy=False)

    prompt = f"{query}\n" \
             f"Answer the question using the following information: {retrieved_info}\n"

    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
    model.to(device)

    # max input token limit of google-t5: 512
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
    attention_mask = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).attention_mask.to(device)
    outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=100, min_length=50, num_beams=2,
                             length_penalty=1.1, repetition_penalty=1.5, pad_token_id=tokenizer.eos_token_id)
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
    # embedding model, 384 dimension
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'  # input token limit: 512

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    client = MilvusClient("milvus_demo.db")
    load_data_to_db(texts, model_name, device, client, max_length=256)

    query = "how to do rag research better"
    res = extract_info(query, model_name, client, limit=2)
    ans = get_llm_response(query, res, device)
    print(ans)
