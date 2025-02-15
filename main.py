import pymupdf
import os
from retrieve_qa_pipeline import load_data_to_db, extract_info, get_llm_response
import torch
from pymilvus import MilvusClient
from typing import List
import logging

def load_pdf(path: str) -> List[str]:
    '''
    Read pdf in the path.

    :param path: The path that containing pdf files
    :return: A list of string
    '''
    texts = []
    for filename in os.listdir(path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(path, filename)
            doc = pymupdf.open(pdf_path)
            s = ""
            for page in doc:
                s += page.get_text()
            texts.append(s)
    return texts


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    print("Hi! This is a PDF analyzer.")
    while True:
        folder_path = input("Please enter the path to the folder containing the PDF files: ")

        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            break
        else:
            print("The specified folder does not exist. Please try again.")

    print("Loading texts from pdf...")
    texts = load_pdf(folder_path)

    # embedding model, 384 dimension
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'  # input token limit: 512

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    client = MilvusClient("milvus_demo.db")
    load_data_to_db(texts, model_name, device, client)
    print("Finish loading!")

    while True:
        query = input("Enter your question (or type 'exit' to end): ")
        if query.lower() == 'exit':
            break

        # Extract relevant information from the database
        retrieved_info = extract_info(query, model_name, client)
        # Generate and display answer using T5
        response = get_llm_response(query, retrieved_info, device)
        print("Answer: " + response)

    print("Conversation ended.")


if __name__=='__main__':
    main()
