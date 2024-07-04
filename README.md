# PDF Analyzer and Question Answering System

This project is a PDF analyzer that can analyze multiple PDF documents and answer questions about their contents.

### Table of Contents
1. Setup Instructions
2. Basic Features
3. API Documentation
4. Hyperparameters fine-tuning
5. Future Work

## Setup Instructions

### Clone the Repository

First, clone this repository to your local machine:

```bash
git clone git@github.com:JWxiaofa/pdfAnalyzer.git
cd pdfAnalyzer
```

### Set up conda environment

Ensure you have conda installed on your system.

```bash
conda create --name pdf_analyzer python=3.10
conda activate pdf_analyzer
pip install -r requirements.txt

```

### Start the application

Now you can start the app!

```bash
python main.py
```

#### Usage Example
```markdown
Hi! This is a PDF analyzer.
Please enter the path to the folder containing the PDF files: </path/to/pdf/folder>
Loading texts from pdf...
Loading data into vector database...
Finish loading!
Enter your question (or type 'exit' to end): <your quesion>
Answer: ...
Enter your question (or type 'exit' to end): exit
Conversation ended.
```

## Features

- Extract text from PDF files in a specified folder
- Create vector embeddings for the extracted text using `all-MiniLM-L6-v2` model
- Store the embeddings in a vector database (`milvus`) for efficient retrieval
- Use a local large language model (`google/flan-t5-basic`) to generate answers based on the relevant context
- Implement a command-line interface for user interaction

## API Documentation
### `retrieve_qa_pipeline.py`

#### `split_text_to_chunk(text, tokenizer, max_length=512)`

Splits the given text into chunks based on the tokenizer and the maximum length.

- **Parameters:**
  - `text` (str): The text to split.
  - `tokenizer` (transformers.AutoTokenizer): The tokenizer to use for splitting.
  - `max_length` (int): The maximum length of each chunk.

- **Returns:**
  - `List[str]`: A list of text chunks.

#### `get_chunks(texts, tokenizer, max_length=512)`

Given a list of texts, split each text into chunks based on the maximum token length.

- **Parameters:**
  - `texts` (List[str]): The list of texts to split.
  - `tokenizer` (transformers.AutoTokenizer): The tokenizer to use for splitting.
  - `max_length` (int): The maximum length of each chunk.

- **Returns:**
  - `List[str]`: A list of text chunks.

#### `mean_pooling(model_output, attention_mask)`

Applies mean pooling to the model output using the attention mask.

- **Parameters:**
  - `model_output` (torch.Tensor): The output from the model.
  - `attention_mask` (torch.Tensor): The attention mask.

- **Returns:**
  - `torch.Tensor`: The pooled embeddings.

#### `generate_embedding(chunked_texts, model_name, device, max_length=512)`

Generates embeddings for the given text chunks using the specified model.

- **Parameters:**
  - `chunked_texts` (List[str]): The text chunks to generate embeddings for.
  - `model_name` (str): The name of the model to use for generating embeddings.
  - `device` (torch.device): The device to use for computation.

- **Returns:**
  - `np.ndarray`: The generated embeddings.

#### `load_data_to_db(texts, model_name, device, client, max_length=256)`

1. Split texts into chunks
2. Generate embeddings for each chunk
3. Load text and embeddings into the vector database.

- **Parameters:**
  - `texts` (List[str]): A list of texts read from PDFs.
  - `model_name` (str): The name of the model to use for embedding generation.
  - `device` (torch.device): The device to use for computation (CPU/GPU).
  - `client` (MilvusClient): The Milvus client for database operations.
  - `max_length` (int): The maximum length of tokens in each chunk.

#### `extract_info(query, model_name, client, limit=2)`

Extracts information relevant to the query from the vector database.

- **Parameters:**
  - `query` (str): The user input.
  - `model_name` (str): The name of the model to use for query encoding.
  - `client` (MilvusClient): The Milvus client for database operations.
  - `limit` (int): The maximum number of results to return.

- **Returns:**
  - `str`: The extracted information as a concatenated string.

#### `get_llm_response(query, retrieved_info, device)`

Generates a response from the LLM based on the query and retrieved information.

- **Parameters:**
  - `query` (str): The user input.
  - `retrieved_info` (str): The relevant information.
  - `device` (torch.device): The device to use for computation (CPU/GPU).

- **Returns:**
  - `str`: The response from the LLM.

### `main.py`

#### `load_pdf(path)`

Reads PDFs from the specified path.

- **Parameters:**
  - `path` (str): The path containing PDF files.

- **Returns:**
  - `List[str]`: A list of strings read from the PDFs.

#### `main()`

Runs the main PDF analyzer script.

- **Description**: Prompts the user for the path to PDF files, loads and processes the PDFs, and allows the user to ask questions about pdf contents.


## Hyperparameter fine-tuning

When using default parameters of T5 model, the answers generated are very short. Fine-tuning some hyperparameters might help to improve the generation quality:

`max_new_token`: Controls the maximum number of new tokens to generate.

`min_length`: Sets the minimum length of the generated sequence, default is 0.

`num_beams`: Using beam search with larger beam width can help generate more coherent and detailed responses. Default is 1.

`length_penalty`: A value greater than 1.0 encourages longer sequences, while a value less than 1.0 discourages them. Default is 1.

`repetition_penalty`: This parameter can help reduce the repetition of tokens, range from 1.0 to 2.0. Default is 1.


### Data and Questions
The pdf data consists of 4 papers about RAG and 2 articles about semantics.

The questions are:
1. What is RAG?
2. How to do NLP research better?
3. What is conversational implicatures?

Here's the result using default parameters and fine-tuned results.

### Default
| Hyperparameter     | Value |
|--------------------|-----|
| max_new_tokens     | 100 |
| min_length         | 0   |
| num_beams          | 1   |
| length_penalty     | 1   |
| repetition_penalty | 1   |

| Question                            | Answer                                                                                             |
|-------------------------------------|----------------------------------------------------------------------------------------------------|
| What is RAG?                        | rag model       |
| How to do nlp research better?      | no             |
| What is conversational implicature? | the concept of conversational implicature|

### Experiment results

Answers with default parameters are generally too short. First, we set the `min_length` to 50.

| Hyperparameter    | Value |
|-------------------|-------|
| max_new_tokens    | 200   |
| min_length        | 50    |
| num_beams         | 1     |
| length_penalty    | 1     |
| repetition_penalty | 1   |

| Question                            | Answer                                                                                             |
|-------------------------------------|----------------------------------------------------------------------------------------------------|
| What is RAG?                        | rag model is a language model that can be used to generate abuse, faked or misleading content in the news or on social media. we rag model is a language model that can be used to impersonate others and automate the production of spam / phishing content.       |
| How to do nlp research better?      | zheng lin1, 2 1institute of information engineering, chinese academy of sciences, beijing, china 2school of cyber security, university of chinese academy of sciences, beijing, china             |
| What is conversational implicature? | the concept of conversational implicature makes it possible to claim that a sentence with two quite distinct effects is nev - 4 ertheless unambiguous from the point of view of its conventional content, and that two sentences that con convey practically the same thing are nevertheless not logically or linguistically equivalent.|

After setting `min_length` to 50, the answer is more detailed. Next, we experimented with different beam width to get better results from beam search.

`num_beams=2`: 

| Hyperparameter    | Value |
|-------------------|-------|
| max_new_tokens    | 200   |
| min_length        | 50    |
| num_beams         | 2     |
| length_penalty    | 1     |
| repetition_penalty | 1   |

| Question                            | Answer                                                                                             |
|-------------------------------------|----------------------------------------------------------------------------------------------------|
| What is RAG?                        | experimented with rag models in a wide variety of scenarios with direct benefit to society, for example by endowing it with a medical index and asking it open - domain questions on that topic, or by helping people be more effective at their jobs.       |
| How to do nlp research better?      | zheng lin1, 2 1institute of information engineering, chinese academy of sciences, beijing, china 2school of cyber security, university of chinese academy of sciences, beijing, china             |
| What is conversational implicature? | conversational implicatures are only indirectly associ - ated with the linguistic content of utterances. they are derived from the content 48 of the sentences used and owe their existence, according to grice, to the fact that participants in a conversation are constrained by the common goal of communi - cation to be cooperative.|

`num_beams=2`, `length_penalty=1.1`: 

| Hyperparameter    | Value |
|-------------------|-------|
| max_new_tokens    | 200   |
| min_length        | 50    |
| num_beams         | 2     |
| length_penalty    | 1.1   |
| repetition_penalty | 1   |

| Question                            | Answer                                                                                             |
|-------------------------------------|----------------------------------------------------------------------------------------------------|
| What is RAG?                        | experimented with rag models in a wide variety of scenarios with direct benefit to society, for example by endowing it with a medical index and asking it open - domain questions on that topic, or by helping people be more effective at their jobs.       |
| How to do nlp research better?      | zheng lin1, 2 1institute of information engineering, chinese academy of sciences, beijing, china 2school of cyber security, university of chinese academy of sciences, beijing, china             |
| What is conversational implicature? | a system for explaining certain aspects of what utterances convey without claiming that they are part of the conven - tional force of the uttered sentence|

Model has better performance when setting beam width as 2.

`num_beams=3`:

| Hyperparameter    | Value |
|-------------------|-----|
| max_new_tokens    | 200 |
| min_length        | 50  |
| num_beams         | 3   |
| length_penalty    | 1   |
| repetition_penalty | 1   |

| Question                            | Answer                                                                                             |
|-------------------------------------|----------------------------------------------------------------------------------------------------|
| What is RAG?                        | experimented with a rag model with a domain adaptation performance of dif - ferent rag models using a domain adaptation performance of dif - ferent rag models using a domain adaptation performance of dif - ferent rag models using a domain adaptation performance of dif - ferent rag models using a domain adaptation performance of dif - ferent rag models using a domain adaptation performance of dif - ferent rag models using a domain adaptation performance of dif - ferent rag models using a domain adaptation performance of dif - ferent rag models using a domain adaptation performance of dif - ferent rag models using a domain adaptation performance of dif - ferent rag models using a domain adaptation performance of dif       |
| How to do nlp research better?      | no response. the responsible nlp checklist used at acl 2023 is adopted from naacl 2022, with the addition of a question on ai writing assistance. 305 [UNK] did you discuss the experimental setup, including hyperparameter search and best - found hyperparameter values?             |
| What is conversational implicature? | conversational implicatures are only indirectly associ - ated with the linguistic content of utterances. they are derived from the content 48 of the sentences used and owe their existence, according to grice, to the fact that participants in a conversation are constrained by the common goal of communi - cation to be cooperative. nonconventional implicatures come in two varieties : first the important class of conversational implicatures that involve the coop - 52 erative principle and its maxims, and then a poorly described class of nonconversational, nonconversational implicatures that are calculated in context on the basis of the conventional meaning, knowledge of the context of utterance, and background knowledge|

Increasing `num_beams` to 3 results in too much repetition. Therefore, we increaseed the `repetition_penalty` to avoid repetition.

`num_beams=3, repetition_penalty=2`

| Hyperparameter    | Value |
|-------------------|-------|
| max_new_tokens    | 200   |
| min_length        | 50    |
| num_beams         | 3     |
| length_penalty    | 1     |
| repetition_penalty | 2     |

| Question                            | Answer                                                                                             |
|-------------------------------------|----------------------------------------------------------------------------------------------------|
| What is RAG?                        |experimented with rag as a language model.       |
| How to do nlp research better?      | a. 3 implementation details we list other important hyperparameters in table 4. hyperparameter rams wikievents t5 - base t5 - large t5 - base t5 - large batch size 16 8 16 8 training epochs 50 50 20 40 optimizer adamw adamw adamw adamw max input length 512 512 512 512 max target length             |
| What is conversational implicature? | a system for explaining certain aspects of what utterances convey without claiming that they are part of the conven - tional force of the uttered sentence|

`num_beams=3, repetition_penalty=1.5`

| Hyperparameter    | Value |
|-------------------|-------|
| max_new_tokens    | 200   |
| min_length        | 50    |
| num_beams         | 3     |
| length_penalty    | 1     |
| repetition_penalty | 1.5   |

| Question                            | Answer                                                                                             |
|-------------------------------------|----------------------------------------------------------------------------------------------------|
| What is RAG?                        |experimented with rag as a language model.       |
| How to do nlp research better?      | a. 3 implementation details we list other important hyperparameters in table 4. hyperparameter rams wikievents t5 - base t5 - large t5 - base t5 - large batch size 16 8 16 8 training epochs 50 50 20 40 optimizer adamw adamw adamw adamw max input length 512 512 512 512 max target length             |
| What is conversational implicature? | semantics )|


Overall, when `num_beams` is set to 2, the generation quality is relatively good. The final hyperparameter configuration used in this task is:


| Hyperparameter    | Value |
|-------------------|-----|
| max_new_tokens    | 200 |
| min_length        | 50  |
| num_beams         | 2   |
| length_penalty    | 1.1 |
| repetition_penalty | 1   |


## Future Work

Future work could include:
1. Experiment with different embedding models. e.g. models with larger dimensions.
2. Move from a local vector database setup to a larger-scale, cloud-based vector database server to handle large dataset and reuse the previous data to enhance efficiency.
3. Fine-tuned the model with customized dataset in specific domains.
4. Currently, previous conversation memory is not implemented due to the input length limitation of T5 (512 tokens). Future work could involve using larger LLM and implement session or window based memory solution.