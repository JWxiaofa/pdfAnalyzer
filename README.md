# PDF Analyzer and Question Answering System

This project is a PDF analyzer that can analyze multiple PDF documents and answer questions about their contents.

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