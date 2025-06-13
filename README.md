# Final Year Project 2025 – Systematic Review App

A web-based application to assist in conducting systematic reviews using large language models (LLMs), featuring automatic summarisation, screening, evidence table generation, and a retrieval-augmented chatbot.

## Environment Setup

### 1. Create Project Directory

Create a new directory for your project:

```bash
mkdir systematic_review_app
cd systematic_review_app
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

- On Linux/MacOS:

```bash
source venv/bin/activate
```

- On Windows

```bash
.\venv\Scripts\activate
```

### 4. Install Dependencies

Run the follwing:

```bash
pip install -r requirements.txt
```

> **_NOTE:_**  Some dependencies (e.g., unstructured) may install more reliably on Linux or WSL. On Windows, compatibility issues have been observed.

### 5. Setup .env file

Create a `.env` file in the root project directory with the following content:

```.env
SQLALCHEMY_DATABASE_URI=sqlite:///sqlite.db
UPLOAD_URL=http://localhost:8050

OPENAI_API_KEY=your-openai-api-key-here

PINECONE_API_KEY=your-pinecone-api-key-here
PINECONE_ENV_NAME=us-east-1
PINECONE_INDEX_NAME=your-index-name-here
```

> **_IMPORTANT:_**

- Do not commit your `.env` file to GitHub. Add `.env` to your `.gitignore`.

- Replace the placeholder values (your-...-here) with your actual API keys and config values.

## Running the Development Server

### Option 1: Start a New Instance (fresh database)

This clears any previous data and starts a new clean instance:

```bash
./start_web.sh
```

### Option 2: Continue from Previous State (retain database)

This retains all previously uploaded files and results:

```bash
inv dev
```

## Tips

- Ensure Python 3.10 or later is installed.

- If facing dependency issues, consider using a Docker container (not included in this README).

## User Instructions

1. On the Home page, click the "Create Project" button to start a new systematic review.
2. Enter your Project Name, Review Question, Review Type, and Search Criteria.
3. Upload your selected PDF files.
4. Once processing is complete, click the project name to view the list of uploaded documents.
5. Click a document to view its summary and screening result, or select documents to generate an evidence table.
6. On a document’s page, use the chat box to interact with the paper, or toggle between summary and screening results.
