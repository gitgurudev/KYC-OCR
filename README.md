# KYC-OCR

AI-powered Indian KYC document scanner using GPT-4o + RAG (Retrieval-Augmented Generation).

## What it does

Upload any Indian identity document and the system will:
1. **Detect** which document type it is (Aadhaar, PAN, Passport, Voter ID, Driving Licence, Ration Card, Utility Bill, Bank Passbook)
2. **Retrieve** the correct field schema using semantic RAG (ChromaDB vector search)
3. **Extract** all relevant fields using GPT-4o vision
4. **Display** extracted data in an editable form

## AI/ML Stack

| Component | Technology |
|-----------|------------|
| **Vision OCR** | GPT-4o (OpenAI multimodal LLM) |
| **RAG** | LangChain + ChromaDB + text-embedding-3-small |
| **Image Preprocessing** | OpenCV (CLAHE contrast, unsharp mask sharpening) |
| **Vector Store** | ChromaDB (persistent, local) |
| **Backend** | FastAPI (async Python) |

## RAG Architecture

The RAG pipeline stores document schema knowledge in a local ChromaDB vector store:

```
Document Templates → Embedding (text-embedding-3-small) → ChromaDB
                                                              ↓
Uploaded Image → GPT-4o (detect type) → Semantic Query → Schema Retrieval
                                                              ↓
                                         GPT-4o (extract with schema prompt)
                                                              ↓
                                              Structured JSON → Form Fields
```

Each document schema contains:
- Visual keyword description (what to look for)
- Exact fields to extract
- Field-specific extraction hints
- Category classification

## Supported Documents

- **Aadhaar Card** — Name, DOB, Gender, Aadhaar Number, Address
- **PAN Card** — Name, Father's Name, DOB, PAN Number
- **Passport** — All MRZ fields + personal details
- **Voter ID (EPIC)** — Name, EPIC Number, Constituency
- **Driving Licence** — License Number, Validity, Vehicle Classes
- **Ration Card** — Card Type, Head of Family, Members
- **Utility Bill** — Consumer Details, Amount, Dates
- **Bank Passbook** — Account Details, IFSC, Branch

## Setup

```bash
# 1. Clone
git clone https://github.com/gitgurudev/KYC-OCR.git
cd KYC-OCR

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your OpenAI API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 5. Run
python run.py
```

Open http://localhost:8000

## Project Structure

```
KYC-OCR/
├── app/
│   ├── data/
│   │   └── document_templates.py   # RAG knowledge base (8 document schemas)
│   ├── models/
│   │   └── schemas.py              # Pydantic response models
│   ├── services/
│   │   ├── image_service.py        # OpenCV preprocessing
│   │   ├── ocr_service.py          # GPT-4o two-pass pipeline
│   │   └── rag_service.py          # ChromaDB vector store + retrieval
│   └── main.py                     # FastAPI application
├── static/
│   ├── css/style.css
│   └── js/app.js
├── templates/
│   └── index.html
├── run.py
└── requirements.txt
```
