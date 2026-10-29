# Multimodal PDF RAG Pipeline

A notebook-based **multimodal Retrieval-Augmented Generation (RAG)** system that lets users ask questions about a PDF using its extracted text, tables, charts, diagrams, and other embedded images.

The pipeline extracts document content, summarizes visual content with a vision-capable Groq model, creates embeddings, stores them in Pinecone, retrieves relevant evidence for a question, and generates an answer with source-page references.

> **Project status:** This repository README documents the supplied `Multimodal_RAG_pipeline.ipynb` notebook. It is a research/demo notebook, not yet a fully production-hardened service.

## Features

- **PDF text extraction** with PyMuPDF.
- **Table extraction** and conversion to Markdown with pandas.
- **Embedded image extraction** for charts, diagrams, screenshots, and illustrations.
- **Vision-based image summaries** stored as text so they can be embedded and searched.
- **Semantic embeddings** using `sentence-transformers/all-MiniLM-L6-v2`.
- **Vector search** using Pinecone.
- **Text RAG** for questions supported by retrieved text or tables.
- **Visual RAG** that can attach retrieved original images when visual evidence is relevant.
- **Page-level metadata** so retrieved sources can be traced back to the PDF.
- **Configurable text and vision models** through Groq.

## Architecture

```text
PDF document
    |
    +--> Extract page text --------------------+
    |                                          |
    +--> Extract tables --> Markdown ----------+--> LangChain Documents
    |                                          |          |
    +--> Extract embedded images               |          v
             |                                 |    Hugging Face embeddings
             v                                 |          |
       Groq vision model                       |          v
             |                                 |       Pinecone
       Concise visual summary -----------------+          |
                                                        Question
                                                           |
                                                           v
                                                    Top-k retrieval
                                                           |
                                             +-------------+-------------+
                                             |                           |
                                      Text/table results           Visual results
                                             |                           |
                                      Text RAG chain            Original retrieved images
                                             |                           |
                                             +-------------+-------------+
                                                           |
                                                    Grounded answer
```

## Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Notebook runtime | Google Colab / Jupyter | Run the pipeline |
| PDF parsing | PyMuPDF (`fitz`) | Extract page text, embedded images, and tables |
| Image handling | Pillow | Resize and encode images for vision requests |
| LLM / vision API | Groq API | Generate answers and describe visuals |
| Text orchestration | LangChain | Prompt templates, model wrapper, and output parsing |
| Embeddings | Hugging Face `all-MiniLM-L6-v2` | Convert extracted text and visual summaries into vectors |
| Vector database | Pinecone serverless | Store and retrieve document chunks |
| Data formatting | pandas / tabulate | Convert extracted tables to Markdown |

## Repository Layout

```text
.
├── Multimodal_RAG_pipeline.ipynb
├── my_doc.pdf                         # Add your own PDF; not included by default
└── uploaded_doc_extracted_images/     # Created by the notebook
```

The notebook also creates a Pinecone index named `multimodal-rag` and uses the namespace `fy2026-demo` by default. These settings are configurable in the notebook.

## Requirements

- Python environment supported by the selected package versions.
- A PDF file to index.
- A Groq API key with access to the selected text and vision models.
- A Pinecone API key and permission to create/use a serverless index.
- Internet access for API calls and model/package downloads.

The notebook currently uses these packages:

```bash
pip install -U \
  langchain \
  langchain-core \
  langchain-groq \
  langchain-huggingface \
  langchain-pinecone \
  pinecone \
  sentence-transformers \
  pymupdf \
  groq \
  pandas \
  tabulate \
  pillow
```

If you run this in Google Colab, use `%pip install ...` in a notebook cell. The supplied notebook also force-reinstalls Pillow because of an environment-specific installation issue; that step is not necessarily needed in every environment.

## Setup

### 1. Open the notebook

Open `Multimodal_RAG_pipeline.ipynb` in Google Colab or Jupyter.

### 2. Install dependencies

Run the installation cell. If the environment asks for a runtime restart after package installation, restart it and run the notebook cells again in order.

### 3. Add the PDF

Place your PDF in the notebook's working directory and name it `my_doc.pdf`, or update `PDF_PATH` in the notebook:

```python
PDF_PATH = Path("my_doc.pdf")
```

The notebook checks `/mnt/data/my_doc.pdf` as a fallback when the relative path does not exist.

### 4. Configure API keys

The supplied notebook uses Google Colab Secrets:

- `GROQ_API_KEY`
- `PINECONE_API_KEY`

Add both secrets in Colab's Secrets panel and allow the notebook to access them. Alternatively, in a local environment, set them as environment variables before running the notebook:

```bash
export GROQ_API_KEY="your_groq_api_key"
export PINECONE_API_KEY="your_pinecone_api_key"
```

For Windows PowerShell:

```powershell
$env:GROQ_API_KEY="your_groq_api_key"
$env:PINECONE_API_KEY="your_pinecone_api_key"
```

Do not commit API keys to Git or place real keys directly in notebook cells.

### 5. Review model and database configuration

The notebook defines:

```python
TEXT_MODEL = "openai/gpt-oss-20b"
VISION_MODEL = "qwen/qwen3.8-27b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

PINECONE_INDEX_NAME = "multimodal-rag"
PINECONE_NAMESPACE = "fy2026-demo"
```

These are the defaults in the supplied notebook. Model availability, vision support, account access, and rate limits can change. Verify that your Groq account supports the selected models before running the pipeline.

### 6. Run cells in order

Run the notebook from top to bottom. The main stages are:

1. Initialize API clients and model settings.
2. Extract PDF text, tables, and embedded images.
3. Summarize images with the vision model.
4. Create embeddings.
5. Create or connect to the Pinecone index.
6. Store extracted LangChain documents.
7. Retrieve relevant documents for questions.
8. Generate text-only or visual-grounded answers.

**Important:** The notebook includes a cell that deletes all vectors in the configured Pinecone namespace before ingestion:

```python
index.delete(delete_all=True, namespace=PINECONE_NAMESPACE)
```

This is destructive for that namespace. Use a dedicated namespace for experiments and remove or disable this cell if you need to preserve existing vectors.

## How It Works

### 1. PDF extraction

PyMuPDF extracts page text and embedded images. It also attempts to identify tables and convert them to Markdown. Each extracted item is stored as a LangChain `Document` with metadata such as page number, modality, source filename, and—when applicable—image path or table number.

### 2. Visual summarization

The notebook converts each extracted image to a JPEG data URI and sends it, with a general-purpose prompt, to the configured Groq vision model. The resulting concise description is stored as searchable text. Raw images are not embedded directly by the text embedding model.

### 3. Embedding and indexing

The notebook creates normalized embeddings with `sentence-transformers/all-MiniLM-L6-v2`, then stores the documents in Pinecone. The index dimension is detected from the embedding model rather than hard-coded.

### 4. Retrieval

The retriever uses semantic similarity and defaults to `k=3`:

```python
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)
```

Increase or decrease `k` based on relevance and context size. Returning more chunks can improve coverage but also increases prompt size and token consumption.

### 5. Answer generation

- If the retrieved results contain visual summaries with valid image paths, the notebook sends the context and up to three original images to the vision model.
- Otherwise, it sends the retrieved context and question through the text RAG chain.
- The prompts instruct the model to answer only from retrieved evidence, avoid unsupported claims, and mention page numbers when available.

## Example Usage

After indexing the PDF, call:

```python
result = ask_rag("What is this document about?")
```

Other example questions:

```python
ask_rag("Summarize the key findings of the document.")
ask_rag("Which model has the highest ROC-AUC according to the retrieved evidence?")
ask_rag("Explain the relevant table and compare its important values.")
ask_rag(
    "Summarize the document's main points.",
    show_sources=True,
    show_images=True,
)
```

`ask_rag()` returns a dictionary containing:

- `answer`: generated answer text
- `documents`: retrieved LangChain documents
- `image_paths`: paths of retrieved visual files

The function also prints retrieved source pages and can display retrieved images when `show_images=True`.

## Token Limits and Rate-Limit Errors

Groq model limits may apply to requests per minute, input tokens per minute, output tokens per minute, and total tokens per day. Limits depend on the model, account, and service tier.

Common errors seen while running this notebook include:

- `model_terms_required`: the organization must accept the model's terms or choose a model already authorized for that organization.
- `429` with `OTPM`: the expected output-token budget exceeds the available output-token-per-minute budget.
- `413` or `429` with `ITPM`: the request or accumulated input tokens exceed the input-token-per-minute limit.
- `429` with `TPD`: the daily token allowance has been exhausted.

Recommended mitigations:

1. **Reuse visual summaries.** Run PDF extraction and vision summarization once, then store the results. Avoid reprocessing the same PDF for every question.
2. **Reduce retrieval context.** Start with a small `k` value and retrieve only relevant chunks. Avoid sending the entire document in every prompt.
3. **Limit image attachments.** The current visual-answer helper attaches at most three images; use fewer when possible.
4. **Use concise prompts and outputs.** Set a suitable output-token limit for each call. Output limits do not reduce input context or image tokens.
5. **Handle rate limits.** Respect the retry interval reported by the API and use bounded exponential backoff for temporary rate limits. Do not retry unchanged oversized requests indefinitely.
6. **Control concurrency.** Parallel page processing can exhaust shared model quotas faster. Limit concurrent vision calls according to your account's actual quotas.
7. **Choose an available model.** Confirm that the replacement model supports image input if the pipeline uses it for vision.

The notebook currently sets `max_tokens=500` for visual summarization and `max_completion_tokens=900` for visual question answering. These are maximum output settings, not guaranteed token usage. Adjust the relevant call rather than assuming one setting controls every model call.

## Important Limitations

- The extraction loop processes **embedded PDF images**. Charts or diagrams drawn as vector shapes may not be extracted as images by `page.get_images(full=True)`.
- Scanned PDFs may produce little or no text unless OCR is added.
- Table detection is best-effort and may not reconstruct complex or borderless tables correctly.
- The embedding model is text-oriented; image content becomes searchable through its generated textual summary.
- Original image paths are local to the notebook runtime. They must remain available for visual answering.
- The current notebook is synchronous and does not implement production-grade job queues, concurrency controls, authentication, access control, observability, or persistent image storage.
- The example namespace is cleared before indexing. Do not use that behavior for a shared or production namespace.
- Retrieval quality depends on extracted text quality, visual summaries, embedding choice, chunking, and the relevance of the retrieved results.

## Security and Privacy

- Treat PDF files as potentially sensitive input.
- Do not commit secrets, private documents, extracted images, or generated sensitive summaries to a public repository.
- Review Groq and Pinecone data-handling terms and your organization's requirements before sending confidential documents to external services.
- For regulated or confidential workloads, evaluate data residency, retention, encryption, access control, and vendor agreements before deployment.

## Troubleshooting

| Problem | Likely cause | What to check |
|---|---|---|
| PDF not found | Incorrect `PDF_PATH` or file not uploaded | Place the PDF in the working directory or update the path |
| Missing Colab secret | Secret name mismatch or notebook access not enabled | Check `GROQ_API_KEY` and `PINECONE_API_KEY` |
| `model_terms_required` | Model terms not accepted for the organization | Ask the organization admin to accept terms or select an authorized model |
| `OTPM` rate limit | Requested output budget exceeds the current output-token budget | Reduce the call's output limit, wait for quota, or use a permitted model with available quota |
| `ITPM` rate limit / oversized request | Context, history, or images make the request too large | Reduce retrieved chunks/context and unnecessary image attachments |
| `TPD` rate limit | Daily model token quota exhausted | Wait for the reported reset or use another authorized model with available quota |
| No visual result retrieved | No visual summary matched the query, or image file is missing | Inspect retrieved document metadata, `modality`, and `image_path` |
| Pinecone dimension error | Existing index dimension differs from the current embedding model | Use an index with the matching dimension |
| Poor answers | Irrelevant chunks, missing OCR, or weak visual summaries | Inspect retrieval results and improve extraction/retrieval before changing the prompt |

## Future Improvements

Potential next steps for a more robust application:

- OCR for scanned PDFs and pages with text embedded in images.
- Page rendering to detect charts and diagrams that are not embedded image objects.
- Chunking long pages while preserving page and section metadata.
- Token-aware context budgeting before model calls.
- Rate-limit-aware retries and bounded concurrency for vision processing.
- Caching visual descriptions by document hash and image hash.
- Persistent object storage for original images.
- Evaluation of retrieval quality, answer faithfulness, and page citations.
- A user-facing interface and an API layer.
- Per-user document isolation, authorization, logging, monitoring, and deletion controls.

## License

No license is specified in the supplied notebook. Add a `LICENSE` file before distributing or publishing the project.
