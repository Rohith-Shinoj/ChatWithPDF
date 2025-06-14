# ChatWithPDF: PDF Question Answering with Text and Image Understanding

`ChatWithPDF` is a tool that I have developed with the intention of learning NLP and how Large language Models (LLMs) can be used for document-based question answering and multimodal retrieval. The project began with a simple implementation (`bot.py`) that used LLMs to answer questions based purely on the text extracted from PDFs. However, I quickly noticed a limitation: many academic and technical PDFs contain **important information in the form of images**, such as graphs, charts and other images.

Since the initial version completely ignored this data, I improved the system by building `chatpdf.py`, which integrates **image captioning (using BLIP)** and **OCR (using Tesseract)**. This allowed the system to extract and understand information from figures and diagrams, enabling more complete and accurate answers. This project leverages **LangChain**, **Hugging Face models**, **OCR**, and **Image recognition** to enhance document understanding.

---

## Features

| Feature                           | `bot.py`             | `chatpdf.py` (Enhanced Version) |
|----------------------------------|----------------------|----------------------------------|
| Text-based PDF QA                | ✅                   | ✅                               |
| Handles multi-page PDFs          | ✅                   | ✅                               |
| Rephrases and summarizes answers | ✅                   | ✅                               |
| OCR support (images in PDFs)     | ❌                   | ✅                               |
| Image captioning with BLIP       | ❌                   | ✅                               |
| Image extraction from PDFs       | ❌                   | ✅                               |

---

## 📁 File Overview

### `bot.py`

- Basic version with only **text-based QA**.
- Extracts and chunks text from the PDF.
- Embeds and stores documents using `FAISS`.
- Uses `Flan-T5` for question-answering.

### `chatpdf.py` (Improved Version)

- Includes all features from `bot.py`.
- Adds:
  - OCR extraction from images using `pytesseract`.
  - Image captioning using BLIP.
  - Embeds both textual and image-based content.
- Allows richer QA from diagrams, graphs, scanned pages, and mixed-content PDFs.

---

## 🖼️ Sample Use Case

**Use Case:** Add the required pdf as a filepath replacing "sample.py". You can then ask questions like:

> *"How is X different from Y?"*

or 

> *"What does the graph on page 3 represent?"*


`chatpdf.py` extracts the image, runs OCR + captioning, embeds it alongside text, and responds accordingly.

---

## Installation

```bash
# Clone the repo
git clone https://github.com/your-username/chatpdf.git
cd chatpdf

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Install tesseract OCR

# macOS:
brew install tesseract

# Ubuntu:
sudo apt-get install tesseract-ocr
```

## 📌 To-Do / Future Improvements
- Add GUI using Streamlit or Flask
- Support for table parsing using pdfplumber
- Multilingual OCR support
