# Plagiarism Guard

An advanced AI-powered plagiarism detection system that analyzes textual similarity using multiple detection techniques including lexical analysis, semantic similarity, stylometry, citation analysis, and AI-generated content detection.

---

## Project Overview

Plagiarism Guard is a multi-engine plagiarism detection platform designed to identify copied, paraphrased, or AI-generated content in documents.

Traditional plagiarism systems rely mainly on direct text matching. This project improves detection accuracy by combining several analysis techniques such as lexical comparison, semantic similarity, writing style analysis, and citation validation.

The system processes documents, extracts textual features, and generates detailed similarity reports to help identify potential plagiarism in academic and research content.

This project demonstrates the integration of Natural Language Processing (NLP), machine learning techniques, and modular backend architecture for intelligent document analysis.

---

## Features

• Multi-engine plagiarism detection
• Lexical similarity detection
• Semantic similarity analysis using NLP techniques
• Stylometry analysis to detect writing style inconsistencies
• Citation and reference validation
• AI-generated text detection
• User authentication system
• Database storage for analyzed documents
• Automated plagiarism scoring
• PDF report generation for results
• Modular architecture for scalability

---

## Detection Techniques

Plagiarism Guard integrates multiple detection strategies to improve accuracy.

### 1. Lexical Analysis

Compares direct word usage between documents and identifies exact or near-exact matches.

### 2. Semantic Analysis

Detects paraphrased content by analyzing meaning rather than just matching words.

### 3. Stylometric Analysis

Examines writing style characteristics such as sentence structure, vocabulary patterns, and readability to identify authorship inconsistencies.

### 4. Citation Analysis

Evaluates references and citations to detect improper or missing citations.

### 5. AI Content Detection

Attempts to identify text that may have been generated using AI tools.

---

## 🏗 System Architecture

The system follows a modular backend architecture to separate different analysis engines.

```
Plagiarism-Guard
│
├── plagiarism_system
│   │
│   ├── app
│   │   ├── routes.py
│   │   ├── models.py
│   │   ├── database.py
│   │   └── auth.py
│   │
│   ├── engines
│   │   ├── lexical_engine.py
│   │   ├── semantic_engine.py
│   │   ├── stylometry_engine.py
│   │   ├── citation_engine.py
│   │   └── ai_detection_engine.py
│   │
│   ├── utils
│   │   ├── preprocessing.py
│   │   ├── feature_extraction.py
│   │   └── text_extractor.py
│   │
│   ├── reports
│   │   └── pdf_export.py
│   │
│   └── run.py
│
├── requirements.txt
└── README.md
```

---

## Tech Stack

### Backend

• Python
• Flask

### Database

• SQLite

### Natural Language Processing

• Scikit-learn
• NLP preprocessing techniques
• Text vectorization methods

### Other Tools

• Git
• GitHub
• Python Virtual Environment

---

## ⚙ Installation

### 1. Clone the repository

```
git clone https://github.com/RitwiKanojia15/Plagiarism-Guard.git
```

### 2. Navigate to the project directory

```
cd Plagiarism-Guard
```

### 3. Create a virtual environment

```
python -m venv .venv
```

### 4. Activate the virtual environment

Windows

```
.venv\Scripts\activate
```

Linux / Mac

```
source .venv/bin/activate
```

### 5. Install dependencies

```
pip install -r requirements.txt
```

### 6. Run the application

```
python plagiarism_system/run.py
```

---

## How the System Works

1. User uploads or enters a document.
2. The system preprocesses the text.
3. Multiple analysis engines process the content.
4. Each engine generates similarity scores.
5. The results are combined to calculate the final plagiarism score.
6. A detailed plagiarism report is generated.

---

## Screenshots

Add screenshots of:

• Upload interface
• Analysis results page
• Plagiarism score output
• Generated report

---

## Future Improvements

• Support for PDF and DOCX document uploads
• Deep learning based semantic similarity detection
• Integration with academic databases
• Real-time plagiarism detection
• Interactive analytics dashboard
• API integration for external platforms

---

## Author

Ritwik Kanojia

GitHub
https://github.com/RitwiKanojia15

---

## 📜 License

This project is licensed under the MIT License.
