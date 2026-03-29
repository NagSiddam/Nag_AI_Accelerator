# Nag AI Accelerator

A curated collection of AI/ML project accelerators to jumpstart development across common use cases. Each project provides a production-ready template with clean architecture, reproducible training pipelines, and clear documentation.

---

## 🚀 Projects

| Project | Domain | Key Technologies |
|---|---|---|
| [Text Classification](./projects/text-classification/) | NLP | Python, scikit-learn, Transformers |
| [Image Classification](./projects/image-classification/) | Computer Vision | Python, PyTorch, torchvision |
| [RAG Chatbot](./projects/rag-chatbot/) | Generative AI | Python, LangChain, FAISS, OpenAI |

---

## 📂 Repository Structure

```
Nag_AI_Accelerator/
├── projects/
│   ├── text-classification/     # Sentiment & topic classification (NLP)
│   ├── image-classification/    # Image recognition (Computer Vision)
│   └── rag-chatbot/             # Retrieval-Augmented Generation chatbot
├── shared/
│   └── utils/                   # Shared helper utilities
└── README.md
```

---

## ⚙️ Prerequisites

- Python 3.9+
- pip or conda package manager
- (Optional) CUDA-enabled GPU for faster model training

---

## 🛠️ Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/NagSiddam/Nag_AI_Accelerator.git
   cd Nag_AI_Accelerator
   ```

2. **Navigate to a project**
   ```bash
   cd projects/text-classification
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Follow the project-specific README** for training and inference instructions.

---

## 📖 Project Details

### 1. Text Classification
Accelerates building NLP pipelines for tasks such as sentiment analysis, spam detection, and topic labeling. Includes data preprocessing, feature extraction, model training (both classical ML and fine-tuned transformers), and evaluation utilities.

### 2. Image Classification
End-to-end image recognition template using PyTorch. Covers data augmentation, transfer learning from pretrained CNNs, training loops with early stopping, and model export for deployment.

### 3. RAG Chatbot
Retrieval-Augmented Generation chatbot that grounds LLM responses in your own documents. Supports PDF/text ingestion, vector store indexing with FAISS, and a simple query interface powered by LangChain and OpenAI.

---

## 🤝 Contributing

Pull requests are welcome! Please open an issue first to discuss proposed changes.

---

## 📄 License

MIT License — see [LICENSE](./LICENSE) for details.
