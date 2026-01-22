# English → Urdu Translator (CPU)

This project implements an **English-to-Urdu conversational translation chatbot** using a **fine-tuned Marian NMT model**.  
It is **CPU-only**, beginner-friendly, and inspired by the research paper  
**“Generalists vs. Specialists: Evaluating Large Language Models for Urdu.”**

The project demonstrates how a **specialist model (fine-tuned for one task)** can produce useful results for a low-resource language like Urdu using limited computational resources.

---

## 📌 Project Overview

- **Task**: English → Urdu conversational translation  
- **Model**: Marian NMT (`opus-mt-en-ur`)  
- **Approach**: Fine-tuning a pre-trained specialist model  
- **Hardware**: CPU only (no GPU required)  
- **Use Case**: Daily conversation, directions, polite phrases, chatbot demo  

---

## 🧠 Model Used

- **Base model**: `Helsinki-NLP/opus-mt-en-ur`
- **Architecture**: Encoder–Decoder (Seq2Seq) with Attention
- **Why Marian NMT?**
  - Specifically designed for translation tasks
  - Lightweight and CPU-friendly
  - Pre-trained on multilingual parallel corpora
  - Suitable for low-resource languages like Urdu

The fine-tuned model is stored locally in:


---

## 📊 Dataset

- **Training set**: `train_1300_clean.tsv` (769 cleaned sentence pairs)
- **Evaluation set**: `bleu_test.tsv` (held-out test data)
- **Domain**: Conversational English → Urdu  
  (greetings, directions, daily talk)

---

## ⚙️ Training Configuration

- **Framework**: Hugging Face Transformers
- **Optimizer**: AdamW
- **Epochs**: 2
- **Hardware**: CPU only
- **Loss**: Reduced significantly during training

---

## 📈 Evaluation

- **Metric**: SacreBLEU
- **BLEU Score**: **23.38**

**Interpretation**:
- BLEU score between **20–30** is considered reasonable for conversational NMT
- Confirms that fine-tuning a specialist model improves translation quality

---

