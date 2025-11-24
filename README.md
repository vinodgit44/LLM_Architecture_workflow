# LLM_Architecture_workflow
This is a beautifully structured,  LLM Training Workflow (Tokenizer → Embeddings → Transformer → Loss → Training).

# 📘 **README — LLM Training Workflow (Step-by-Step Architecture)**

A complete, beginner-friendly and industry-accurate explanation of how **Large Language Models (LLMs)** are trained — from **raw text → tokenization → embeddings → transformer layers → logits → loss → backpropagation → trained LLM**.



# 📑 **Table of Contents**

* [🔍 1. Raw Data Collection](#-1-raw-data-collection)
* [🧹 2. Data Cleaning & Preprocessing](#-2-data-cleaning--preprocessing)
* [🔤 3. Tokenization](#-3-tokenization)
* [🔢 4. Input Embeddings](#-4-input-embeddings)
* [🏗️ 5. Transformer Architecture](#️-5-transformer-architecture)

  * Multi-head Self-Attention
  * Feed Forward Networks
  * Residuals + LayerNorm
* [📊 6. Output Layer → Logits](#-6-output-layer--logits)
* [📈 7. Loss Calculation](#-7-loss-calculation)
* [🔁 8. Backpropagation](#-8-backpropagation)
* [🌀 9. Training Loop](#-9-training-loop)
* [🎯 10. Fine-tuning & Instruction Tuning](#-10-fine-tuning--instruction-tuning)
* [📘 Full Workflow Diagram](#-full-workflow-diagram)



# 🔍 **1. Raw Data Collection**

The LLM pipeline begins with massive amounts of text from:

* Web pages
* Books & articles
* Code repositories
* Conversation datasets
* Domain-specific text (medical, legal, finance)

High-quality data = high-quality model.



# 🧹 **2. Data Cleaning & Preprocessing**

Before using the dataset:

* Remove HTML tags
* Remove boilerplate (menus, ads)
* Normalize Unicode
* Remove duplicates
* Filter toxic / low-quality content
* Split into documents

Goal: **clean, consistent, meaningful text**.



# 🔤 **3. Tokenization**

LLMs cannot operate on raw text — only on **tokens**
(usually numbers representing subwords).

### Example text

```
"Large language models are amazing!"
```

### Tokenizer steps

1. Normalize text
2. Split words into subwords
3. Convert to IDs

### Example output

```
[1256, 9432, 4421, 17, 204, 33091, 0]
```

Popular tokenizers:

* **BPE (GPT, Llama)**
* **SentencePiece (T5, ALBERT)**
* **WordPiece (BERT)**

---

# 🔢 **4. Input Embeddings**

Tokens → vectors

Each token ID is mapped to an embedding vector:

```python
Embedding(token_id) → [0.12, -0.77, 1.02, ...]
```

### Types of embeddings:

* **Token embeddings**
* **Positional embeddings** (absolute, learned, rotary)

Final input =

```
TokenEmbedding + PositionalEmbedding
```



# 🏗️ **5. Transformer Architecture**

Every LLM consists of **N repeated Transformer blocks**
(e.g., 32, 48, 70, or 120 layers).

Each block contains:



## 🧠 5.1 **Multi-Head Self-Attention (MHSA)**

For each token, compute 3 vectors:

* **Query (Q)**
* **Key (K)**
* **Value (V)**

### Attention formula:

```
Attention = Softmax( (Q · Kᵀ) / √d ) × V
```

Meaning:

* Query → "What am I looking for?"
* Key → "What information do I contain?"
* Value → "The actual information"

Multi-head attention lets the model focus on:

* grammar
* long-range dependencies
* meaning
* entities
* relationships



## 🔧 5.2 **Feed Forward Network (FFN)**

After attention, each token passes through:

```
Linear → GELU → Linear
```

This gives the model non-linear reasoning power.



## 🔄 5.3 **Residual Connections + LayerNorm**

Improve stability by:

* Normalizing activations
* Adding residual connections around attention & FFN layers

Pipeline (GPT-style):

```
LayerNorm → Attention → Add
LayerNorm → FFN → Add
```



# 📊 **6. Output Layer → Logits**

Final hidden states go to a **linear projection**:

```
Hidden State → Linear → Logits (size = vocab)
```

Example:

```
[3.1, -1.2, 0.5, ... ] (50,000 logits)
```

These are raw, unnormalized scores.



# 📈 **7. Loss Calculation**

Use **Cross-Entropy Loss**.

If true next token = “are”:

```
Loss = -log( P("are") )
```

Lower loss → better prediction.


# 🔁 **8. Backpropagation**

Gradients are computed and back-propagated through:

* Output layer
* Transformer blocks
* Embeddings

Then optimizer updates weights.

Popular optimizers:

* **AdamW**
* **Lion**
* **LAMB**



# 🌀 **9. Training Loop**

Typical training cycle:

```python
for batch in dataset:
    tokens = tokenizer(batch)
    embeddings = embed(tokens)
    hidden = transformer(embeddings)
    logits = linear(hidden)
    loss = cross_entropy(logits, targets)
    loss.backward()
    optimizer.step()
```

Training runs for:

* **millions of steps**
* **billions of tokens**



# 🎯 **10. Fine-Tuning & Instruction Tuning**

After base training:

## 🔹 **1. Supervised Fine-Tuning (SFT)**

Teach the model to follow instructions.

## 🔹 **2. RLHF**

Train using human preference comparisons.

## 🔹 **3. DPO / KTO / ORPO**

Newer methods → simpler & more stable.

## 🔹 **4. LoRA**

Efficient fine-tuning on small GPUs.



# 🧩 **Full LLM Training Workflow Diagram (ASCII)**

```
                   RAW TEXT
                       │
                       ▼
         ┌─────────────────────────┐
         │ Data Cleaning + Prep    │
         └─────────────────────────┘
                       │
                       ▼
              TOKENIZATION
        (Text → Subwords → IDs)
                       │
                       ▼
               EMBEDDING LAYER
      TokenEmb + PositionalEmb = InputEmb
                       │
                       ▼
        ┌──────────────────────────────┐
        │    TRANSFORMER (N Layers)    │
        │ ┌──────────────────────────┐ │
        │ │ Multi-Head Attention     │ │
        │ └──────────────────────────┘ │
        │ │ Feed Forward Network     │ │
        └──────────────────────────────┘
                       │
                       ▼
                LINEAR PROJECTION
             (Hidden → Vocabulary)
                       │
                       ▼
                    LOGITS
                       │
                       ▼
                 SOFTMAX PROBS
                       │
                       ▼
                   LOSS (CE)
                       │
                       ▼
                BACKPROPAGATION
                       │
                       ▼
                WEIGHT UPDATES
```



# 📜 **License**

MIT License — free to use, modify, share.


