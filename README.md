# Social Media Pharmacovigilance and RAG System

An end-to-end NLP and retrieval project for social media pharmacovigilance built on a large drug review dataset, combining exploratory data analysis, ternary sentiment classification, transformer fine-tuning, vector search with ChromaDB, and a Streamlit chatbot with Tavily web search fallback.

## Project Overview

This project analyzes patient drug reviews to support pharmacovigilance-style insights on effectiveness, side effects, satisfaction, and general sentiment. The workflow moves from data cleaning and exploratory analysis to classical and transformer-based text classification, then extends into retrieval-augmented generation (RAG) so an analyst can query the review corpus and enrich answers with external medical information when needed.

The repository includes:

- EDA and preprocessing notebooks
- Classical text-classification experiments
- Transformer-based binary and ternary sentiment classification
- ChromaDB vector store construction from the review dataset
- A Streamlit sentiment classifier app
- A RAG + Tavily chatbot and agent-based executor


## Important Disclaimer

This repository, its notebooks, models, visualizations, chatbot outputs, and Streamlit applications are provided strictly for educational, research, and exploratory data-science purposes.
- This project does **not** provide medical advice, diagnosis, treatment recommendations, prescribing guidance, or individualized clinical judgment and should not be used as a substitute for consultation with a qualified healthcare professional.
- Any health concern, symptom, suspected adverse reaction, or question about a drug should be discussed with a qualified healthcare professional.
- This project is not medical software, a medical device, a clinical decision support system, or a regulated diagnostic product.
- Any use in healthcare, pharmacovigilance, regulatory, or patient-facing settings requires independent validation, governance review, and oversight by appropriately qualified professionals.
- The author and any contributors disclaim liability for any direct, indirect, incidental, consequential, special, exemplary, or punitive damages arising from use of, reliance on, or inability to use this project.
- Users are solely responsible for verifying outputs against authoritative sources and professional judgment before taking any action.

## Dataset

The project uses `DrugRev`, a drug review dataset with user's views of particular drugs. The dataser was obtained from [here](https://data.mendeley.com/datasets/64cc5w5dxy/1) and is licenced under CC BY 4.0 licence.
The original dataset has 392,510 rows and 9 columns centered on medicines, conditions and reviews related information (review text, length, rating and likes). The working dataset contains 386,137 rows and 16 columns after cleaning and feature engineering.               

### Main columns used

- `MedicineName`
- `MedicineFor`
- `Reviews`
- `ReviewLength`
- `Rating`
- `NumberOfLikes`
- `ReviewDate`
- `ReviewDay`
- `ReviewMonth`
- `ReviewYear`
- `Label`
- cleaned text variants such as `reviews_clean`, `reviews_no_stopwords`, and `reviews_lemmatized`

### Label definition

The project evaluates two approaches for sentiment labels:

**Ternary**

- `0` = Negative
- `1` = Neutral
- `2` = Positive

**Binary**

- `0`= Negative
- `1`= Non-negative

## Repository Structure

```text
drug-review-pharmacovigilance/
├── 1.cleaning_eda.ipynb
├── 2.classification.ipynb
├── 3.transformer_ternary_classification.ipynb
├── 4.transformer_binary_classification.ipynb
├── 5.build_chromadb.ipynb
└── apps/
│   └── sentiment_classifier.py
│   └── rag_tavily_agent_executor.py
│   └── rag_tavily_chatbot.py
├── README.md
└── requirements.txt
```

## Workflow

### 1. Data cleaning and EDA

The EDA notebook prepares the dataset, inspects the schema, and supports visual analysis of the most frequent drugs, conditions, review-length distribution, rating distribution, likes distribution, and rating patterns by drug and condition.

EDA goals:

- audit missing values and duplicates
- inspect class balance across the 3 sentiment classes
- analyze top medicines and top uses
- study `ReviewLength`, `Rating`, and `NumberOfLikes`
- compare rating distributions with violin plots

### 2. Classical NLP classification

The classification notebook covers non-transformer modelling for sentiment prediction. This stage is useful as a baseline before moving to deep transformer models.

The steps include:

- text cleaning and removal of stopwords
- vectorization with TF-IDF features
- train/test split
- model training and evaluation
- comparison of metrics such precision, recall, and F1 scores

Tested classical classifiers:
- Logistic Regression Classifier
- Logistic Regression Classifier with hyperparameter tuning
- SVM Classifier
- SVM Classifier with hyperparameter tuning
- Random Forest Classifier
- XGBoost Classifier

### 3. Transformer-based ternary classification

The transformer notebook fine-tunes a RoBERTa sentiment model for 3-class prediction. The implementation uses `cardiffnlp/twitter-roberta-base-sentiment-latest`, mapped to the project label scheme Negative, Neutral, and Positive.

The notebook includes:

- train/test split on review text and labels
- Hugging Face `Dataset` conversion
- tokenization with max length control
- `Pipeline` model
- `Trainer`-based fine-tuning
- evaluation with macro F1, weighted F1, recall, and confusion matrix
- model export for later inference in Streamlit

The fine-tuned setup achieved the best performance:

- F1 Macro: `0.820`
- F1 Weighted: `0.895`
- Class 0 Recall: `0.901`
- Class 0 F1: `0.896`

### 4. Transformer-based binary classification

The transformer notebook fine-tunes a BERT sentiment model for 2-class prediction. The implementation uses `dmis-lab/biobert-base-cased-v1.1`, mapped to the project label scheme Negative and Non-negative. The aim was to improve the detection of negative sentiment.

The notebook includes:

- train/test split on review text and labels
- Hugging Face `Dataset` conversion
- tokenization with max length control
- `Trainer`-based fine-tuning
- evaluation with macro F1, weighted F1, recall, and confusion matrix
- model export

The fine-tuned setup achieved the best performance:

- F1 Macro: `0.917`
- F1 Weighted: `0.925`
- Class 0 Recall: `0.900`
- Class 0 F1: `0.890`

### 5. Sentiment classifier app

The `sentiment_classifier.py` script provides a Streamlit interface for single-review inference using the saved transformer model.

Current app features:

- load tokenizer and model from `model`
- accept free-text user input
- predict Negative, Neutral, or Positive sentiment
- display class confidence
- color-code the result for usability

### 6. ChromaDB construction

The ChromaDB notebook converts dataset rows into LangChain `Document` objects and persists a local vector database using `sentence-transformers/all-MiniLM-L6-v2` embeddings.

This enables semantic retrieval over the local review corpus for downstream RAG applications.

### 7. RAG + Tavily chatbot

The chatbot layer lets an analyst ask natural-language questions about the review corpus. It first queries the local ChromaDB and can then enrich answers with Tavily web search when broader medical context is required.

The implementation is split across:

- `rag_tavily_agent_executor.py`: builds the agent executor, tools, prompt, and routing logic
- `rag_tavily_chatbot.py`: Streamlit chat interface for interactive use

Key chatbot features:

- local-first retrieval from the review vector database
- Tavily fallback for external medical information
- ReAct-style agent loop with tool use
- chatbot memory for continuous interactions
- extracted source list for Tavily-supported answers

## Model and App Details

### Transformer models

The transformer workflow uses:

- `AutoTokenizer`
- `AutoModelForSequenceClassification`
- `TrainingArguments`
- `Trainer`
- `DataCollatorWithPadding`

The model cards used in the notebook are:

- `cardiffnlp/twitter-roberta-base-sentiment-latest`
- `dmis-lab/biobert-base-cased-v1.1`

The label mapping used in both training and inference is:

- roBERTa
```python
id2label = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}
```
- bioBERT
```python
id2label = {
    0: "Negative",
    1: "Non-negative",
}
```

### RAG agent tools

The RAG agent uses two main tools:

1. `DrugReviewRetriever`
   - searches the local Chroma database built from `DrugReviews.csv`
   - best for patient-review questions, side effects, effectiveness, and satisfaction patterns

2. `TavilySafeSearch`
   - performs external web search
   - trims the result to stay within token limits
   - adds a short source list for traceability

## Installation

### 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 2. Install dependencies

`requirements.txt`

```txt
streamlit
langchain-openai
langchain-community
langchain-chroma
langchain-classic
chromadb
sentence-transformers
tavily-python
python-dotenv
tiktoken
```

Run:

```bash
pip install -r requirements.txt
```

## How to Run

### 1. Run the Streamlit sentiment classifier

```bash
streamlit run sentiment_classifier.py
```

### 2. Build the Chroma database

Open and run:

```text
4.build_chromadb.ipynb
```

This creates the local vector store in:

```text
drug-review-chromadb/
```

### 3. Run the RAG + Tavily chatbot

Before running, set environment variables for API access if needed in a `.env` file:

```bash
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key
```

Then launch:

```bash
streamlit run rag_tavily_chatbot.py
```

## Example Questions for the Chatbot

- What do patient reviews say about sertraline for anxiety?
- What side effects are most commonly mentioned for lexapro?
- Summarize patient sentiment for phentermine.
- Compare what the reviews say about a drug with broader medical information from the web.

## Key Results

**Classifier comparison**

| Classifier | F1 macro | F1 weighted | Class 0 recall score | Class 0 F1 score |
| ------ | ------ | ------ | ------ | ------ |
| XGBoost| 0.399 | 0.580 | 0.295 | 0.425 |
| Random Forest | 0.774 | 0.821 | 0.790 | 0.812 |
| Logistic Regression | 0.765 | 0.823 | 0.806 | 0.798 |
| SVM | 0.819 | 0.867 | 0.833 | 0.846 |
| Logistic Regression with hyperparameter tuning | 0.816 | 0.866 | 0.831 | 0.845 |
| SVM with hyperparameter tuning | 0.813 | 0.866 | 0.838 | 0.846 |
| Pipeline twitter-roberta-base-sentiment-latest (ternary) | 0.460 | 0.560 | 0.780 | 0.650 |
| Trainer Class twitter-roberta-base-sentiment-latest (ternary) | 0.820 | 0.895 | 0.901 | 0.896 |
| Trainer Class biobert-base-cased-v1.1 (binary) | 0.917 | 0.925 | 0.900 | 0.890 |


### Fine-tuned transformer performance

The fine-tuned trainer versions improves materially and reaches about:

**Ternary classification**

| Metric | Value |
|---|---:|
| F1 Macro | 0.820 |
| F1 Weighted | 0.895 |
| Class 0 Recall | 0.901 |
| Class 0 F1 | 0.896 |


**Binary classification**

| Metric | Value |
|---|---:|
| F1 Macro | 0.917 |
| F1 Weighted | 0.925 |
| Class 0 Recall | 0.900 |
| Class 0 F1 | 0.890 |


These results suggest the fine-tuned transformer is the strongest classification component currently implemented in the repository.

## Future Improvements

Possible extensions include:

- batch inference in the Streamlit sentiment app (eg. from CSV file)
- aspect-based sentiment analysis for side effects, effectiveness, and ease of use
- SHAP-based text explanation for the transformer model
- use a proxy-label method, such as multi-view training, for labeling
- stronger metadata-aware Chroma retrieval by drug and condition
- deployment of the chatbot and classifier apps
- stricter evaluation of hallucination and retrieval quality in the RAG layer
- create a social media scraper to monitor drug reviews

## Notes and Limitations

- The RAG system is intended as an analytical support tool, not a medical diagnosis system.
- External web search should be treated as supporting context, not as personalized medical advice.
- The ChromaDB notebook currently stores each row as a document; future chunking and metadata filtering could improve retrieval quality.
- The Streamlit classifier assumes the saved transformer model path exists and matches the training label order.

## Acknowledgements

This project combines NLP, deep learning, vector retrieval, and agentic search to analyze patient-generated drug reviews in a pharmacovigilance-oriented workflow.