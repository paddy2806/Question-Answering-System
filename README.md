# RNN-Based Question Answering System

## Project Overview

This project demonstrates a simple question answering (QA) system using a Recurrent Neural Network (RNN) with PyTorch. It leverages a dataset of question-answer pairs and trains a neural network to predict concise answers to factual questions by modeling the relationship between question text and answer.

## Features

- Custom tokenizer for text preprocessing (lowercasing, punctuation removal, and splitting)
- Vocabulary builder for mapping unique words to indices, including robust handling of unknown tokens
- PyTorch `Dataset` and `DataLoader` abstractions for efficient batching and data management
- RNN model with embedding, RNN, and fully connected output layer for text understanding
- Training loop with per-epoch reporting of cross-entropy loss using the Adam optimizer
- Prediction function capable of confidence thresholding, outputting “I don’t know” for low-confidence queries

## Dataset

The dataset includes 100 unique question-answer pairs in CSV format with `question` and `answer` columns.

**Example rows:**

| Question                                    | Answer      |
|----------------------------------------------|-------------|
| What is the capital of France?               | Paris       |
| Who wrote 'To Kill a Mockingbird'?           | Harper-Lee  |
| What is the boiling point of water in Celsius? | 100         |

## Usage

1. Ensure your `100_Unique_QA_Dataset.csv` file with question-answer pairs is in your project directory.
2. Run the project notebook or script to:
    - Load and preprocess data
    - Build vocabulary
    - Train the RNN model (20 epochs by default with loss reported each epoch)
    - Predict answers for new, simple text queries

**Example for running prediction in code:**
```python
predict(model, "What is the largest planet in our solar system?")
# Expected output: jupiter
```

## Code Structure

- Data loading and preprocessing (tokenization, vocab mapping, text indexing)
- Model definition: 
    - Embedding layer (dimension: 50)
    - RNN layer (hidden size: 64)
    - Fully connected output for answer classification
- Training loop: batch data, compute loss, backpropagation, model update per epoch
- Prediction logic: converts text to indices, gets model output, returns predicted word if confident

## Model Architecture

| Layer            | Description                                   |
|------------------|-----------------------------------------------|
| Embedding        | 50-dimensional dense vectors per token         |
| RNN              | Single layer with 64 hidden units (batch-first mode) |
| Fully Connected  | Linear layer mapping hidden vector to vocab size|

## Results

- Training loss consistently reduced from 500+ to ~11 in 20 epochs, indicating the model learned to map questions to answers.
- Can predict correct answers for questions similar to those in the dataset.
- Includes a threshold to output “I don’t know” when the model lacks confidence.

## Future Improvements

- Use more advanced sequence models (LSTM or GRU)
- Expand dataset for better generalization
- Enable prediction of full-sentence answers
- Add quantitative evaluation metrics (accuracy, etc.)
- Integrate results/stats storage, potentially leveraging PostgreSQL for scalable logging and retrieval, in line with database habits on Mac/pgAdmin 4.
