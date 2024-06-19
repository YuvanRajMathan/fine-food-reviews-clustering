Text-Embedding-and-Clustering

Text Embedding and Clustering with OpenAI's GPT Models

This repository contains a Python project that uses OpenAI's API to generate embeddings for text data, cluster the embeddings using K-means clustering, and visualize the clusters using t-SNE dimensionality reduction. The project also includes text cleaning and preprocessing functions.
Features

  Text Cleaning and Preprocessing: Cleans and preprocesses text data.
  Text Embedding: Generates text embeddings using OpenAI's API.
  Clustering: Performs K-means clustering on the embeddings.
  Visualization: Visualizes the clusters in 2D using t-SNE.
  Theme Extraction: Extracts themes from the clusters using OpenAI's API.

Dataset

The dataset used in this project contains fine food reviews along with their embeddings. The dataset is stored in the file fine_food_reviews_with_embeddings.csv.
Requirements

    Python 3.x
    pandas
    scikit-learn
    matplotlib
    numpy
    openai
Installation

Clone the repository:

bash

git clone https://github.com/your_username/your_repository.git
cd your_repository

Install the required dependencies:

bash

    pip install pandas scikit-learn matplotlib numpy openai

Usage

  Set up your OpenAI API key:
  Create a .env file in the root directory of the project.
  Add your OpenAI API key to the .env file:

  makefile

    OPENAI_API_KEY=your-api-key

Run the main script:

bash

    python your_main_script.py
