import pickle
import pandas as pd
import os
# import logging
from sklearn.metrics.pairwise import cosine_similarity
import openai
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

with open('data.pkl', 'rb') as file:
    df = pickle.load(file)

# response = ''' [{"query":"what is Readyly ?"}]'''


@app.route('/process_query', methods=['GET'])
def process_query():
    # print("response", response)
    # app.logger.info('Processing default request')
    # query = response.json()['query']
   # query = request.json['query']
    # query = "What is Readyly ?"
    # query = "How fast is Readyly ?"
    query = response.json()['query']
  #  os.environ["OPENAI_API_KEY"] = "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"
    #openai.api_key = "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"
    embedding_model = "text-embedding-ada-002"
    embedding_encoding = "cl100k_base"
    max_tokens = 1000

    def get_embedding(text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']

    query_embedding = get_embedding(query)

    def find_most_similar(query_embedding):
        query_embedding = np.array(query_embedding)
        similarities = cosine_similarity(
            np.vstack(df['embedding']), query_embedding.reshape(1, -1))
        most_similar_index = similarities.argmax()
        return most_similar_index

    most_similar_index = find_most_similar(query_embedding)

    most_similar_row = df.iloc[most_similar_index]

    article_content = most_similar_row['context']

    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=article_content,
        max_tokens=100,
        temperature=0.7
    )

    generated_response = response.choices[0].text.strip()

    return jsonify({'query': query, 'response': generated_response})


if __name__ == '__main__':
    app.run()
