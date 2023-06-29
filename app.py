import numpy as np
import pandas as pd
import pickle
# json_data = '''
# [
#   {
#     "article_id": 1,
#     "title": "what is readyly ?",
#     "content": "Readyly provides customer support AI-as-a-Service to mid-market companies. Our comprehensive solution reduces support costs by up to 10X, while automating up to 100 of Tier 1 and Tier 2 customer support workflows.",
#     "org_id": 12345
#   },
#   {
#     "article_id": 2,
#     "title": "what services does it offer?",
#     "content": We build, scale and manage digitized customer support solutions for fast-growth companies using the latest AI and automation technologies.  We call it Customer Support AI-as-a-Service because we do all the heavy lifting when it comes to making sure your team and system are operating at maximum efficiency and effectiveness.  And we do it affordably. ",
#     "org_id": 67890
#   },
#   {
#     "article_id": 3,
#     "title": "how quick is readyly ? ",
#     "content": "Readyly installs in under two minutes. Our customers start receiving insights in as little as 5 minutes. Our virtual agent and AI-powered help center is set up within 24 hours and provides value out of the box. In as little as 7 days, Readyly can take over your Tier 1 and Tier 2 tickets, automating them up to 100%. ",
#     "org_id": 54321
#   }
# ]
# '''

json_data = '''
[
  {
    "article_id": 1,
    "title": "what is Readyly?",
    "content": "Readyly provides customer support AI-as-a-Service to mid-market companies. Our comprehensive solution reduces support costs by up to 10X faster than tier1 and tier 2 comapny.Our proprietary algorithms are over 90 percent accurate at data tagging and data mapping, over 6x more accurate than Zendesk at search, and can cut a customer's support costs by over 50 percent in as little as 7 days. Our founding team has a strong background in AI and customer support technology, including McKinsey and Palantir experience, as well as advisors from leading AI and customer support technology companies.",
    "org_id": 12345
  },
  {
    "article_id": 2,
    "title": "how fast is Readyly ?",
    "content": "Our virtual agent and AI-powered help center is set up within 24 hours and provides value out of the box. In as little as 7 days, Readyly can take over your Tier 1 and Tier 2 tickets, automating them up to 100%. ",
    "org_id": 67890
  },
  {
    "article_id": 3,
    "title": "how Readyly survive competiton ?",
    "content": "Readyly is the first platform built for fast-growth companies that builds, scales, and maintains AI and advanced automations for you, enabled by our Readyly technology suite. Companies will spend up to 98 percent of their customer support budget on labor.  In-house teams are incredibly expensive and outsourcing agencies use limited technology (or none at all) to manage their resources, leading to high overhead costs. They pass these added costs onto you, the customer.One of the biggest sources of waste is manual workflows, typically 50 percent + of all the customer support labor spend, which require agents to take multiple steps, often logging into one or more systems, in order to accomplish a task for customers. The best way to reduce or even eliminate this waste is to build advanced automations that replace these manual steps.",
    "org_id": 54321
  }
  ,
    {
    "article_id": 4,
    "title": "what are the benefits of using  Readyly ?",
    "content": "An AI-powered help center that responds to customers with 6x the accuracy of leading ticketing solutions. An integrated NPS-based auto-survey solution captures customer voice in real-time and eliminates the need for data reconciliation from third-party survey systems.A robust analytics module that identifies and removes customer pain points.",
    "org_id": 54328
  }

]
'''
# cloud-based software platform benefits include: 

#An AI-powered help center that responds to customers with 6x the accuracy of leading ticketing solutions. 



df = pd.read_json(json_data, orient='records')
print(df)
articles = df['content'].tolist()

# model_name = 'bert-base-uncased'
# from transformers import BertTokenizer, TFBertModel
# tokenizer = BertTokenizer.from_pretrained(model_name)
# bert_model = TFBertModel.from_pretrained(model_name)

# import tensorflow_text as text
# import tensorflow as tf
# import tensorflow_hub as hub
# bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
# bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")


# def get_sentence_embeding(sentences):
#     preprocessed_text = bert_preprocess(sentences)
#     return bert_encoder(preprocessed_text)['pooled_output']

# get_sentence_embeding([
#     "500$ discount. hurry up", 
#     "Bhavin, are you up for a volleybal game tomorrow?"]
# )
# train_embeddings =  get_sentence_embeding(articles)\
df['context'] = df['title']+df['content']
import os
import  openai
#os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = os.environ["OPENAI_API_KEY"]
import  openai
#openai.api_key = "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"
from openai.embeddings_utils import get_embedding
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 1000  # the maximum for text-embedding-ada-002 is 8191
df["embedding"] = df.context.apply(lambda x: get_embedding(x, engine=embedding_model))
print(df)
import pickle
file_path = 'data.pkl'

# Dump the DataFrame as a pickle file
with open(file_path, 'wb') as file:
    pickle.dump(df, file)
