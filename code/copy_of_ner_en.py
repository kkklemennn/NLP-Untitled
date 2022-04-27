# # Install PySpark and Spark NLP
# ! pip install -q pyspark==3.1.2 spark-nlp

# # Install Spark NLP Display lib
# ! pip install --upgrade -q spark-nlp-display

# """## 2. Start the Spark session

# Import dependencies and start Spark session.
# """

import json
import pandas as pd
import numpy as np
import sys
import os


os.environ['HADOOP_HOME'] = "S:\spark-3.1.2-bin-hadoop3.2"
#os.environ['HADOOP_HOME'] = "S:\spark-3.2.1-bin-hadoop3.2"
#os.environ['HADOOP_HOME'] = "C:/Mine/Spark/hadoop-2.6.0"
sys.path.append("S:/spark-3.1.2-bin-hadoop3.2/bin")
#sys.path.append("C:/Mine/Spark/hadoop-2.6.0/bin")

from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from sparknlp.annotator import *
from sparknlp.base import *
import sparknlp
from sparknlp.pretrained import PretrainedPipeline

spark = sparknlp.start()

"""## 3. Select the DL model"""

# If you change the model, re-run all the cells below
# Other applicable models: ner_dl, ner_dl_bert
MODEL_NAME = "onto_100"
#MODEL_NAME = "ner_dl"
#MODEL_NAME = "ner_dl_bert"

# ner_dl and onto_100 model are trained with glove_100d, so the embeddings in
  # the pipeline should match
if (MODEL_NAME == "ner_dl") or (MODEL_NAME == "onto_100"):
    embeddings = WordEmbeddingsModel.pretrained('glove_100d') \
        .setInputCols(["document", 'token']) \
        .setOutputCol("embeddings")

# Bert model uses Bert embeddings
elif MODEL_NAME == "ner_dl_bert":
    embeddings = BertEmbeddings.pretrained(name='bert_base_cased', lang='en') \
        .setInputCols(['document', 'token']) \
        .setOutputCol('embeddings')

ner_model = NerDLModel.pretrained(MODEL_NAME, 'en') \
      .setInputCols(['document', 'token', 'embeddings']) \
      .setOutputCol('ner')

"""## 4. Some sample examples"""

import glob
from sparknlp_display import NerVisualizer
i = 0
empty_dict_ct = 0
if MODEL_NAME == "onto_100": # short stories - 26 without entities  
  for file in glob.iglob('../material/medium_stories/*.txt'): 
    print(file)
    f = open(file, "r")
    text_list = [f.read()]

    documentAssembler = DocumentAssembler() \
      .setInputCol('text') \
      .setOutputCol('document')

    tokenizer = Tokenizer() \
        .setInputCols(['document']) \
        .setOutputCol('token')

    ner_converter = NerConverter() \
        .setInputCols(['document', 'token', 'ner']) \
        .setOutputCol('ner_chunk')

    nlp_pipeline = Pipeline(stages=[
        documentAssembler, 
        tokenizer,
        embeddings,
        ner_model,
        ner_converter
    ])
    
    empty_df = spark.createDataFrame([['']]).toDF('text')
    pipeline_model = nlp_pipeline.fit(empty_df)
    df = spark.createDataFrame(pd.DataFrame({'text': text_list}))
    result = pipeline_model.transform(df)


    rows = result.select("ner").collect()[0]
    #entities = set()
    entities_dict = {}

    for row in rows[0]: # BERT - without rows[0]
      #print(row)
      #print(row.result)
      if(row.result.endswith('PERSON')):# BERT - PER
        if(row.metadata["word"] not in entities_dict):
          entities_dict[row.metadata["word"]] = 1
        else:
          entities_dict[row.metadata["word"]] += 1
          #print(row.metadata["word"])
          #entities.add(row.metadata["word"])

    #print(entities_dict)
    curr_entities = (dict((k, v) for k, v in entities_dict.items() if v >= 2))
    if not curr_entities:
      empty_dict_ct += 1
    #print(dict((k, v) for k, v in entities_dict.items() if v >= 2))

    #print(result.select("ner").collect())
    NerVisualizer().display(
        result = result.collect()[0],
        label_col = 'ner_chunk',
        document_col = 'document'
    )

    i = i + 1
    if i>0: break

  print(empty_dict_ct)
if MODEL_NAME == "ner_dl_bert": # short stories - 12 without entities
  for file in glob.iglob('../material/medium_stories/*.txt'): #('./*.txt')
    
    #file = "/content/sample_data/medium_stories/The Most Dangerous Game_modified.txt"
    print(file)
    f = open(file, "r")
    text_list = [f.read()]

    documentAssembler = DocumentAssembler() \
      .setInputCol('text') \
      .setOutputCol('document')

    tokenizer = Tokenizer() \
        .setInputCols(['document']) \
        .setOutputCol('token')

    ner_converter = NerConverter() \
        .setInputCols(['document', 'token', 'ner']) \
        .setOutputCol('ner_chunk')

    nlp_pipeline = Pipeline(stages=[
        documentAssembler, 
        tokenizer,
        embeddings,
        ner_model,
        ner_converter
    ])
    
    empty_df = spark.createDataFrame([['']]).toDF('text')
    pipeline_model = nlp_pipeline.fit(empty_df)
    df = spark.createDataFrame(pd.DataFrame({'text': text_list}))
    result = pipeline_model.transform(df)


    rows = result.select("ner").collect()[0]
    #entities = set()
    entities_dict = {}

    for row in rows[0]: # BERT - without rows[0]
      #print(row)
      #print(row.result)
      if(row.result.endswith('PER') or row.result.endswith('MISC') or row.result.endswith('ORG')):# BERT - PER
        if(row.metadata["word"] not in entities_dict):
          entities_dict[row.metadata["word"]] = 1
        else:
          entities_dict[row.metadata["word"]] += 1
          #print(row.metadata["word"])
          #entities.add(row.metadata["word"])

    #print(entities_dict)
    curr_entities = (dict((k, v) for k, v in entities_dict.items() if v >= 2))
    if not curr_entities:
      empty_dict_ct += 1
    print(dict((k, v) for k, v in entities_dict.items() if v >= 2))

    print(result.select("ner").collect())
    NerVisualizer().display(
        result = result.collect()[0],
        label_col = 'ner_chunk',
        document_col = 'document'
    )

    i = i + 1
    if i>0: break

  print(empty_dict_ct)