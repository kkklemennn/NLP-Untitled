import pandas as pd
import numpy as np
import json

def read_json(path):
    f = open(path,"r",encoding='utf-8')
    data = json.load(f)
    return data

def read_data(input):
    if (input == 'short'):
        spacy_data = read_json('../results/performance_spacy_short.json')
        stanza_data = read_json('../results/performance_stanza_short.json')
    elif (input == 'medium'):
        spacy_data = read_json('../results/performance_spacy_medium.json')
        stanza_data = read_json('../results/performance_stanza_medium.json')
    elif (input == 'litbank'):
        spacy_data = read_json('../results/performance_spacy_litbank.json')
        stanza_data = read_json('../results/performance_stanza_litbank.json')
    else:
        print("Input should be 'short', 'medium' or 'litbank'")
        exit()
    return spacy_data, stanza_data

def compare_running_time(spacy_data, stanza_data):
    
    rt_spacy = []
    rt_stanza = []

    for story in spacy_data:
        rt_spacy.append(float(spacy_data[story]['runtime']))
        rt_stanza.append(float(stanza_data[story]['runtime']))
    print("Average runtime")
    print("Spacy:", np.mean(rt_spacy))
    print("Stanza:", np.mean(rt_stanza))

def compare_entities(spacy_data, stanza_data):
    score_spacy = 0
    score_stanza = 0
    equal = 0
    equald = 0
    stanza0 = 0
    spacy0 = 0
    for story in spacy_data:
        spacy_entities = list(spacy_data[story]['entities'].keys())
        stanza_entities = list(stanza_data[story]['entities'].keys())
        if len(stanza_entities) == 0:
            stanza0 += 1
        if len(spacy_entities) == 0:
            spacy0 += 1
        if len(spacy_entities) == len(stanza_entities)  and len(stanza_entities) > 0:
            if list(set(spacy_entities) & set(stanza_entities)):
                equal += 1
            else:
                equald += 1
        elif len(spacy_entities) > len(stanza_entities):
            score_spacy += 1
        else: score_stanza += 1
    print("Equal:", equal)
    print("Equal, different output:", equald)
    print("Stanza recognized more entities:", score_stanza)
    print("Spacy recognized more entities:", score_spacy)
    print("Stanza did not recognize any entity:", stanza0)
    print("Spacy did not recognize any entity:", spacy0)


if __name__ == "__main__":
    mode = 'litbank'
    spacy_data, stanza_data = read_data(mode)
    print(mode + " stories")
    compare_running_time(spacy_data, stanza_data)
    compare_entities(spacy_data, stanza_data)