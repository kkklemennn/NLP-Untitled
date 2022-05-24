import sys
import numpy as np
import json
import os
import timeit

from named_entity_extraciton import get_book_string, get_common_entities_stanza, get_common_entities_spacy

short_stories_path = "../material/short_stories_corpus/"
medium_stories_path = "../material/medium_stories_corpus/"
litbank_stories_path = "../material/litbank_corpus/"

def get_performance(model, stories_path):
    files = os.listdir(stories_path)
    performance_model = {}
    for file in files:
        # print(file)
        performance = {}
        novel = get_book_string(stories_path + file)
        start = timeit.default_timer()
        if model == 'stanza':
            entities = get_common_entities_stanza(novel)
        elif model == 'spacy':
            entities = get_common_entities_spacy(novel)
        else:
            print("Enter either stanza or spacy")
            exit()
        occurences = {}
        for entity in entities:
            occurences[entity] = novel.lower().count(entity)
        stop = timeit.default_timer()
        performance['entities'] = occurences
        performance['runtime'] = stop - start
        performance_model[file] = performance
        # print(performance)
        # break
    return performance_model

def generate_performance_results(stories_path):
    files = os.listdir(stories_path)
    current_model = stories_path.split("/")[-2].split("_")[-0]
    performance_stanza = get_performance('stanza')
    performance_spacy = get_performance('spacy')
    with open("../results/performance_stanza_" + current_model + ".json", "w+", encoding="utf-8") as outfile:
       json.dump(performance_stanza, outfile, indent=4, ensure_ascii=False)
    with open("../results/performance_spacy_" + current_model + ".json", "w+", encoding="utf-8") as outfile:
        json.dump(performance_spacy, outfile, indent=4, ensure_ascii=False)

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
    
    if (sys.argv[1] == "litbank" or sys.argv[1] == "short" or sys.argv[1] == "long"):
        mode = sys.argv[1]
        spacy_data, stanza_data = read_data(mode)
        print(mode + " stories analysis")
        compare_running_time(spacy_data, stanza_data)
        compare_entities(spacy_data, stanza_data)
    
    elif (sys.argv[1] == "generate"):
        # Generate JSON results for each corpus
        generate_performance_results(short_stories_path)
        generate_performance_results(medium_stories_path)
        generate_performance_results(litbank_stories_path)
    
    else:
        print("Invalid argument.")
        print("Please input one of the following corpora to analyze performance:")
        print('"short"')
        print('"medium"')
        print('"litbank"')
        print("To generate results, input argument:")
        print('"generate"')