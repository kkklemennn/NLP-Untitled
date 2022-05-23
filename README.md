# Literacy situation models knowledge base creation

Literacy situation models knowledge base creation from english short fiction stories for Natural Language Processing class @ Faculty of Computer and Information Science, University of Ljubljana

## Repository structure

### Code

This folder contains all the necessary code to reproduce the results obtained in our report.\

- **train_ner_spacy_litbank** folder contains the neccesary files to convert the annnoteted data from the LitBank corpus to Spacy format and train the NER classifier.
- **character_sentiment_analysis.py** contains code to perform sentiment analysis for provided list of fiction characters
- **character_sentiment_aspect_experimental.py** contains code we used to perform Part-Of-Speech tagging and Dependency Parsing aspect based sentiment analysis, but without good results
- **named_entity_extraction.py** contains code used to read the _.txt_ books and perform basic text preprocessing and character extraction. This file currently contains the character co-occurence calculation and co-occurence visualization.
- **performance_analysis.py** contains code used to run tests of different models and methods for character extraction and sentiment analysis, Example `python performance_analysis.py litbank`

### Material

- Contains three different corpora
  - **litbank_corpus** folder contains 100 medium fiction stories in .txt format from the LitBank corpus
  - **medium_stories_corpus** folder contains 7 medium fiction stories in .txt format
  - **short_stories_corpus** folder contains 82 short fiction stories in .txt format

### Results

- JSON files containing the results of our analysis

### Report

- PDF version of our report

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install missing dependencies.

```bash
pip install -r requirements.txt
```

## Usage

```python
import foobar

# returns 'words'
foobar.pluralize('word')

# returns 'geese'
foobar.pluralize('goose')

# returns 'phenomenon'
foobar.singularize('phenomena')
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
