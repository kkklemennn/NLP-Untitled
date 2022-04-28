#!/bin/bash

for file in "./entities_tsv"/*.tsv; do
	#echo $file
	powershell.exe "python -m spacy convert $file -c ner -n 10  ./entities_spacy"

done