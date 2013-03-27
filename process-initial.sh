

# A script to initially process google data. The significantly speeds up everything later.

nohup pigz -dc /CorpusA/GoogleBooks/eng-gb-all/2/* | python process-google.py > /CorpusA/GoogleBooks/Processed/eng-gb-2 &
nohup pigz -dc /CorpusA/GoogleBooks/eng-us-all/2/* | python process-google.py > /CorpusA/GoogleBooks/Processed/eng-us-2 &

