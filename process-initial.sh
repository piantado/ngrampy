

# A script to initially process google data. The significantly speeds up everything later.

DIR=/home/piantado/Corpus/GoogleBooks/
for L in eng-us-all fre-all heb-all ita-all rus-all spa-all ger-all; do
	myDIR=$DIR/Processed/bigram/$L
	mkdir $myDIR
	pypy process-google.py --in=$DIR/$L/2/* --out=$myDIR/bigram --quiet & ## TODO: IF YOU CHANGE N, CHANGE THE 
done

