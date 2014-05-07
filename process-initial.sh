

# A script to initially process google data. The significantly speeds up everything later.

DIR=/home/piantado/Corpus/GoogleBooks/
#for L in eng-us-all fre-all heb-all ita-all rus-all spa-all ger-all; do
for L in eng-us-all fre-all heb-all; do
for N in 1 2 3; do
	myDIR=$DIR/Processed/$L/$N

	mkdir $DIR/Processed/$L
	mkdir $myDIR

	pypy process-google.py --in=$DIR/$L/$N/* --out=$myDIR/processed-google --N=$N --quiet & ## TODO: IF YOU CHANGE N, CHANGE THE 
done
done

