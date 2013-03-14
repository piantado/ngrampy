

# Run on all 
DIR=/home/piantado/Desktop/mit/Corpora/Web1T5gramEuropean/data
Ngms=3gms

for LANGUAGE in CZECH DUTCH FRENCH GERMAN ITALIAN POLISH PORTUGUESE ROMANIAN SPANISH SWEDISH;
do
	echo Running $LANGUAGE
	python compute_avg_surprisal.py --in=$DIR/$LANGUAGE/$Ngms/ --path=/tmp/$LANGUAGE > Surprisal/$LANGUAGE.txt
done

