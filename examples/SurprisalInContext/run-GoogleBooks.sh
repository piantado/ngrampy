
# Pass in the google directory with each file you'd like to process

# > bash run-GoogleBooks.sh /CorpusA/GoogleBooks/Processed/eng-us-2/*

# This then computes, stores the google archive to Archive/xxx.7z and surprisal to Surprisal

for f in $@
do
	x=$(basename $f) # the base file name
	python surprisal-2gram.py --in=$f --path=/tmp/$x.google > Surprisal/$x.txt
	7z a -mx=9 Archive/$x.7z /tmp/$x.google && rm /tmp/$x.google & # run in background since its slow
done