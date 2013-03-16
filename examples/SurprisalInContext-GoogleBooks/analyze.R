
# A script for analyzing the results of run-all.sh, which will populate the Surprisal directory
# This will compute all of the correlations. 

D <- NULL
for(L in c("FRENCH","GERMAN","ITALIAN","POLISH","PORTUGUESE","ROMANIAN","SPANISH","SWEDISH")) {

	d <- read.table(paste("Surprisal/",L,".txt", sep=""), skip=20)
	names(d) <- c("word", "length", "surprisal.3", "log.frequency", "ccnt")
	v <- read.table(paste("Extract_Vocabularies/Vocabularies/",L,".txt",sep=""))
	v <- v$V1 # make this just a list
	
	# analyze the top 25k, since there is lots of garbage in google
	v <- v[is.element(as.character(v), as.character(d$word))] # keep only things in v and in d
	v <- v[1:25000] # these are already trimmed to 25k
	d <- d[is.element(as.character(d$word),as.character(v)),] # and trim d according to top 25k in v
	
	# Very simple--just nonparametric correlations
	# NOTE: Email Steve for fancier scripts and analysis (partials, bootstrapping, etc.)
	sc <- cor.test(d$surprisal.3, d$length, method="spearman")
	fc <- cor.test(-d$log.frequency, d$length, method="spearman")  ## Negative log freq here so that its on the same scale (we didn't normalize freq--that's slower

	D <- rbind(D, data.frame( Language=L,
				  surprisal.cor=sc$estimate,
				  frequency.cor=fc$estimate,
				  surprsial.p.value=sc$p.value,
				  frequency.p.value=fc$p.value))
}

print(D)
