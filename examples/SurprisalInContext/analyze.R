
# A script for analyzing the results of run-all.sh, which will populate the Surprisal directory
# In the original work, we used Opensubtlex to define vocabularies, but now for simplicity
# Let's just use the most frequent strings

D <- NULL
for(Y in c("1500", "1525", "1550")) { #c("1800", "1825", "1850", "1875", "1900")) {
for(L in c("eng-gb-2", "eng-us-2")){

	d <- read.table(paste("Surprisal/",L,".", Y, ".txt", sep=""), header=T)
	
	d <- d[order(d$Log.Frequency, decreasing=T), ]
	d <- d[ 1:5000, ] # keep only the top 25k
	
	# Very simple--just nonparametric correlations
	# NOTE: Email Steve for fancier scripts and analysis (partials, bootstrapping, etc.)
	sc <- cor.test(d$Surprisal, d$Orthographic.Length, method="spearman")
	fc <- cor.test(-d$Log.Frequency, d$Orthographic.Length, method="spearman")  ## Negative log freq here so that its on the same scale (we didn't normalize freq--that's slower

	D <- rbind(D, data.frame( Language=L,
				   Year=Y,
				  Surprisal.cor=sc$estimate,
				  Frequency.cor=fc$estimate,
				  Surprsial.p.value=sc$p.value,
				  Frequency.p.value=fc$p.value))
}
}

print(D)
