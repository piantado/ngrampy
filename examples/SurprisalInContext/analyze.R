
# A script for analyzing the results of run-all.sh, which will populate the Surprisal directory
# In the original work, we used Opensubtlex to define vocabularies, but now for simplicity
# Let's just use the most frequent strings

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Some handy functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

stdize <- function(x, ...) { (x - mean(x,...)) / sd(x, ...) }

sort.by.frequency <- function(d) { d[order(d$Log.Frequency, decreasing=T),] }

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define the vocabulary -- take the most frequent words in some year
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VOCAB <- as.character(sort.by.frequency(read.table("Surprisal/eng-us-2.1950.txt", header=T))[1:25000,"Word"])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Now analyze:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

D <- NULL
for(Y in c("1900", "1925", "1950", "1975", "2000")){#"1500", "1525", "1550", "1600", "1625", "1650", "1675", "1700", "1725", "1750", "1775", "1800")) { 
for(L in c("eng-gb-2", "eng-us-2")){

	d <- read.table(paste("Surprisal/", L, ".", Y, ".txt", sep=""), header=T)
	d <- d[is.element(d$Word, VOCAB),]
	
	# Very simple--just nonparametric correlations
	# NOTE: Email Steve for fancier scripts and analysis (partials, bootstrapping, etc.)
	sc <- cor.test(d$Surprisal, d$Orthographic.Length, method="spearman")
	fc <- cor.test(-d$Log.Frequency, d$Orthographic.Length, method="spearman")  ## Negative log freq here so that its on the same scale (we didn't normalize freq)

	l <- lm( stdize(Orthographic.Length) ~ stdize(Surprisal), data=d)

	D <- rbind(D, data.frame( Language=L,
				   Year=Y,
				   Surprisal.cor=sc$estimate,
				   Frequency.cor=fc$estimate,
#				   Surprisal.p.value=sc$p.value,
#				   Frequency.p.value=fc$p.value, 
				   mean.surprisal=mean(d$Surprisal),
       				   sd.surprisal=sd(d$Surprisal),
				   lm.icpt=coef(l)[1],
				   lm.slope=coef(l)[2],
				   mean.fw.surprisal=weighted.mean(d$Surprisal, exp(d$Log.Frequency)),
				   total.freq=sum(2.0**d$Log.Frequency)
			))
}
}

print(D)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Build the monster data frame
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 
# D <- NULL
# for(Y in c("1500", "1525", "1550", "1600", "1625", "1650", "1675", "1700", "1725", "1750", "1775", "1800")) { 
# for(L in c("eng-us-2")){
# 
# 	d <- read.table(paste("Surprisal/", L, ".", Y, ".txt", sep=""), header=T)
# 	d <- d[is.element(d$Word, VOCAB),]
# 	d$Total.Log.Frequency <- log(sum(2.0**d$Log.Frequency)) # TODO:Logsumexp
# 	d$Year <- as.numeric(Y)
# 	d$Language <- L
# 	
# 	D <- rbind(D, d)
# }
# }
