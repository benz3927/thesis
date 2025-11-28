rm(list=ls()); cat("\014")
suppressMessages(source("/if/appl/R/Functions/IFfunctions.r"))
#-----------Sentiment Analysis-------------#

#install.packages("SentimentAnalysis")
library(SentimentAnalysis)
data(DictionaryLM)

#Bring in data
#data <- read.csv("fomc_transcript/data/processed/sets/owen_initial.csv")
data <- read.csv('fomc_transcript/data/processed/sets/all.csv')
length(unique(data$date))
sentiment <- analyzeSentiment(data$text_between_speakers)
data$sentiment <- sentiment$SentimentQDAP

write.csv(data, "fomc_transcript/data/processed/sets/after_drive.csv")