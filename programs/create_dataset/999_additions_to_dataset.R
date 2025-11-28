rm(list = ls())


library(plm)
library(fastDummies)
library(dplyr)




library(ggtext)
library(xlsx)
library(readxl)
library(ggpubr, warn.conflicts = F)
#-------------------------------------------------------------------
data <- read.csv("/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/data/processed/sets/final.csv")
data$date <- gsub("(\\d{4})(\\d{2})(\\d{2})$","\\1-\\2-\\3",data$date)
data$date <- as.Date(data$date)
data <- subset(data, select = -c(X))
data$quarter <- lubridate::quarter(data$date)
data$year <- lubridate::year(data$date)
data$month <- lubridate::month(data$date)

#NA means that the person did not vote
#  1 - Means the person voted and dissented
#  0 - Means the person voted and did NOT dissent
voter_dumm <- (ifelse(is.na(data$dissenter_dummy), 0, 1))
data$voter_dummy <- voter_dumm
#-------------------------------------------------------------------
#---Bring in Financial Crisis Dates
#Financial crisis is 2007Q3:2009Q4
data$fin_crisis <- 0
data$fin_crisis[data$quarter == 4 & data$year == 2007] <- 1
data$fin_crisis[data$quarter == 1 & data$year == 2008] <- 1
data$fin_crisis[data$quarter == 2 & data$year == 2008] <- 1
data$fin_crisis[data$quarter == 3 & data$year == 2008] <- 1
data$fin_crisis[data$quarter == 4 & data$year == 2008] <- 1
data$fin_crisis[data$quarter == 1 & data$year == 2009] <- 1
data$fin_crisis[data$quarter == 2 & data$year == 2009] <- 1
data$fin_crisis[data$quarter == 3 & data$year == 2009] <- 1
data$fin_crisis[data$quarter == 4 & data$year == 2009] <- 1
#-------------------------------------------------------------------
#Bring in Economic Crisis Dates
#NBER based Recession Indicators for the United States from the Period following the Peak through the Trough
#Pull from FRED
library(fredr)
library(pkgconfig)
library(httr)

set_config(use_proxy('wwwproxy.frb.gov',8080), override = FALSE)
set_config(user_agent('Lynx'), override = FALSE)

key <- "d3a9f3aa2d0bb929998418ee84ef9ba5"
FRED_API_KEY <- fredr_set_key(key)
econ_dates <- fredr_series_observations(series_id = "USREC",
                                      observation_start = as.Date('1994-01-01'),
                                      observation_end = Sys.Date())

econ_dates <- subset(econ_dates, select = c("date", "value"))
econ_dates$quarter <- lubridate::quarter(econ_dates$date)
econ_dates$year <- lubridate::year(econ_dates$date)
econ_dates$econ_crisis <- econ_dates$value
econ_dates <- subset(econ_dates, select = -c(date, value))
econ_dates <- unique(econ_dates[,c("quarter", "year", "econ_crisis")])
econ_dates <- econ_dates[!duplicated(econ_dates[c('year', 'quarter')]), ]
total <- merge(data, econ_dates, by = c("year", "quarter"), all.x = TRUE)

#UPPER LIMIT
upper_limit <- fredr_series_observations(series_id = "DFEDTARU",
                                        observation_start = as.Date('1994-01-01'),
                                        observation_end = Sys.Date())

upper_limit <- subset(upper_limit, select = c("date", "value"))
upper_limit$target_upper <- upper_limit$value
upper_limit <- subset(upper_limit, select = -c(value))

upper_limit <- unique(upper_limit[,c("date", "target_upper")])
total <- merge(total, upper_limit, by = c("date"), all.x = TRUE)


#LOWER LIMIT
lower_limit <- fredr_series_observations(series_id = "DFEDTARL",
                                         observation_start = as.Date('1994-01-01'),
                                         observation_end = Sys.Date())

lower_limit <- subset(lower_limit, select = c("date", "value"))
lower_limit$target_lower <- lower_limit$value
lower_limit <- subset(lower_limit, select = -c(value))

lower_limit <- unique(lower_limit[,c("date", "target_lower")])
total <- merge(total, lower_limit, by = c("date"), all.x = TRUE)
total$target_mid <- rowMeans(total[,c("target_lower", "target_upper")], na.rm = TRUE)

length(unique(total$date))
#------------------------------------------------------------------------------Chair names
#Chair names
chair_names <- read.csv("fomc_transcript/data/metadata/chairman_dates.csv")
chair_names$date <- as.Date(chair_names$date)
chair_names <- distinct(chair_names, date, chair, .keep_all = TRUE)
total <- merge(total, chair_names, by = c("date"), all.X = TRUE)
length(unique(total$date))
#----Let's add Dummy for Bucket
add_buckets <- read.csv("fomc_transcript/data/processed/everyonce.csv")
add_buckets$date <- gsub("(\\d{4})(\\d{2})(\\d{2})$","\\1-\\2-\\3",add_buckets$date)
add_buckets$date <- as.Date(add_buckets$date)
add_buckets$quarter <- lubridate::quarter(add_buckets$date)
add_buckets$year <- lubridate::year(add_buckets$date)
add_buckets$month <- lubridate::month(add_buckets$date)
length(unique(add_buckets$date))
total <- merge(total, add_buckets, by = c("date", "year", "month", "quarter", "short_name"))
length(unique(total$date))

# test_buckets_subset <- subset(buckets_testing, select = c("date", "short_name", "position"))

#b1_dummy
bucket_1_dum = c('chair','chairman','preisdent','president elect','Chairman','Vice Chairman','president','vice chairman','governor','president-elect')
total <- total %>%
  mutate(b1_dummy = if_else(position %in% bucket_1_dum, 1,0))
#b2_dummy
bucket_2_dum = c('executice vice president','senior vice preisdent','vice president','executvie vice president','senior vice president','assistant vice president','first vice preisndet','first vice president','executive vice president','group vice president')
total <- total %>%
  mutate(b2_dummy = if_else(position %in% bucket_2_dum, 1,0))
#b3_dummy
bucket_3_dum = c('deputy general counsel','senior special adviser to the board','deputy secretary','senior special adviser to the chair','senior sepcial adviser to the board','secretary of the board','deputy general cousnel','special policy advisor to the president','assistant to the board','special adviser to the chair','deputy congressional liason','deputy generl counsel','special assitant to the board','special advisor to the board','assitant to the board','special counsel','adviser to the board','acting director','special assistant to the board','special policy adviser to the president','advisor to the president','special adviser to the board', 'director','deputy secretary counsel','senior special advisor to the chair')
total <- total %>%
  mutate(b3_dummy = if_else(position %in% bucket_3_dum, 1,0))
#b4_dummy
bucket_4_dum = c('associate economist', 'manager','senior associate','secretary and economist','senior advisor','economic policy advisor','research','secretariat assistant','special assistant','assistant to the director','general counsel','deputy staff director','senior information manager','assistant directors','group manager','senior counsel','deputy director','financial economist','senior research economist','associate director','visiting associate director','assistant to the secretary','research assistant','visiting senior adviser','assistant general counsel','senior research advisor senior economist','policy adviser','open market secretariant assistant','open market operations manager','system open market account mananger','senior economic advisor','assistant director','visiting research bank officer','senior professional economist','system open market manager','assistant secretary','senior associate director','senior economist','senior special adviser','information manager','senior research officer','senior economic project manager','information management analyst','system open market account manger','senior financial analyst ','adivser','visiting reserve bank officer','special adviser','system open market account manager','monetary advisor','economic adviser','section chief','research economist','sectio chief','dpeuty associate director','markets officer','senior research adviser','economist','manager for domestic operations','open market secretariat assistant','open market secretary specialist','consultant','senior economic adviser','open market secretariat specialist','open market secretariat','open makret secretariat specialist','dpeuty director','associate secretary','deputy manager','project manager','records management analyst','special assistant to the director','deputy associate director','associate general counsel', 'senior associate directgor','secretary','assistant economist','assistant congressional liasion','seniro associate director','economic advisor','senior attorney','associate economists','staff assistant','senior financial analyst','manager for foreign operations','seciton chief','research officer','records project manager','senior research advisor','principal economist','temporary manager','senior project manager','managing senior counsel','financial analyst','research adviser','visitng reserve bank officer','senior adviser','officer','senior techincal editor','special policy advisor','adviser')
total <- total %>%
  mutate(b4_dummy = if_else(position %in% bucket_4_dum, 1,0))

#Female Dummy
total$fem_dummy[total$greeting=="Mr"] <- 0
total$fem_dummy[total$greeting=="Ms"] <- 1
total$fem_dummy[total$greeting=="Mrs"] <- 1

#Briefing Dummy
briefing_dummy <- readxl::read_xlsx("./data/processed/sets/briefing_dummy.xlsx")

#Merge in set
total <- merge(briefing_dummy, total, by = c("date", "short_name", "greeting"), all.y = TRUE)
total$briefer_dummy[is.na(total$briefer_dummy)] <- 0
reserve <- total
#-------------------------------------------------------------------------------------Bring in political ideologies
pol_ideo <- readxl::read_xlsx("fomc_transcript/data/external/election/board_of_governors_election_data.xlsx", sheet = "pol_data")
pol_ideo <- pol_ideo[!duplicated(pol_ideo[c('speaker', 'position', 'pol_dummy')]), ]
total <- merge(total, pol_ideo, by = c('speaker', 'position'), all.x = TRUE)

length(unique(total$date))
#-------------------------------------------------------------------------------------------------

#-----------------------------------Add Baker, Bloom, and Davis 
baker_bloom_davis <- readxl::read_xlsx("fomc_transcript/data/external/baker_bloom_davis/Categorical_EPU_Data.xlsx", sheet = "Indices")
baker_bloom_davis$date <- as.Date(baker_bloom_davis$date)
baker_bloom_davis <- subset(baker_bloom_davis, select = c(date, epu, mpu))
baker_bloom_davis$quarter <- lubridate::quarter(baker_bloom_davis$date)
baker_bloom_davis$year <- lubridate::year(baker_bloom_davis$date)
baker_bloom_davis$month <- lubridate::month(baker_bloom_davis$date)
total <- merge(total, baker_bloom_davis, by = c('year', 'quarter', 'month'), all.x = TRUE)
#-----------------------------------Shadow Interest Rates
shadow_i_r <- read_csv("fomc_transcript/data/external/shadow_interest_rate/shadowrate_US.csv")
shadow_i_r$quarter <- lubridate::quarter(shadow_i_r$date)
shadow_i_r$year <- lubridate::year(shadow_i_r$date)
shadow_i_r$month <- lubridate::month(shadow_i_r$date)
total$date <- total$date.x
shadow_i_r <- subset(shadow_i_r, select = -c(date))
total <- merge(total, shadow_i_r, by = c("year", "month", "quarter"), all.X = TRUE)
write_csv(dat, "fomc_transcript/data/processed/data_01292024.csv")
#Now clean the csv above
#--------------------------------------------------------
cleaned_data <- read.csv("fomc_transcript/data/processed/save_data_01292024.csv")

#Add Unique Identifer
uniqueid <- distinct(cleaned_data, short_name, greeting)
uniqueid$unique_id <- 1:nrow(uniqueid)


data_merge <- merge(cleaned_data, uniqueid, by = c("short_name", "greeting"), all.X = TRUE)

dat <- data_merge %>%
  distinct(short_name, greeting, position, date, unique_id, .keep_all = TRUE)
length(unique(dat$date))

write_csv(dat, "fomc_transcript/data/processed/with_u_is_data_01292024.csv")

