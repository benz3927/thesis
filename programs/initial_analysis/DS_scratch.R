# Introduction ---------------------------------------------------------------------------------------------------
# Script purpose/description: Create Fed Funds Rate and Gender Percentage Chart for Transcript Analysis Paper
# *_ Once we get BoardEx Data
rm(list=ls()); cat("\014")


library(ggtext)
library(xlsx)
library(readxl)
library(tidyr)
library(ggpubr)
library(ggbreak)
library(ggpattern)

select <- dplyr::select
mutate <- dplyr::mutate
filter <- dplyr::filter
group_by <- dplyr::group_by
ungroup <- dplyr::ungroup
summarize <- dplyr::summarize
rename <- dplyr::rename

# Briefing theme
title_font_size = 22
caption_font_size = 10
briefing_theme =
  theme(legend.title = element_blank(),
        legend.text=element_text(size = 16),
        legend.background = element_blank(),
        plot.title = element_text(size = title_font_size, hjust = 0.5),
        plot.subtitle = element_text(size = 16),
        axis.title.y = element_text(face="bold"),
        axis.title.x = element_text(face="bold"),
        axis.text.x.bottom = element_text(size = 16, face = "bold"),
        axis.text.y.right = element_text(size = 16),
        plot.caption = element_text(size =caption_font_size, hjust =0))



#Bring in data
#The data I will give you will be a list of the transcripts that are first name and last name of the speaker
#You will use the BoardEx data to identify the gender of the speaker. The gender categories we will be using are the ones previously used in the transcripts before 2009 - MR | MS | SPEAKER | GOVERNOR
#Because the BoardEx dat aiwll be available on Tuesday, the 20th, we will work on settingup the FRED API to gather the interest rates Professor Owen wanted to see


#Pull from FRED
library(fredr)
library(pkgconfig)
library(httr)


#You need to first create an account here and collect an API key


FRED_API_KEY <- fredr_set_key(key)
FFER_FRED <- fredr_series_observations(series_id = "DFF",
                                       observation_start = as.Date('1994-01-01'),
                                       observation_end = Sys.Date())
FFER_FRED <- select(FFER_FRED, subset = -c(realtime_start, realtime_end))
#---Graph 1 Make a line chart of the federal funds rate and Percentage of Males to Female
total <- read.csv('./output/attendance_voter_breakdown.csv') #Percent can be charted until and through December 2008
#ifgraphics: https://figs.web.rsma.frb.gov/

total$date <- as.Date(gsub("(\\d{4})(\\d{2})(\\d{2})$","\\1-\\2-\\3",total$date))
merged <- merge(FFER_FRED, total, by = ("date"), all.Y = TRUE)

#----
merged <- as.data.frame(merged)
merged <- subset(merged, select = -c(series_id, X))
merged$date <- as.Date(merged$date)


#lm
att_graph_lm <- ggplot() +
  geom_line(data = merged, aes(x = date, y = Percent.Male.Att, color = "Male Attendees(%)")) +
  geom_line(data = merged, aes(x = date, y = percent_male_voter, color = "Male Voters(%)")) +
  geom_smooth(data = merged, aes (x = date, y = Percent.Male.Att), method = lm, color = "black") +
  geom_smooth(data = merged, aes (x = date, y = percent_male_voter), method = lm, color = "firebrick") +
  geom_rect(data = merged, aes(xmin = as.Date("1994-01-01", "%Y-%m-%d"), xmax = as.Date("2006-01-31", "%Y-%m-%d")),ymin = 0, ymax = 1.1, color = "lightgoldenrod2", fill = "lightgoldenrod2", alpha = .01)+ 
  #annotate("text", x = as.Date("1997-01-01", "%Y-%m-%d"), y = .75, label = "Chairman Greenspan", family = "Helvetica", hjust = 0, size = 2.7) +
  geom_rect(data = merged, aes(xmin = as.Date("2014-05-01", "%Y-%m-%d"), xmax = as.Date("2018-01-01", "%Y-%m-%d")), ymin = 0, ymax = 1.1, color = "lightgoldenrod2", fill = "lightgoldenrod2", alpha = .01) +
  scale_color_manual(values = c("Male Attendees(%)" = "black",
                                "Male Voters(%)" = "firebrick")) + 
  
  #geom_line(mapping = aes(x = imf_small$date, y = imf_small$CHINA_FINAL), color = "blue") +
  theme_if_exhibits +
  labs(title = "FOMC Gender Breakdown",
       subtitle = NULL,
       x = NULL,
       y = "Percentage (%)",
       caption = paste0("Data as of December 31, 2017."))+
  theme(legend.title = element_blank(),
        legend.position = "top",
        legend.text=element_text(size = 8),
        plot.title = element_text(size = 14, hjust = 0.5),
        plot.subtitle = element_text(size = 12, hjust = 0),
        axis.title.y = element_text(face="bold", size = 14),
        axis.title.x = element_text(face="bold", size = 14),
        axis.text.x.bottom = element_text(size = 10, face = "bold"),
        axis.title.y.left = element_blank(),
        axis.text.y.right = element_text(size = 10, face = "bold"),
        plot.caption = element_text(size =caption_font_size, hjust = 0)) +
  scale_x_date(date_labels = "%b\n%Y",date_breaks = "21 months", limits = c(as.Date("1994-01-01"), as.Date("2018-01-01"))) +
  board_y_axis(.50, 1, .10)

att_graph_lm


#loess
att_graph_loess <- ggplot() +
  geom_line(data = merged, aes(x = date, y = Percent.Male.Att, color = "Male Attendees(%)")) +
  geom_line(data = merged, aes(x = date, y = percent_male_voter, color = "Male Voters(%)")) +
  geom_smooth(data = merged, aes (x = date, y = Percent.Male.Att), method = loess, color = "black") +
  geom_smooth(data = merged, aes (x = date, y = percent_male_voter), method = loess, color = "firebrick") +
  geom_rect(data = merged, aes(xmin = as.Date("1994-01-01", "%Y-%m-%d"), xmax = as.Date("2006-01-31", "%Y-%m-%d")),ymin = 0, ymax = 1.1, color = "lightgoldenrod2", fill = "lightgoldenrod2", alpha = .01)+ 
  #annotate("text", x = as.Date("1997-01-01", "%Y-%m-%d"), y = .75, label = "Chairman Greenspan", family = "Helvetica", hjust = 0, size = 2.7) +
  geom_rect(data = merged, aes(xmin = as.Date("2014-05-01", "%Y-%m-%d"), xmax = as.Date("2018-01-01", "%Y-%m-%d")), ymin = 0, ymax = 1.1, color = "lightgoldenrod2", fill = "lightgoldenrod2", alpha = .01) +
  scale_color_manual(values = c("Male Attendees(%)" = "black",
                                "Male Voters(%)" = "firebrick")) + 
  
  #geom_line(mapping = aes(x = imf_small$date, y = imf_small$CHINA_FINAL), color = "blue") +
  theme_if_exhibits +
  labs(title = "FOMC Gender Breakdown",
       subtitle = NULL,
       x = NULL,
       y = "Percentage (%)",
       caption = paste0("Data as of December 31, 2017."))+
  theme(legend.title = element_blank(),
        legend.position = "top",
        legend.text=element_text(size = 8),
        plot.title = element_text(size = 14, hjust = 0.5),
        plot.subtitle = element_text(size = 12, hjust = 0),
        axis.title.y = element_text(face="bold", size = 14),
        axis.title.x = element_text(face="bold", size = 14),
        axis.text.x.bottom = element_text(size = 10, face = "bold"),
        axis.title.y.left = element_blank(),
        axis.text.y.right = element_text(size = 10, face = "bold"),
        plot.caption = element_text(size =caption_font_size, hjust = 0)) +
  scale_x_date(date_labels = "%b\n%Y",date_breaks = "21 months", limits = c(as.Date("1994-01-01"), as.Date("2018-01-01"))) +
  board_y_axis(.50, 1, .10)

att_graph_loess





#-------------------------------------------------------------------------------------------------------------------#
#                                                      NON VOTERS                                                   #
#-------------------------------------------------------------------------------------------------------------------#
#lm
fem_att_graph_3 <- ggplot() +
  geom_line(data = merged, aes(x = date, y = percent_female_NONvoters, color = "Female Non-Voters (%)")) +
  geom_line(data = merged, aes(x = date, y = Percent.Female.Att, color = "Female Attendees (%)")) +
  geom_line(data = merged, aes(x = date, y = percent_fem_voter, color = "Female Voters (%)")) +
  #geom_line(data = merged, aes(x = date, y = percent_male_NONvoters, color = "Male Non-Voters(%)")) +
  #geom_smooth(data = merged, aes (x = date, y = Percent.Male.Att), method = lm, color = "black") +
  #geom_smooth(data = merged, aes (x = date, y = percent_male_voter), method = lm, color = "firebrick") +
  geom_rect(data = merged, aes(xmin = as.Date("1994-01-01", "%Y-%m-%d"), xmax = as.Date("2006-01-31", "%Y-%m-%d")),ymin = 0, ymax = 1.1, color = "lightgoldenrod2", fill = "lightgoldenrod2", alpha = .01)+ 
  #annotate("text", x = as.Date("1997-01-01", "%Y-%m-%d"), y = .75, label = "Chairman Greenspan", family = "Helvetica", hjust = 0, size = 2.7) +
  geom_rect(data = merged, aes(xmin = as.Date("2014-05-01", "%Y-%m-%d"), xmax = as.Date("2018-01-01", "%Y-%m-%d")), ymin = 0, ymax = 1.1, color = "lightgoldenrod2", fill = "lightgoldenrod2", alpha = .01) +
  scale_color_manual(values = c("Female Attendees (%)" = "black",
                                "Female Non-Voters (%)" = "forestgreen",
                                "Female Voters (%)" = "darkblue")) + 
  
  #geom_line(mapping = aes(x = imf_small$date, y = imf_small$CHINA_FINAL), color = "blue") +
  theme_if_exhibits +
  labs(title = "FOMC Gender Breakdown",
       subtitle = NULL,
       x = NULL,
       y = "Percentage (%)",
       caption = paste0("Data as of December 31, 2017."))+
  theme(legend.title = element_blank(),
        legend.position = "top",
        legend.text=element_text(size = 8),
        plot.title = element_text(size = 14, hjust = 0.5),
        plot.subtitle = element_text(size = 12, hjust = 0),
        axis.title.y = element_text(face="bold", size = 14),
        axis.title.x = element_text(face="bold", size = 14),
        axis.text.x.bottom = element_text(size = 10, face = "bold"),
        axis.title.y.left = element_blank(),
        axis.text.y.right = element_text(size = 10, face = "bold"),
        plot.caption = element_text(size =caption_font_size, hjust = 0)) +
  scale_x_date(date_labels = "%b\n%Y",date_breaks = "21 months", limits = c(as.Date("1994-01-01"), as.Date("2018-01-01"))) +
  board_y_axis(0, .50, .10)

fem_att_graph_3

#----
fem_att_graph_2 <- ggplot() +
  geom_line(data = merged, aes(x = date, y = percent_female_NONvoters, color = "Female Non-Voters (%)")) +
  geom_line(data = merged, aes(x = date, y = Percent.Female.Att, color = "Female Attendees (%)")) +
  #geom_line(data = merged, aes(x = date, y = percent_fem_voter, color = "Female Voters (%)")) +
  #geom_line(data = merged, aes(x = date, y = percent_male_NONvoters, color = "Male Non-Voters(%)")) +
  #geom_smooth(data = merged, aes (x = date, y = Percent.Male.Att), method = lm, color = "black") +
  #geom_smooth(data = merged, aes (x = date, y = percent_male_voter), method = lm, color = "firebrick") +
  geom_rect(data = merged, aes(xmin = as.Date("1994-01-01", "%Y-%m-%d"), xmax = as.Date("2006-01-31", "%Y-%m-%d")),ymin = 0, ymax = 1.1, color = "lightgoldenrod2", fill = "lightgoldenrod2", alpha = .01)+ 
  #annotate("text", x = as.Date("1997-01-01", "%Y-%m-%d"), y = .75, label = "Chairman Greenspan", family = "Helvetica", hjust = 0, size = 2.7) +
  geom_rect(data = merged, aes(xmin = as.Date("2014-05-01", "%Y-%m-%d"), xmax = as.Date("2018-01-01", "%Y-%m-%d")), ymin = 0, ymax = 1.1, color = "lightgoldenrod2", fill = "lightgoldenrod2", alpha = .01) +
  scale_color_manual(values = c("Female Attendees (%)" = "black",
                                "Female Non-Voters (%)" = "forestgreen",
                                "Female Voters (%)" = "darkblue")) + 
  
  #geom_line(mapping = aes(x = imf_small$date, y = imf_small$CHINA_FINAL), color = "blue") +
  theme_if_exhibits +
  labs(title = "FOMC Gender Breakdown",
       subtitle = NULL,
       x = NULL,
       y = "Percentage (%)",
       caption = paste0("Data as of December 31, 2017."))+
  theme(legend.title = element_blank(),
        legend.position = "top",
        legend.text=element_text(size = 8),
        plot.title = element_text(size = 14, hjust = 0.5),
        plot.subtitle = element_text(size = 12, hjust = 0),
        axis.title.y = element_text(face="bold", size = 14),
        axis.title.x = element_text(face="bold", size = 14),
        axis.text.x.bottom = element_text(size = 10, face = "bold"),
        axis.title.y.left = element_blank(),
        axis.text.y.right = element_text(size = 10, face = "bold"),
        plot.caption = element_text(size =caption_font_size, hjust = 0)) +
  scale_x_date(date_labels = "%b\n%Y",date_breaks = "21 months", limits = c(as.Date("1994-01-01"), as.Date("2018-01-01"))) +
  board_y_axis(0, .50, .10)

fem_att_graph_2



#--------------------------------------------------------------------------------------------------------------

#----Breakdown of Voter Gender

#----Export
pdf("fomc_transcript/output/graphs_for_07062023.pdf", 
    paper="letter",
    width = 9,
    height = 11)



plot_grid(att_graph_lm,
          att_graph_loess,
          NULL,
          nrow = 3,
          ncol =1)

plot_grid(fem_att_graph_3,
          fem_att_graph_2,
          NULL,
          nrow = 3,
          ncol =1)


dev.off()
#----