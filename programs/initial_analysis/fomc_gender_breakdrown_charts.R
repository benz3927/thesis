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
total <- read.csv('./output/jokes.csv') #Percent can be charted until and through December 2008
total <- subset(total, select = -c(X))
laughs <- read.csv('./output/laughs.csv')
laughs <- subset(laughs, select = -c(X))

merged <- merge(total, laughs, by = 'date', all = TRUE)
#----
merged$date <- as.Date(gsub("(\\d{4})(\\d{2})(\\d{2})$","\\1-\\2-\\3",merged$date))
merged$date <- as.Date(merged$date)

merged <- subset(merged, select = -c(chair_dummy.y,  All.laughs.y))
merged$perc_laughs_after_mr <- (merged$Laughs.After.MR.SPEAKER / merged$All.laughs.x) * 100
merged$perc_laughs_after_ms <- (merged$Laughs.After.MS.SPEAKER / merged$All.laughs.x) * 100
merged$perc_laughs_after_mr_chair <- (merged$Laughs.After.CHAIRMAN / merged$All.laughs.x) * 100
merged$perc_laughs_after_ms_chair <- (merged$Laughs.after.FEMALE.CHAIR/ merged$All.laughs.x) * 100


#lm
laughs_i_breakdown <- ggplot() +
  geom_line(data = merged, aes(x = date, y = perc_joke_i, color = "'I' Jokes")) +
  geom_line(data = merged, aes(x = date, y = perc_laughs_after_mr, color = "Laughs After MR Speaker(%)")) +
  geom_line(data = merged, aes(x = date, y = perc_laughs_after_ms, color = "Laughs After MS Speaker(%)")) +
  #geom_smooth(data = merged, aes (x = date, y = Percent.Male.Att), method = lm, color = "black") +
  #geom_smooth(data = merged, aes (x = date, y = percent_male_voter), method = lm, color = "firebrick") +
  geom_rect(data = merged, aes(xmin = as.Date("1994-01-01", "%Y-%m-%d"), xmax = as.Date("2006-01-31", "%Y-%m-%d")),ymin = 0, ymax = 100, color = "lightgoldenrod2", fill = "lightgoldenrod2", alpha = .01)+ 
  #annotate("text", x = as.Date("1997-01-01", "%Y-%m-%d"), y = .75, label = "Chairman Greenspan", family = "Helvetica", hjust = 0, size = 2.7) +
  geom_rect(data = merged, aes(xmin = as.Date("2014-05-01", "%Y-%m-%d"), xmax = as.Date("2018-01-01", "%Y-%m-%d")), ymin = 0, ymax = 100, color = "lightgoldenrod2", fill = "lightgoldenrod2", alpha = .01) +
  scale_color_manual(values = c("'I' Jokes" = "black",
                                "Laughs After MR Speaker(%)" = "firebrick",
                                "Laughs After MS Speaker(%)" = "forestgreen")) + 
  
  #geom_line(mapping = aes(x = imf_small$date, y = imf_small$CHINA_FINAL), color = "blue") +
  theme_if_exhibits +
  labs(title = "FOMC Laugh-Gender Breakdown",
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
  board_y_axis(0, 100, 25)

laughs_i_breakdown



#YOU
laughs_u_breakdown <- ggplot() +
  geom_line(data = merged, aes(x = date, y = perc_joke_u, color = "'U' Jokes")) +
  geom_line(data = merged, aes(x = date, y = perc_laughs_after_mr, color = "Laughs After MR Speaker(%)")) +
  geom_line(data = merged, aes(x = date, y = perc_laughs_after_ms, color = "Laughs After MS Speaker(%)")) +
  #geom_smooth(data = merged, aes (x = date, y = Percent.Male.Att), method = lm, color = "black") +
  #geom_smooth(data = merged, aes (x = date, y = percent_male_voter), method = lm, color = "firebrick") +
  geom_rect(data = merged, aes(xmin = as.Date("1994-01-01", "%Y-%m-%d"), xmax = as.Date("2006-01-31", "%Y-%m-%d")),ymin = 0, ymax = 100, color = "lightgoldenrod2", fill = "lightgoldenrod2", alpha = .01)+ 
  #annotate("text", x = as.Date("1997-01-01", "%Y-%m-%d"), y = .75, label = "Chairman Greenspan", family = "Helvetica", hjust = 0, size = 2.7) +
  geom_rect(data = merged, aes(xmin = as.Date("2014-05-01", "%Y-%m-%d"), xmax = as.Date("2018-01-01", "%Y-%m-%d")), ymin = 0, ymax = 100, color = "lightgoldenrod2", fill = "lightgoldenrod2", alpha = .01) +
  scale_color_manual(values = c("'U' Jokes" = "black",
                                "Laughs After MR Speaker(%)" = "firebrick",
                                "Laughs After MS Speaker(%)" = "forestgreen")) + 
  
  #geom_line(mapping = aes(x = imf_small$date, y = imf_small$CHINA_FINAL), color = "blue") +
  theme_if_exhibits +
  labs(title = "FOMC Laugh-Gender Breakdown",
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
  board_y_axis(0, 100, 25)

laughs_u_breakdown


#Neutral
#YOU
laughs_neutral_breakdown <- ggplot() +
  geom_line(data = merged, aes(x = date, y = perc_joke_np, color = "Neutral Jokes")) +
  geom_line(data = merged, aes(x = date, y = perc_laughs_after_mr, color = "Laughs After MR Speaker(%)")) +
  geom_line(data = merged, aes(x = date, y = perc_laughs_after_ms, color = "Laughs After MS Speaker(%)")) +
  #geom_smooth(data = merged, aes (x = date, y = Percent.Male.Att), method = lm, color = "black") +
  #geom_smooth(data = merged, aes (x = date, y = percent_male_voter), method = lm, color = "firebrick") +
  geom_rect(data = merged, aes(xmin = as.Date("1994-01-01", "%Y-%m-%d"), xmax = as.Date("2006-01-31", "%Y-%m-%d")),ymin = 0, ymax = 100, color = "lightgoldenrod2", fill = "lightgoldenrod2", alpha = .01)+ 
  #annotate("text", x = as.Date("1997-01-01", "%Y-%m-%d"), y = .75, label = "Chairman Greenspan", family = "Helvetica", hjust = 0, size = 2.7) +
  geom_rect(data = merged, aes(xmin = as.Date("2014-05-01", "%Y-%m-%d"), xmax = as.Date("2018-01-01", "%Y-%m-%d")), ymin = 0, ymax = 100, color = "lightgoldenrod2", fill = "lightgoldenrod2", alpha = .01) +
  scale_color_manual(values = c("Neutral Jokes" = "black",
                                "Laughs After MR Speaker(%)" = "firebrick",
                                "Laughs After MS Speaker(%)" = "forestgreen")) + 
  
  #geom_line(mapping = aes(x = imf_small$date, y = imf_small$CHINA_FINAL), color = "blue") +
  theme_if_exhibits +
  labs(title = "FOMC Laugh-Gender Breakdown",
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
  board_y_axis(0, 100, 25)

laughs_neutral_breakdown





#--------------------------------------------------------------------------------------------------------------

#----Breakdown of Voter Gender

#----Export
pdf("fomc_transcript/output/laughs_breakdown_gender.pdf", 
    paper="letter",
    width = 9,
    height = 11)



plot_grid(laughs_i_breakdown,
          laughs_u_breakdown,
          laughs_neutral_breakdown,
          NULL,
          nrow = 3,
          ncol =1)


dev.off()
#----