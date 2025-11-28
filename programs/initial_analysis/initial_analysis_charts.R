#######################################################################################
#
#######################################################################################

rm(list=ls()); cat("\014")

packs = c("tidyr","plyr","dplyr","data.table", "readstata13", "xlsx", "qpdf", "ggpubr",
          "ggplot2", "readxl", "lubridate", "foreign", "haven", "directlabels", "miscTools")
lapply(packs, require, character.only = TRUE)
library(data.table)
library(tidyverse)
library(policyPlot)
library(readxl)
library(haven)
library(rlist)
library(zoo)
library(fst)
library(dplyr)


select <- dplyr::select
mutate <- dplyr::mutate
filter <- dplyr::filter
group_by <- dplyr::group_by
ungroup <- dplyr::ungroup
summarize <- dplyr::summarize
rename <- dplyr::rename

fonts_for_ppt$plot.title <- element_text(size = 14, hjust = .5)
fonts_for_ppt$plot.caption <- element_text(size = 11)
fonts_for_ppt$legend.text <- element_text(size = 10)
fonts_for_ppt$axis.text.x <- element_text(size = 9)
fonts_for_ppt$axis.text.y.left <- element_text(size = 11)
fonts_for_ppt$axis.text.y.right <- element_text(size = 11)
fonts_for_ppt$plot.subtitle <- element_text(size = 11)

fonts_for_ppt$text$family <- "Helvetica"

options(scipen = 999)
#Read in data ------------------------------------------------------------------
data <- read_csv("fomc_transcript/output/total_clean.csv")
data <- subset(data, select = -c(...1, Transcript_y))
data$date <- gsub("(\\d{4})(\\d{2})(\\d{2})$","\\1-\\2-\\3",data$date)
data$date<- as.Date(data$date)
data <- data %>%
  relocate(date) %>%
  drop_na(`Laughs After MR SPEAKER`)


laughter_plot_mr_s <- 
  ggplot() +
  geom_line(data, mapping = aes(x = date,
                                           y = `Laughs After MR SPEAKER`,
                                           color = "After MR")) +
  
  geom_line(data, mapping = aes(x = date,
                                y = `Laughs After MS SPEAKER`,
                                color = "After MS")) +
  #annotate("text", x = as.Date("04-15-2021", "%m-%d-%Y"), y = 100000, label = "Russia Invades Ukraine\n on February 24, 2022", family = "Helvetica", hjust = 0, size = 3) +
  theme_if_exhibits + 
  small_margins +
  fonts_for_ppt +
  scale_color_manual(values = c("After MR" = "darkblue", "After MS" = "goldenrod")) + 
  labs(x = NULL,
       y = NULL,
       title = "Laughs After Gender of Speaker") +
  theme(#plot.margin=grid::unit(c(0,0,0,0), "mm"),
    panel.spacing = unit(0, 'lines'),
    legend.title = element_blank(),
    legend.position = "top", 
    legend.spacing.y = unit(-0.3, 'lines'),
    axis.text.y.left = element_text(hjust = 1, margin = margin(t = 0, r = 12, b = 0, l = 0)),
    legend.background = element_blank(),
    legend.box.background = element_blank()) +
  board_y_axis(0, 50, 10)

laughter_plot_mr_s


att_plot_mr_s <- 
  ggplot() +
  # geom_line(data, mapping = aes(x = date,
  #                               y = Attendees,
  #                               color = "Attendees")) +
  
  geom_line(data, mapping = aes(x = date,
                                y = Male,
                                color = "MR")) +
  geom_line(data, mapping = aes(x = date,
                                y = Female,
                                color = "MS"))+
  #annotate("text", x = as.Date("04-15-2021", "%m-%d-%Y"), y = 100000, label = "Russia Invades Ukraine\n on February 24, 2022", family = "Helvetica", hjust = 0, size = 3) +
  theme_if_exhibits + 
  small_margins +
  fonts_for_ppt +
  scale_color_manual(values = c("MR" = "darkblue", "MS" = "goldenrod")) + 
  labs(x = NULL,
       y = NULL,
       title = "Attendance") +
  theme(#plot.margin=grid::unit(c(0,0,0,0), "mm"),
    panel.spacing = unit(0, 'lines'),
    legend.title = element_blank(),
    legend.position = "top", 
    legend.spacing.y = unit(-0.3, 'lines'),
    axis.text.y.left = element_text(hjust = 1, margin = margin(t = 0, r = 12, b = 0, l = 0)),
    legend.background = element_blank(),
    legend.box.background = element_blank()) +
  board_y_axis(0, 60, 10)

att_plot_mr_s

att_not_vote_plot_mr_s <- 
  ggplot() +
  # geom_line(data, mapping = aes(x = date,
  #                               y = Attendees,
  #                               color = "Attendees")) +
  
  geom_line(data, mapping = aes(x = date,
                                y = Attendees)) +
  #annotate("text", x = as.Date("04-15-2021", "%m-%d-%Y"), y = 100000, label = "Russia Invades Ukraine\n on February 24, 2022", family = "Helvetica", hjust = 0, size = 3) +
  theme_if_exhibits + 
  small_margins +
  fonts_for_ppt +
  #scale_color_manual(values = c("MR" = "darkblue", "MS" = "goldenrod")) + 
  labs(x = NULL,
       y = NULL,
       title = "Attendance") +
  theme(#plot.margin=grid::unit(c(0,0,0,0), "mm"),
    panel.spacing = unit(0, 'lines'),
    legend.title = element_blank(),
    legend.position = "top", 
    legend.spacing.y = unit(-0.3, 'lines'),
    axis.text.y.left = element_text(hjust = 1, margin = margin(t = 0, r = 12, b = 0, l = 0)),
    legend.background = element_blank(),
    legend.box.background = element_blank()) +
  board_y_axis(0, 100, 25)

att_not_vote_plot_mr_s





# Combine ----------------------------------------------------------------------

pdf("fomc_transcript/output/initial_analysis.pdf", 
    paper="letter", 
    width = 9, 
    height = 11)

plot_grid(laughter_plot_mr_s,
          att_plot_mr_s,
          att_not_vote_plot_mr_s,
          nrow = 3,
          ncol = 1)



dev.off()
########################################################################################
#End
