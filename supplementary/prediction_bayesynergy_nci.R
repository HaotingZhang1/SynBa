#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly=TRUE)


library(devtools)
library(rstan)
library(bayesynergy)
library(tictoc)
library(tidyverse)


name_to_data_nci_almanac_subset <- function(combination_id, cell_line, xmin=NULL, add_epsilon=FALSE) {
  combination_id <- gsub(" ", "-", combination_id)
  cell_line <- gsub(" ", "-", cell_line)
  cell_line <- gsub("/", "-", cell_line)
  data <- read.csv(paste0("data/nci/combinations/", combination_id, ".", cell_line, ".csv"), header=FALSE, sep=",")
  ind0 <- which(data[, 1] == 0)
  ind0c <- which(data[, 1] > 0)
  non_0_min <- min(data[ind0c, 1])
  ind1 <- which(data[, 2] == 0)
  ind1c <- which(data[, 2] > 0)
  non_1_min <- min(data[ind1c, 2])
  if (add_epsilon == TRUE) {
    data[ind0, 1] <- non_0_min * 0.0001 #1e-12
    data[ind1, 2] <- non_1_min * 0.0001 #1e-12
  }
  if (!is.null(xmin)) {
    data[ind0, 1] <- xmin
    data[ind1, 2] <- xmin
  }
  x_1 <- as.numeric(data[, 1])
  x_2 <- as.numeric(data[, 2])
  y <- as.numeric(data[, 3])
  return(list(x_1=x_1, x_2=x_2, y=y))
}


# setwd("...")

data <- read_csv('data/nci/NCI-ALMANAC_subset_555300.csv', na = c('.', 'ND'))
data <- as.matrix(data)
compound_list <- unique(data[, 3])
cell_list <- unique(data[, 5])
D <- length(compound_list)
C <- length(cell_list)
stopifnot(D == 50)
stopifnot(C == 60)

C1_upper_bound <- 1e1
C1_lower_bound <- 1e-15
C2_upper_bound <- 1e1
C2_lower_bound <- 1e-15


cell_no <- as.numeric(args[1])  # 1
cell_line <- cell_list[cell_no]
te_rmse_bayesynergy <- c()
for (j in 1:D) {
  for (k in 1:D) {
    combination_id <- paste0(compound_list[j], '.', compound_list[k])
    combo <- paste0(combination_id, '.', cell_line)
    combo <- gsub(' ', '-', combo)
    combo <- gsub('/', '-', combo)
    if (file.exists(paste0(getwd(), '/data/nci/combinations/', combo, '.csv'))) {
      result <- name_to_data_nci_almanac_subset(combination_id, cell_line, add_epsilon=FALSE)
      x1 <- unlist(result['x_1'])
      x2 <- unlist(result['x_2'])
      y <- unlist(result['y'])
      assertthat::assert_that(length(y) / 5 == 3 | length(y) / 5 == 6)
      if (min(y) >= 0) {
        leave_out <- sample.int(length(y), size = floor(length(y) / 5), replace = FALSE)

        y_train <- y
        y_train[leave_out] <- NA
        x1_test <- x1[leave_out]
        x2_test <- x2[leave_out]
        y_test <- y[leave_out]
        
        x_train <- matrix(c(x1, x2), nrow=length(x1), ncol=2, byrow=FALSE)
        tic()
        fit <- bayesynergy(y=y_train, x=x_train, bayes_factor =T)
        toc()

        x_test <- matrix(c(x1_test, x2_test), # the data elements 
                         nrow=length(x1_test), ncol=2, byrow=FALSE)
        stanfit <- fit[['stanfit']]
        summary(fit)
        summary(stanfit)
        
        y_pred_matrix <- fit[['posterior_mean']]$p0 + fit[['posterior_mean']]$Delta
        y_pred <- c(t(y_pred_matrix)) * 100.0
        rmse <- sqrt(mean((y_pred[leave_out] - y[leave_out])^2))
        cat("RMSE:", rmse)
        
        te_rmse_bayesynergy <- append(te_rmse_bayesynergy, rmse)
        cat("Mean RMSE after", length(te_rmse_bayesynergy), "examples:", mean(te_rmse_bayesynergy))
        write.csv(te_rmse_bayesynergy, file=paste0("rmse_nci/te_rmse_bayesynergy_nci_", cell_no, ".csv"))
      }
    }
  }
}

