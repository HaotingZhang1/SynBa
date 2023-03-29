library(devtools)
library(rstan)
library(bayesynergy)

name_to_data_dream <- function(combination_id, cell_line, rep_no='1', xmin=NULL, add_epsilon=FALSE) {
  dose_mat <- read.csv(paste0('data/dream/Raw_Data/ch1_training_combinations_csv/', combination_id, '.', cell_line, '.Rep', rep_no, '.csv'), 
                       header=F, na.strings=c('.', 'ND'))
  agent2_mono_dose <- as.numeric(dose_mat[1, 2:7])
  agent1_mono_dose <- as.numeric(dose_mat[2:7, 1])
  if (add_epsilon) {
    if (!is.null(xmin)) {
      agent1_mono_dose[1] <- xmin
      agent2_mono_dose[1] <- xmin
    } else {
      agent1_mono_dose[1] <- agent1_mono_dose[2] * 0.0001
      agent2_mono_dose[1] <- agent2_mono_dose[2] * 0.0001
    }
  }
  dose_mat <- as.matrix(sapply(dose_mat[2:7, 2:7], as.numeric))
  x_1 <- rep(agent1_mono_dose, each=6)
  x_2 <- matrix(rep(agent2_mono_dose, 6), ncol=6)
  y <- as.numeric(t(dose_mat))
  df <- data.frame(drug1.conc=x_1, drug2.conc=x_2, effect=y)
  return(list(df=df, x_1=x_1, x_2=x_2, y=y, dose_mat=dose_mat))
}

# setwd("...")
ch1_combo_mono_df <- read.csv("data/dream/ch1_train_combination_and_monoTherapy.csv", na.strings = c(".", "ND"))
ch1_combo_mono <- as.matrix(ch1_combo_mono_df)
print(paste0("Length of the DREAM dataset: ", nrow(ch1_combo_mono)))

C1_upper_bound <- 1e6
C1_lower_bound <- 1e-10
C2_upper_bound <- 1e6
C2_lower_bound <- 1e-10

te_rmse_bayesynergy <- c()
for (row in 1:nrow(ch1_combo_mono)) {
    combination_id <- ch1_combo_mono[row, 14]
    cell_line <- ch1_combo_mono[row, 1]
    
    # Filter out the following:
    # (1) rows with Rep2 data
    # (2) rows that have not passed QA
    # (3) rows with negative combination response(s)
    
    if (!file.exists(paste0(getwd(), "/data/dream/Raw_Data/ch1_training_combinations_csv/", 
                            combination_id, ".", cell_line, ".Rep2.csv"))) {
      result <- name_to_data_dream(combination_id, cell_line, rep_no = "1", add_epsilon = FALSE)
      df <- result['df']
      x1 <- unlist(result['x_1'])
      x2 <- unlist(result['x_2'])
      y <- unlist(result['y'])
      dose_mat <- unlist(result['dose_mat'])
      
      if (as.numeric(ch1_combo_mono[row, 13]) == 1 && min(y) >= 0) {
        print(paste0(combination_id, " ", cell_line))
        print(unname(dose_mat))
        leave_out_mono_1 <- sample(c(7, 13, 19, 25, 31), 1, replace = FALSE)
        leave_out_mono_2 <- sample(c(2, 3, 4, 5, 6), 1, replace = FALSE)
        leave_out_mono <- c(leave_out_mono_1, leave_out_mono_2)
        leave_out_combo <- sample(c(8:12, 14:24, 26:30, 32:36), 5, replace = FALSE)
        leave_out <- c(leave_out_mono, leave_out_combo)

        y_train <- y
        y_train[leave_out] <- NA
        x1_test <- x1[leave_out]
        x2_test <- x2[leave_out]
        y_test <- y[leave_out]
        
        x_train <- matrix(c(x1, x2), nrow=length(x1), ncol=2, byrow=FALSE)
        fit <- bayesynergy(y=y_train, x=x_train, bayes_factor =T)

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
        write.csv(te_rmse_bayesynergy, "rmse_dream/te_rmse_bayesynergy_dream.csv")
      }
    }
}


