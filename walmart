#install.packages("caret")
#install.packages("randomForest")
#install.packages("tune")
#install.packages("gbm")
#install.packages("glmnet")
#install.packages("corrplot")
#install.packages("ranger")
#install.packages("rpart.plot)
#install.packages("gridExtra")
library(caret)
library(gbm)
library(glmnet)
library(randomForest)
library(corrplot)
library(tune)
library(dplyr)
library(ranger)
library(rpart)
library(rpart.plot)
library(gridExtra)
data <- read.csv("~/Downloads/walmart.csv")

data$Date <- as.Date(data$Date, format="%d-%m-%Y")
data$Year <- as.numeric(format(data$Date, "%Y"))
data$Month <- as.numeric(format(data$Date, "%m"))
data$WeekOfYear <- as.numeric(format(data$Date, "%U")) 

data <- subset(data, select = -Date)

data$Store<-as.factor(data$Store)
data$Holiday_Flag<-as.logical(data$Holiday_Flag)

store_one_hot <- model.matrix(~ Store - 1, data)
data_notrf <- cbind(data, store_one_hot)
data_notrf$Store <- NULL

data_notrf[, c("Temperature", "Fuel_Price", "CPI", "Unemployment")] <- 
  scale(data_notrf[, c("Temperature", "Fuel_Price", "CPI", "Unemployment")])

# Splitting the data
set.seed(123)
trainIndex <- createDataPartition(data$Weekly_Sales, p = .8, 
                                  list = FALSE, 
                                  times = 1)
trainData <- data[trainIndex, ]
testData  <- data[-trainIndex, ]


# Create a control object for training with cross-validation
control <- trainControl(method = "cv", # Use cross-validation
                        number = 10)  # 10-fold CV

# Train Random Forest model
set.seed(123)
rf_model_cv <- ranger(Weekly_Sales ~ ., data = trainData, importance = 'permutation')

trainData_notrf <- data_notrf[trainIndex, ]
testData_notrf  <- data_notrf[-trainIndex, ]

# Train GBM model with cross-validation
set.seed(123)
gbm_model_cv <- train(Weekly_Sales ~ ., data = trainData_notrf, method = "gbm", 
                      trControl = control, verbose = FALSE)

# Train Lasso regression model with cross-validation
set.seed(123)
lasso_model_cv <- train(Weekly_Sales ~ ., data = trainData_notrf, method = "glmnet",
                        trControl = control, lambda = 0.1)

# Access the best model (cross-validated)
best_model_gbm <- gbm_model_cv$finalModel

# Access the best model (cross-validated)
best_model_lasso <- lasso_model_cv$finalModel

response <- testData$Weekly_Sales
predictor <- as.matrix(testData[,-1])

response_notrf <- testData_notrf$Weekly_Sales
predictor_notrf <- as.matrix(testData_notrf[,-1])

# Predictions
rf_predictions_cv <- predict(rf_model_cv, data = testData)$predictions
gbm_predictions_cv <- predict(best_model_gbm, predictor_notrf)
lasso_predictions_cv <- predict(best_model_lasso, predictor_notrf)

# Evaluate models
rf_evaluation_cv <- postResample(rf_predictions_cv, response)
gbm_evaluation_cv <- postResample(gbm_predictions_cv, response)
lasso_evaluation_cv <- postResample(lasso_predictions_cv, response)

# Store the evaluation metrics in separate variables
rf_r_squared <- rf_evaluation_cv["Rsquared"] * 100
rf_rmse <- rf_evaluation_cv["RMSE"]
rf_mae <- rf_evaluation_cv["MAE"]

gbm_r_squared <- gbm_evaluation_cv["Rsquared"] * 100
gbm_rmse <- gbm_evaluation_cv["RMSE"]
gbm_mae <- gbm_evaluation_cv["MAE"]

lasso_r_squared <- lasso_evaluation_cv["Rsquared"] * 100
lasso_rmse <- lasso_evaluation_cv["RMSE"]
lasso_mae <- lasso_evaluation_cv["MAE"]

# Combine the evaluation metrics into a data frame
evaluation_table <- data.frame(
  Model = c("Random Forest", "GBM", "Lasso Regression"),
  R_squared = c(rf_r_squared, gbm_r_squared, lasso_r_squared),
  RMSE = c(rf_rmse, gbm_rmse, lasso_rmse),
  MAE = c(rf_mae, gbm_mae, lasso_mae)
)

evaluation_table$R_squared <- paste(round(evaluation_table$R_squared, 2), "%")

# PLOT! the evaluation table
print("Evaluation Metrics:")
print(evaluation_table)
grid.arrange(tableGrob(evaluation_table))

# Variable importance
importance_rf <- rf_model_cv$variable.importance
print(importance_rf)

# Create a data frame for plotting
importance_df <- data.frame(
  Variable = names(importance_rf),
  Importance = importance_rf
)

# Sort the data frame by importance
importance_df <- importance_df[order(importance_df$Importance, decreasing = TRUE), ]

# PLOT! variable importance using ggplot2
ggplot(importance_df, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  xlab("Variable") +
  ylab("Importance") +
  ggtitle("Variable Importance from Random Forest") +
  theme_minimal()


total_importance <- sum(importance_rf)

# Normalize importance values to percentages
rf_importance_percentage <- importance_rf / total_importance * 100

# Create a data frame for plotting
importance_df_perc <- data.frame(
  Variable = names(importance_rf),
  Percentage = rf_importance_percentage
)


# Sort the importance values by decreasing importance
importance_df_perc <- importance_df_perc[order(importance_df_perc$Percentage, 
                                               decreasing = TRUE),]

# PLOT! variable importance as percentages
ggplot(importance_df_perc, 
       aes(x = reorder(Variable, Percentage), y = Percentage))+
         geom_bar(stat="identity")+
            geom_text(aes(label = round(Percentage, 2)),  
            position = position_stack(vjust = 0.5),
            size = 3, color = "white") +
         coord_flip() +
         xlab("Variable") + ylab("Percentage") +
        ggtitle("Variable Importance (%) from Random Forest") + theme_minimal()

# PLOT! Correlation between predictors
numeric_data <- data[sapply(data, is.numeric)]
cor_matrix <- cor(numeric_data)
corrplot(cor_matrix, method="circle")
