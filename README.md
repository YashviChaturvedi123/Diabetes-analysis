library(class)  # for KNN
library(e1071)  # for Naive Bayes
library(rpart)  # for Decision Tree
library(ROCR)   # for ROC curve plotting

url <- "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names <- c("Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome")
pima_data <- read.csv(url, header = FALSE, col.names = column_names)
head(pima_data)

set.seed(123)
train_index <- sample(1:nrow(pima_data), 0.8 * nrow(pima_data))
train_data <- pima_data[train_index, ]
test_data <- pima_data[-train_index, ]

train_x <- train_data[, -9]  
train_y <- train_data$Outcome
test_x <- test_data[, -9]
test_y <- test_data$Outcome

# Model 1: K-Nearest Neighbors (KNN)
knn_pred <- knn(train = train_x, test = test_x, cl = train_y, k = 3)

# Model 2: Decision Tree
dt_model <- rpart(Outcome ~ ., data = train_data, method = "class")
dt_pred <- predict(dt_model, test_data, type = "class")

# Model 3: Naive Bayes
nb_model <- naiveBayes(Outcome ~ ., data = train_data)
nb_pred <- predict(nb_model, test_data)

# Confusion matrices for each models
knn_cm <- table(Predicted = knn_pred, Actual = test_y)
dt_cm <- table(Predicted = dt_pred, Actual = test_y)
nb_cm <- table(Predicted = nb_pred, Actual = test_y)

print("KNN Confusion Matrix:")
print(knn_cm)
print("Decision Tree Confusion Matrix:")
print(dt_cm)
print("Naive Bayes Confusion Matrix:")
print(nb_cm)

# Accuracy for each model
knn_accuracy <- mean(knn_pred == test_y)
dt_accuracy <- mean(dt_pred == test_y)
nb_accuracy <- mean(nb_pred == test_y)

cat("KNN Accuracy: ", knn_accuracy, "\n")
cat("Decision Tree Accuracy: ", dt_accuracy, "\n")
cat("Naive Bayes Accuracy: ", nb_accuracy, "\n")

#ROC curves
par(mfrow = c(1, 3)) 

# KNN ROC curve
knn_prob <- as.numeric(knn_pred) 
knn_pred_obj <- prediction(knn_prob, test_y)
knn_perf <- performance(knn_pred_obj, "tpr", "fpr")
plot(knn_perf, main = "KNN ROC Curve", col = "blue", lwd = 2)

# Decision Tree ROC curve
dt_prob <- as.numeric(dt_pred) 
dt_pred_obj <- prediction(dt_prob, test_y)
dt_perf <- performance(dt_pred_obj, "tpr", "fpr")
plot(dt_perf, main = "Decision Tree ROC Curve", col = "red", lwd = 2)

# Naive Bayes ROC curve
nb_prob <- as.numeric(nb_pred)  
nb_pred_obj <- prediction(nb_prob, test_y)
nb_perf <- performance(nb_pred_obj, "tpr", "fpr")
plot(nb_perf, main = "Naive Bayes ROC Curve", col = "green", lwd = 2)

par(mfrow = c(1, 1))

#A bar plot for the comparative analysis of accuracies
model_names <- c("KNN", "Decision Tree", "Naive Bayes")
accuracies <- c(knn_accuracy, dt_accuracy, nb_accuracy)

#Bar chart
barplot(accuracies, 
        names.arg = model_names, 
        col = c("blue", "red", "green"), 
        main = "Model Accuracy Comparison", 
        ylab = "Accuracy", 
        ylim = c(0, 1), 
        border = "white", 
        cex.names = 1.2)
text(x = 1:length(accuracies), y = accuracies, 
     labels = round(accuracies, 2), pos = 3, cex = 1.2, col = "black")
