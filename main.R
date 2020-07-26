# === Install packages as required
if(!require(h2o)){ install.packages("h2o") }
if(!require(tidyverse)){ install.packages("tidyverse") }
if(!require(gridExtra)){ install.packages("gridExtra") }

# === Load the required libraries
library(h2o)
library(tidyverse)
library(gridExtra)

# === Start H2O on local machine using all available cores.
h2o.init(ip = "localhost", port = 54321, nthreads = -1, min_mem_size = "20g")

# === Import data into h2o
train_data <- h2o.importFile(
  path = 'cifar-10-batches-py/data_train.csv',
  destination_frame = 'train_data'
)

splits <- h2o.splitFrame(train_data, 0.75, seed=1234)
train = splits[[1]]
validate = splits[[2]]

test <- h2o.importFile(
  path = 'cifar-10-batches-py/data_test.csv',
  destination_frame = 'test'
)

#h2o.ls()

# === Specify the response and predictor columns
y <- "C3073"
x <- setdiff(names(train), y)

# === Preprocess data
train <- train/255.0
validate <- validate/255.0
test <- test/255.0

# === Encode the response column as categorical
train[,y] <- as.factor(train[,y])
validate[,y] <- as.factor(validate[,y])
test[,y] <- as.factor(test[,y])

# # === Cratesian grid search (hyperparameter tuning)
# hidden_opt <- list(c(32,32), c(32,16,8), c(100))
# l1_opt <- c(1e-4,1e-3)
# hyper_params <- list(hidden = hidden_opt, l1 = l1_opt)
# 
# model_grid <- h2o.grid(
#   "deeplearning",
#   grid_id = "mygrid",
#   hyper_params = hyper_params,
#   x = x,
#   y = y,
#   distribution = "multinomial",
#   training_frame = train,
#   validation_frame = validate,
#   score_interval = 2,
#   epochs = 100,
#   activation = "RectifierWithDropout",
#   stopping_rounds = 3,
#   stopping_tolerance = 0.05,
#   stopping_metric = "misclassification",
#   ignore_const_cols = FALSE,
#   export_weights_and_biases = TRUE
# )
# 
# # View prediction errors and run times of the models
# model_grid
# 
# # Obtain validation MSE of the models
# for (model_id in model_grid@model_ids) {
#   mse <- h2o.mse(h2o.getModel(model_id), valid = TRUE)
#   print(sprintf("Model %d Validation MSE: %f", model_id, mse))
# }

# === Create neural network model
s <- proc.time()

set.seed(1105)

nn_1 = h2o.deeplearning(
  x = x,
  y = y,
  training_frame = train,
  validation_frame = validate,
  distribution = "multinomial",
  activation = "RectifierWithDropout",
  hidden = c(128, 64),
  epochs = 20,
  balance_classes = TRUE,
  model_id = "nn_1"
)

e <- proc.time()
d = e-s

print(sprintf("Model trained in %f", d))

# === Examine the performance of the trained model
h2o.performance(nn_1) # training metrics
h2o.performance(nn_1, valid = TRUE) # validation metrics

# === Get MSE of the trained model based on validation set
h2o.mse(nn_1, valid = TRUE)

# === Make predictions
pred <- h2o.predict(nn_1, test)
head(pred)

# === Obtain Confusion matrix
h2o.confusionMatrix(nn_1, test)
