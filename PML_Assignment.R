data = read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv");
cases = read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv");

#install.packages("caret");
library(caret);

#transform the data into the format we need before splitting into train and test
data$year <- as.numeric(substr(data$cvtd_timestamp, 7, 11)); 
data$month <- as.numeric(substr(data$cvtd_timestamp, 4, 5)); 
data$day <- as.numeric(substr(data$cvtd_timestamp, 1, 2)); 
data$hour <- as.numeric(substr(data$cvtd_timestamp, 12, 13)); 
data$minute <- as.numeric(substr(data$cvtd_timestamp, 15, 16));

#remove raw timestamp and index columns from the data
data = data[, !(colnames(data) %in% c("X", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp"))];

data$kurtosis_roll_belt       <- as.numeric(data$kurtosis_roll_belt);
data$kurtosis_picth_belt      <- as.numeric(data$kurtosis_picth_belt);
data$kurtosis_yaw_belt        <- as.numeric(data$kurtosis_yaw_belt);
data$skewness_roll_belt       <- as.numeric(data$skewness_roll_belt);
data$skewness_roll_belt.1     <- as.numeric(data$skewness_roll_belt.1);
data$skewness_yaw_belt        <- as.numeric(data$skewness_yaw_belt);
data$max_yaw_belt             <- as.numeric(data$max_yaw_belt);
data$min_yaw_belt1            <- as.numeric(data$min_yaw_belt);
data$amplitude_yaw_belt       <- as.numeric(data$amplitude_yaw_belt);
data$kurtosis_yaw_arm         <- as.numeric(data$kurtosis_yaw_arm);
data$skewness_roll_arm        <- as.numeric(data$skewness_roll_arm);
data$skewness_pitch_arm       <- as.numeric(data$skewness_pitch_arm);
data$skewness_yaw_arm         <- as.numeric(data$skewness_yaw_arm);
data$kurtosis_roll_dumbbell   <- as.numeric(data$kurtosis_roll_dumbbell);
data$kurtosis_picth_dumbbell  <- as.numeric(data$kurtosis_picth_dumbbel);
data$kurtosis_yaw_dumbbell    <- as.numeric(data$kurtosis_yaw_dumbbell);
data$skewness_roll_dumbbell   <- as.numeric(data$skewness_roll_dumbbell);
data$skewness_pitch_dumbbell  <- as.numeric(data$skewness_pitch_dumbbell);
data$skewness_yaw_dumbbell    <- as.numeric(data$skewness_yaw_dumbbell);
data$max_yaw_dumbbell         <- as.numeric(data$max_yaw_dumbbell);
data$min_yaw_dumbbell         <- as.numeric(data$min_yaw_dumbbell);
data$amplitude_yaw_dumbbell   <- as.numeric(data$amplitude_yaw_dumbbell);
data$kurtosis_roll_forearm    <- as.numeric(data$kurtosis_roll_forearm);
data$kurtosis_picth_forearm   <- as.numeric(data$kurtosis_picth_forearm);
data$kurtosis_yaw_torearm     <- as.numeric(data$kurtosis_yaw_forearm);
data$skewness_roll_forearm    <- as.numeric(data$skewness_roll_forearm);
data$skewness_pitch_forearm   <- as.numeric(data$skewness_pitch_forearm);
data$skewnessyawforearm       <- as.numeric(data$skewness_yaw_forearm);
data$max_yaw_forearm          <- as.numeric(data$max_yaw_forearm);
data$min_yaw_forearm          <- as.numeric(data$min_yaw_forearm);
data$amplitude_yaw_forearm    <- as.numeric(data$amplitude_yaw_forearm);
data$kurtosis_roll_arm        <- as.numeric(data$kurtosis_roll_arm);
data$kurtosis_picth_arm       <- as.numeric(data$kurtosis_picth_arm);

inTrain <- createDataPartition(data$classe,
                               p=3/4,
                               list = FALSE);

training <- data[inTrain,];
testing <- data[-inTrain,];

#preprocess with the median
preObj <- preProcess(training[,-156], method="medianImpute");
training.imputed <- predict(preObj, training);

#set our training control to leverage cross-validation, 5 folds
control <- trainControl(method='repeatedcv',
                        number=5,
                        repeats=3);

#train the random forest model using the training subset
modFit <- train(classe ~ ., data=training.imputed, method="rf", trControl=control);
modFit

#check the variable importance on the training set
rfImp <- varImp(modFit, scale=TRUE);
p = plot(rfImp, top=50);
p;

#keep only the top 20 predictor variables in the model, this will help prevent over fitting
p = plot(rfImp, top=20);
lst = c(p$y.limits, "classe");
training.optimized <- training.imputed[, (colnames(training.imputed) %in% lst)];

#re-train the model with only the top 20 predictors from our previous run
modFit <- train(classe ~ ., data=training.optimized, method="rf", trControl=control);
modFit

#truncate the predictor variables that we don't need (insignificant) from our testing dataset
testing.optimized <- testing[, (colnames(testing) %in% lst)];
combPred <- predict(modFit, testing.optimized);
confusionMatrix(as.factor(testing.optimized$classe), combPred);


