install.packages("dplyr")
install.packages("lmtest")
library(zoo)
library(lmtest)
library(dplyr)


# ------------Data Cleaning--------------------------------------------------------
# here, first we discard the following factors:
# Name, Ticket, Cabin and Embarked. It seems obvious that particular name, ticket and the port of embarkation
# cannot really affect the probability of a person dying in the voyage. The variable "Cabin" may have 
# some effect, but the same effect should be covered by the variable "PClass".


# Following our observation, we take the dataset "train.csv" and prune the dataset by keeping the factors 
# we think may have an effect on the result.
#----------------------------------------------------------------------------------

data <- read.csv("train.csv")
attach(data)
data <- select(data, Survived,Pclass,Sex,Age,SibSp,Parch,Fare)

#------------------------------------------------------------------------------------
# The difficulty we faced with the pruned dataset was the missing age values. To remedy this problem, 
# we first checked the histogram to determine the distribution of age, which came out to be approximately
# normal. After calculating the parameters, we created the required number of age values to replace the
# missing age values.
#------------------------------------------------------------------------------------

miss_count <- as.numeric(sum(is.na(data$Age)))
avg_age <- mean(data$Age, na.rm = TRUE)
sd_age <- sd(data$Age, na.rm = TRUE)

# This part is to ensure that the generated age values are nonnegative. Obviously, age cannot be negative!
repeat {
  impute_age_NA <- rnorm(miss_count,avg_age,sd_age)
  if ((length(which(impute_age_NA<0)))==0){break}
}

# Imputuing the generated datasets into the missing age values
data <- data[order(data$Age,na.last = F), ]
data$Age[1:miss_count] <- impute_age_NA

#------------------------------------------------------------------------------------
# Here, we divide the dataset into two parts: train_data and valid_data. Train_data contains the first
# 600 data, while valid_data contains the rest. The idea is to generate a good enough model, based on 
# statistical methodologies; then apply the model on valid_data to check the result. 
#------------------------------------------------------------------------------------

train_data<- data[1:600,]
valid_data <- data [601:dim(data)[1],]

#------------ Convert the classes to numeric-----------------------------------------
# Here, we transform the classes of each explanatory variable to numeric, so the function that generates
# the model works smoothly.
#------------------------------------------------------------------------------------

train_data$Sex<- sapply(as.character(train_data$Sex),switch, "male"=1,"female"=0, USE.NAMES = F)
train_data$Pclass <- as.numeric(train_data$Pclass)
train_data$Sex <- as.numeric(train_data$Sex)
train_data$Age <- as.numeric(train_data$Age)
train_data$SibSp <- as.numeric(train_data$SibSp)
train_data$Parch <- as.numeric(train_data$Parch)
train_data$Fare <- as.numeric(train_data$Fare)


valid_data$Sex<- sapply(as.character(valid_data$Sex),switch, "male"=1,"female"=0, USE.NAMES = F)
valid_data$Pclass <- as.numeric(valid_data$Pclass)
valid_data$Sex <- as.numeric(valid_data$Sex)
valid_data$Age <- as.numeric(valid_data$Age)
valid_data$SibSp <- as.numeric(valid_data$SibSp)
valid_data$Parch <- as.numeric(valid_data$Parch)
valid_data$Fare <- as.numeric(valid_data$Fare)

#------------Statistical methods to determine the effective variables---------------------

model_1 <- glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare, data = train_data)
summary(model_1)

#----------Correlation Check---------------------------------------------------------
# Here, we generate the correlation matrix to find out the variables that are highly correlated 
# with each other. Then we develop another model to check if the correlations are important or not.

install.packages("Hmisc")
library(Hmisc)
rcorr(as.matrix(train_data),type = ("spearman"))

# We can see from the result that, the correlation between the important explanatory variables are
# insignificant. So we do not need to consider the correlations in our model

#--------Multiple Logistics Regression Model------------------------------------------
# As the outcomes are binary, it is logical to start with multiple logistics regression. Here, we used the
# variables that has statistical importance.
#--------------------------------------------------------------------------------------

model_5 <- glm(Survived ~ Pclass + Sex + Age + SibSp, data = train_data, family = binomial)
summary(model_5)
anova(model_5)

#------------Results------------------------------------------------------------

fitted.results <- predict(model_5,newdata=subset(valid_data,select=c(2,3,4,5)),type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)

misClasificError <- mean(fitted.results != valid_data$Survived)
print(paste('Accuracy',1-misClasificError))

install.packages("ROCR")
library(ROCR)

p <- predict(model_5, newdata=subset(valid_data,select=c(2,3,4,5)), type="response")
pr <- prediction(p, valid_data$Survived)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)

mlr.auc <- performance(pr, measure = "auc")
mlr.auc <- mlr.auc@y.values[[1]]
mlr.auc



#---------------------Random Forest------------------------------------------
# Here, we used random forest modeling to see if it generates a better predictive model than 
# multiple logistics regression model
#---------------------------------------------------------------------------

install.packages("randomForest")
install.packages("MASS")
library(randomForest)
library(MASS)

# It is imperative to change the class of response variable to "Factor" to apply random forest.
data$Survived <- as.factor(data$Survived)
train_data$Survived <- as.factor(train_data$Survived)
valid_data$Survived <- as.factor(valid_data$Survived)

# As random forest generates the confusion matrix for any given dataset, we do not need to use the 
# train_data to generate the model; random forest can be applied to the whole dataset, "train.csv".

surv.rf <- randomForest(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare, data = data)
print(surv.rf)

#------------------Results-----------------------------------------------------

predictions=as.vector(surv.rf$votes[,2])
pred=prediction(predictions,Survived)

rf.AUC=performance(pred,"auc") #Calculate the AUC value
rf.AUC=rf.AUC@y.values[[1]]

perf_ROC=performance(pred,"tpr","fpr") #plot the actual ROC curve
plot(perf_ROC, main="ROC plot")
text(0.5,0.5,paste("AUC = ",format(AUC, digits=5, scientific=FALSE)))
rf.AUC

# According to the AUC curves, multiple logictics regression model is a better model than the random forest model.