## Предварительный код для 
## Coursera Machine Learning Course project
## march 2015

## MAIN
require(randomForest)
require(caret)
setwd("~/Документы/coursera/data-science/m-learning/project/")
## download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", "data/pml-training.csv", method = "curl")
## download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", "data/pml-testing.csv", method = "curl")

## Six young health participants were asked to perform one set of 10 repetitions 
## of the Unilateral Dumbbell Biceps Curl in five different fashions: 
## exactly according to the specification (Class A), 
## throwing the elbows to the front (Class B), 
## lifting the dumbbell only halfway (Class C), 
## lowering the dumbbell only halfway (Class D) 
## and throwing the hips to the front (Class E).
## For data recording we used four 9 degrees of freedom Razor
## inertial measurement units (IMU), which provide three-axes
## acceleration,  gyroscope  and  magnetometer  data  at  a  joint
## sampling rate of 45 Hz.  Each IMU also featured a Bluetooth
## Figure 1:  Sensing setup
## module to stream the recorded data to a notebook running
## the Context Recognition Network Toolbox [3].  We mounted
## the  sensors  in  the  users'  glove,  armband,  lumbar  belt  and
## dumbbell (see Figure 1).  We designed the tracking system
## to be as unobtrusive as possible, as these are all equipmentm
## commonly used by weight lifters.

## Read more: http://groupware.les.inf.puc-rio.br/har#ixzz3UlZb8ZjP

trainData <- read.csv("data/pml-training.csv")
testData <- read.csv("data/pml-testing.csv")

## Проверяем идентичность структуры данных:
dim(trainData); dim(testData)

## Вариант один - по наличию во втором наборе данных
which(!(names(trainData) %in% names(testData)))
which(!(names(testData) %in% names(trainData)))

## Вариант два - по совпадению позиции во втором наборе данных
dif <- which(names(trainData) != names(testData))


names(trainData[dif])
names(testData[dif])

## Удаляем (или просто фиксируем как неиспользуемые) неполные, испорченные колонки, строки. 

## Проверяем строки
sum(complete.cases(trainData))
sum(complete.cases(testData))

## Полных строк ничтожно мало, поэтому имеет смысл проверить колонки

sum(complete.cases(t(trainData)))
sum(complete.cases(t(testData)))


## Тоже немного. В тестовом наборе всего 60. И важная деталь: в тестовом наборе меньше полных колонок, чем в исследуемом. 
## Поскольку строк в тестовом всего 20, то моделировать данные нет смысла - принимаем решение удалить из 
## исследуемого набора все колонки, которые являются неполными в тестовом наборе

## Ищем заведомо пустые колонки 
## В тестовом наборе!!!
## создаем вектор, значения которого - количество NA в колонке с тем же номером, что и номер значения в векторе:
naTst <- apply(testData, MARGIN = 2, FUN = function(n){sum(is.na(n))})
## Преобразуем его в вектор с номерами колонок, в которых все значения NA
naTst <- which(naTst == 20)
## Проверяем количество колонок
length(naTst)
testData <- testData[,-naTst]
trainData <- trainData[,-naTst]
dim(testData)
dim(trainData)
## Теперь всё значительно лучше. 

## Проверим на полноту данных
sum(is.na(trainData))
sum(is.na(testData))

## Уберем технические переменные: 
trainData <- trainData[,8:60]
testData <- testData[,8:60]




fitModel <- train(classe ~ ., data = trainData, method = "rf", ntree=50, 
                  trControl = trainControl(method = "cv", number = 10))

fitModel$resample
fitModel$finalModel

varImp(fitModel)

dotPlot(varImp(fitModel))

mean(fitModel$resample$Accuracy)

predVal <- predict(fitModel, newdata = testData)

predVal


pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(predVal)

## confusionMatrix(fitModel)

str(trainData)
str(testData)

## ШАГИ ПО ДАЛЬНЕЙШЕМУ СОКРАЩЕНИЮ ЧИСЛА ПЕРЕМЕННЫХ - ОПЦИОНАЛЬНО
## проверим ещё колонки на низкую дисперсию: 
nearZeroVar(x = trainData, saveMetrics = T)$nzv


## смотрим состав данных
str(testData)


## Проверим сравнимость данных по users
summary(testData$user_name); summary(trainData$user_name)
summary(testData$new_window); summary(trainData$new_window)
summary(testData$num_window); summary(trainData$num_window)

## 
corMatrix <- cor(trainData[,-53])
hiCor <- findCorrelation(corMatrix, cutoff = .75)
hiCor
trainData <- trainData[,-hiCor]
str(trainData)
dim(trainData)
testData <- testData[,-hiCor]

## используем PCA
pcaTrain <- preProcess(trainData[,-53], method = "pca", thresh = .95, outcome = trainData$classe)
pcaTrain
pcaTrain$ranges

pcaTrain$rotation

## Bootstrap                            train_control1 <- trainControl(method="boot", number=100)
## k-fold Cross Validation              train_control2 <- trainControl(method="cv", number=10)
## Repeated k-fold Cross Validation     train_control3 <- trainControl(method="repeatedcv", number=10, repeats=3)
## Leave One Out Cross Validation       train_control4 <- trainControl(method="LOOCV")






filterVarImp(trainData[,-53], trainData[,53])





str(trainData)
plot(trainData[,6], trainData$classe)

i <- 10
boxplot(trainData[,i] ~ trainData$classe, main = names(trainData)[i])



table(trainData$num_window, trainData$raw_timestamp_part_1)

plot(trainData[,13] ~ trainData$raw_timestamp_part_1, main = names(trainData)[14])

## проверим отношения

