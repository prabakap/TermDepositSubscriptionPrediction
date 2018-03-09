# Term Deposit Subscription Prediction
Our [dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing) is related with direct marketing campaigns of a Portuguese banking institution. Our objective here is to analyze and find feature which leads to the **product (bank term deposit) subscription** so our bank can optimize the future marketing campaigns.

## Challenge - Balancing Class Imbalance Problem
One of the important problem we have to tackle is [Class Imbalance problem](http://www.chioka.in/class-imbalance-problem/) - Our dependent variable is *not equally distributed*, the number of customer who subscribed our campaign are very less when compared to the other, this often lead to the [Accuracy Paradox Problem](https://en.wikipedia.org/wiki/Accuracy_paradox), if our model blindly predict all of our customer will not subscribe we will have our *Accuracy Score closely to 88%*.

I have used [SMOTE](https://arxiv.org/pdf/1106.1813.pdf) to solve the Class Imbalance problem. For the implementation details please look at my .rmd file

## Machine Learning Models
-[Decision Trees](https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/)
-[Random Forest](https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/)

## Model Evaluated On
-[Recall Score](https://en.wikipedia.org/wiki/Precision_and_recall)
-[F1 Score](https://en.wikipedia.org/wiki/F1_score)
-[Accuracy](https://en.wikipedia.org/wiki/Precision_and_recall)
