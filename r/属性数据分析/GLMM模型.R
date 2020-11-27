library('lme4')

table<-read.csv('D:/书籍资料整理/属性数据分析/抑郁症治疗_展开.csv',encoding='UTF-8')
(fm1 <- glmer(zhi ~ severity + drug + zhous+drug:zhous + (1 | zu), table, family = binomial))
summary(fm1)


table<-read.csv('D:/书籍资料整理/属性数据分析/老鼠给R.csv',encoding='UTF-8')
#这个结果与书中的结果一致
#nAGQ设置高斯-埃尔米特求积 求积点个数
(fm1 <- glmer(siwang ~ factor(zu)  + (1 | cu), table, family = binomial, nAGQ = 9))
summary(fm1)


