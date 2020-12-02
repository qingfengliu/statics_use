library(CCA)
ccdata<-read.csv("D:/书籍资料整理/多元统计分析/例8-2.csv",encoding='UTF-8')
ECO<-ccdata[,2:7]
ECOS<-scale(ECO)  #标准化

AIR<-ccdata[,8:13]
AIRS<-scale(AIR)  #标准化

matcor(ECOS,AIRS)  #这里是求相关系数矩阵
cc1<-cc(ECOS,AIRS)   
cc1[1]  #典型相关系数
#第一个参数为0.875代表 U1和V1相关系数为0.875  然后依次类推

ccl[3:4] #输出典型系数,标准化的典型系数即典型权重
#列[,1] 代表相应变量(x或y)求出 U1或V1的公式

