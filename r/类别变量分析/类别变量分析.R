###########################单变量的期望#########################
#期望频数相等
load("D:\\书籍资料整理\\统计R-贾俊平\\example7_1.RData")
chisq.test(example7_1$人数)
#期望频数不相等
load("D:\\书籍资料整理\\统计R-贾俊平\\example7_2.RData")
chisq.test(example7_2$离婚家庭数,p=example7_2$期望比例)


############################两个类别变量的独立性检验###############
#手动创建列联表
x<-c(126,158,35,34,82,65)
M<-matrix(x,nr=2,nc=3,byrow = TRUE,dimnames = list(c("满意","不满意"),c("东部","中部","西部")))
chisq.test(M)

#从文件中加载数据创建列联表
load("D:\\书籍资料整理\\统计R-贾俊平\\example7_1.RData")
chisq.test(example7_1$人数)
#期望频数不相等
load("D:\\书籍资料整理\\统计R-贾俊平\\example7_3.RData")
count<-table(example7_3)
chisq.test(count)

#############################列联表相关性度量##################
load("D:\\书籍资料整理\\统计R-贾俊平\\example7_3.RData")
count<-table(example7_3)
library(vcd)
assocstats(count)
#得到结果Phi-Coefficient φ系数，Contingency Coeff.列联系数，Cramer's V Cramer's V 系数