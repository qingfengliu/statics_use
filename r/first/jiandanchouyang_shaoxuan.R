#抽样与筛选

table1_2<-read.xlsx("D:\\书籍资料整理\\统计R-贾俊平\\1-2学生成绩50名.xlsx",sheetName = "Sheet1",encoding="UTF-8")

#无放回简单随机抽样
sample(table1_2$姓名,10,replace = FALSE)

#放回简单随机抽样
sample(table1_2$姓名,10,replace = TRUE)

#筛选出考试分数小于60的所有学生
sample(table1_2$姓名[table1_2$考试分数<60])


#筛选出考试分数大于等于90的所有分数
sample(table1_2$考试分数[table1_2$考试分数>=90])


