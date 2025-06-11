# 打开TXT文件
prune_rate = 0.9
file_name1 = 'l2_src8.txt'
file_name2 = 'l2_tree8.txt'


file = open(file_name1, 'r')

# 读取文件内容
content = file.readlines()

# 将内容存储到列表中
data = [line.strip() for line in content]

print(data[0:int(len(data)*prune_rate)])
set1 = set(data[0:int(len(data)*prune_rate)])

file1 = open(file_name2, 'r')

# 读取文件内容
content1 = file1.readlines()

# 将内容存储到列表中
data1 = [line1.strip() for line1 in content1]

print(data1[0:int(len(data1)*prune_rate)])
set2 = set(data1[0:int(len(data1)*prune_rate)])

U_data = set1 & set2
print(U_data)

print(1 - len(U_data)/len(data1[0:int(len(data1)*prune_rate)]))
# 关闭文件
file.close()
file1.close()