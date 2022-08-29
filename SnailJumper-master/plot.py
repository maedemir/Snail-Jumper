import matplotlib.pyplot as plt

f = open("result/result.txt", "r")
lines = f.readlines()
generation_index = 0
x = []
mins = []
maxs = []
averages = []
for line in lines:
    generation_index += 1
    x.append(generation_index)
    fitness_values = line.split(" ")
    fitness_values.remove('\n')
    fitness_values = [int(i) for i in fitness_values]
    # print(numbers)
    avg = 0
    max_num = int(fitness_values[0])
    min_num = int(fitness_values[0])
    avg = sum(fitness_values) / len(fitness_values)
    max_num = max(fitness_values)
    min_num = min(fitness_values)
    averages.append(avg)
    mins.append(min_num)
    maxs.append(max_num)

plt.plot(x, averages, 'b', label='average')
plt.plot(x, mins, 'r', label='min')
plt.plot(x, maxs, 'g', label='max')
plt.legend(loc="upper right")
plt.xlabel("generation")
plt.ylabel("fitness")
plt.show()