import csv
import matplotlib.pyplot as plt

filename = "generateStudents/studentsmarks.csv"

less_50 = 0
between_50_80 = 0
between_80_100 = 0

with open(filename, "r") as file:
    reader = csv.reader(file)

    next(reader)#skip first row

    for row in reader:
        total = int(row[1]) + int(row[2]) + int(row[3]) + int(row[4])
        average = total / 4

        if average < 50:
            less_50 += 1
        elif average < 80:
            between_50_80 += 1
        else:
            between_80_100 += 1

categories = ["Less than 50", "50-80", "80-100"]
students = [less_50, between_50_80, between_80_100]

plt.bar(categories, students)

plt.xlabel("Overall Marks")
plt.ylabel("Number of Students")
plt.title("Students Based on Overall Marks")

plt.show()