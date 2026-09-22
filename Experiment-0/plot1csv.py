import csv
import matplotlib.pyplot as plt

filename = "generateStudents/studentsmarks.csv"

math = 0
physics = 0
chemistry = 0
computer = 0

with open(filename, "r") as file:
    reader = csv.reader(file)

    next(reader)

    for row in reader:
        math += int(row[1])
        physics += int(row[2])
        chemistry += int(row[3])
        computer += int(row[4])

subjects = ["Math", "Physics", "Chemistry", "Computer Science"]
marks = [math, physics, chemistry, computer]

plt.pie(marks, labels=subjects, autopct="%1.1f%%")

plt.title("Subject-wise Total Marks")

plt.show()