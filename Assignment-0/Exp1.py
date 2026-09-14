import csv
import random


filename = "studentsmarks.csv"

subjects = ["Math", "Physics", "Chemistry", "Computer Science"]

students = []

for i in range(1, 101):
    roll = i

    marks = {
        "Math": random.randint(20, 100),
        "Physics": random.randint(20, 100),
        "Chemistry": random.randint(20, 100),
        "Computer Science": random.randint(20, 100)
    }

    students.append([roll, marks["Math"], marks["Physics"], marks["Chemistry"], marks["Computer Science"]])

with open(filename, "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow(["Roll"] + subjects)

    writer.writerows(students)