import pandas as pd
data={
    "Student_Name":["Raihan","Yammen","Obaid","Sohel","Rahul"],
    "Roll_Number":[1,2,3,4,5],
    "Marks":[90,85,65,56,39],
    "Attendence":[75,72,63,89,74]
}
df=pd.DataFrame(data)

# Add New Column Grade
def Calculation_Grade(marks):
    if(marks>=80):
        return "A"
    elif(marks>=60):
        return "B"
    elif(marks>=40):
        return "C"
    else:
        return "D"
df["Grade"]=df["Marks"].apply(Calculation_Grade)
print(df)