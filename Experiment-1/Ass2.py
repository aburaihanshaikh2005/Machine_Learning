'''Construct a DataFrame containing the following columns:Student_Name,Roll_Number,Marks,and Attendence.
Filter and display only the records of students who have scored above 80 marks.'''
import pandas as pd
data={
    "Student_Name":["Raihan","Obaid","Aniket","Prashang","Yameen","Sanjeev","Raj"],
    "Roll_Number":[1,2,3,4,5,6,7],
    "Marks":[96,68,78,70,81,88,89],
    "Attendence":[78,75,72,68,81,82,66]
}
df=pd.DataFrame(data)
print(df)
filtered_df = df[df["Marks"] > 80]
print(filtered_df)