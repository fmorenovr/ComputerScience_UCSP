#! /usr/bin/env python

# -*- coding: utf-8 -*-
# encoding: utf-8

def getNota(nota):
  if 17<=nota and nota<=20:
    return 4.0, "A"
  elif 14<=nota and nota<17:
    return 3.0, "B"
  elif 12<=nota and nota<14:
    return 2.0, "C"
  else:
    return 1.0, "D"

def getGPA(courses_list, depurar=False):
  total_credits = 0.0
  ponderate_credits = 0.0
  ponderate_grades = 0.0
  for name_couse, credit, grade in courses_list:
    points, letter = getNota(grade)
    if depurar and letter=="D":
      continue    
    total_credits += credit
    ponderate_credits += credit*points
    ponderate_grades += credit*grade
    #print(credit, points, grade, total_credits, ponderate_credits)
  
  return ponderate_credits/total_credits, ponderate_grades/total_credits


courses_list = [
# 2018-I
("Lineal", 4, 15.63), ("EDA", 4, 13.66), ("Paralelas", 4, 15.73), ("Seminario de Tesis I", 4, 20),
# 2018-II
("Imágenes", 4, 12.75), ("IA", 4, 14.02), ("Gráficos", 4, 14.60), ("Seminario de Tesis II", 4, 19),
# 2019-I
("Seminarios Avanzados de Computación", 4, 14), ("Seminario de Tesis III", 12, 15),
# 2019-II
("Seminario de Tesis IV", 16, 16),
]

GPA, AVG = getGPA(courses_list, False)

print("No Depurado GPA: ", GPA, "Average: ", AVG)

GPA, AVG = getGPA(courses_list, True)

print("Depurado GPA: ", GPA, "Average: ", AVG)
