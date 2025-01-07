students = {
    1: {"name": "ani", "age": 19},
    2: {"name": "minnu", "age": 20},
    3: {"name": "ish", "age": 21},
    4: {"name": "siri", "age": 22},
    5: {"name": "chinnu", "age": 18}
}

for student_id, details in students.items():
    print(f"Student ID: {student_id}, Name: {details['name']}, Age: {details['age']}")
