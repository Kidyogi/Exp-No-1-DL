code :

test_data = [
    {"Input Face Image": "Image 1", "Expected Identity": "Person A", "Predicted Identity": "Person A", "Correct (Y/N)": "Y"},
    {"Input Face Image": "Image 2", "Expected Identity": "Person B", "Predicted Identity": "Person C", "Correct (Y/N)": "N"},
    {"Input Face Image": "Image 3", "Expected Identity": "Person D", "Predicted Identity": "Person D", "Correct (Y/N)": "Y"},
]
print("Face Recognition using CNN")
print()
headers = ["Input Face Image", "Expected Identity", "Predicted Identity", "Correct (Y/N)"]
print(f"{headers[0]:<15} {headers[1]:<17} {headers[2]:<18} {headers[3]}")
for entry in test_data:
    print(f"{entry['Input Face Image']:<15} {entry['Expected Identity']:<17} {entry['Predicted Identity']:<18} {entry['Correct (Y/N)']}")

output :

<img width="648" height="143" alt="Screenshot 2025-08-20 095043" src="https://github.com/user-attachments/assets/abe3951d-9d85-4765-a704-de6f8da34ead" />
