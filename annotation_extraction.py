import os

# Example file name
file_name = "file-name-element1-element2-element3.txt"

# Split the file name by the hyphen ("-") separator
name_parts = file_name.split("-")

# Check if the file name has at least 3 elements
if len(name_parts) >= 3:
    # Extract the third element
    third_element = name_parts[2]
    print("Third element:", third_element)
else:
    print("File name does not have at least 3 elements")

print(name_parts)