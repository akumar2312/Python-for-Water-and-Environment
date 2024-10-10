# Writing to a file
with open('myfile.txt', 'w') as file:
    file.write('Hello World!\n')
    file.write('This is a simple Python script '
               'illustrating file operations.\n')

# Reading from a file
with open('myfile.txt', 'r') as file:
    print("Contents of myfile.txt:")
    print(file.read())

# Appending to a file
with open('myfile.txt', 'a') as file:
    file.write('This line was appended to the '
               'file.\n')

# Reading from a file after appending
with open('myfile.txt', 'r') as file:
    print("Contents of myfile.txt after "
          "appending:")
    print(file.read())
