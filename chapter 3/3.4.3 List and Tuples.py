# List Creation
fruits_list = ["Apple", "Banana", "Cherry",
               "Dates", "Elderberry"]

# Accessing List Items
print("First fruit in the list: ", fruits_list[0])

# Modifying List Items
fruits_list[1] = "Blueberry"
print("Fruit list after modification: ", fruits_list)

# Adding Items to the List
fruits_list.append("Fig")
print("Fruit list after adding a new item: ", fruits_list)

# Removing an item from the list
fruits_list.remove("Cherry")
print("Fruit list after removing an item: ", fruits_list)



# Tuple Creation
fruits_tuple = ("Grapes", "Honeydew", "Ice-Apple")

# Accessing Tuple Items
print("First fruit in the tuple: ", fruits_tuple[0])

# Trying to modify tuple items (This will result in an error)
# fruits_tuple[1] = "Jackfruit"

# Tuples are immutable, which means you can't add or
# remove items after the tuple is defined.

# Thus, we can't demonstrate adding or removing items
# in a tuple like we did with the list
