# This is a single line comment in Python

"""
This is a
multi-line comment (or docstring)
in Python
"""

# Import the built-in math module
import math


# Define a function
def calculate_circle_area(radius):
    """
    This function calculates the area of a circle
    given a radius.
    """
    if radius < 0:
        print("Error: Radius cannot be negative.")
        return
    area = math.pi * (radius ** 2)
    return area


# Call the function
radius = 10
area = calculate_circle_area(radius)
print(f"The area of the circle with radius "
      f"{radius} is {area:.2f}")
