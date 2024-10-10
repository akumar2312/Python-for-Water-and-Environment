def outer_function(outer_arg):
    print(f"Outer function argument: {outer_arg}")

    # This is a nested or inner function
    def inner_function(inner_arg):
        return outer_arg * inner_arg

    # Calling the nested function
    return inner_function(5)


result = outer_function(10)
print(f"Result from inner function: {result}")
