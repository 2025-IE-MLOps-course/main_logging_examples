# %%
import sys
x = 10
y = 2
print(x / y)
print("Hello, World!")
# %%
x = 10
y = 0
print(x / y)
print("Hello, World!")
# %%
x = 10
y = 0
try:
    print(x / y)
except ZeroDivisionError as e:
    print(f"An error occurred: {e}")
print("Hello, World!")

# %%
x = 10
y = 0
try:
    print(x / y)
except ZeroDivisionError as e:
    print(f"An error occurred: {e}")
    raise
print("Hello, World!")
# %%
x = 10
y = 0
try:
    print(x / y)
except ZeroDivisionError as e:
    print(f"An error occurred: {e}")
    sys.exit(1)
print("Hello, World!")
