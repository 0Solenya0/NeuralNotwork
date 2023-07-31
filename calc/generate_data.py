import random


def generate_simple_expression(max_num=100_000_000):
    """Generate a random expression"""
    num1 = random.randint(0, max_num)
    num2 = random.randint(0, max_num)
    op = '+'  # random.choice(['+', '-', '*', '/'])
    return f'{num1}{op}{num2}={eval(f"{num1} {op} {num2}")}'


print(f'simple expression: {generate_simple_expression()}')
