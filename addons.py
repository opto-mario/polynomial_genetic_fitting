import random
import numpy as np
import matplotlib.pyplot as plt


# takes in the order of the equation, and then the arguments, C0, C1, C2... Cn , such as the equation is
# in the form : C0 + C1 X + C2 X^2 + .... + Cn X^n
# write the jitter in percent
def generate_fuzzy_data(order, start, stop, jitter=0,  *args):
    data = []
    coef = []
    abscisse = []

    number_of_points = 100

    if args:
        if len(args) == order:
            for i in range(len(args)):
                coef.append(args[i])
        else:
            print("number of coefficients should be equal to the order of the equation")
    else:
        coef = [random.uniform(-1, 1) for _i in range(order)]

    order_list = list(range(order))

    for x in range(number_of_points):
        data.append(0)
        abscisse.append(start+x*(stop-start)/number_of_points)
        for index in range(order):
            data[x] += coef[index] * pow(abscisse[x], order_list[index])

    span = (np.max(data) - np.min(data))

    return abscisse, list(map(lambda y: y + span * random.uniform(-jitter/100, jitter/100), data))


def create_random_couples(list_of_elements:list):
    random.shuffle(list_of_elements)
    index = 0
    result = []
    for _ in range(len(list_of_elements) // 2):
        result.append([list_of_elements[index], list_of_elements[index+1]])
        index += 2
    if len(list_of_elements) % 2:
        result.append([list_of_elements[-1]])
    return result


if __name__ == '__main__':
    data_x, data_y = generate_fuzzy_data(2,-3,3,0,5,-3)
    print("G")
