from FloatGA import FloatGA
import numpy as np
import matplotlib.pyplot as plt


def graph(formula, x_range):
    x = np.array(x_range)
    y = eval(formula)
    plt.plot(x, y)
    plt.show()

def get_mse(chromosome, x_points, y_points):
    x_array = np.array([[x_points[i] ** j for j in range(np.size(chromosome))]for i in range(np.size(x_points))])
    y_predicted = chromosome.dot(x_array.transpose())
    mse = (1 / np.size(y_points)) * np.sum((y_points - y_predicted) ** 2, axis=1)
    return mse


t = int(input())
for i in range(t):
    p, d = tuple((map(int, input().strip().split(' '))))
    x_points = np.empty(p)
    y_points = np.empty(p)
    for j in range(p):
        x, y = tuple((map(float, input().strip().split(' '))))
        x_points[j] = x
        y_points[j] = y

    plt.figure()
    plt.scatter(x_points, y_points)
    plt.show()
    ga = FloatGA(pop_num=2000, chromosome_length=d+1, points_x=x_points, points_y=y_points)
    chromosome, mse = ga.apply_algorithm(max_gen=1000)
    genes = np.array([[0.429163, 1.18487, -0.717967, 0.0854301]])
    print(get_mse(genes, x_points, y_points))
    print(chromosome, " ", mse)

