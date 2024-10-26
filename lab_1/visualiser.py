import math
import matplotlib.pyplot as plt
import numpy as np
v0 = 50
g = 9.81
h0 = 100
class Visualiser:

    def count_height(self, x: float, alpha: float) -> float:
        alpha_rad = math.radians(alpha)

        t = x / (v0 * math.cos(alpha_rad))

        return h0 + v0 * math.sin(alpha_rad) * t - 0.5 * g * t ** 2

    def visualise_trajectory(self, alpha: float, distance: int) -> None:
        x_values = np.linspace(0, distance, 500)
        y_values = np.array([self.count_height(x, alpha) for x in x_values])
        plt.plot(x_values, y_values)
        plt.xlabel("Distance[m]")
        plt.ylabel("Height[m]")
        plt.title("Projectile trajectory")
        plt.savefig("trajectory.png")