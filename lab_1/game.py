import math
from random import randint

from visualiser import Visualiser

v0 = 50
g = 9.81
h0 = 100

class Game:

    distance = 0
    counter = 0

    def __init__(self):
        self._visualizer = Visualiser()

    def randomize_distance(self)-> None:
        self.distance = randint(50,340)

    def ask_for_alpha(self)-> float:
        while True:
            alpha = float(input("Enter alpha: "))
            if alpha < 0 or alpha > 90:
                print("alpha must be between 0 and 90 degrees")
            else:
                return alpha

    def count_distance(self, alpha: float) -> float:
        alpha_rad = math.radians(alpha)

        part1 = (v0 * math.cos(alpha_rad)) / g
        part2 = v0 * math.sin(alpha_rad)
        part3 = math.sqrt((v0 * math.sin(alpha_rad)) ** 2 + 2 * g * h0)

        distance = part1 * (part2 + part3)

        return distance

    def target_hit(self, distance: float) -> bool:
        return abs(distance - self.distance) < 5

    def game_round(self):
        self.randomize_distance()
        self.counter = 0
        print(f"Distance: {self.distance}")
        print("Round starts!")
        while True:
            alfa = self.ask_for_alpha()
            self.counter += 1
            distance = self.count_distance(alfa)
            if self.target_hit(distance):
                print(f"Hit! It took you {self.counter} shots")
                self._visualizer.visualise_trajectory(alfa, self.distance)
                break
            print(f"Distance: {distance}. Try again!")



