from biorhythm_counter import BiorhythmCounter
from game import Game
from person import Person


def exercise_1():
    person = Person.from_input()
    print(f"Hello {person._name}! You are {person.day_of_life} days old.")
    print(f"Your biorhythm today is:")
    BiorhythmCounter.get_all_diagnosis(person.day_of_life)

def exercise_2():
    Game().game_round()

if __name__ == "__main__":
    #exercise_1()
    exercise_2()