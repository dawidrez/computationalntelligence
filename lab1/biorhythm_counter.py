from math import sin, pi


class BiorhythmCounter:
    @staticmethod
    def get_physical(day_of_life)-> float:
        return sin(2 * pi * day_of_life / 23)

    @staticmethod
    def get_emotional(day_of_life)-> float:
        return sin(2 * pi * day_of_life / 28)
    @staticmethod
    def get_intellectual(day_of_life)-> float:
        return sin(2 * pi * day_of_life / 33)

    @staticmethod
    def get_all(day_of_life)-> dict:
        return {
            'physical': BiorhythmCounter.get_physical(day_of_life),
            'emotional': BiorhythmCounter.get_emotional(day_of_life),
            'intellectual': BiorhythmCounter.get_intellectual(day_of_life)
        }

    @staticmethod
    def get_physical_diagnosis(day_of_life)-> tuple[float, str]:
        physical = BiorhythmCounter.get_physical(day_of_life)
        diagnosis = "Your physical biorhythm is neutral."
        if physical > 0.5:
            diagnosis = "Your physical biorhythm is at its peak"
        if physical < -0.5:
            diagnosis =  "Your physical biorhythm is at its lowest."
            physical_tomorrow = BiorhythmCounter.get_physical(day_of_life + 1)
            if physical_tomorrow > physical:
                diagnosis += " Tomorrow will be better."
        return physical, diagnosis

    @staticmethod
    def get_intellectual_diagnosis(day_of_life) -> tuple[float, str]:
        intellectual = BiorhythmCounter.get_intellectual(day_of_life)
        diagnosis = "Your intellectual biorhythm is neutral."
        if intellectual > 0.5:
            diagnosis = "Your intellectual biorhythm is at its peak"
        if intellectual < -0.5:
            diagnosis = "Your intellectual biorhythm is at its lowest."
            intellectual_tomorrow = BiorhythmCounter.get_intellectual(day_of_life + 1)
            if intellectual_tomorrow > intellectual:
                diagnosis += " Tomorrow will be better."
        return intellectual, diagnosis

    @staticmethod
    def get_emotional_diagnosis(day_of_life) -> tuple[float, str]:
        emotional = BiorhythmCounter.get_emotional(day_of_life)
        diagnosis = "Your emotional biorhythm is neutral."
        if emotional > 0.5:
            diagnosis = "Your emotional biorhythm is at its peak"
        if emotional < -0.5:
            diagnosis = "Your emotional biorhythm is at its lowest."
            emotional_tomorrow = BiorhythmCounter.get_physical(day_of_life + 1)
            if emotional_tomorrow > emotional:
                diagnosis += " Tomorrow will be better."
        return emotional, diagnosis

    @staticmethod
    def get_all_diagnosis(day_of_life) -> None:
        physical, physical_diagnosis = BiorhythmCounter.get_physical_diagnosis(day_of_life)
        emotional, emotional_diagnosis = BiorhythmCounter.get_emotional_diagnosis(day_of_life)
        intellectual, intellectual_diagnosis = BiorhythmCounter.get_intellectual_diagnosis(day_of_life)
        print(f"Physical: {physical} - {physical_diagnosis}")
        print(f"Emotional: {emotional} - {emotional_diagnosis}")
        print(f"Intellectual: {intellectual} - {intellectual_diagnosis}")