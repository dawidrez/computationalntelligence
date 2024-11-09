from datetime import datetime


class Person:

    def __init__(self, name: str, date_of_birth: datetime):
        self._name = name
        self._date_of_birth = date_of_birth

    @property
    def day_of_life(self):
        return (datetime.now() - self._date_of_birth).days

    @staticmethod
    def _enter_name() -> str:
        return input("Enter your name: ")

    @staticmethod
    def _validate_date_of_birth(date_of_birth: datetime):
        if date_of_birth > datetime.now():
            raise ValueError("Date of birth cannot be in the future")
        if datetime.now().year - date_of_birth.year > 150:
            raise ValueError("You are too old")

    @staticmethod
    def _enter_date_of_birth() -> datetime:
        while True:
            try:
                year = int(input("Enter your year of birth: "))
                month = int(input("Enter your month of birth: "))
                day = int(input("Enter your day of birth: "))
                date_birth = datetime(year, month, day)
                Person._validate_date_of_birth(date_birth)
                return date_birth
            except ValueError as e:
                print(e)

    @classmethod
    def from_input(cls):
        name = cls._enter_name()
        date_of_birth = cls._enter_date_of_birth()
        return cls(name, date_of_birth)



