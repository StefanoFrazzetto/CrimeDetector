import datetime


def __get_timestamp():
    return str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def info(string):
    print(f"{[__get_timestamp()]} # {string}")


def warning(string):
    print(f"{[__get_timestamp()]} ! {string}")
