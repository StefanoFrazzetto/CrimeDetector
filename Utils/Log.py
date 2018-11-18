import datetime


class Log(object):
    @staticmethod
    def get_timestamp():
        return str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    @staticmethod
    def info(string):
        print(f"{[Log.get_timestamp()]} # {string}")

    @staticmethod
    def warning(string):
        print(f"{[Log.get_timestamp()]} ! {string}")
