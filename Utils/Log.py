import datetime


class Log(object):
    @staticmethod
    def get_timestamp():
        return str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    @staticmethod
    def info(message: str, timestamp=True, newline=True):
        print(f"{create_string(message, timestamp)}", end=get_newline(newline))

    @staticmethod
    def warning(message: str, timestamp=True, newline=False):
        print(f"WARNING: {create_string(message, timestamp)}", end=get_newline(newline))


def create_string(content, timestamp=True):
    string = ""
    if timestamp:
        string = f"[{Log.get_timestamp()}] "
    string += content

    return string


def get_newline(newline) -> str:
    return '\n' if newline is True else ''
