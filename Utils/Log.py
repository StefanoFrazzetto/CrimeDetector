import datetime


class Log(object):
    @staticmethod
    def get_timestamp():
        return str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    @staticmethod
    def info(message: str, timestamp=True, newline=True, header=False):
        print(f"{create_string(message, timestamp, header)}", end=get_newline(newline))

    @staticmethod
    def warning(message: str, timestamp=True, newline=True, header=False):
        message = f"WARNING: {message}"
        print(f"\n\n{create_string(message, timestamp, header)}", end=get_newline(newline))


def create_string(content, timestamp=True, header=False):
    string = "" if not header else "\n"
    if timestamp:
        string += f"[{Log.get_timestamp()}] "
    string += content

    return string


def get_newline(newline) -> str:
    return '\n' if newline is True else ''
