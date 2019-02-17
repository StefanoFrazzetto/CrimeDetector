import datetime
from enum import Enum

from Utils import File


class LogOutput(Enum):
    CONSOLE = 0
    FILE = 1
    BOTH = 2


class Log(object):
    output: LogOutput = LogOutput.CONSOLE
    path: str
    filename: str = "logfile.log"

    @staticmethod
    def get_timestamp():
        return str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    @staticmethod
    def info(message: str, timestamp=True, newline=True, header=False):
        Log.to_output(
            message=message,
            timestamp=timestamp,
            newline=newline,
            header=header
        )

    @staticmethod
    def warning(message: str, timestamp=True, newline=True, header=False):
        Log.to_output(
            message=f"WARNING: {message}",
            timestamp=timestamp,
            newline=newline,
            header=header
        )

    @staticmethod
    def get_log_file():
        return f"{Log.path}/{Log.filename}"

    @staticmethod
    def to_output(message: str, timestamp=True, newline=True, header=False):
        message = create_string(message, timestamp, header)

        if Log.output == LogOutput.CONSOLE or Log.output == LogOutput.BOTH:
            print(message, end=get_newline(newline))

        if Log.output == LogOutput.FILE or Log.output == LogOutput.BOTH:
            File.create_directory(Log.path)
            message = message + get_newline(newline)
            File.write_file(Log.get_log_file(), message, "a")

    @staticmethod
    def clear():
        if File.file_exists(Log.get_log_file()):
            File.delete_file(Log.get_log_file())


def create_string(content, timestamp=True, header=False):
    string = "" if not header else "\n"
    if timestamp:
        string += f"[{Log.get_timestamp()}] "
    string += content

    return string


def get_newline(newline) -> str:
    return '\n' if newline is True else ''
