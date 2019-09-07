from __future__ import print_function
from enum import IntEnum, unique
from datetime import datetime

class FileWriter:

    def __init__(self, filename = None, keepopen = False, append = False):
        self.__filename = filename
        self.__keepopen = keepopen
        self.__fobj = None
        self.__write_mode = 'a+' if append else 'w+'

        if keepopen:
            self.__open_file()

    def __open_file(self):
        if self.__filename != None:
            try:
                self.__fobj = open(self.__filename, self.__write_mode)
            except IOError as err:
                prRed("WARNING: Could not open file: {0}. Error: {1}".format(self.__filename, err.strerror))

    def write(self, string):
        
        if self.__filename == None:
            pr_red("WARNING: Tried to write: {0} to file but no file open.".format(string))
            return
        
        if self.__keepopen:
            if self.__fobj == None:
                self.__open_file()
            self.__fobj.write(string)

        else:
            with open(self.__filename, self.__write_mode) as f:
                f.write(string)

    def writeline(self, string):
        self.write(string + '\n')

    def __del__(self):
        if self.__fobj != None:
            self.__fobj.close()
        
    def pr_red(string): 
        print("\033[91m {}\033[00m" .format(string)) 


@unique
class LogLevel(IntEnum):
    Always = 1
    Warn   = 2
    Info   = 4
    Debug  = 8

class LogWriter:

    def __init__(self, log_file = None, log_level = LogLevel.Info):
        self.__log_file_writer = None
        self.__log_level = log_level

        if log_file:
            self.__log_file_writer = FileWriter(log_file, True)

    def write_log(self, log, timestamp = datetime.utcnow(), log_level = LogLevel.Info):
        
        if log_level <= self.__log_level:    
            timestamp = timestamp.strftime("%Y%m%d-%H%M%S.%f")
            log_line = "[{0}] {1} ---- {2}".format(log_level.name, timestamp, log)

            if self.__log_file_writer:
                self.__log_file_writer.writeline(log_line)
            else:
                print(log_line + '\n')

