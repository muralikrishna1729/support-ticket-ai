import sys 
import logging

def error_message_detail(error,error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    ## (type, value, traceback)
    ## give the which file the excpetion is occured and which line all such details

    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error Occured in python Script name [{0}] line number [{1}] error message is [{2}]".format(file_name,exc_tb.tb_lineno, str(error)) 


class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_detail=error_detail)
        logging.error(self.error_message)

    def __str__(self):
        return self.error_message


