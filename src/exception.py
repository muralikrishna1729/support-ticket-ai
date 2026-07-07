from copy import error
import sys 
from src.logger import logging

def error_message_detail(error,error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    ## (type, value, traceback)
    ## give the which file the excpetion is occured and which line all such details

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    message = (
        f"Error in script [{file_name}] "
        f"at line [{line_number}] "
        f"--->{str(error)}"
    )
    return message

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_detail=error_detail)
        logging.error(self.error_message)

    def __str__(self):
        return self.error_message

if __name__ == "__main__":
    try:
        a = 1/0
    except Exception as e:
        logging.info("Divided By Zero")
        raise CustomException(e,sys)
    

