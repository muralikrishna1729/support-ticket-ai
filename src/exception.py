import sys 
from src.logger import logger
def error_message_detail(error, error_detail:sys):
    """Extract file name, line number and error message."""

    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_msg = str(error)

    message = (
        f"Error in script [{file_name}]"
        f"at line [{line_number}]"
        f"--->{error_msg}"
    )

    return message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_detail)

    def __str__(self):
        return self.error_message



