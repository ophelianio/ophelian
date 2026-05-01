from datetime import datetime


class OphelianLogger:

    @staticmethod
    def get_current_time():
        time_format = "%H:%M:%S.%f"
        return datetime.now().time().strftime(time_format)[:-3]

    def get_message(self, message, level):
        time = self.get_current_time()
        ophelian_name = "Ophelian"
        return f"{time} {ophelian_name} [{level}] {message}"

    def debug(self, message):
        print(self.get_message(message, "DEBUG"))

    def info(self, message):
        print(self.get_message(message, "INFO"))

    def warning(self, message):
        print(self.get_message(message, "WARN"))

    def error(self, message):
        print(self.get_message(message, "ERROR"))

    def mask(self, message):
        print(self.get_message(message, "MASK"))

    def tape(self, message, adjust_tape=1):
        length = len(message) - adjust_tape
        print(self.get_message("+" + "-" * length + "+", "TAPE"))
