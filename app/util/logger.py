import os

class Logger:
    def __init__(self, name):
        self.name = name
        self.logger_path = "./logs/app.log"

        if not os.path.exists("./logs"):
            os.makedirs("./logs")
        
        if not os.path.exists(self.logger_path):
            with open(self.logger_path, "w") as log_file:
                log_file.write(f"[INFO] [{self.name}] Logger initialized.\n")

    def info(self, message):
        with open(self.logger_path, "a") as log_file:
            log_file.write(f"[INFO] [{self.name}] {message}\n")
    
    def error(self, message):
        with open(self.logger_path, "a") as log_file:
            log_file.write(f"[ERROR] [{self.name}] {message}\n")

    def debug(self, message):
        with open(self.logger_path, "a") as log_file:
            log_file.write(f"[DEBUG] [{self.name}] {message}\n")
