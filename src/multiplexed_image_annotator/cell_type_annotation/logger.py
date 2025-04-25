import time
import os

class Logger:
    def __init__(self, main_dir): 
        os.makedirs(os.path.join(main_dir, "results"), exist_ok=True)
        self.log_file_path = os.path.join(main_dir, "results/log.txt")
        self.log_file = open(self.log_file_path, "w")
        self.log_file.write("Log file created at {}\n".format(time.ctime()))

    def log(self, message):
        self.log_file.write(message + "\n")

    def log_all_hyperparameters(self, hyperparameters):
        self.log_file.write("Hyperparameters:\n")
        for key, value in hyperparameters.items():
            self.log_file.write(f"{key}: {value}\n")

    def close(self):
        self.log_file.close()

        