"""
Name: executor.py
Description: Module to handle commandline interaction
Author: MT
"""
from typing import List
import subprocess
import json
import sys
from pathlib import Path

class Executor:
    def __init__(self):
        self.default_folder = Path(__file__).parent
        self.exec_status = None
        self.cmd_output = None
        self.error_messages = []

    
    def execute_command(self, cmd: str):
        try:
            proc = subprocess.Popen(cmd.split(), stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            ret = proc.wait()
            if ret < 0:
                self.exec_status = ret
        except Exception as e:
            self.exec_status = 1
            self.error_messages.append(f"Exception: {e} executing: {cmd}")
        
    def execute_return_output(self, cmd: List):
        p=subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
        try:
            stdout, err = p.communicate(timeout=10)
            self.cmd_output = stdout.splitlines()
            self.exec_status = 0
        except subprocess.TimeoutExpired:
            self.exec_status = 1
            self.error_messages.append(f"Process timeout: {err}: {cmd.join('')}")
            p.kill()

    def exit_graceful(self, error_msg=None):
        msg = error_msg if error_msg else self.error_messages[0]
        print(f"{msg}")
        sys.exit(1)
        

    def save_file_json(self, data: List, fpath: str):
        try:
            with open(fpath, 'w') as f:
                json.dump(data, f)
            f.close()
            self.exec_status = 0
        except Exception as e:
            print(e)
            self.exec_status = 1
            self.error_messages.append(f"Exception: {e} save_file_json: {fpath}")