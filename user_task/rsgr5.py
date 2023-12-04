import subprocess
import time

command = ["git add .",
           "git commit -m 'hello_from_home' ",
           "git push"
           ]

while True:
    time.sleep(5)#600

    for cmd in command:
        returned_output = subprocess.check_output(cmd)
        print('Результат выполнения команды:', returned_output.decode("utf-8"))
        time.sleep(2)

         
