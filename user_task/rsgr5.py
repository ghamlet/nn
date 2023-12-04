import subprocess

cmd = "git add ." # Здесь вместо date Ваша команда для git

returned_output = subprocess.check_output(cmd) # returned_output содержит вывод в виде строки байтов

print('Результат выполнения команды:', returned_output.decode("utf-8")) 