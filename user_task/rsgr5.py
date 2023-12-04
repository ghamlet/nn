import subprocess

#cmd = "git commit -m 'efer' "
#cmd = "git push"
cmd  = "git add ."
returned_output = subprocess.check_output(cmd) # returned_output содержит вывод в виде строки байтов

print('Результат выполнения команды:', returned_output.decode("utf-8")) 
#dfghjkop;