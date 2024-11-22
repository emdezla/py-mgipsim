import os, shutil

test_directory = './Test'
for dirpath, dirnames, filenames in os.walk(test_directory):
    if '__pycache__' in dirnames:
        pycache_path = os.path.join(dirpath, '__pycache__')
        shutil.rmtree(pycache_path)

pymgipsim_directory = os.path.dirname(os.getcwd()) + '\\pymgipsim'
if os.path.isdir(os.path.join(pymgipsim_directory, '__pycache__')):
    shutil.rmtree(os.path.join(pymgipsim_directory, '__pycache__'))

directory=os.path.join(pymgipsim_directory, '.idea')
if os.path.isdir(directory):
    shutil.rmtree(directory)

directory=os.path.join(pymgipsim_directory, '.vscode')
if os.path.isdir(directory):
    shutil.rmtree(directory)

directory=os.path.join(pymgipsim_directory, '.idea')
if os.path.isdir(directory):
    shutil.rmtree(directory)

directory=os.path.join(pymgipsim_directory, '.vscode')
if os.path.isdir(directory):
    shutil.rmtree(directory)


for dirpath, dirnames, filenames in os.walk(pymgipsim_directory):
    if '__pycache__' in dirnames:
        pycache_path = os.path.join(dirpath, '__pycache__')
        shutil.rmtree(pycache_path)

