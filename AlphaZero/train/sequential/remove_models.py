import os
import shutil


def remove_dir(path):
    try:
        shutil.rmtree(path)
        print(path, 'is removed.')
    except:
        print('Failed to remove', path)


if __name__ == '__main__':
    """Do not use this script anywhere in the program
    """
    remove_dir('data/models')
    remove_dir('data/selfplay')
    remove_dir('data/evaluations')
    try:
        os.remove('data/config.json')
        print('Configuration removed.')
    except:
        print('Failed to remove configuration')
