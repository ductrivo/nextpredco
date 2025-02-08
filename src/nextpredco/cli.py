import subprocess
from pathlib import Path

from nextpredco.core import examples, tasks
from nextpredco.core._logger import logger
from nextpredco.core.settings import create_settings_template
from nextpredco.core.tools import print_dict

BACK_KEY: int = 88
BACK_STR: str = 'Back'
EXIT_KEY: int = 99
EXIT_STR: str = 'Exit'
YES_STR: str = 'Yes'


def menu_main():
    _clear_screen()
    print('Welcome to NextPredCo!')
    menu = [
        'Initialize the project',
        'Create settings file',
        'Compare simulation data',
        'Compare different settings',
        'Test',
    ]

    choice = _get_choice(menu, add_back=False)

    if choice == 'Initialize the project':
        menu_init_project()

    elif choice == 'Create settings file':
        create_settings_template()

    elif choice == 'Compare simulation data':
        tasks.simulation_with_example_data()

    elif choice == 'Compare different settings':
        tasks.simulation_different_settings()

    elif choice == 'Test':
        tasks.simulate_control_system()

    elif choice == EXIT_KEY:
        return


def menu_init_project():
    menu = [
        'Initialize project in a new directory',
        'Initialize project in the current directory',
    ]
    choice = _get_choice(menu=menu)

    if choice == 'Initialize project in a new directory':
        dir_name = _input_dir_name()
        menu_choose_example_project(
            dir_name=dir_name,
            back_func=menu_init_project,
        )

    elif choice == 'Initialize project in the current directory':
        menu_choose_example_project(
            dir_name='',
            back_func=menu_init_project,
        )

    elif choice == BACK_STR:
        menu_main()

    elif choice == EXIT_STR:
        exit()


def menu_choose_example_project(dir_name: str, back_func):
    projects = examples.list_example_projects()
    choice = _get_choice(
        projects,
        title='\nChoose an example project:',
        add_back=True,
    )
    if choice == BACK_STR:
        back_func()
    elif choice == EXIT_STR:
        exit()
    else:
        tasks.init_dir(work_dir=Path.cwd() / dir_name, example_project=choice)


def _input_dir_name():
    while True:
        dir_name = input('Enter the name of the new directory: ')
        if _is_valid_dir_name(dir_name):
            break
        print('Invalid directory name. Please try again: ')
    return dir_name


def _is_valid_dir_name(dir_name: str) -> bool:
    # Check for invalid characters
    invalid_chars = '<>:"/\\|?*'
    if any(char in dir_name for char in invalid_chars):
        return False
    # Check for reserved names (Windows)
    reserved_names = [
        'CON',
        'PRN',
        'AUX',
        'NUL',
        'COM1',
        'COM2',
        'COM3',
        'COM4',
        'COM5',
        'COM6',
        'COM7',
        'COM8',
        'COM9',
        'LPT1',
        'LPT2',
        'LPT3',
        'LPT4',
        'LPT5',
        'LPT6',
        'LPT7',
        'LPT8',
        'LPT9',
    ]
    return dir_name.upper() not in reserved_names


def _clear_screen():
    try:
        subprocess.run(
            ['/usr/bin/bash', '-c', 'clear || cls'],
            check=True,
        )
    except subprocess.CalledProcessError:
        logger.warning('Could not clear the screen. Please do it manually.')


def _print_menu(menu_dict: dict[int, str], title: str = ''):
    print(title)
    print('Please select an option:')
    print_dict(menu_dict)


def _get_choice(menu: list, title: str = '', add_back: bool = True) -> str:
    menu_dict: dict[int, str] = dict(enumerate(menu))
    if add_back:
        menu_dict[BACK_KEY] = BACK_STR
    menu_dict[EXIT_KEY] = EXIT_STR

    _print_menu(menu_dict, title=title)

    key = input('Your choice: ')
    if key == '':
        print('You entered nothing, thus choosing the default option: 0.')
        key = '0'
    while True:
        if key.isdigit() and int(key) in menu_dict:
            return menu_dict[int(key)]
        key = input('Invalid choice, please try again: ')
        if key == '':
            print('You entered nothing, thus choosing the default option: 0.')
            key = '0'


if __name__ == '__main__':
    menu_main()
