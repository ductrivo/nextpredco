import argparse
import contextlib
import sys

from nextpredco.cli import menu_main
from nextpredco.core.tasks import (
    init_dir,
    simulate_transient,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Welcome to NextPredCo - Your NEXT PREDICTive Control and '
            'Optimization Framework!'
        ),
    )
    subparsers = parser.add_subparsers(dest='command', help='Sub-commands')

    # Subparser for the 'init' command
    init_parser = subparsers.add_parser(
        name='init',
        help='Initialize a new working directory',
    )
    init_parser.add_argument(
        'work_dir',
        type=str,
        nargs='?',
        help='Name of the new directory',
    )

    # Subparser for the 'create-settings' command
    subparsers.add_parser(
        name='create-settings',
        help='Create settings.csv file',
    )

    # Subparser for the 'task' command
    run_parser = subparsers.add_parser('run', help='Perform a task')
    run_parser.add_argument('--task', type=str, help='Task to perform')
    run_parser.add_argument(
        '--args',
        nargs=argparse.REMAINDER,
        help='Arguments for the task',
    )

    args = parser.parse_args()

    if args.command == 'init':
        init_dir(args.work_dir)

    elif args.command == 'create-settings':
        print('Settings file created successfully.')

    elif args.command == 'run':
        if args.task:
            print(f'Running task: {args.run} with arguments: {args.args}')

            # Add logic to handle different tasks
            if args.task == 'sim-transient':
                simulate_transient(args.args)
            else:
                print(f'Unknown task: {args.task}')
                sys.exit(1)
        else:
            run_parser.print_help()
    else:
        # parser.print_help()
        menu_main()


if __name__ == '__main__':
    with contextlib.suppress(ImportError):
        from rich import print  # noqa: A004
        from rich.pretty import install

        install()
    main()
