import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Welcome to NextPredCo - Your NEXT PREDICTive Control and '
            'Optimization Framework!'
        ),
    )
    parser.add_argument('--task', type=str, help='Task to perform')
    parser.add_argument(
        '--args',
        nargs=argparse.REMAINDER,
        help='Arguments for the task',
    )

    args = parser.parse_args()

    if args.task:
        print(f'Running task: {args.task} with arguments: {args.args}')
        # Add logic to handle different tasks
        if args.task == 'example_task':
            example_task(args.args)
        else:
            print(f'Unknown task: {args.task}')
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


def example_task(args):
    print(f'Executing example task with arguments: {args}')
    # Add your task logic here


if __name__ == '__main__':
    main()
