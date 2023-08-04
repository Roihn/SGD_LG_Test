from src.train_act_limited import train as train_act
from src.train_slot_limited import train as train_slot
from src.train_value_trainer_limited import train as train_value
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--proportion', type=int, default=100)
    parser.add_argument('-m', '--model', type=str, choices=['act', 'slot', 'value'], default='act')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    if args.model == 'act':
        train_act(proportion=args.proportion)
    elif args.model == 'slot':
        train_slot(proportion=args.proportion)
    elif args.model == 'value':
        train_value(proportion=args.proportion)


