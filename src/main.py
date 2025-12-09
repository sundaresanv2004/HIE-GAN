import argparse
from trainer.train import train_phase1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train",
                        help="train | resume (same as train w/resume=True)")
    args = parser.parse_args()

    if args.mode == "train" or args.mode == "resume":
        train_phase1()
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
