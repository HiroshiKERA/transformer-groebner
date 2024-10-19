import argparse

def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Language transfer")

    # path
    parser.add_argument("--data_path", type=str, default="data/")
    parser.add_argument("--load_path", type=str, default="results/")
    parser.add_argument("--save_path", type=str, default="results/")
    parser.add_argument("--correct_file", type=str, default="prediction_correct.txt")
    parser.add_argument("--incorrect_file", type=str, default="prediction_incorrect.txt")

    # setup
    parser.add_argument("--data_encoding", type=str, default="prefix")
    parser.add_argument("--term_order", type=str, default="lex")
    parser.add_argument("--field", type=str, default='QQ', help='QQ or FP with some integer P (e.g., F7).')
    parser.add_argument("--num_variables", type=int, default=2)

    # evaluation parameters
    parser.add_argument("--num_prints", type=int, default=10)


    return parser

