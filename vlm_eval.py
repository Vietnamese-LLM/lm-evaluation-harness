# coding: utf-8

# Author: Mingzhe Du
# Date: 2026-01-05
# Description: Evaluate the performance of VLMs.

import lm_eval
import argparse

def main(args):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_args", type=str, required=True)
    parser.add_argument("--tasks", type=str, required=True)
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    main(args)