import sys

from argparse import ArgumentParser

from runner import run_inference


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--do_paraphrase', default=False, action='store_true')
    parser.add_argument('--do_masking', default=False, action='store_true')
    parser.add_argument('--do_save', default=False, action='store_true')
    parser.add_argument('--do_eval', default=False, action='store_true')

    parser.add_argument('--model', '-m', type=str,
                        help='Encoder model path or type', dest='path_enc', required=True)
    parser.add_argument('--paraphraser', '-p', type=str,
                        help='Path to paraphraser model', dest='path_par',
                        required=('--do_paraphrase' in sys.argv))
    parser.add_argument('--task', '-t', type=str,
                        help='Task to compute', choices=['atis', 'snips', 'clinc150', 'massive'],
                        dest='task', required=True)
    parser.add_argument('--descriptor', '-d', type=str,
                        help='Path to descriptor for task, either intent label description or tokenized intent labels',
                        dest='path_desc', required=True)

    parser.add_argument('--output', '-o', type=str,
                        help='Path to output directory', dest='path_out', default='output')
    parser.add_argument('--device', type=str,
                        help='Device on which to run experiments', dest='device', default='cpu')
    parser.add_argument('--seed', '-s', type=int,
                        help='Seed value for reproducibility', dest='seed', default=1234567890)
    parser.add_argument('--topk', '-tk', type=int,
                        help='Select top k candidate classes for overlap detection', dest='top_k', default=3)

    args = parser.parse_args()
    args = vars(args)

    run_inference(**args)
