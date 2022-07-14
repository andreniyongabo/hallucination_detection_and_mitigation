import argparse
import json
import sys

def main():
    parser = argparse.ArgumentParser(description="Convert TriviaQA and WebQuestions to NQ-Open format")
    parser.add_argument("format", choices=["triviaqa", "webquestions"])
    parser.add_argument("-i", "--input", default=sys.stdin.fileno(), help="the original dataset (defaults to stdin)")
    parser.add_argument("-o", "--output", default=sys.stdout.fileno(), help="the output dataset (defaults to stdout)")
    parser.add_argument("--encoding", default="utf-8", help="the character encoding for input/output (defaults to utf-8)")
    args = parser.parse_args()

    data = []
    with open(args.input, encoding=args.encoding) as f:
        orig_data = json.load(f)
        if args.format == "triviaqa":
            for sample in orig_data["Data"]:
                question = sample["Question"]
                answer = sample["Answer"]["Aliases"]
                data.append({"question": question, "answer": answer})
        elif args.format == "webquestions":
            for sample in orig_data:
                question = sample["qText"]
                answer = sample["answers"]
                data.append({"question": question, "answer": answer})


    with open(args.output, mode="w", encoding=args.encoding) as f:
        for sample in data:
            print(json.dumps(sample), file=f)

if __name__ == "__main__":
    main()
