import argparse
import sys
import pathlib
import glob

def split_file(input_file, split_ratios, split_names):
    with open(input_file, mode="r") as f_in:
        lines = f_in.readlines()

        header_lines = []
        if args.has_header:
            header_lines.append(lines[0])
            lines = lines[1:]
        
        total_lines = len(lines)

        def get_ranges_from_ratios(total_lines, split_ratios):
            ratio_sum = sum(split_ratios)
            ratio_part = total_lines // ratio_sum

            split_ranges = []
            prev_end = 0
            for i, ratio in enumerate(split_ratios):
                split_start = prev_end
                
                if i == (len(split_ratios) - 1): # last ratio:
                    split_end = total_lines
                else:
                    split_end = ratio * ratio_part
                
                split_ranges.append((split_start, split_end))
                prev_end = split_end

            return split_ranges

        file_extension = pathlib.Path(input_file).suffix
        for i, split_range in enumerate(get_ranges_from_ratios(total_lines, split_ratios)):
            split_name_parts = [] if split_names is None else [split_names[i]]
            out_file_suffix = "_".join(["split"]+[str(x) for x in split_ratios] + split_name_parts)
            split_start, split_end = split_range
            out_file_path = f"{input_file}.{out_file_suffix}" + file_extension

            with open(out_file_path, mode="w") as f_out:
                f_out.writelines(header_lines + lines[split_start:split_end])

            print(f"Created {out_file_path} with {split_end - split_start+1} items")
            
if __name__ == "__main__":
    """
        # note that the file can be a wildcard that will be extended
        examples/few_shot/scripts/split_file.py --input-file "/private/home/tbmihaylov/data/xlmg/few_shot/story_cloze/translation/spring2016.val.*.tsv"  --split-ratios 20:80 --split-names train:eval --has-header
        examples/few_shot/scripts/split_file.py --input-file "/private/home/tbmihaylov/data/xlmg/few_shot/story_cloze/translation/spring2016.val.en_original.tsv"  --split-ratios 20:80 --split-names train:eval --has-header
    """
    parser = argparse.ArgumentParser(description="Split data set")
    parser.add_argument(
        "-i", 
        "--input-file",
        #require=True,
        default="spring2016.val.es.tsv",
        help="Input file"
    )
    parser.add_argument(
        "--split-ratios",
        #require=True,
        default="20:80",
        help="Split rations"
    )
    parser.add_argument(
        "--split-names",
        #require=True,
        default="train:eval",
        help="split names"
    )
    parser.add_argument(
        "--has-header",
        action="store_true",
        help="If the file has header",
    )
    args = parser.parse_args()

    split_ratios = None
    if args.split_ratios:
        split_fields = args.split_ratios.strip().split(":")
        split_ratios = [int(x) for x in split_fields]

    split_names = None
    if args.split_names is not None:
        split_names = args.split_names.split(":")
        assert len(split_names) == len(split_ratios)

    input_files = glob.glob(args.input_file)
    print(f"{len(input_files)} to split. Ratios: {args.split_ratios}")
    for input_file in input_files:
        print()
        print(f"Splitting {input_file}")
        split_file(input_file, split_ratios, split_names)

        
    
    



        

        
    