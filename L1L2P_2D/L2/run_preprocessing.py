import argparse
from data.preprocess import L2DataPreprocessor

language_name = ''
path_to_load_data = ''
path_to_save_data = ''


class CommandLine:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Description for my parser")
        parser.add_argument("--language_name", help="name of processed language (required)", required=True)
        parser.add_argument("--path_to_load_data", help="folder to load data (required)", required=True)
        parser.add_argument("--path_to_save_data", help="folder to save data", required=False, default="Data/csv/")
        parser.add_argument("--require_debug", help="print debug data", required=False, default="False")

        self.argument = parser.parse_args()


if __name__ == '__main__':
    args = CommandLine().argument
    language_name = args.language_name
    path_to_load_data = args.path_to_load_data
    debug = False

    if args.path_to_save_data:
        path_to_save_data = args.path_to_save_data
    else:
        print('will save to Data/csv/')
    if args.require_debug and args.require_debug == "True":
        debug = True

    l2_preprocessor = L2DataPreprocessor(language_name=language_name,
                                         path_to_load_data=path_to_load_data,
                                         path_to_save_data=path_to_save_data,
                                         print_debug=debug)
    if debug:
        l2_preprocessor.print_missing_data()
    l2_preprocessor.midterm_conclusion()
