# Command line interaction
from argparse import ArgumentParser
# Json to parse files
import json


def process_searchqa():
    pass

def process_quasar(folder, set_type, doc_size):
    # Question File and Path
    question_dic = {}
    question_file = set_type + "_questions.json"
    question_file_path = "\\".join([folder, "questions", question_file, question_file])
    with open(question_file_path, "r") as qf:
        # Parse each line separate to avoid memory issues
        for line in qf:
            parsed_line = json.loads(line)
            question = parsed_line["question"]
            question_id = parsed_line["uid"]
            question_answer = parsed_line["answer"]
            question_dic[question_id] = {"question":question, "answer":question_answer, "contexts":[]}


    # Contexts File and Path
    context_file = set_type + "_contexts.json"
    context_file_path = "\\".join([folder, "contexts", doc_size, context_file, context_file])
    print(context_file_path)
    with open(context_file_path, "r") as qf:
        pass
        # todo: throw an exception if there is no question for a found answer, i.e. the uid is not matching

    pass

def main(type, folder, set_type, doc_size):
    '''
    :param type of qa-set: either searchqa or quasar
    :param folder: folderpath
    :param set_type: either train, dev, or test
    :return: dictionary of type {question_id : {question:"", category:"", snippets:[]}}
    '''
    print(type, folder, set_type, doc_size)

    if type == "quasar":
        return process_quasar(folder, set_type, doc_size)
    elif type == "searchqa":
        return process_searchqa()
    else:
        return ValueError("type must be either 'quasar' or 'searchqa'")

    # Return None for now
    return None


def save_to_file():
    pass

if __name__ == '__main__':
    parser = ArgumentParser()

    # Specify the arguments the script will take from the command line
    # Type
    parser.add_argument("-t", "--type", required=True, dest="TYPE",
                        help="Specify type of question answer set. either searchqa or quasar", choices=['searchqa', 'quasar'])
                        # Dest specifies how the attribute is referred to by arparse
    # Folderpath
    parser.add_argument("-f", "--folder", required=True, dest="FOLDERPATH",
                        help="Specify source folder")
    # Set
    parser.add_argument("-s", "--set", required=True, dest="SETTYPE", help="specify set: either train, dev or test", choices=['train', 'dev', 'test'])

    # Optional Argument: Size of pseudo documents (only relevant for quasar)
    parser.add_argument("-ds", "--docsize", dest="DOCSIZE", help="specify size of pseudo documents", choices=['long', 'short'], default="short")

    # Return an argparse object by taking the commands from the command line (using sys.argv)
    args = parser.parse_args() # Argparse returns a namespace object

    # Call the main function with with the argparse arguments
    main(type=args.TYPE, folder=args.FOLDERPATH, set_type=args.SETTYPE, doc_size=args.DOCSIZE)

    # Sample call
    #   python preprocessing.py -t "quasar" -f "F:\1QuestionAnswering\quasar\quasar_t\qt" -s "train"