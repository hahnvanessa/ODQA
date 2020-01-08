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
            parsed_question = json.loads(line)
            question = parsed_question["question"]
            question_id = parsed_question["uid"]
            question_answer = parsed_question["answer"]
            question_dic[question_id] = {"question":question, "answer":question_answer, "contexts":[]}

    # Check whether the question dictionary has key-value pairs
    assert len(question_dic) > 0, "question dictionary is empty"

    # Contexts File and Path
    context_file = set_type + "_contexts.json"
    context_file_path = "\\".join([folder, "contexts", doc_size, context_file, context_file])
    print(context_file_path)
    with open(context_file_path, "r") as cf:
        for line in cf:
            parsed_answer = json.loads(line)
            # Answer ID should have a corresponding question ID
            answer_id = parsed_answer["uid"]
            # List of contexts with retrieval scores, contexts are sorted from highest to lowest score
            answer_contexts = parsed_answer["contexts"]
            question_dic[answer_id]["contexts"] = answer_contexts

        # todo: determine whether it is computationally more efficient to save a list of tuplles instead of a nested list
        # todo: throw an exception if there is no question for a found answer, i.e. the uid is not matching
        # todo: check if the encoding of the contexts is correct, think I saw a "u355" wrongly encoded piece

        # Check if every question has contexts
        for qid, q in question_dic.items():
            assert len(q["contexts"])>0, "Question {} missing context".format(qid)

        return question_dic

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
        # A wrong type should be identified by argparse already but this is another safeguard
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