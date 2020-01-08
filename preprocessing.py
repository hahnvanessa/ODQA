from argparse import ArgumentParser




def main(type, folder, set_type, doc_size):
    '''
    :param type of qa-set: either searchqa or quasar
    :param folder: folderpath
    :param set_type: either train, dev, or test
    :return: dictionary of type {question_id : {question:"", category:"", snippets:[]}}
    '''
    print(type, folder, set_type, doc_size)
    return None


def save_to_file():
    pass

if __name__ == '__main__':
    parser = ArgumentParser()
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
    args = parser.parse_args()

    # Argparse returns a namespace object
    print(args)
    main(type=args.TYPE, folder=args.FOLDERPATH, set_type=args.SETTYPE, doc_size=args.DOCSIZE)