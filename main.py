
from argparse import ArgumentParser
import BILSTM
import embeddings


def get_file_paths(data_dir):
    # Get paths for all files in the given directory

    file_names = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(args.data):
        for file in f:
            if '.pkl' in file:
                file_names.append(os.path.join(r, file))
    return file_names


if __name__ == '__main__':

    parser = ArgumentParser(
        description='Main ODQA script')
    parser.add_argument(
        'embeddings', help='Path to the pkl file')
    parser.add_argument(
        'data', help='Path to the folder with the pkl files')


    qp_bilstm = BILSTM(embedding_matrix=pickle.load(args.embeddings, 'rb'), embedding_dim=300, hidden_dim=100, batch_size=30)
    interaction_bilstm = BILSTM(embedding_matrix=pickle.load(args.embeddings, 'rb'), embedding_dim=300, hidden_dim=100, batch_size=30)

    file_paths = get_file_paths(args.data)

    qp_representations = {}

    for file in file_paths:
        with open(os.path.join(file, 'r')) as f:
            content = pickle.load(f, 'rb')
            for item in content:
                item_id = item.key()
                question = item[item_id]['question']
                q_representation = qp_bilstm.forward(question) #get the question representation
                contexts = item[item_id]['contexts']
                c_representations = []
                for context in contexts:
                    c_representation = qp_bilstm.forward(context) #get the context representation
                    c_representations.append(c_representation)
                qp_representations[item_id] = {'q_repr': q_representation,
                                                'c_repr': c_representations}
            #TODO: get question and context interaction
