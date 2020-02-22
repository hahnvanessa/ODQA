import torch
from torch.utils.data import Dataset, DataLoader

MAX_SEQUENCE_LENGTH = 100

class Question_Answer_Set(Dataset):
    '''
    Store and load data for training and testing. This can be used to
    create minibatches. Also handles converting the numpy-encoded
    embeddings to LongTensors.
    '''

    def __init__(self, file_content):
        # todo: implement a max size parameter to allow for mini-sets, it should stop reading in content when
        # some index is reached
        self.set_len = None

        # Tensor storage
        self.questionid_context_answerid = []
        self.questions = {}
        self.answers = {}

        # Read in Dataset
        self.read_in_dataset(file_content)


    def read_in_dataset(self, content):
        '''
        :param
        :return:
        '''
        self.set_len = None

        # Read q,c,a and convert to torch tensors
        for idx, item in enumerate(content):
            item_id = item
            question = content[item_id]['encoded_question']
            contexts = content[item_id]['encoded_contexts']
            answer = content[item_id]['encoded_answer']
            # Store questions, contexts and answers as tensors
            for context in contexts:
                self.questionid_context_answerid.append((idx, torch.from_numpy(context).type(torch.LongTensor), idx))
            self.questions[idx] = torch.from_numpy(question).type(torch.LongTensor)
            self.answers[idx] = torch.from_numpy(answer).type(torch.LongTensor)
        self.set_len = len(self.questionid_context_answerid)

    def get_question(self, index):
        return self.questions[self.questionid_context_answerid[index][0]]

    def get_answer(self, index):
        return self.answers[self.questionid_context_answerid[index][2]]

    def get_context(self, index):
        return self.questionid_context_answerid[index][1]

    def determine_length(self, tensor):
        ''' Determines length without padding '''
        return len(tensor[tensor.nonzero()])

    def __getitem__(self, index):
        '''
        Returns a single question, context, answer pair and the corresponding
        lenghts (lengths exclude padding)
        :param index:
        :return:
        '''
        question = self.get_question(index)
        context = self.get_context(index)
        answer = self.get_answer(index)
        return question, context, answer, self.determine_length(question), self.determine_length(context), self.determine_length(answer)


    def __len__(self):
        return self.set_len