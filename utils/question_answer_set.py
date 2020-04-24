import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

MAX_SEQUENCE_LENGTH = 100

class Question_Answer_Set(Dataset):
    '''
    Store and load data for training and testing. This can be used to
    create minibatches. Also handles converting the numpy-encoded
    embeddings into LongTensors.
    '''

    def __init__(self, file_content):
        # some index is reached
        self.set_len = None

        # Tensor storage
        self.questionid_context_answerid = []
        self.common_word_encoding_list = []
        self.questions = {}
        self.answers = {}
        self.gt_contexts = []
        self.gt_spans = []

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
            if idx % 1000 == 0:
                print(idx)
            item_id = item
            question = content[item_id]['encoded_question']
            contexts = content[item_id]['encoded_contexts']
            answer = content[item_id]['encoded_answer']
            # Store questions, contexts and answers as tensors
            for context in contexts:
                # Save context tensor to storage
                self.questionid_context_answerid.append((idx, torch.from_numpy(context).type(torch.LongTensor), idx))
                # Encode common_words and append to storage
                self.common_word_encoding_list.append(self.common_word_encoding(question, context))
                # Check if context contains ground truth answer and return its answer span
                answer_span = self.search_sequence_numpy(context, answer[np.nonzero(answer)])
                self.gt_spans.append(answer_span)
                self.gt_contexts.append(1) if answer_span.shape[0] is 0 else self.gt_contexts.append(0)

            self.questions[idx] = torch.from_numpy(question).type(torch.LongTensor)
            self.answers[idx] = torch.from_numpy(answer).type(torch.LongTensor)
        self.set_len = len(self.questionid_context_answerid)
        print(self.gt_spans)

    def search_sequence_numpy(self, arr, seq):
        """ Find sequence in an array using NumPy only.

        Parameters
        ----------
        arr    : input 1D array
        seq    : input 1D array

        Output
        ------
        Output : 1D Array of indices in the input array that satisfy the
        matching of input sequence in the input array.
        In case of no match, an empty list is returned.
        """
        #Source = https://stackoverflow.com/questions/36522220/searching-a-sequence-in-a-numpy-array/36524045#36524045
        # Store sizes of input array and sequence
        Na, Nseq = arr.size, seq.size

        # Range of sequence
        r_seq = np.arange(Nseq)

        # Answer length
        answer_len = seq.shape[0]

        # Create a 2D array of sliding indices across the entire length of input array.
        # Match up with the input sequence & get the matching starting indices.
        M = (arr[np.arange(Na - Nseq + 1)[:, None] + r_seq] == seq).all(1)


        # Get the range of those indices as final output
        if M.any() > 0:
            answer_start = np.where(np.convolve(M, np.ones((Nseq), dtype=int)) > 0)[0][0] # just find the first exact match
            answer_end = answer_start + (answer_len - 1)
            answer_span = torch.LongTensor([answer_start, answer_end])
            return answer_span
        else:
            return torch.LongTensor([-1, -1]) # todo: find a better way to avoid using these


    def get_question(self, index):
        '''
        Returns both the question as LongTensor and the id of that question. Id can later be used to reference the
        question.
        :param index:
        :return:
        '''
        return self.questions[self.questionid_context_answerid[index][0]], self.questionid_context_answerid[index][0]

    def get_answer(self, index):
        return self.answers[self.questionid_context_answerid[index][2]]

    def get_context(self, index):
        return self.questionid_context_answerid[index][1]

    def determine_length(self, tensor):
        ''' Determines length without padding '''
        return len(tensor[tensor.nonzero()])

    def common_word_encoding(self, question, context):
        '''
        For every word in context returns whether that word also occurs anywhere in
        the question. Returns boolean values as LongTensor..
        :param question:
        :param context:
        :return:
        '''
        # Find the words that also appear in the question
        common_words = np.isin(context, question).astype(int)
        # Convert to LongTensor
        return torch.from_numpy(common_words).type(torch.FloatTensor).view(-1, 1)


    def __getitem__(self, index):
        '''
        Returns a single question, context, answer pair and the corresponding
        lenghts (lengths exclude padding). Also returns the question id and
        the common_word_encoding.
        :param index:
        :return:
        '''
        question, q_id = self.get_question(index)
        context = self.get_context(index)
        answer = self.get_answer(index)
        common_word_encoding = self.common_word_encoding_list[index]
        gt_contexts = self.gt_contexts[index]
        gt_span = self.gt_spans[index]
        return question, context, gt_contexts, answer, self.determine_length(question), self.determine_length(context), self.determine_length(answer), q_id, common_word_encoding, gt_span


    def set_max_len(self, max_len):
        '''
        EXPERIMENTAL
        todo: Delete if necessary
        Manually reduces the maximum number of datapoints the DataLoader can retrieve.
        :param max_len:
        :return:
        '''
        self.set_len = max_len

    def __len__(self):
        return self.set_len
