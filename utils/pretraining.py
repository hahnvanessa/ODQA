# to be continued...
'''
questions, contexts, gt_contexts, answers, q_len, c_len, a_len, q_id, common_word_encodings = data
#filter out the ground-truth passages for pre-training
pretrain_contexts = contexts[gt_contexts.nonzero(),:]
pretrain_questions = questions[:pretrain_contexts.shape[0],:]
pretrain_answers = answers[:pretrain_contexts.shape[0],:]
#then use the pretrain versions for pre-training below
'''
