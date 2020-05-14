import torch


def remove_data(batch, remove_passages='no_ground_truth'): 
	'''
	Removes passages that either do not contain the ground truth or consist of padding only.
	'''
	questions, contexts, gt_contexts, answers, q_len, c_len, a_len, q_id, common_word_encodings, gt_spans = batch
	if remove_passages=='no_ground_truth': # use this for first part of pretraining
		selector = gt_contexts
	elif remove_passages=='empty': # use this for every other part
		selector = c_len
	else:
		raise ValueError('Respecify argument to remove data')
	cleaned_contexts = torch.squeeze(contexts[selector.nonzero(),:])
	cleaned_questions =  torch.squeeze(questions[selector.nonzero(),:])
	cleaned_gt_contexts = torch.squeeze(gt_contexts[selector.nonzero()])
	cleaned_answers =  torch.squeeze(answers[selector.nonzero(),:])
	cleaned_q_len = torch.squeeze(q_len[selector.nonzero()])
	cleaned_a_len = torch.squeeze(a_len[selector.nonzero()])
	cleaned_c_len = torch.squeeze(c_len[selector.nonzero()])
	cleaned_q_id =  torch.squeeze(q_id[selector.nonzero()])
	cleaned_common_word_encodings = torch.squeeze(common_word_encodings[selector.nonzero(),:])# .unsqueeze(2)
	if cleaned_common_word_encodings.dim() > 1:
		cleaned_common_word_encodings = cleaned_common_word_encodings.unsqueeze(2)
	cleaned_gt_spans = torch.squeeze(gt_spans[selector.nonzero(),:])
  
	if cleaned_q_len.dim() == 0:
			cleaned_contexts = cleaned_contexts.view(1,-1)
			cleaned_questions = cleaned_questions.view(1,-1)
			cleaned_gt_contexts = cleaned_gt_contexts.view(1,-1)
			cleaned_answers = cleaned_answers.view(1,-1) 
			cleaned_q_len = cleaned_q_len.view(-1)
			cleaned_a_len = cleaned_a_len.view(-1) 
			cleaned_c_len = cleaned_c_len.view(-1)
			cleaned_q_id = cleaned_q_id.view(-1)
			cleaned_common_word_encodings = cleaned_common_word_encodings.view(1,-1)
			if cleaned_common_word_encodings.dim() > 1:
				cleaned_common_word_encodings = cleaned_common_word_encodings.unsqueeze(2)
			cleaned_gt_spans = cleaned_gt_spans.view(1,-1)
	return cleaned_questions, cleaned_contexts, cleaned_gt_contexts, cleaned_answers, cleaned_q_len, cleaned_c_len, cleaned_a_len, cleaned_q_id, cleaned_common_word_encodings, cleaned_gt_spans


def pretrain_candidate_scoring(model, dataset, MAX_SEQUENCE_LENGTH):
	questions, contexts, gt_contexts, answers, q_len, c_len, a_len, q_id, common_word_encodings, gt_spans = dataset
	questions = questions.to(model.device)
	contexts = contexts.to(model.device)
	gt_contexts = gt_contexts.to(model.device)   
	answers = answers.to(model.device)
	q_len = q_len.to(model.device)
	c_len = c_len.to(model.device)
	a_len = a_len.to(model.device)
	q_id  = q_id.to(model.device)
	common_word_encodings = common_word_encodings.to(model.device)
	gt_spans = gt_spans.to(model.device)

<<<<<<< HEAD
	C_spans, k_max_list = model.extract_candidates(questions, contexts, q_len, c_len, k=MAX_SEQUENCE_LENGTH*MAX_SEQUENCE_LENGTH, pretraining=False)
	#_, max_idx = k_max_list.max(1) # return indicies of max probabilities
	#max_spans = C_spans[torch.arange(C_spans.shape[0]).unsqueeze(-1), max_idx] # get spans with max probability (https://stackoverflow.com/questions/55628014/indexing-a-3d-tensor-using-a-2d-tensor)
	sgt_span_idxs = []
=======
	C_spans, k_max_list = model.extract_candidates(questions, contexts, q_len, c_len, k=MAX_SEQUENCE_LENGTH*MAX_SEQUENCE_LENGTH, pretraining=True)
	gt_span_idxs = []
>>>>>>> 14783b0ed65ec871ded3d917aa76e1d7da95a182
	for i, gt_span in enumerate(gt_spans):
		gt_span_idx = torch.where((C_spans[i]==gt_span).all(dim=1)) # find ground truth index in the spans
		gt_span_idxs.append(gt_span_idx)
	gt_span_idxs = torch.LongTensor(gt_span_idxs).view(-1).cuda()
	k_max_list = k_max_list.view(k_max_list.shape[0],k_max_list.shape[1])

	return k_max_list, gt_span_idxs


