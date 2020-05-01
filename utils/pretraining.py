import torch

def select_pretrain_data(batch): 
	questions, contexts, gt_contexts, answers, q_len, c_len, a_len, q_id, common_word_encodings, gt_spans = batch
	pretrain_contexts = torch.squeeze(contexts[gt_contexts.nonzero(),:])
	pretrain_questions =  torch.squeeze(questions[gt_contexts.nonzero(),:])
	pretrain_gt_contexts = torch.squeeze(gt_contexts[gt_contexts.nonzero()])
	pretrain_answers =  torch.squeeze(answers[gt_contexts.nonzero(),:])
	pretrain_q_len = torch.squeeze(q_len[gt_contexts.nonzero()])
	pretrain_a_len = torch.squeeze(a_len[gt_contexts.nonzero()])
	pretrain_c_len = torch.squeeze(c_len[gt_contexts.nonzero()])
	pretrain_q_id =  torch.squeeze(q_id[gt_contexts.nonzero()])
	pretrain_common_word_encodings = torch.squeeze(contexts[gt_contexts.nonzero(),:])
	pretrain_gt_spans = torch.squeeze(gt_spans[gt_contexts.nonzero(),:])
	if pretrain_q_len.dim() == 0:
            pretrain_contexts = pretrain_contexts.view(1,-1)
            pretrain_questions = pretrain_questions.view(1,-1)
            pretrain_gt_contexts = pretrain_gt_contexts.view(1,-1)
            pretrain_answers = pretrain_answers.view(1,-1) 
            pretrain_q_len = pretrain_q_len.view(-1)
            pretrain_a_len = pretrain_a_len.view(-1) 
            pretrain_c_len = pretrain_c_len.view(-1)
            pretrain_q_id = pretrain_q_id.view(-1)
            pretrain_common_word_encodings = pretrain_common_word_encodings.view(1,-1)
            pretrain_gt_spans = pretrain_gt_spans.view(1,-1)
	return pretrain_questions, pretrain_contexts, pretrain_gt_contexts, pretrain_answers, pretrain_q_len, pretrain_c_len, pretrain_a_len, pretrain_q_id, pretrain_common_word_encodings, pretrain_gt_spans

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

	C_spans, k_max_list = model.extract_candidates(questions, contexts, q_len, c_len, k=MAX_SEQUENCE_LENGTH*MAX_SEQUENCE_LENGTH)
	#_, max_idx = k_max_list.max(1) # return indicies of max probabilities
	#max_spans = C_spans[torch.arange(C_spans.shape[0]).unsqueeze(-1), max_idx] # get spans with max probability (https://stackoverflow.com/questions/55628014/indexing-a-3d-tensor-using-a-2d-tensor)
	gt_span_idxs = []
	for i, gt_span in enumerate(gt_spans):
		gt_span_idx = torch.where((C_spans[i]==gt_span).all(dim=1)) # find ground truth index in the spans
		gt_span_idxs.append(gt_span_idx)
	gt_span_idxs = torch.LongTensor(gt_span_idxs).view(-1)
	k_max_list = k_max_list.view(k_max_list.shape[0],k_max_list.shape[1])
	# So the input to the CrossEntropyLoss would be 
	# k_max_list : where each row is a context and a columns contain probabilities of candidates
	# gd_span_idxs: a tensor with ground truth span indicies (classes)
	#S_p = model.compute_passage_representation(questions, contexts, common_word_encodings, q_len=q_len, c_len=c_len)
	return k_max_list, gt_span_idxs

def pretrain_answer_selection(dataset):
	pass

