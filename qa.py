import nltk
import string
from bllipparser import RerankingParser
import re

#Python 3
# download and install a basic unified parsing model (Wall Street Journal)
# sudo python -m nltk.downloader bllip_wsj_no_aux

#In the required Python3, no need to typecast to str
model_dir = nltk.data.find('models/bllip_wsj_no_aux').path
rrp = RerankingParser.from_unified_model_dir(model_dir)

def textToSentences(text):
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	sentences = tokenizer.tokenize(text)
	punc = string.punctuation
	punc = punc.replace('!','').replace('.','').replace(',','').replace('\'','')
	translation_table = str.maketrans({key: None for key in punc})
	first_line = sentences[0].split('\n')[0].strip().split()
	topic = ''
	for word in first_line:
		if not word.startswith('('):
			topic += word + ' '
	topic = topic.strip()
	for i in range(len(sentences)):
		sentences[i] = sentences[i].translate(translation_table)
	first_paragraph = tokenizer.tokenize(text.split('\n\n')[1])
	it_count = 0
	he_count = 0
	she_count = 0
	for i in range(len(first_paragraph)):
		lower_sen = first_paragraph[i].lower()
		it_count += lower_sen.count(' it ') + lower_sen.count(' its ')
		he_count += lower_sen.count(' he ') + lower_sen.count(' his ')
		she_count += lower_sen.count(' she ') + lower_sen.count(' her ')
	max_topic = max(it_count,max(he_count,she_count))
	if it_count == max_topic:
		topic_type = 'it'
	elif he_count == max_topic:
		topic_type = 'he'
	else:
		topic_type = 'she'
	return (topic,topic_type,sentences)

def getTags(sentence):
	return nltk.pos_tag(nltk.word_tokenize(sentence))

def scoredParseToTree(scored_parse):
	return nltk.tree.Tree.fromstring(str(scored_parse.ptb_parse))

def getParseTree(sentence):
	words = []
	tag_map = {}
	for i,(word,tag) in enumerate(getTags(sentence)):
		words.append(word)
		if tag != None:
			tag_map[i] = tag

	try:
		nbest_list = rrp.parse_tagged(words,tag_map)
	except ValueError:
		print(sentence)
		raise ValueError

	for scored_parse in nbest_list:
		#return best parse tree
		return re.sub('\s+', ' ',str(scoredParseToTree(scored_parse))).strip()

class Tree(object):
	def __init__(self):
		self.child_tags = []
		self.children = []
		self.sentence = ''
		self.tag = ''

def makeTreeHelp(parsed,parsed_len,start,tag):
	curr_tree = Tree()
	curr_tree.tag = tag
	leaf = True
	i = start
	while i < parsed_len:
		if parsed[i] == '(':
			leaf = False
			space_ind = parsed.find(' ',i)
			new_tag = parsed[i + 1:space_ind]
			new_start = space_ind + 1
			(child,i) = makeTreeHelp(parsed,parsed_len,new_start,new_tag)
			#child sentence should not be empty
			if not child.sentence[0] in string.punctuation:
				curr_tree.sentence += ' ' + child.sentence
				curr_tree.sentence = curr_tree.sentence.strip()
			else:
				curr_tree.sentence += child.sentence
			curr_tree.child_tags.append((child.tag,child.sentence))
			curr_tree.children.append(child)
		elif parsed[i] == ')':
			if leaf:
				curr_tree.sentence = parsed[start:i]
			return (curr_tree,i + 1)
		else:
			i += 1

	return (curr_tree,i + 1)

def makeTree(parsed):
	parsed = parsed[parsed.index(' ') + 1:][:-1]

	(tree,i) = makeTreeHelp(parsed,len(parsed),0,'S1')
	if len(tree.children) == 1:
		return tree.children[0]
	return tree

def treeFromSentence(sentence):
	parsed = getParseTree(sentence)
	return makeTree(parsed)

def isPerson(np,topic,topic_type):
	if np in topic and topic_type != 'it':
		return True
	np_tags = getTags(np)
	if np_tags[0][1] == 'PRP':
		return True
	else:
		entities = nltk.chunk.ne_chunk(np_tags)
		for entity in entities:
			if isinstance(entity,nltk.tree.Tree) and entity._label == 'PERSON':
				return True
	return False	

def questionNPVP(np,vp,topic,topic_type):
	if isPerson(np,topic,topic_type):
		it_true = not np in topic and not topic in np and len(getTags(np)) < 4
		if topic_type != 'it' or it_true:
			return 'Who ' + vp + '?'
		return 'What ' + vp + '?'
	return None

def questionModify(questions,topic,topic_type):
	for i in range(len(questions)):
		if topic_type == 'it':
			#pattern_1 = re.compile(re.escape(' it '),re.IGNORECASE)
			pattern_2 = re.compile(re.escape(' its '),re.IGNORECASE)
			pattern_3 = re.compile(re.escape(' it?'),re.IGNORECASE)
			questions[i] = pattern_3.sub(' ' + topic + '?',questions[i])
		elif topic_type == 'he':
			#pattern_1 = re.compile(re.escape(' he '),re.IGNORECASE)
			pattern_2 = re.compile(re.escape(' his '),re.IGNORECASE)
			pattern_3 = re.compile(re.escape(' he?'),re.IGNORECASE)
			questions[i] = pattern_3.sub(' ' + topic + '?',questions[i])
		else:
			#pattern_1 = re.compile(re.escape(' she '),re.IGNORECASE)
			pattern_2 = re.compile(re.escape(' her '),re.IGNORECASE)
			pattern_3 = re.compile(re.escape(' she?'),re.IGNORECASE)
			questions[i] = pattern_3.sub(' ' + topic + '?',questions[i])
		#questions[i] = pattern_1.sub(' ' + topic + ' ',questions[i])
		questions[i] = pattern_2.sub(' ' + topic + '\'s ',questions[i])

def generateQuestions(text,n):
	(topic,topic_type,sentences) = textToSentences(text)
	questions = []
	for sentence in sentences:
		t = treeFromSentence(sentence)
		parsed = t.child_tags
		if len(parsed) > 1:
			first = parsed[0]
			second = parsed[1]
			if first[0] == 'NP' and second[0] == 'VP':
				question = questionNPVP(first[1],second[1],topic,topic_type)
				if question != None:
					questions.append(question)
				else:
					parsed_vp = t.children[1].child_tags
					if len(parsed_vp) > 1:
						vp_first = parsed_vp[0]
						vp_second = parsed_vp[1]
						if vp_first[0].startswith('VB') and vp_second[0] == 'NP':
							questions.append('What ' + vp_first[1] + ' ' +
											vp_second[1] + '?')
							questions.append('What ' + vp_first[1] + ' ' +
											first[1] + '?')

		elif len(parsed) > 2:
			first = parsed[0]
			second = parsed[1]
			third = parsed[2]
			if second[0].startswith('VB') and third[0] == 'NP':
				questions.append('What ' + second[1] + ' ' + third[1] + '?')
		if len(questions) == n:
			questionModify(questions,topic,topic_type)
			return questions
	questionModify(questions,topic,topic_type)
	return questions

def questionsFromText(file_in,n):
	f = open(file_in)
	text = f.read()
	f.close()
	questions = generateQuestions(text,n)
	for question in questions:
		print(question)

'''
Function that computes the Damerau-Levenshtein
distance between an input string and the
string it needs to be altered to. Valid
operations are insert, remove, substitute, and
transposition (swapping) between adjacent characters.
The difference between Damerau-Levenshtein and OSA
is that OSA does not allow a substring to be edited
more than once, while Damerau-Levenshtein does not have
such a restriction.
Uses dynamic programming to run in O(nm) time,
where n and m are the lengths of the strings.
From HW 5, modified for the tokens of strings, updated weights
to less penalize inserts (substrings), more penalize deletes.
Calculates WER.
'''
def damerLev(initial_str,final_str):
	if initial_str == final_str:
		return 0
	translation_table = str.maketrans({key: None for key in string.punctuation})
	initial_str = nltk.word_tokenize(initial_str.translate(translation_table))
	initial_str = [word for word in initial_str
					if not word in nltk.corpus.stopwords.words('english')]
	final_str = nltk.word_tokenize(final_str.translate(translation_table))
	final_str = [word for word in final_str
				if not word in nltk.corpus.stopwords.words('english')]
	n = len(initial_str)
	m = len(final_str)
	if n == 0:
		return m
	if m == 0:
		return n * 5

	table = [[j for j in range(m + 1)] if i == 0
			else [i * 5 if k == 0 else 0 for k in range(m + 1)]
			for i in range(n + 1)]

	alpha_map = {}
	for ch in (initial_str + final_str):
		alpha_map[ch] = 0

	i = 1
	while i <= n:
		db = 0
		j = 1
		while j <= m:
			k = alpha_map[final_str[j - 1]]
			l = db
			cost = 7
			if initial_str[i - 1] == final_str[j - 1]:
				cost = 0
				db = j
			table[i][j] = min(table[i][j - 1] + 1, #insert
							min(table[i - 1][j] + 7, #remove
								table[i - 1][j - 1] + cost)) #substitute
			if k > 0 and l > 0:
				trans = table[k - 1][l - 1] + (i - k - 1) + (j - l - 1) + 1
				table[i][j] = min(table[i][j],trans) #transpose

			j += 1
		alpha_map[initial_str[i - 1]] = i
		i += 1

	return table[n][m]

def answerModify(answer,topic,topic_type):
	if topic_type == 'it':
		pattern_1 = re.compile(re.escape(' it '),re.IGNORECASE)
		pattern_2 = re.compile(re.escape(' its '),re.IGNORECASE)
	elif topic_type == 'he':
		pattern_1 = re.compile(re.escape(' he '),re.IGNORECASE)
		pattern_2 = re.compile(re.escape(' his '),re.IGNORECASE)
	else:
		pattern_1 = re.compile(re.escape(' she '),re.IGNORECASE)
		pattern_2 = re.compile(re.escape(' her '),re.IGNORECASE)

	answer = pattern_1.sub(' ' + topic + ' ',answer)
	return pattern_2.sub(' ' + topic + '\'s ',answer)

def answerQuestion(question,text):
	(topic,topic_type,sentences) = textToSentences(text)
	#Currently just supports fuzzy matching, Who and What
	t = treeFromSentence(question)
	if t.child_tags[0][0] == 'WHNP':
		sq = t.child_tags[1][1]
		best_response = None
		best_wer = None
		for sentence in sentences:
			t = treeFromSentence(sentence)
			parsed = t.child_tags
			if len(parsed) > 1:
				first = parsed[0]
				second = parsed[1]
				if first[0] == 'NP' and second[0] == 'VP':
					wer = damerLev(sq,second[1])
					if best_wer == None or wer < best_wer:
						best_response = first[1]
						best_wer = wer
					wer = damerLev(sq,first[1])
					if best_wer == None or wer < best_wer:
						best_response = second[1]
						best_wer = wer
		return answerModify(' ' + best_response + ' ',topic,topic_type).strip()

	#Need to do a non Who and What case, currently return topic
	return topic

def answerQuestions(file_questions,file_in):
	f = open(file_in)
	text = f.read()
	f.close()
	f = open(file_questions)
	questions = f.readlines()
	f.close()
	for question in questions:
		question = question.strip()
		if len(question) > 0:
			print(answerQuestion(question,text))

'''
def writeRuleHelp(preterminal,t,rules):
	if len(t.children) == 0:
		rhs = ' -> ' + t.sentence
		if not t.tag in rules:
			rules[t.tag] = {rhs:1}
		elif not rhs in rules[t.tag]:
			rules[t.tag][rhs] = 1
		else:
			rules[t.tag][rhs] += 1
		return
	rhs = ' -> ' + ' '.join([tag[0] for tag in t.child_tags])
	if not preterminal in rules:
		rules[preterminal] = {rhs:1}
	elif not rhs in rules[preterminal]:
		rules[preterminal][rhs] = 1
	else:
		rules[preterminal][rhs] += 1
	for child in t.children:
		writeRuleHelp(child.tag,child,rules)

def writeRules(file_in,file_out):
	f = open(file_in)
	arr = f.readlines()
	f.close()

	rules = {'ROOT': {}}
	for i in range(len(arr)):
		rules['ROOT'][' -> S1'] = 1
		writeRuleHelp('S1',treeFromSentence(arr[i]),rules)
	f = open(file_out,'w+')
	for key in rules.keys():
		for rhs in rules[key].keys():
			count = rules[key][rhs]
			f.write(key + rhs + ' ' + str(count) + '\n')
	f.close()
'''
			