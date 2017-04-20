import nltk
import string
from bllipparser import RerankingParser
import re

'''
Python 3

download and install a basic unified parsing model (Wall Street Journal) via:
sudo python -m nltk.downloader bllip_wsj_no_aux
'''

'''
Global variables created on startup of the file
'''
model_dir = nltk.data.find('models/bllip_wsj_no_aux').path
rrp = RerankingParser.from_unified_model_dir(model_dir)


'''
Functions for the Question-Answering System
'''

'''
Turns input article text to sentences, and extracts the main
topic of the article as well as the type of topic (male/female/object).
'''
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
		if '\n' in sentences[i]:
			sentences[i] = sentences[i].split('\n')[1].strip()
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
	return (topic,topic_type,list(filter(None,sentences)))

'''
POS tagging.
'''
def getTags(sentence):
	return nltk.pos_tag(nltk.word_tokenize(sentence))

'''
From a scored parse object, returns an NLTK Tree.
'''
def scoredParseToTree(scored_parse):
	return nltk.tree.Tree.fromstring(str(scored_parse.ptb_parse))

'''
Obtain the best parse tree for a sentence, string representation.
'''
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

'''
Tree object representing a parse tree. Easier access of parse tags.
'''
class Tree(object):
	def __init__(self):
		self.child_tags = []
		self.children = []
		self.sentence = ''
		self.tag = ''

'''
Recursive helper function to make a Tree object from the
string representation of the parse tree (DFS)
'''
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

'''
Makes a Tree object from a string representation of the parse tree.
'''
def makeTree(parsed):
	parsed = parsed[parsed.index(' ') + 1:][:-1]

	(tree,i) = makeTreeHelp(parsed,len(parsed),0,'S1')
	if len(tree.children) == 1:
		return tree.children[0]
	return tree

'''
Obtains the parse tree of a sentence.
'''
def treeFromSentence(sentence):
	parsed = getParseTree(sentence)
	return makeTree(parsed)

'''
Detects if the Noun Phrase is referring to a person.
'''
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

'''
Creates simple Who/What questions from NP-VP sentences.
'''
def questionNPVP(np,vp,topic,topic_type):
	if isPerson(np,topic,topic_type):
		it_true = not np in topic and not topic in np and len(getTags(np)) < 4
		if topic_type != 'it' or it_true:
			return 'Who ' + vp + '?'
		return 'What ' + vp + '?'
	return None

'''
Modifies question so that any mentions of generic pronouns
maps to the topic of the article. Only modifies if the topic
is not already mentioned in the question
(which is the highest chance pronoun refers to the topic).
'''
def questionModify(questions,topic,topic_type):
	for i in range(len(questions)):
		if not topic in questions[i]:
			if topic_type == 'it':
				pattern_1 = re.compile(re.escape(' its '),re.IGNORECASE)
				pattern_2 = re.compile(re.escape(' it?'),re.IGNORECASE)
			elif topic_type == 'he':
				pattern_1 = re.compile(re.escape(' his '),re.IGNORECASE)
				pattern_2 = re.compile(re.escape(' he?'),re.IGNORECASE)
			else:
				pattern_1 = re.compile(re.escape(' her '),re.IGNORECASE)
				pattern_2 = re.compile(re.escape(' she?'),re.IGNORECASE)
			questions[i] = pattern_2.sub(' ' + topic + '?',questions[i])
			questions[i] = pattern_1.sub(' ' + topic + '\'s ',questions[i])

'''
Main function to generate questions from text, using
parsing of sentences into parse trees and finding
sentences that can be turned into good questions.
'''
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

'''
Generates questions based on an input article.
'''
def questionsFromText(file_in,n):
	f = open(file_in)
	text = f.read()
	f.close()
	questions = generateQuestions(text,n)
	for question in questions:
		print(question)

'''
Removes puntuation and stop words from a sentence.
'''
def removePuncAndStop(s,translation_table):
	s = nltk.word_tokenize(s.translate(translation_table))
	return [word for word in s
			if not word in nltk.corpus.stopwords.words('english')]

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
	n = len(initial_str)
	m = len(final_str)
	if n == 0:
		return m
	if m == 0:
		return n * 6

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

'''
Most likely, when the answer contains the generalized pronoun
and no reference to the topic entity, the generalized pronoun
will represent the topic entity.
'''
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

'''
Checks if a sentence contains a GPE
'''
def containsLocation(sentence):
	for t in nltk.chunk.ne_chunk(getTags(sentence)):
		if isinstance(t,nltk.tree.Tree) and t._label == 'GPE':
			return True
	return False

'''
Function to answer a question based on the article text.
'''
def answerQuestion(question,text):
	(topic,topic_type,sentences) = textToSentences(text)
	translation_table = str.maketrans({key: None for key in string.punctuation})

	#Common cases of to be verb tuple seeding created questions
	if (question.startswith('What is ') or question.startswith('what is ') or
		question.startswith('Who is ') or question.startswith('who is ')):
		starter = removePuncAndStop(question,translation_table)[1:]
		for sentence in sentences:
			if ' is ' in sentence:
				start = removePuncAndStop(sentence.split(' is ')[0],translation_table)
				answer = True
				for word in starter:
					if not word in start:
						answer = False
						break
				if answer:
					return sentence
	elif (question.startswith('What are ') or question.startswith('what are ') or
			question.startswith('Who are ') or question.startswith('who are ')):
		starter = removePuncAndStop(question,translation_table)[1:]
		for sentence in sentences:
			if ' are ' in sentence:
				start = removePuncAndStop(sentence.split(' are ')[0],translation_table)
				answer = True
				for word in starter:
					if not word in start:
						answer = False
						break
				if answer:
					return sentence
	elif question.startswith('Where ') or question.startswith('where '):
		starter = removePuncAndStop(question,translation_table)[1:]
		best_response = None
		best_count = 0
		for sentence in sentences:
			if containsLocation(sentence):
				sentence_list = removePuncAndStop(sentence,translation_table)
				temp_count = 0
				for word in starter:
					if word in sentence_list:
						temp_count += 1
				if temp_count > best_count:
					best_count = temp_count
					best_response = sentence
		if best_response != None:
			return best_response

	#Fuzzy matching based solution using modified WER (deletion penalized)
	question = removePuncAndStop(question,translation_table)[1:]
	best_response = None
	best_wer = None
	for sentence in sentences:
		#Checks for non-trivial sentence
		if len(sentence) > 2:
			wer = damerLev(question,removePuncAndStop(sentence,translation_table))
			if best_wer == None or wer < best_wer:
				best_response = sentence
				best_wer = wer
	if best_response == None:
		return topic
	modify_answer = True
	for word in topic.split():
		if word in best_response:
			modify_answer = False
			break
	if modify_answer:
		return answerModify(' ' + best_response + ' ',topic,topic_type).strip()
	return best_response

'''
Answers questions from the questions file based
on the text in the article file.
'''
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
			