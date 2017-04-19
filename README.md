# NLP_QA
The Question-Answer system for team HALT.
Uses the Natural Language Toolkit (Bird, Steven, Edward Loper and Ewan Klein (2009),
Natural Language Processing with Python. O’Reilly Media Inc.) and the BLLIP parser.

Uses the BLLIP WSJ model for the parser (sudo python -m nltk.downloader bllip_wsj_no_aux)

Question Generation portion based on parse tree parsing of sentences, and generating
questions from the sentences whose parse trees make it easy to create good and coherent
questions.

Question Answering portion based on tuple seeding of the verb "to be" (since this makes up many questions),
and fuzzy string matching for other questions, based on a modified Damerau-Levenshtein process (deletion penalized)
to calculate word error rate between the stop word removed question and stop word removed sentences.