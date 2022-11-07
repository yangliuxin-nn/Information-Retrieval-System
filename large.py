import math
import os
import re
import sys
import time
import json
from files_large import porter

"""
    This file is for the large corpus
    There are two options: manual and evaluation
"""

def load_stopwords():
    """
        This method creates and returns the stopwords by loading from stopwords.txt
        :return: a set of stopwords
    """
    stopwords = set()
    # load stopwords and store it as a set
    # the reason for using a set rather than a list is that their efficiency are O(1) and O(N) respectively
    with open('files_large/stopwords.txt', 'r') as f:
        for line in f:
            # store the stopwords
            stopwords.add(line.rstrip())

    return stopwords


def indexing(dirs, stopwords):
    """
        This method does as much preprocessing as possible
        the stored index file is a dictionary with the term as the key,
        a dict containing the bm25 that it contributes to each document it appears in as the value
        :param dirs: documents directory
        :param stopwords: a set of stopwords
    """

    # {docID: length of that doc}
    # key is the docID, and value is the length of that doc
    doc_lengths = {}

    # {docID: {term: frequency of the term in that doc}}
    # key is the docID, and value is the frequency of the term in that doc
    doc_term_freqs = {}

    # {term: stemmed term}
    # key is the word, and value is its stemmed term
    stemmed = {}

    # total lengths of all documents in the corpus
    total_lengths = 0

    # {term: the number of documents that the term occur in}
    # key is the term, and value is the number of documents that the term occur in
    occurrences = {}

    # final dict to be stored as the index file in the json format
    indexes = {}

    p = porter.PorterStemmer()

    for dir_child in dirs:
        dir = os.listdir('documents_large/' + dir_child)
        for file in dir:
            # {term: frequency of the term in that doc}
            # key is the term, and value is the frequency of that term in that doc
            freqs = {}

            with open('documents_large/' + dir_child + '/' + file, 'r', encoding='utf-8') as f:
                doc = f.read()
                # preprocessing
                # remove 's, punctuations between non-digit
                regex = r"'s|(?<!\d)[^\w\s](?!\d)"
                result = re.sub(regex, " ", doc, 0)
                # remove digits, letters and separate them
                # store the seperated words into a list for further preprocessing
                terms = re.compile("[a-z|\']+|[\d.]+", re.I).findall(result)
                # length is the length of the current document
                # the document length is the number of terms that are contained in it
                length = 0

                for term in terms:
                    # preprocessing should remove the punctuation and convert to the lower case
                    term = term.strip(',./():!@#$%^&*=\'-<>?`~|_+[]""').lower()

                    # stopwords removal
                    # Also remove term with only one character because they are very less likely to be searched
                    if term not in stopwords and term != "" and term != "-" and len(term) != 1:
                        # stemming
                        # the reason for using a dictionary to store the stemmed words is for better efficiency
                        if term not in stemmed:
                            stemmed[term] = p.stem(term)
                        term = stemmed[term]

                        # first time I've seen 'term' in the current document
                        if term not in freqs:
                            # store the term into the freqs
                            freqs[term] = 1

                            # calculate the number of documents that the term occurs in
                            # the following condition check is not written in the "not first" condition because
                            # if the term is not the first time seen in the current document,
                            # it must have already updated its occurrence number in the collection,
                            # so there is no need to check again
                            if term in occurrences:
                                occurrences[term] += 1
                            else:
                                occurrences[term] = 1

                        # not the first time I've seen 'term' in the current document
                        else:
                            # update the term in the freqs
                            freqs[term] += 1

                        # the length of the document increases by 1
                        length += 1

                        # draft index will be updated with the bm25 of each individual term later on
                        # now key of indexes is term, and value is a dict
                        # with the key of the docID that that term occurs in and value of 0 (updated later on)
                        if term not in indexes:
                            dic = {file: 0}
                            indexes[term] = dic
                        else:
                            indexes[term][file] = 0  # this will be updated later on

                # record the length of the current document
                doc_lengths[file] = length
                # record the frequencies (freqs is a dict) of each term of the current document
                doc_term_freqs[file] = freqs

                # update the total length for further calculation of average document length: avg_doclen
                total_lengths += length

    # calculate the average document length in the collection
    avg_doc_len = total_lengths / len(doc_lengths)

    # store the indexes as a dictionary
    # indexes: key is the term, value is the a dictionary
    # in the value dictionary, the key is the docID, the value is the bm25 that the term contributes to that document
    removed = []  # used to record words with negative bm25 so it will be removed later on
    for term in indexes:
        # doc_dic is a dictionary with the docID as the key, and bm25 that term contributes to it as the value
        doc_dic = indexes[term]
        for doc in doc_dic:
            percentage = doc_lengths[doc] / avg_doc_len
            # calculate the bm25 that the term contributes to each document it occurs in
            bm25V = bm25_term(doc_term_freqs[doc][term], 1, 0.75, percentage, len(doc_term_freqs),
                             occurrences[term])

            # if its contribution is positive, store it in the indexes
            if bm25V > 0:
                indexes[term][doc] = bm25V
            # if its contribution is negative, it appears in too many documents in the collection
            # it's too common to use, so do not store it in the indexes
            else:
                removed.append(term)
                break
    # remove the word with negative bm25 from indexes
    for i in removed:
        indexes.pop(i)

    # store the index file called large.json
    tf = open("large.json", "w")
    json.dump(indexes, tf)
    tf.close()


# load the documents relevance for the evaluation mode
def load_relevance():
    """
        This method stores the relevance for standard queries evaluation
        The key of the qrels is the query ID
        The value of the qrels is a dictionary, which key is the docID or 'non'
        If the key is the docID, the value stored in that key is relevance
        If the key is "non', the value stored in that key is a set, which contains a set of documents judged to be non-relevant
        The reason to combine dictionary and set is to make 'search' operation more efficient
        :return: a dict of relevance to each query
    """
    qrels = {}

    # The relevance judgment. For judged relevant documents, a higher score means a higher level of relevance.
    # Any document that does not appear in this file has been judged non-relevant, except for bm25.
    with open('files_large/qrels.txt', 'r') as f:
        for line in f:
            # split the qrels into a list
            # [queryID, 0, docID, relevance]
            words = line.strip().split()
            # words[0] is the query ID
            # query ID is the key of the qrels dictionary
            # if the query ID has not been stored
            if words[0] not in qrels:
                # the value of the qrels dictionary is also a dictionary called tmp
                # tmp {docID: relevance}
                # words[2] is the docID
                # words[3] is the relevance
                # if relevance > 0, it is a relevant document;
                # otherwise, mark is as non-relevant to distinguish it from unjudged documents
                if int(words[3]) > 0:
                    tmp = {words[2]: int(words[3])}
                # if it is the first time to see a non-relevant document, construct a set as the value of the key 'non'
                else:
                    # the reason to use set is detect whether a document is non-relevant or unjudged quickly
                    nonSet = set()
                    nonSet.add(words[2])
                    tmp = {'non': nonSet}
            else:
                # tmp {docID: relevance}
                # words[2] is the docID
                # words[3] is the relevance
                # if relevance > 0, it is a relevant document
                # otherwise, mark is as non-relevant to distinguish it from unjudged documents
                if int(words[3]) > 0:
                    tmp[words[2]] = int(words[3])
                # the reason to use set is detect whether a document is non-relevant or unjudged quickly
                else:
                    # if it is the first time to see a non-relevant document, construct a set
                    if 'non' not in tmp:
                        nonSet = set()
                        nonSet.add(words[2])
                        tmp['non'] = nonSet
                    else:
                        tmp['non'].add(words[2])
            qrels[words[0]] = tmp

    return qrels


def bm25_term(f_ij, k, b, percentage, N, n_i):
    """
        calculate the bm25 value that the individual term contributes to each document that it appears in
        :param f_ij: frequency of that term in the document
        :param k: constant 1, used to suit the document collection and the desired behaviour
        :param b: constant 0.75, used to suit the document collection and the desired behaviour
        :param percentage: the length of the document / average document length
        :param N: the total number of documents in the collection
        :param n_i: the total number of documents in the collection that contain that term
        :return: the bm25 that the term contributes to a document
    """
    # f left upper
    f_l_u = f_ij * (1 + k)
    # f left down
    f_l_d = f_ij + k * ((1 - b) + (b * percentage))
    # f left
    f_l = f_l_u / f_l_d

    # f right upper
    f_r_u = N - n_i + 0.5
    # f right down
    f_r_d = n_i + 0.5
    # f right
    f_r = math.log2(f_r_u / f_r_d)

    # a term: f_l * f_r
    return f_l * f_r


def process_query(qry, manual, stopwords):
    """
        This method processes query and returns a list of query terms for manual and evaluation option
        :param qry: query string
        :param manual: Boolean (true for manual, false for evaluation)
        :param stopwords: a set of stopwords
        :return: a list of processed query terms
    """
    # process_query has two modes: manual and evaluation

    # repetitive terms in queries should also be considered
    qry_terms = []

    p = porter.PorterStemmer()

    # preprocessing
    # remove 's, punctuations between non-digit
    regex = r"'s|(?<!\d)[^\w\s](?!\d)"
    result = re.sub(regex, " ", qry, 0)
    # remove digits, letters and separate them
    # store the seperated words into a list for further preprocessing
    terms = re.compile("[a-z|\']+|[\d.]+", re.I).findall(result)

    # if it is the evaluation mode, we need to remove the docID from the query
    # if it is the manual mode, there is no docID in the query, so we do nothing here
    if manual is not True:
        terms = terms[1:]

    for term in terms:
        # preprocessing should remove the punctuation and convert to the lower case
        term = term.strip(',./():!@#$%^&*=\'-<>?`~|_+[]""').lower()

        # stopwords removal
        # Also remove term with only one character because indexes does not contain a term with one character
        if term not in stopwords and term != "" and term != "-" and len(term) != 1:
            # stemming
            term = p.stem(term)
            qry_terms.append(term)
    return qry_terms


def bm25_doc(qry_terms, large_index):
    """
        This method calculates the total bm25 that the query terms contribute to the documents that they appear in
        :param qry_terms: a list of processed query terms returned by process_query(qry, manual, stopwords) method
        :param large_index: the dictionary index file for the small corpus
        :return: bm25 score of that document to a query
    """
    # store all similarities
    # key is the docID, value is the bm25 score
    sims = {}

    # iterate the term in the query
    for term in qry_terms:
        # if the term is contained in the document collection
        if term in large_index:
            # doc_dict is a dictionary
            # its key is the docID, value is the score that term contributes to that document's bm25
            doc_dict = large_index[term]
            # iterate the doc in the doc_dict
            for doc in doc_dict:
                # update the bm25 for that document
                if doc not in sims:
                    sims[doc] = large_index[term][doc]
                else:
                    sims[doc] += large_index[term][doc]

    return sims


def process_standard_queries(stopwords, large_index):
    """
        This method processes standard queries for evaluation mode
        It also uses the process_query method as the manual mode to process a single query
        :param stopwords: a set of stopwords
        :param large_index: the dictionary index file for the small corpus
        :return: results of bm25 in response to standard queries
    """
    # the key of the eva_dict is the query ID the value of the eva_dict is a list,
    # list contains the 'Q0', docID, rank of the document in the results, similarity score, name of the run
    results = {}

    with open('files_large/queries.txt', 'r') as f:
        with open('output.txt', 'w', encoding='utf-8') as fWrite:
            for line in f:
                query = line.strip()

                # query ID
                qry_id = query.split()[0]

                if qry_id not in results:
                    results[qry_id] = []

                # perform the same preprocessing as the documents
                qry_terms = process_query(query, False, stopwords)

                # calculate the bm25 for this standard query
                sims = bm25_doc(qry_terms, large_index)

                '''The number of returned documents is dynamic and changes with different queries because some 
                queries are easy and some queries are difficult. 
                My design is firstly sorting the similarity and store the ranked similarity in the list called tmp.
                If the number of documents to be returned is larger than 30, I restrict it to the top 42 most relevant documents.
                My experiments show that this method improves all evaluation metrics, except for the precision a lot 
                while precision just reduces from 0.5 to 0.41 when I change the fixed number to dynamic number.
                However, I think this decrease in precision can be justified because recall and other metrics increase.
                
                The design reason for the large corpus is that the large corpus has many documents with many words, 
                therefore, the similarity score between different documents is small.
                So I do not use bm25 as the threshold but combines the number 30 and 42 as the threshold. 
                '''
                # rank/sort the similarity in the decreasing order
                tmp = sorted(sims.keys(), key=sims.get, reverse=True)
                # sorted_sims contains documents that are returned for evaluation
                sorted_sims = []
                for i in tmp:
                    sorted_sims.append(i)
                # if the number of documents to be returned for evaluation is larger than 30
                if len(sorted_sims) > 30:
                    # restrict the number to the 42 most relevant documents
                    sorted_sims = sorted_sims[:42]

                # store the ranked similarity score of the 15 most relevant documents for each query
                count = 1

                for doc in sorted_sims:
                    # store in the dictionary
                    results[qry_id].append(['Q0', doc, count, sims[doc], '19206207'])
                    fWrite.write(qry_id + ' Q0 ' + doc + ' ' + str(count) + ' ' + str(sims[doc]) + ' 19206207\n')
                    count += 1

    return results


def manual(stopwords, large_index):
    """
        This method is implemented for manual option
        It continues asking for query until the user inputs "QUIT" and print the 15 most relevant documents
        :param stopwords: a set of stopwords
        :param large_index: the dictionary index file for the small corpus
    """
    active = True  # to avoid break
    while active:
        query = input("Enter query: ")
        if query == "QUIT":
            active = False
        else:
            start_time = time.process_time()

            # perform the same preprocessing as the documents
            qry_terms = process_query(query, True, stopwords)

            # sims (similarity) is a dict containing documents' bm25 in response to the current query
            sims = bm25_doc(qry_terms, large_index)

            # rank/sort the similarity in the decreasing order and restrict it to at most 15 most relevance documents
            sorted_sims = sorted(sims.keys(), key=sims.get, reverse=True)[:15]

            # If nothing matches the query
            if len(sims) == 0:
                print("Sorry, nothing matches the query. Please try another query.")
            # If some documents matches the query
            else:
                print('\nResults for query [{}]'.format(query))
                # print the rank, the document's ID, and the similarity score
                count = 1
                for doc in sorted_sims:
                    print(count, doc, sims[doc])
                    count += 1

            end_time = time.process_time()
            # Manual query takes 0.0 seconds.
            print("Manual query takes {} seconds".format(end_time - start_time))

            print("------------------------------------------------------")


def evaluation(results, qrels, n):
    """
        This method is implemented for evaluation option
        Everytime it executes an output.txt is written
        It prints the precision, recall, P@10, R-precision, bpref and NDCG
        The reason why I combine all evaluation metrics in a single method
        rather than writing them into separated methods is to save the loop times
        so that I do not need to loop every time for each evaluation metric
        :param results: a dictionary with the query ID as the key, and the list containing similarity etc. as the value
        :param qrels: a dictionary containing relevance information of standard queries
        :param n: n is 10 to calculate P@10
    """
    # total precision, recall, P@n, R_Precision, MAP, bpref and NDCG for all queries
    precision_value = 0
    recall_value = 0
    p_at_n_value = 0
    r_pre_value = 0
    map_value = 0
    bpref_value = 0
    NDCG10 = 0

    # the key of the results is the query ID
    # the value of the results is a list, containing the string 'Q0', docID, rank, similarity, and the name of the run
    # iterate all results and perform evaluation
    for j in results:
        # count marks the place whenever a relevant document is found for the current query
        count = 0
        count_p_at_n = 0
        count_r_pre = 0
        p = 0

        # count the number of non-relevant documents
        count_non = 0
        bpref = 0
        # for the large corpus, the key of qrels is either docID or 'non'
        # so the number of relevant documents is the length of qrels dictionary minus 1
        len_qrels = len(qrels[j]) - 1

        # DCG
        dcg = 0
        idcg = 0

        for i in range(0, len(results[j])):
            # doc ID: doc
            doc = results[j][i][1]
            # if relevant
            if doc in qrels[j]:
                count += 1

                # whenever a recall point (at each rank where a relevant document) is found,
                # update p for the calculation of MAP
                # count is the number of relevant documents that we found till that position
                # the denominator (i+1) is rank of the current position
                p += count / (i + 1)

                # bpref: restrict bpref to the range (0, 1)
                if count_non <= len_qrels:
                    bpref = bpref + (1 - count_non / len_qrels)

                # if the rank is 1, ie. i is 0
                if i == 0:
                    dcg = qrels[j][doc]
                # stop when i = 9, because when i = 9, the current rank is 10
                # OR stop when i reaches the end of the returned documents so the loop terminates
                # we can stop here to calculate the NDCG@10 or NDCG@(i+1) if the number of returned documents < 10
                elif i < 10:
                    dcg = dcg + (float(qrels[j][doc]) / math.log2(i + 1))

                # if i < n, also update the values for P@n
                if (i + 1) <= n:
                    count_p_at_n += 1
                # if i < n, also update the values for R-precision
                # Adding the following if condition is to deal with the situation that:
                # the number of returned documents is less than the number of relevant documents
                if (i + 1) <= len(qrels[j]):
                    count_r_pre += 1
            # if it is judged to be non-relevant
            # update the number of non-relevant documents
            elif doc in qrels[j]['non']:
                count_non += 1

        # Precision: number of relevant documents in the retrieved set / number of retrieved documents for the query
        precision_value += count / len(results[j])

        # Recall: the number of relevant documents in the retrieved set / the number of relevant documents for the query
        recall_value += count / len_qrels

        # P@n: the number of relevant documents in the retrieved set / n
        # if the number of returned documents is less than n (which is 10 here), then it is the same as the precision
        if len(results[j]) < n:
            p_at_n_value += count_p_at_n / len(results[j])
        # if the number of returned documents is larger than n (which is 10 here), divide by n
        else:
            p_at_n_value += count_p_at_n / n

        # R_Precision: the number of relevant documents in the retrieved set / n
        # no matter whether the number of returned documents is smaller or larger than the number of relevant documents
        # the denominator should always be the number of relevant documents
        r_pre_value += count_r_pre / len_qrels

        # AP of MAP
        map_value += p / len_qrels

        # bpref
        bpref_value += bpref / len_qrels

        # sort the relevant documents by their relevance for the current query
        # before sorting the relevance of relevant documents, first get rid of non-relevant documents
        qrels[j].pop('non')
        sortedQ = sorted(qrels[j].keys(), key=qrels[j].get, reverse=True)

        # variable "end" marks the upper bound of calculating IDCG
        end = min(len(sortedQ), 10)
        # The following if condition considers some extreme situations, though they may not occur in this corpus
        # if the number of returned documents: p < the number of relevant: q
        # if the number of returned documents: p > the number of relevant: q
        # then the NDCG should be NDCG@p = DCG@p / IDCG@p
        # the reason for comparing (i+1) with "end" rather than i with "end" is that:
        # when the above "for" loop terminates, (i+1) should be the number of returned documents
        if (i+1) < end:
            end = i

        # calculate the IDCG
        # initialize IDCG
        idcg = float(qrels[j][sortedQ[0]])
        for k in range(1, end):
            idcg = idcg + (float(qrels[j][sortedQ[k]]) / math.log2(k + 1))

        # NDCG score
        # (not necessarily at 10: if the number of returned documents is less than 10, it will not be NDCG@10)
        NDCG10 = NDCG10 + dcg / idcg

        """
            The following is used to test my evaluation implementation
            I output my program and compare it with my handwritten calculation 
            to check the correctness of my evaluation implementation
        """
        # if j == '702':
        #     print("count:", count)
        #     print("number of relevant documents:", len_qrels)
        #     print("precision 702:", count / len(results[j]))
        #     print("recall 702:", count / len_qrels)
        #     print("p@10 702:", count_p_at_n / n)
        #     print("R precision 702:", count_r_pre / len_qrels)
        #     print("MAP 702:", p / len_qrels)
        #     print("bpref 702:", bpref / len_qrels)
        #     print("NDCG@10 702:", NDCG10 + dcg / idcg)

    # calculate the average Precision, Recall, P_at_n, R_Precision, MAP, bpref, NDCG score
    # total number of queries is the length of results, which key is the query ID
    # the denominator len(results) is the number of total queries
    avg_precision = precision_value / len(results)
    avg_recall = recall_value / len(results)
    avg_p_at_n = p_at_n_value / len(results)
    avg_r_precision = r_pre_value / len(results)
    avg_map = map_value / len(results)
    avg_bpref = bpref_value / len(results)
    avg_NDCG10 = NDCG10 / len(results)

    print("Precision:".ljust(12), avg_precision)
    print("Recall:".ljust(12), avg_recall)
    print("P@10:".ljust(12), avg_p_at_n)
    print("R-precision:".ljust(12), avg_r_precision)
    print("MAP:".ljust(12), avg_map)
    print("bpref:".ljust(12), avg_bpref)
    print("NDCG:".ljust(12), avg_NDCG10)  # try NDCG@10 but if len(results) < 10, it will be return NDCG@10


options = ['manual', 'evaluation']

if __name__ == '__main__':
    # check whether the command follows the rules
    try:
        # -m
        m = sys.argv[1]
        # Two possible options of execution parameter: 'manual' and 'evaluation'
        option = sys.argv[2]
        # store the stopwords into a set
        stopwords = load_stopwords()
    except IndexError:
        print("Sorry, please follow the format 'python large.py -m [option]'.")
        sys.exit()

    else:
        # further check whether the option is correct; if not, give hints and exit
        if option not in options:
            print("Sorry, You should enter 'manual' or 'evaluation' after '-m'.")
            sys.exit()

        # if there is an index file
        try:
            f = open('large.json', 'r')
        # if there is not an index file, create it at first
        except FileNotFoundError:
            print("Creating index file...")
            # start time
            start_time = time.process_time()

            # read each doc under the document directory
            # do as much preprocessing as possible
            # [so I record the bm25 score that each term contributes to each document it appears in in the index file]
            indexing(os.listdir("documents_large"), stopwords)

            # end time
            end_time = time.process_time()

            # Preprocessing time is 65.4375 seconds
            print("Preprocessing time is {} seconds".format(end_time - start_time))
            print("--------------------------------------------------------")

        print("Loading BM25 index from file, please wait.")

        # load the index at first before performing manual and evaluation mode
        start_time = time.process_time()
        tf = open("large.json", "r")
        large_index = json.load(tf)
        end_time = time.process_time()
        # Loading index takes 3.390625 seconds.
        print("Loading index takes {} seconds".format(end_time - start_time))

        # evaluation mode
        if option == "evaluation":

            # store the relevance of documents for the 'evaluation' option
            qrels = load_relevance()

            # process queries
            start_time = time.process_time()
            # process standard queries and return a ranked dict containing bm25
            results = process_standard_queries(stopwords, large_index)
            end_time = time.process_time()
            # Process standard queries takes 0.15625 second
            print("Processing standard queries takes is {} seconds".format(end_time - start_time))

            print("Evaluation results--------------------------------------")

            # start time
            start_time = time.process_time()

            evaluation(results, qrels, 10)

            # end time
            end_time = time.process_time()

            # Evaluation time is 0.0 seconds
            print("Evaluation time is {} seconds".format(end_time - start_time))

        # manual mode
        if option == 'manual':
            manual(stopwords, large_index)
