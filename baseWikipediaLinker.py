import re

from wikireader import WikiRegexes, WikipediaReader
from wordvecs import WordVectors, WordTokenizer


def PreProcessedQueries(
        wikipedia_dump_fname,
        vectors,#=wordvectors,
        queries,#=queries,
        redirects,#,=page_redirects,
        surface,#=surface_counts
):

    get_words = re.compile('[^a-zA-Z0-9 ]')
    get_link = re.compile('.*?\[(.*?)\].*?')

    wordvec = WordTokenizer(vectors, sentence_length=200)
    documentvec = WordTokenizer(vectors, sentence_length=1)

    queried_pages = set()
    for docs, q in queries.iteritems():
        wordvec.tokenize(docs)
        for sur, v in q.iteritems():
            wrds_sur = get_words.sub(' ', sur)
            wordvec.tokenize(wrds_sur)
            link_sur = get_link.match(sur).group(1)
            wordvec.tokenize(link_sur)
            for link in v['vals'].keys():
                wrds = get_words.sub(' ', link)
                wordvec.tokenize(wrds)
                tt = WikiRegexes.convertToTitle(link)
                documentvec.get_location(tt)
                queried_pages.add(tt)


    added_pages = set()
    for title in queried_pages:
        if title in redirects:
            #wordvec.tokenize(self.redirects[title])
            documentvec.get_location(redirects[title])
            added_pages.add(redirects[title])
    queried_pages |= added_pages

    for w in queried_pages:
        wordvec.tokenize(get_words.sub(' ', w))

    page_content = {}

    class GetWikipediaWords(WikipediaReader, WikiRegexes):

        def readPage(ss, title, content, namespace):
            if namespace != 0:
                return
            tt = ss.convertToTitle(title)
            if tt in queried_pages:
                cnt = ss._wikiToText(content)
                page_content[tt] = wordvec.tokenize(cnt)

    GetWikipediaWords(wikipedia_dump_fname).read()

    rr = redirects
    rq = queried_pages
    rc = page_content
    rs = surface

    qp = queried_pages
    qq = queries

    class PreProcessedQueriesCls(object):

        wordvecs = wordvec
        documentvecs = documentvec
        queries = qq
        redirects = rr
        queried_pages = rq
        page_content = rc
        surface_counts = rs
        queried_pages = qp


    return PreProcessedQueriesCls
