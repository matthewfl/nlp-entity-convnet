import re
import json

import regex


class WikipediaReader(object):

    title_rg = re.compile('.*<title>(.*)</title>.*')
    link_rg = re.compile('\[\[([^\]]*)\]\]')
    redirect_rg = re.compile('.*<redirect title="(.*)" />')
    not_link_match = re.compile('[^a-zA-Z0-9_]')

    def __init__(self, fname):
        self.wikidump_fname = fname

    def read(self):
        current_page = None
        look_for_next_page = True
        page_text = None

        title_rg = self.title_rg

        with open(self.wikidump_fname) as f:
            try:
                while True:
                    line = f.next()
                    if look_for_next_page:
                        if '<page>' not in line:
                            continue
                        else:
                            look_for_next_page = False
                    if '<title>' in line:
                        current_page = title_rg.match(line).group(1)
                    elif '<redirect' in line:
                        redirect_page = self.redirect_rg.match(line).group(1)
                        self.readRedirect(current_page, redirect_page)
                        look_for_next_page = True
                    elif '<text' in line:
                        lines = [ line[line.index('>')+2:] ]
                        if '</text>' in lines[0]:
                            page_text = lines[0][:lines[0].index('</text>')]
                            look_for_next_page = True
                            self.readPage(current_page, page_text)
                        else:
                            while True:
                                line = f.next()
                                if '</text>' in line:
                                    lines.append(line[:line.index('</text>')])
                                    look_for_next_page = True
                                    page_text = '\n'.join(lines)
                                    self.readPage(current_page, page_text)
                                    break
                                else:
                                    lines.append(line)
            except StopIteration as e:
                pass

    @classmethod
    def getLinkTargets(cls, content):
        ret = cls.link_rg.findall(content)
        def s(v):
            a = v.split('|')
            pg = a[0].replace(' ', '_').replace('(', '_lrb_').replace(')', '_rrb_').lower()
            pg = cls.not_link_match.sub('', pg)
            txt = a[-1]
            if '://' not in v:
                return pg, txt
        return [a for a in [s(r) for r in ret] if a is not None]

    def readPage(self, title, content):
        pass

    def readRedirect(self, title, target):
        pass


class WikiRegexes(object):

    redirects = {}

    page_titles = set()

    _wiki_re_pre = [
        (re.compile('&amp;'), '&'),
        (re.compile('&lt;'), '<'),
        (re.compile('&gt;'), '>'),
        (re.compile('<ref.+?<\/ref>'), ''),
        (re.compile('<.*?>'), ''),
        (re.compile('\[http[^\] ]*', re.IGNORECASE), ''),
        (re.compile('[a-zA-Z]+:\/\/[^\] ]+', re.IGNORECASE), ''),
        (re.compile('\|(thumb|left|right|\d+px)', re.IGNORECASE), ''),
        (re.compile('\[\[image:[^\[\]]*\|([^\[\]]*)\]\]', re.IGNORECASE), '\\1'),
        (re.compile('\[\[category:([^\|\]\[]*)[^\]\[]*\]\]', re.IGNORECASE), '[[\\1]]'),  # make category into links
        (re.compile('\[\[[a-z\-]*:[^\]]\]\]'), ''),
        (re.compile('\[\[[^\|\]]*\|'), '[['),
        #(regex.compile('\{((?R)|[^\{\}]*)*\}'), ''),  # this is a recursive regex
        (re.compile('{{[^\{\}]*}}'), ''),
        (re.compile('{{[^\{\}]*}}'), ''),
        (re.compile('{{[^\{\}]*}}'), ''),
        (re.compile('{[^\{\}]*}'), ''),
        (re.compile('{[^\{\}]*}'), ''),
        (re.compile('{[^\{\}]*}'), ''),
    ]

    _wiki_re_post = [
        (re.compile('[\[|\]]'), ''),
        (re.compile('&[^;]*;'), ' '),
        (re.compile('\(|\)'), ''),
        #(re.compile('\)'), '_rrb_'),
        (re.compile('\n+'), ' '),
        (re.compile('[^a-zA-Z0-9_ ]'), ''),
        (re.compile(' \d+ '), ' ### '),  # numbers on their own in text are replaces with ###, maintain numbers in page titles
        (re.compile('\s+'), ' '),
    ]

    _wiki_links_to_text = [
        (re.compile('\[\[([^\[\|\n\]]*)\]\]'), '\\1'),
        (re.compile('\[\[([^\|\]\[\{\}]+?)\|([^\]\[\{\}]*)\]\]'), '\\2'),
    ]

    _wiki_re_all = _wiki_re_pre + _wiki_links_to_text + _wiki_re_post

    _wiki_link_re = [
        re.compile('\[\[([^\|\n\]]*)\]\]'),
        re.compile('\[\[([^\|\]\[\{\}]+?)\|([^\]\[\{\}]*)\]\]'),
    ]

    _wiki_non_title = re.compile('[^a-z0-9_]')

    def _wikiResolveLink(self, match):
        #print match.groups()
        #import ipdb; ipdb.set_trace()
        m = match.group(1)
        if m:
            mg = self.convertToTitle(m)
            tit = self.redirects.get(mg, mg)
            if tit in self.page_titles:
                return tit
            else:
                return match.group(0)
        else:
            return match.group(0)

    @classmethod
    def convertToTitle(cls, tit):
        return cls._wiki_non_title.sub('', tit.replace(' ', '_').replace('(', '_lrb_').replace(')', '_rrb_').lower())

    @classmethod
    def _wikiToText(cls, txt):
        txt = txt.lower()
        for r in cls._wiki_re_all:
            txt = r[0].sub(r[1], txt)
        return txt

    def _wikiToLinks(self, txt):
        txt = txt.lower()
        for r in self._wiki_re_pre:
            txt = r[0].sub(r[1], txt)
        for r in self._wiki_link_re:
            txt = r.sub(self._wikiResolveLink, txt)
        for r in self._wiki_re_post:
            txt = r[0].sub(r[1], txt)
        return txt


class WikipediaW2VParser(WikipediaReader, WikiRegexes):

    def __init__(self, wiki_fname, redirect_fname, output_fname):
        super(WikipediaW2VParser, self).__init__(wiki_fname)
        self.redirect_fname = redirect_fname
        self.output_fname = output_fname
        self.read_pages = False
        self.redirects = {}
        self.page_titles = set()

    def save_redirects(self):
        cnt = 0
        cont_iters = True
        # resolve double or more redirects
        while cnt < 10 and cont_iters:
            cont_iters = False
            cnt += 1
            for k, v in self.redirects.iteritems():
                v2 = self.redirects.get(v)
                if v2:
                    cont_iters = True
                    self.redirects[k] = v2

        with open(self.redirect_fname, 'w+') as f:
            json.dump(self.redirects, f)

    def readRedirect(self, title, target):
        if not self.read_pages:
            self.redirects[self.convertToTitle(title)] = self.convertToTitle(target)

    def readPage(self, title, content):
        if self.read_pages:
            self.save_f.write(self._wikiToLinks(content))
            self.save_f.write('\n')
        else:
            self.page_titles.add(self.convertToTitle(title))

    def run(self):
        # read the reidrects first
        self.read_pages = False
        self.read()
        self.save_redirects()
        self.read_pages = True
        self.save_f = open(self.output_fname, 'w+')
        self.read()
        self.save_f.close()



def main():
    # wikipedia_raw_dump output_redirects output_text
    import sys
    parser = WikipediaW2VParser(sys.argv[-3], sys.argv[-2], sys.argv[-1])
    parser.run()


if __name__ == '__main__':
    main()
